import os
import numpy as np
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

from .GroundedSAM2.sam2.build_sam import build_sam2_video_predictor, build_sam2
from .GroundedSAM2.sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from .GroundedSAM2.utils.track_utils import sample_points_from_masks
from .GroundedSAM2.utils.video_utils import create_video_from_images

import pdb

class Segmentation_wrapper(nn.Module):
    def __init__(self, sam_ckpt_path):
        super().__init__()
        
        # init sam image predictor and video predictor model
        # sam2_checkpoint = "/workspace/workspace/checkpoints/sam2/sam2.1_hiera_base_plus.pt"
        # model_cfg = "sam2_hiera_b+.yaml"
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        
        try:
            self.video_predictor = build_sam2_video_predictor(model_cfg, sam_ckpt_path)
        except:
            sam_ckpt_path: "/mnt/workspace/workspace/checkpoints/sam2/sam2.1_hiera_base_plus.pt"
            self.video_predictor = build_sam2_video_predictor(model_cfg, sam_ckpt_path)
            
        sam2_image_model = build_sam2(model_cfg, sam_ckpt_path)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)
        
        # init grounding dino model from huggingface
        model_id = "IDEA-Research/grounding-dino-tiny"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        
        self.text = "people <and> moving object."
    
    def forward(self, inference_type, condition_image, pixel_values=None):
        # SAM2 uses bfloat16 as a default dtype
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            if inference_type=='video':
                return self.get_mask_from_video(condition_image, pixel_values)
            elif inference_type=='image':
                return self.get_mask_from_image(condition_image)
            else:
                raise NotImplementedError("Set inference_type between ['video', 'image']")
        
    @torch.no_grad()
    def get_mask_from_video(self, condition_image, pixel_values):
        """
            Used in Training time.
            
            Args:
            - condition_image: torch.Tensor, bchw, [-1,1]
            - pixel_values: torch.Tensor, (bf)chw, [-1,1]
            Returns:
            - Mask of static region in pixel_values   
        """
        
        v = pixel_values.shape[0] // condition_image.shape[0]
        input_images = condition_image.unsqueeze(1)
        video_frames = rearrange(pixel_values, "(b v) c h w -> b v c h w", v=v)
        device = condition_image.device
        
        static_mask_list = []
        sam_feature_list = []
        for image, frames in zip(input_images, video_frames):
            # init video predictor state
            inference_state = self.video_predictor.init_state_pt(frames=frames)
            
            ann_frame_idx = 0  # the frame index we interact with
            ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
            
            """
            Prompt Grounding DINO and SAM image predictor to get the box and mask for specific frame
            """
            image_pil = Image.fromarray(((image[0]+1)/2*255).permute(1,2,0).cpu().numpy().astype(np.uint8))
            inputs = self.processor(images=image_pil, text=self.text, return_tensors="pt").to(device)
            outputs = self.grounding_model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.25,
                text_threshold=0.3,
                target_sizes=[image_pil.size[::-1]]
            )
            
            # prompt SAM image predictor to get the mask for the object
            self.image_predictor.set_image(np.array(image_pil.convert("RGB")))
            sam_feature = F.interpolate(self.image_predictor._features['image_embed'].detach().clone(), size=image.shape[2:], mode='bicubic')
            sam_feature_list.append(sam_feature)
            
            # process the detection results
            input_boxes = results[0]["boxes"].cpu().numpy()
            OBJECTS = results[0]["labels"]
            
            if len(input_boxes) == 0:
                # There are no objects to be detected.
                v, _, h, w = frames.shape
                static_mask = torch.ones(v, 1, h, w).to(frames.device, frames.dtype)
                static_mask_list.append(static_mask)
                continue
            
            # prompt SAM 2 image predictor to get the mask for the object
            masks, scores, logits = self.image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
                )

            # convert the mask shape to (n, H, W)
            if masks.ndim == 3:
                masks = masks[None]
                scores = scores[None]
                logits = logits[None]
            elif masks.ndim == 4:
                masks = masks.squeeze(1)
            
            """
            Register each object's positive points to video predictor with seperate add_new_points call
            """
            # Using box prompt
            PROMPT_TYPE_FOR_VIDEO = "box"
            for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    box=box,
                )
            
            """
            Propagate the video predictor to get the segmentation results for each frame
            """
            video_segments = {}  # video_segments contains the per-frame segmentation results
            object_mask_list = []
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
                mask_list = []
                for i in range(len(out_obj_ids)):
                    mask_list.append(out_mask_logits[i] > 0.0)
                object_mask_list.append(torch.sum(torch.stack(mask_list, dim=0), dim=0))
                
            object_mask = torch.stack(object_mask_list, dim=0)
            object_mask[object_mask>=1] = 1
            object_mask = object_mask.to(torch.float32)
            
            static_mask_list.append(1-object_mask)
            
            # torchvision.utils.save_image(torchvision.utils.make_grid((frames+1)/2, nrow=14), "image.png")
            # torchvision.utils.save_image(torchvision.utils.make_grid((frames+1)/2 * object_mask, nrow=14), "masked_image.png")
            # pdb.set_trace()
        
        static_mask = torch.cat(static_mask_list, dim=0)
        sam_features = torch.cat(sam_feature_list, dim=0)
        
        return static_mask, sam_features # (bf)chw

    @torch.no_grad()
    def get_mask_from_image(self, condition_image):
        """
            Used in Inference time.
            
            Args:
            - condition_image: torch.Tensor, bchw, [-1,1]
            Returns:
            - Mask of static region in condition_image   
        """
        
        # v = pixel_values.shape[0] // condition_image.shape[0]
        input_images = condition_image.unsqueeze(1)
        
        device = condition_image.device
        
        static_mask_list = []
        sam_feature_list = []
        for image in input_images:
            """
            Prompt Grounding DINO and SAM image predictor to get the box and mask for specific frame
            """
            image_pil = Image.fromarray(((image[0]+1)/2*255).permute(1,2,0).cpu().numpy().astype(np.uint8))
            inputs = self.processor(images=image_pil, text=self.text, return_tensors="pt").to(device)
            outputs = self.grounding_model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.25,
                text_threshold=0.3,
                target_sizes=[image_pil.size[::-1]]
            )
            
            # prompt SAM image predictor to get the mask for the object
            self.image_predictor.set_image(np.array(image_pil.convert("RGB")))
            sam_feature = F.interpolate(self.image_predictor._features['image_embed'].detach().clone(), size=image.shape[2:], mode='bicubic')
            sam_feature_list.append(sam_feature)
            
            
            # process the detection results
            input_boxes = results[0]["boxes"].cpu().numpy()
            OBJECTS = results[0]["labels"]
            
            if len(input_boxes) == 0:
                # There is no detected object.
                v, _, h, w = image.shape
                static_mask = torch.ones(v, 1, h, w).to(image.device, image.dtype)
                static_mask_list.append(static_mask)
                continue
            
            # prompt SAM 2 image predictor to get the mask for the object
            masks, scores, logits = self.image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            
            assert masks.ndim == 3 or masks.ndim == 4
            if masks.ndim == 3:
                masks = masks[np.newaxis]
            
            object_mask = torch.tensor(masks).to(image.device, image.dtype)

            # Filter out too large mask
            object_mask = filter_mask(object_mask, th=0.75)
            if object_mask is None:
                # No Mask after filtering
                v, _, h, w = image.shape
                static_mask = torch.ones(v, 1, h, w).to(image.device, image.dtype)
                static_mask_list.append(static_mask)
                continue
            
            object_mask = torch.sum(object_mask, dim=0, keepdim=True)
            object_mask[object_mask>=1] = 1
            static_mask_list.append(1-object_mask)
            
                
            # torchvision.utils.save_image((image+1)/2, "image_inf.png")
            # torchvision.utils.save_image((image+1)/2 * torch.tensor(object_mask).to(image.device), "image_mask_inf.png")
            # pdb.set_trace()

        static_mask = torch.cat(static_mask_list, dim=0)
        sam_features = torch.cat(sam_feature_list, dim=0)
        
        return static_mask, sam_features #bchw (only for condition_image, not pixel_values)
    
    
    # @torch.no_grad()
    # def get_mask_from_image(self, condition_image):
    #     """
    #         Used in Inference time.
            
    #         Args:
    #         - condition_image: torch.Tensor, bchw, [-1,1]
    #         Returns:
    #         - Mask of static region in condition_image   
    #     """
    #     device = condition_image.device
    #     # v = pixel_values.shape[0] // condition_image.shape[0]
    #     condition_image_sam = condition_image.unsqueeze(1)
    #     condition_image_sam = ((condition_image_sam / 2 + 0.5) * 255).permute(0,2,3,1).detach().cpu().numpy().astype(np.uint8)
    #     condition_image_sam = [Image.fromarray(x) for x in condition_image_sam]
        
    #     inputs = self.processor(images=condition_image_sam, text=text, return_tensors="pt").to(local_rank)
    #     outputs = self.grounding_model(**inputs)
        
    #     results = self.processor.post_process_grounded_object_detection(
    #         outputs,
    #         inputs.input_ids,
    #         box_threshold=0.25,
    #         text_threshold=0.3,
    #         target_sizes=[condition_image_sam[0].size[::-1]]
    #     )
        
    #     # prompt SAM image predictor to get the mask for the object
    #     condition_image_sam = [np.array(x.convert("RGB")) for x in condition_image_sam]
    #     self.image_predictor.set_image(condition_image_sam)
    #     sam_feature = F.interpolate(self.image_predictor._features['image_embed'].detach().clone(), size=image.shape[2:], mode='bicubic')
    #     sam_feature_list.append(sam_feature)
        
        
        
    #     static_mask_list = []
    #     sam_feature_list = []
    #     for image in input_images:
    #         """
    #         Prompt Grounding DINO and SAM image predictor to get the box and mask for specific frame
    #         """
    #         image_pil = Image.fromarray(((image[0]+1)/2*255).permute(1,2,0).cpu().numpy().astype(np.uint8))
    #         inputs = self.processor(images=image_pil, text=self.text, return_tensors="pt").to(device)
    #         outputs = self.grounding_model(**inputs)
    #         results = self.processor.post_process_grounded_object_detection(
    #             outputs,
    #             inputs.input_ids,
    #             box_threshold=0.25,
    #             text_threshold=0.3,
    #             target_sizes=[image_pil.size[::-1]]
    #         )
            
    #         # prompt SAM image predictor to get the mask for the object
    #         self.image_predictor.set_image(np.array(image_pil.convert("RGB")))
    #         sam_feature = F.interpolate(self.image_predictor._features['image_embed'].detach().clone(), size=image.shape[2:], mode='bicubic')
    #         sam_feature_list.append(sam_feature)
            
            
    #         # process the detection results
    #         input_boxes = results[0]["boxes"].cpu().numpy()
    #         OBJECTS = results[0]["labels"]
            
    #         if len(input_boxes) == 0:
    #             # There is no detected object.
    #             v, _, h, w = image.shape
    #             static_mask = torch.ones(v, 1, h, w).to(image.device, image.dtype)
    #             static_mask_list.append(static_mask)
    #             continue
            
    #         # prompt SAM 2 image predictor to get the mask for the object
    #         masks, scores, logits = self.image_predictor.predict(
    #             point_coords=None,
    #             point_labels=None,
    #             box=input_boxes,
    #             multimask_output=False,
    #         )
            
    #         assert masks.ndim == 3 or masks.ndim == 4
    #         if masks.ndim == 3:
    #             masks = masks[np.newaxis]
            
    #         object_mask = torch.tensor(masks).to(image.device, image.dtype)

    #         # Filter out too large mask
    #         object_mask = filter_mask(object_mask)
    #         if object_mask is None:
    #             # No Mask after filtering
    #             v, _, h, w = image.shape
    #             static_mask = torch.ones(v, 1, h, w).to(image.device, image.dtype)
    #             static_mask_list.append(static_mask)
    #             continue
            
    #         object_mask = torch.sum(object_mask, dim=0, keepdim=True)
    #         object_mask[object_mask>=1] = 1
    #         static_mask_list.append(1-object_mask)
            
                
    #         # torchvision.utils.save_image((image+1)/2, "image_inf.png")
    #         # torchvision.utils.save_image((image+1)/2 * torch.tensor(object_mask).to(image.device), "image_mask_inf.png")
    #         # pdb.set_trace()

    #     static_mask = torch.cat(static_mask_list, dim=0)
    #     sam_features = torch.cat(sam_feature_list, dim=0)
        
    #     return static_mask, sam_features #bchw (only for condition_image, not pixel_values)


def filter_mask(mask, th=0.5):
    H, W = mask.shape[-2:]
    total_pixel = H * W
    pixel_num = mask.sum(dim=(1,2,3))
    ratio = pixel_num / total_pixel
    
    mask_filtered = []
    for idx, rat in enumerate(ratio):
        if rat <= th:
            mask_filtered.append(mask[idx])
    
    if mask_filtered:
        return torch.stack(mask_filtered)
    else:
        return None
    