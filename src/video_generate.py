import os
from PIL import Image
import numpy as np
import torch
from diffusers.utils import export_to_video

def inference(
    pipe,
    args,
    pipeline_args,
    resizing: bool = True,
):
    # Move pipeline to device and eval mode
    device = pipe._execution_device
    pipe = pipe.to(device)
    pipe.transformer.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()

    # Set deterministic generation
    generator = torch.Generator(device=device)
    if args.seed is not None:
        generator.manual_seed(args.seed) # init seed

    current_pipeline_args = pipeline_args.copy()
    ref_image = Image.open(pipeline_args['reference_image']).convert('RGB')
    ref_image = np.array(ref_image)
    ref_image = np.expand_dims(ref_image, axis=0)  # Add frame dimension
    ref_image = torch.from_numpy(ref_image).float() / 255.0 * 2.0 - 1.0
    ref_image = ref_image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    ref_image = ref_image.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device=device, dtype=pipe.transformer.dtype)
    
    with torch.no_grad():
        ref_image_latents = pipe.vae.encode(ref_image).latent_dist
        ref_image_latents = ref_image_latents.sample() * pipe.vae.config.scaling_factor
    ref_image_latents = ref_image_latents.permute(0, 2, 1, 3, 4)

    current_pipeline_args['ref_img_states'] = ref_image_latents
    current_pipeline_args.pop('reference_image', None)
                
    # Generate the video
    print(f"Generating video with prompt: {pipeline_args['prompt']}")
    
    # Add inference parameters explicitly
    inference_args = {
        'num_inference_steps': 50,
        'output_type': "np",
        'guidance_scale': args.guidance_scale,
        'use_dynamic_cfg': args.use_dynamic_cfg,
        'height': args.height,
        'width': args.width,
        'num_frames': args.max_num_frames, #args.max_num_frames,
        'eval': True
    }
    current_pipeline_args.update(inference_args)
    
    # Run inference with torch.no_grad()
    with torch.no_grad():
        output = pipe(**current_pipeline_args)
        video = output.frames[0]


    filename = os.path.join(args.output_dir, 'output.mp4')
    export_to_video(video, filename, fps=args.fps)