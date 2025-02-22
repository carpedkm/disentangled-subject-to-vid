import argparse
import os
import torch
import json
from transformers import AutoTokenizer, T5EncoderModel
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXPipeline
from diffusers.utils import export_to_video
from PIL import Image
import numpy as np
from peft import LoraConfig, set_peft_model_state_dict

# Custom transformer module (still used)
from custom_cogvideox import CustomCogVideoXTransformer3DModel

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for CogVideoX-5b with LoRA weights.")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to the trained checkpoint directory (e.g., checkpoint-5000).")
    parser.add_argument("--prompt", type=str, default=None, 
                        help="Text prompt for video generation (used with --inference).")
    parser.add_argument("--reference_image", type=str, default=None, 
                        help="Optional path to a reference image (for --inference).")
    parser.add_argument("--output_dir", type=str, default="inference_outputs", 
                        help="Directory to save the generated video(s).")
    parser.add_argument("--num_frames", type=int, default=25, 
                        help="Number of frames in the output video.")
    parser.add_argument("--guidance_scale", type=float, default=6.0, 
                        help="Classifier-free guidance scale.")
    parser.add_argument("--num_inference_steps", type=int, default=50, 
                        help="Number of denoising steps.")
    parser.add_argument("--fps", type=int, default=8, 
                        help="Frames per second for the output video.")
    parser.add_argument("--height", type=int, default=480, 
                        help="Height of the generated video.")
    parser.add_argument("--width", type=int, default=720, 
                        help="Width of the generated video.")
    parser.add_argument("--inference", action="store_true", 
                        help="Run default inference with provided prompt and reference image.")
    parser.add_argument("--dreambooth_inference", action="store_true", 
                        help="Run inference with predefined dreambooth validation prompts.")
    parser.add_argument("--seen_validation", action="store_true", 
                        help="Use seen validation prompts from metadata with --dreambooth_inference.")
    return parser.parse_args()

def get_val_instance_prompt_dict(seen_validation=False):
    add_special = True
    prefix = "<cls> "
    if seen_validation:
        val_instance_prompt_dict = {}
        path_for_seen_meta = '../seen_samples/omini_meta/'
        if not os.path.exists(path_for_seen_meta):
            raise FileNotFoundError(f"Seen validation metadata directory {path_for_seen_meta} not found.")
        for file in os.listdir(path_for_seen_meta):
            with open(os.path.join(path_for_seen_meta, file), 'r') as f:
                meta_seen = json.load(f)
            id_ = 'right_' + file.split('_')[1].split('.')[0]
            tmp_desc = meta_seen['description_0']
            val_instance_prompt_dict[id_] = prefix + tmp_desc
    else:
        val_instance_prompt_dict = {
            'oranges_omini': prefix + "A close up view. A bowl of oranges are placed on a wooden table. The background is a dark room, the TV is on, and the screen is showing a cooking show.",
            'clock_omini': prefix + "In a Bauhaus style room, the clock is placed on a shiny glass table, with a vase of flowers next to it. In the afternoon sun, the shadows of the blinds are cast on the wall.",
            'rc_car_omini': prefix + "A film style shot. On the moon, toy car goes across the moon surface. The background is that Earth looms large in the foreground.",
            'shirt_omini': prefix + "On the beach, a lady sits under a beach umbrella. She's wearing hawaiian shirt and has a big smile on her face, with her surfboard behind her. The sun is setting in the background. The sky is a beautiful shade of orange and purple.",
            'cat': prefix + "cat is rollerblading in the park",
            'dog': prefix + "dog is flying in the sky",
            'red_toy': prefix + "red toy is dancing in the room",
            'dog_toy': prefix + "dog toy is walking around the grass",
        }
    return val_instance_prompt_dict

def load_model_hook(transformer, checkpoint_path):
    lora_config = LoraConfig(
        r=128,
        lora_alpha=64,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    if not hasattr(transformer, 'peft_config'):
        transformer.add_adapter(lora_config)
    lora_state_dict = CogVideoXPipeline.lora_state_dict(checkpoint_path)
    transformer_state_dict = {
        k.replace("transformer.", ""): v 
        for k, v in lora_state_dict.items() 
        if k.startswith("transformer.")
    }
    from diffusers.utils import convert_unet_state_dict_to_peft
    transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
    set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")

def run_inference(pipe, prompt, reference_image_path, args):
    pipeline_args = {
        "prompt": prompt,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
    }
    # Only add ref_img_states if a reference image is provided
    if reference_image_path:
        ref_image = Image.open(reference_image_path).convert("RGB")
        ref_image = ref_image.resize((args.width, args.height))
        ref_image = np.array(ref_image) / 255.0 * 2.0 - 1.0
        ref_image = torch.from_numpy(ref_image).float().permute(2, 0, 1).unsqueeze(0)
        ref_image = ref_image.to(pipe.device, dtype=pipe.transformer.dtype)
        with torch.no_grad():
            ref_latents = pipe.vae.encode(ref_image).latent_dist.sample() * pipe.vae.config.scaling_factor
        pipeline_args["ref_img_states"] = ref_latents.unsqueeze(2)
    if "<cls>" not in prompt:
        pipeline_args["prompt"] = "<cls> " + prompt
    with torch.no_grad():
        output = pipe(**pipeline_args)
        video = output.frames[0]
    return video

def main():
    args = parse_args()
    if not (args.inference or args.dreambooth_inference):
        raise ValueError("Must specify either --inference or --dreambooth_inference.")
    if args.inference and not args.prompt:
        raise ValueError("--prompt is required when --inference is set.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16

    # Load transformer separately with custom class
    transformer = CustomCogVideoXTransformer3DModel.from_pretrained(
        "THUDM/CogVideoX-5b", subfolder="transformer", torch_dtype=weight_dtype,
        customization=True, vae_add=True
    )
    load_model_hook(transformer, args.checkpoint_path)

    # Use standard CogVideoXPipeline with custom transformer
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-5b",
        transformer=transformer,
        torch_dtype=weight_dtype
    )
    pipe = pipe.to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.dreambooth_inference:
        val_prompts = get_val_instance_prompt_dict(args.seen_validation)
        for key, prompt in val_prompts.items():
            output_file = os.path.join(args.output_dir, f"{key}_video.mp4")
            video = run_inference(pipe, prompt, None, args)
            export_to_video(video, output_file, fps=args.fps)
            print(f"Generated video for {key} saved to {output_file}")
    elif args.inference:
        output_file = os.path.join(args.output_dir, "inference_video.mp4")
        video = run_inference(pipe, args.prompt, args.reference_image, args)
        export_to_video(video, output_file, fps=args.fps)
        print(f"Generated video saved to {output_file}")

if __name__ == "__main__":
    main()