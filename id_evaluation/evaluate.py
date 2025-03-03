import torch
import os
from evaluator import Evaluator
from evaluator.distributed import dist_init, print0, get_rank
from datetime import datetime
import argparse
import json

def parse_args():
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='VBench', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--output_path",
        type=str,
        default='./evaluation_results/',
        help="output path to save the evaluation results",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        help="path to the video directory",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="path to the image directory",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        help="path to save the json file that contains the video and image location",
    )
    parser.add_argument(
        "--dimension",
        nargs='+',
        required=False,
        default=None,
        help="list of evaluation dimensions, usage: --dimension <dim_1> <dim_2>",
    )
    parser.add_argument(
        "--load_ckpt_from_local",
        type=bool,
        required=False,
        default=False,
        help="whether load checkpoints from local default paths (assuming you have downloaded the checkpoints locally",
    )
    parser.add_argument(
        "--read_frame",
        type=bool,
        required=False,
        default=False,
        help="whether directly read frames, or directly read videos",
    )
    parser.add_argument(
        "--prompt_json",
        type=str,
        required=False,
        default=None,
        help="path to the prompt json file. key:videoname, value:prompt",
    )
    parser.add_argument(
        '--regional_suffix_video',
        type=str,
        required=False,
        default='regional',
        help="subdir to regional data, default: regional"
    )
    parser.add_argument(
        '--regional_suffix_image',
        type=str,
        required=False,
        default='regional',
        help="subdir to regional data, default: regional"
    )
    parser.add_argument(
        '--mode',
        type=str,
        required=False,
        default='default', #default is center_sampling
        help="mode of frame sampling"
    )
    parser.add_argument(
        '--n_frames',
        type=int,
        default='full',
        help="number of frames to be sampled to evaluate"
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    dist_init()
    print0(f'args: {args}')
    device = torch.device("cuda")
    evaluator = Evaluator(device, args.output_path)

    print0(f'start evaluation')

    current_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    kwargs = {
        'video_dir': args.video_dir,
        'image_dir': args.image_dir,
        'regional_suffix_video': args.regional_suffix_video,
        'regional_suffix_image': args.regional_suffix_image,
        'n_frames': args.n_frames,
        'mode': args.mode,
    }

    if args.prompt_json is not None:
        kwargs['prompt_json'] = args.prompt_json

    evaluator.evaluate(
        json_path=args.json_path,
        name=f'results_{current_time}',
        dimension_list=args.dimension,
        local=args.load_ckpt_from_local,
        read_frame=args.read_frame,
        **kwargs,
    )


if __name__ == '__main__':
    main()