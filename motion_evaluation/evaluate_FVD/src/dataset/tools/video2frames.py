import os
import os.path as osp
import argparse
import glob
import tqdm
import sys
from multiprocessing import Pool


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_video_dir')
    parser.add_argument('out_frame_dir')
    parser.add_argument('--ext', default='webm', type=str)
    args = parser.parse_args()
    return args


def extract_frame(vid_item):
    video_path, out_path, vid_id = vid_item
    video_name = osp.splitext(osp.basename(video_path))[0]
    out_video_dir = os.path.join(out_path, video_name)
    os.makedirs(out_video_dir, exist_ok=True)
    out_name = f'{out_video_dir}/{video_name}_%08d.jpg'

    cmd = f'ffmpeg -i "{video_path}" -r 30 -q:v 1 "{out_name}" -loglevel error'
    os.system(cmd)
    # print(f'{vid_id} {video_name}  done')
    sys.stdout.flush()
    return True

def main(args):

    if not osp.isdir(args.out_frame_dir):
        print(f'Creating folder: {args.out_frame_dir}')
        os.makedirs(args.out_frame_dir)

    print('Reading videos from folder: ', args.in_video_dir)

    video_list = glob.glob(args.in_video_dir + '/*' + '.' + args.ext)

    pool = Pool(32)
    for _ in tqdm.tqdm(pool.imap_unordered(
        extract_frame,
        zip(video_list, len(video_list) * [args.out_frame_dir], range(len(video_list)))), total=len(video_list)):
        pass

    # for vid in tqdm.tqdm(video_list):
    #     extract_frame([vid, args.out_frame_dir])


if __name__ == '__main__':
    args = parse_arg()
    main(args)
