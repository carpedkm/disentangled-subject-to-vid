import cv2
import decord
from PIL import Image
import numpy as np
import av
import matplotlib.pyplot as plt
import tqdm
import os
from multiprocessing import Pool, cpu_count
import argparse
import pickle
import pandas as pd


def remove_watermark_image(img, shift, blur=False):
    a = 0.125
    img = img.astype(np.float32)
    im_mask_high = cv2.imread('./pics/watermark_mask_high.png')
    im_mask_low = cv2.imread('./pics/watermark_mask_low.png')
    refer = cv2.imread('./pics/refer.png')
    refer = refer.astype(np.float32)

    mask_high = np.where(im_mask_high[:,:,0] == 255)
    mask_low = np.where(im_mask_low[:,:,2] == 255) # 168
    pixel_high = refer[mask_high]
    # print(pixel_high)
    pixel_low = refer[mask_low]

    mask_high = list(mask_high)
    mask_low = list(mask_low)

    mask_high[0] += shift[0]
    mask_high[1] += shift[1]
    mask_low[0] += shift[0]
    mask_low[1] += shift[1]

    mask_low = tuple(mask_low)
    mask_high = tuple(mask_high)

    # pixel_low = 30
    img[mask_low] -= pixel_low
    img[mask_low] /= (1 - a)
    img[mask_high] -= pixel_high
    img[mask_high] /= (1 - a)

    img[np.where(img < 0)] = 0
    img[np.where(img > 255)] = 255
    img = img.astype(np.uint8)

    if blur != False:
        img_blur = cv2.blur(img, (5, 5))
        if blur == True:
            img[mask_low] = img_blur[mask_low]
            img[mask_high] = img_blur[mask_high]
        elif blur == 'low':
            img[mask_low] = img_blur[mask_low]
        elif blur == 'high':
            img[mask_high] = img_blur[mask_high]
        else:
            img = img_blur
    return img


def predict_bbox_shift(img, verbose=False):
    orig_shape = (193, 132)
    watermark = cv2.imread('./pics/refer_bbox.png', cv2.IMREAD_GRAYSCALE)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(img, watermark, cv2.TM_CCOEFF_NORMED)

    sort_res = np.argsort(res.flatten())[::-1]
    for loc_res in sort_res:
        loc = np.unravel_index(loc_res, res.shape)
        shift = [loc[0] - orig_shape[0], loc[1] - orig_shape[1]]
        if abs(shift[0]) < 4 and abs(shift[1]) < 4:
            break
    if verbose:
        print(f'max probability:{np.max(res)}')
    return shift


def remove_watermark_video(video_path, shift=None, blur=False, output_dir=None, verbose=True):
    vr = decord.VideoReader(video_path)
    if shift is None:
        img = vr[0].asnumpy()[:, :, ::-1]
        shift = predict_bbox_shift(img)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not (width == 596 and height == 336): #should adapt to different after
         return video_path
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if blur is False:
        blur_name = 'no_blur'
    elif blur is True:
        blur_name = 'both'
    else:
        blur_name = 'blur_' + blur
    if output_dir is None:
        # writer = cv2.VideoWriter(f"./videos/after/{video_path.rsplit('/')[-1][:-4]}_{blur_name}.mp4", fourcc, fps, (width, height))
        writer = cv2.VideoWriter(f"./videos/after/{video_path.rsplit('/')[-1][:-4]}.mp4", fourcc, fps, (width, height))
    else:
        output_dir += '/' if not output_dir.endswith('/') else ''
        os.makedirs(output_dir, exist_ok=True)
        # writer = cv2.VideoWriter(f"{output_dir}{video_path.rsplit('/')[-1][:-4]}_{blur_name}.mp4", fourcc, fps, (width, height))
        writer = cv2.VideoWriter(f"{output_dir}{video_path.rsplit('/')[-1][:-4]}.mp4", fourcc, fps, (width, height))
    if verbose:
        gen = tqdm.tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    else:
        gen = range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    for i in gen:
        frame = vr[i].asnumpy()[:, :, ::-1]
        writer.write(remove_watermark_image(frame, shift, blur))

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    return True


def remove_watermark_videos(video_dir, blur=False, output_dir=None, verbose=True, multiprocessing=False): #entrance for batch video processing
    video_dir += '/' if not video_dir.endswith('/') else ''
    output_dir += '/' if not output_dir.endswith('/') else ''
    fail_list = []
    pool_list = []
    p = Pool(cpu_count())

    for video in os.listdir(video_dir):
        if not video.endswith('.mp4'):
            continue
        print(f'processing video {video}')
        video_path = video_dir + video
        if multiprocessing:
            res = p.apply_async(remove_watermark_video, args = (video_path, None, blur, output_dir, verbose))
        else:
            res = remove_watermark_video(video_path, blur=blur, output_dir=output_dir, verbose=verbose)
        if not multiprocessing:
            if res is not True:
                fail_list.append(res)
        else:
            pool_list.append(res)
    if multiprocessing:
        p.close()
        p.join()
        for res in pool_list:
            res = res.get()
            if res is not True:
                fail_list.append(res)
    return fail_list

def remove_watermark_split(video_dir, meta_dir, split_n, split_i, save_fail, blur=False, output_dir=None, verbose=True, multiprocessing=False): #entrance for batch video processing
    video_dir += '/' if not video_dir.endswith('/') else ''
    output_dir += '/' if not output_dir.endswith('/') else ''
    fail_list = []
    pool_list = []
    p = Pool(cpu_count())
    meta_files = [
        "results_10M_train_cleaned.csv",
        "results_2M_train_cleaned.csv",
        "results_10M_val_cleaned.csv",
        "results_2M_val_cleaned.csv",
    ]
    video_ids = None
    for mf in meta_files:
        mf_path = os.path.join(meta_dir, mf)
        label_data = pd.read_csv(mf_path)
        video_id = label_data['videoid']
        if video_ids is None:
            video_ids = video_id
        else:
            video_ids = pd.concat([video_ids, video_id], ignore_index=True)
    split_l = len(video_ids) // split_n

    if split_i == split_n - 1:
        video_ids_split = video_ids[int(split_i * split_l):]
    else:
        video_ids_split = video_ids[int(split_i * split_l): int(split_i * split_l + split_l )]

    for video_id in tqdm.tqdm(video_ids_split):
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        # print(f'processing video {video}')
        if multiprocessing:
            res = p.apply_async(remove_watermark_video, args = (video_path, None, blur, output_dir, verbose))
        else:
            res = remove_watermark_video(video_path, blur=blur, output_dir=output_dir, verbose=verbose)
        if not multiprocessing:
            if res is not True:
                fail_list.append(res)
        else:
            pool_list.append(res)
    if multiprocessing:
        p.close()
        p.join()
        for res in pool_list:
            res = res.get()
            if res is not True:
                fail_list.append(res)

    if save_fail:
        args.output_dir = args.output_dir[:-1] if args.output_dir.endswith('/') else args.output_dir
        with open(f'{args.output_dir}/fail_list_{split_i}.pkl', 'wb') as f:
            pickle.dump(fail_list, f)

    return fail_list


if __name__ == '__main__':
    # fail_list = remove_watermark_videos('./videos_before', blur='high', output_dir='./videos_after', verbose=False, multiprocess=True)
    # print(fail_list)
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--blur', type=str, default='high', required=False)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--multiprocessing', default=False, action='store_true')
    parser.add_argument('--save_fail', default=False, action='store_true')

    parser.add_argument('--meta_dir', type=str, required=False)
    parser.add_argument('--split_n', type=int, required=False)
    parser.add_argument('--split_i', type=int, required=False)

    args, _ = parser.parse_known_args()
    if args.blur.lower() == 'true':
        args.blur = True
    elif args.blur.lower() == 'false':
        args.blur = False
    remove_watermark_split(args.video_dir, args.meta_dir, args.split_n, args.split_i, args.save_fail, blur=args.blur, output_dir=args.output_dir, verbose=args.verbose, multiprocessing=args.multiprocessing)

    '''
    fail_list = remove_watermark_videos(args.video_dir, blur=args.blur, output_dir=args.output_dir, verbose=args.verbose, multiprocessing=args.multiprocessing)

    if args.save_fail:
        args.output_dir = args.output_dir[:-1] if args.output_dir.endswith('/') else args.output_dir
        with open(f'{args.output_dir}/fail_list.pkl', 'wb') as f:
            pickle.dump(fail_list, f)
    '''
