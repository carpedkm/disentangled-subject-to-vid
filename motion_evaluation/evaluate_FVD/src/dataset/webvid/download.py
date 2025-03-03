import pandas as pd
import os
import numpy as np
import argparse
import requests
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
# from mpi4py import MPI
import warnings
import queue

'''
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
'''

'''
class ThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(self, max_workers=None, thread_name_prefix=''):
        super().__init__(max_workers, thread_name_prefix)
        self._work_queue = queue.Queue(self._max_workers * 2)
'''




def request_save(url, save_fp):
    try:
        img_data = requests.get(url, timeout=50).content
        with open(save_fp, 'wb') as handler:
            handler.write(img_data)
        print(f"============ saved {save_fp}")
    except:
        print(f"============ failed {save_fp}")


def main(args):
    ### preproc
    video_dir = os.path.join(args.data_dir, 'videos')
    '''
    if RANK == 0:
        if not os.path.exists(os.path.join(video_dir, 'videos')):
            os.makedirs(os.path.join(video_dir, 'videos'))

    COMM.barrier()
    '''

    # ASSUMES THE CSV FILE HAS BEEN SPLIT INTO N PARTS
    partition_dir = args.csv_path.replace('.csv', f'_{args.partitions}')

    # if not, then split in this job.
    if not os.path.exists(partition_dir):
        os.makedirs(partition_dir)
        full_df = pd.read_csv(args.csv_path)
        df_split = np.array_split(full_df, args.partitions)
        for idx, subdf in enumerate(df_split):
            subdf.to_csv(os.path.join(partition_dir, f'{idx}.csv'), index=False)

    # relevant_fp = os.path.join(args.data_dir, 'relevant_videos_exists.txt')

    '''
    relevant_fp = args.csv_path.replace(".csv", "_cleaned.csv")
    if os.path.isfile(relevant_fp):
        exists = pd.read_csv(relevant_fp, names=['videoid'])
    else:
        exists = []
    '''

    # ASSUMES THE CSV FILE HAS BEEN SPLIT INTO N PARTS
    # data_dir/results_csvsplit/results_0.csv
    # data_dir/results_csvsplit/results_1.csv
    # ...
    # data_dir/results_csvsplit/results_N.csv


    df = pd.read_csv(os.path.join(partition_dir, f'{args.part}.csv'))

    # df['rel_fn'] = df.apply(lambda x: os.path.join(str(x['page_dir']), str(x['videoid'])), axis=1)

    # df['rel_fn'] = df['rel_fn'] + '.mp4'

    # df = df[~df['videoid'].isin(exists)]

    # remove nan
    df.dropna(subset=['page_dir'], inplace=True)

    playlists_to_dl = np.sort(df['page_dir'].unique())

    '''
    for page_dir in playlists_to_dl:
        vid_dir_t = os.path.join(video_dir, page_dir)
        pdf = df[df['page_dir'] == page_dir]
        if len(pdf) > 0:
            if not os.path.exists(vid_dir_t):
                os.makedirs(vid_dir_t)

            urls_todo = []
            save_fps = []

            for idx, row in pdf.iterrows():
                video_fp = os.path.join(vid_dir_t, str(row['videoid']) + '.mp4')
                if not os.path.isfile(video_fp):
                    urls_todo.append(row['contentUrl'])
                    save_fps.append(video_fp)

            print(f'Spawning {len(urls_todo)} jobs for page {page_dir}')
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.processes) as executor:
                future_to_url = {executor.submit(request_save, url, fp) for url, fp in zip(urls_todo, save_fps)}
            # request_save(urls_todo[0], save_fps[0])
    '''

    urls_todo = []
    save_fps = []

    p = Pool(args.processes)
    for idx, row in df.iterrows():
        video_fp = os.path.join(video_dir, str(row['videoid']) + '.mp4')
        print(f"============ {video_fp} exists")
        if not os.path.isfile(video_fp):
            url = row['contentUrl']
            urls_todo.append(url)
            save_fps.append(video_fp)
            p.apply_async(request_save, args = (url, video_fp))



    '''
    print(f"============ start download ==============")
    with ThreadPoolExecutor(max_workers=args.processes) as executor:
        future_to_url = {executor.submit(request_save, url, fp) for url, fp in zip(urls_todo, save_fps)}
    '''
    # request_save(urls_todo[0], save_fps[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shutter Image/Video Downloader')
    parser.add_argument('--partitions', type=int, default=8, # 8, # azure todo
                        help='Number of partitions to split the dataset into, to run multiple jobs in parallel')
    parser.add_argument('--part', type=int, required=True,
                        help='Partition number to download where 0 <= part < partitions')
    parser.add_argument('--processes', type=int, default=63) # azure todo
    parser.add_argument('--processes_id', type=int, default=0) # azure todo
    parser.add_argument('--vmid', type=int, default=1) # azure todo

    parser.add_argument('--data_dir', type=str, default="/home/zhiyzh/workspace/t2vg_tools/video2frame", # '~/mount', # azure todo
                        help='Directory where webvid data is stored.')
    parser.add_argument('--csv_path', type=str, default='~/mount/webvid10m_meta/results_2M_train.csv',
                        help='Path to csv data to download')
    args = parser.parse_args()

    '''
    if SIZE > 1:
        warnings.warn("Overriding --part with MPI rank number")
        args.part = RANK

    if args.part >= args.partitions:
        raise ValueError("Part idx must be less than number of partitions")
    '''
    csvs = [
        "results_2M_val.csv",
        "results_2M_train.csv",
        "results_10M_val.csv",
        "results_10M_train.csv", # azure todo
            ]
    csv_dir = f"/home/wmrm{args.vmid}/mount/webvid10m_meta" # azure todo
    # csv_dir = "/home/zhiyzh/workspace/t2vg_tools/label_analysis"
    args.data_dir = f"/home/wmrm{args.vmid}/mount"
    for csv in csvs:
        args.csv_path = os.path.join(csv_dir, csv)
        main(args)
    '''
    while True:
        for csv in csvs:
            args.csv_path = os.path.join(csv_dir, csv)
            main(args)
    '''
