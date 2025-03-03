# -*- coding: utf-8 -*-
#   Auther: William Zhao    #
# Stay foolish, stay hungry.#
# ------------------------- #

import os
import pandas as pd


def update_meta(video_dir, old_meta_path, new_meta_path):
    videos = os.listdir(video_dir)
    ids = [int(id_[:-4]) for id_ in videos if id_[-4:] == ".mp4"]

    label_data = pd.read_csv(old_meta_path)
    label_data = label_data[label_data['videoid'].isin(ids)]

    label_data.to_csv(new_meta_path, index=False)


def update_newdata_meta(video_dir, all_meta_path, new_meta_path):
    # list videos downloaded
    videos = os.listdir(video_dir)
    ids = [int(id_[:-4]) for id_ in videos if id_[-4:] == ".mp4"]

    # all meta data
    label_data = pd.read_csv(all_meta_path)
    # not processd data meta
    cleaned_meta_path = all_meta_path.replace(".csv", "_cleaned.csv")
    cleaned_label_data = pd.read_csv(cleaned_meta_path)

    delta_label_data = label_data[~label_data['videoid'].isin(cleaned_label_data['videoid'])]

    # exists videos in not processed data
    delta_label_data = delta_label_data[delta_label_data['videoid'].isin(ids)]

    delta_label_data.to_csv(new_meta_path, index=False)


def update_meta_wmrm():
    meta_dir = "/home/wmrm0/mount/webvid10m_meta"
    video_dir = "/home/wmrm0/mount/videos_rmwm"
    csvs = [
        "results_10M_train_cleaned.csv",
        "results_2M_train_cleaned.csv",
        "results_10M_val_cleaned.csv",
        "results_2M_val_cleaned.csv",
    ]
    '''
    meta_dir = "/home/zhiyzh/workspace/t2vg_tools/label_analysis"
    video_dir = "/home/zhiyzh/workspace/t2vg_tools/video2frame"
    csvs = [
        "results_2M_val_cleaned_test.csv",
    ]
    '''
    for csv in csvs:
        meta_path = os.path.join(meta_dir, csv)
        new_csv = csv.replace(".csv", ".wmrm.csv")
        new_meta_path = os.path.join(meta_dir, new_csv)
        update_meta(video_dir, meta_path, new_meta_path)


def update_meta_new_download():
    meta_dir = "/home/wmrm0/mount/webvid10m_meta"
    video_dir = "/home/wmrm0/mount/videos"
    csvs = [
        "results_10M_train.csv",
        "results_2M_train.csv",
        "results_10M_val.csv",
        "results_2M_val.csv",
    ]
    '''
    meta_dir = "/home/zhiyzh/workspace/t2vg_tools/label_analysis"
    video_dir = "/home/zhiyzh/workspace/t2vg_tools/video2frame/videos"
    csvs = [
        "results_2M_val_test.csv",
    ]
    '''
    for csv in csvs:
        meta_path = os.path.join(meta_dir, csv)
        new_csv = csv.replace(".csv", ".newdownload.csv")
        new_meta_path = os.path.join(meta_dir, new_csv)
        update_newdata_meta(video_dir, meta_path, new_meta_path)


def main():
    # update_meta_wmrm()

    update_meta_new_download()


if __name__ == "__main__":
    main()
