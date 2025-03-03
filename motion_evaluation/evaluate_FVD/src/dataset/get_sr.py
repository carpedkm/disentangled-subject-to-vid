# -*- coding: utf-8 -*-
#   Auther: William Zhao    #
# Stay foolish, stay hungry.#
# ------------------------- #

from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import cv2
import subprocess
import ffmpeg
from decord import VideoReader
import time


def get_video_sr_vfc(video_path):
    clip = VideoFileClip(video_path)
    audio = clip.audio
    sample_rate = audio.fps
    return sample_rate


def get_video_sr_as(video_path):
    audio = AudioSegment.from_file(video_path, format='mp4')
    sample_rate = audio.frame_rate
    return sample_rate


def get_video_sr_cv(video_path):
    cap = cv2.VideoCapture(video_path)
    sample_rate = cap.get(cv2.CAP_PROP_FPS)
    return sample_rate


def get_video_sr_ffmpeg(video_path):
    command = f'ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of default=noprint_wrappers=1:nokey=1 {video_path}'
    output = subprocess.check_output(command, shell=True)
    sample_rate = int(output.strip())
    return sample_rate


def get_video_sr_ffmpeg_p(video_path):
    probe = ffmpeg.probe(video_path)
    audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
    sample_rate = int(audio_stream['sample_rate'])
    return sample_rate


def get_video_sr_decord(video_path):
    vr = VideoReader(video_path)
    sample_rate = vr.get_avg_fps()
    return sample_rate


def main():
    video_path = "/home/zhiyzh/workspace/montage/serverdata/video_test_top/video_test_top_0/video_test_top.mp4"

    start = time.time()
    sr = get_video_sr_vfc(video_path)
    print(time.time() - start, sr)

    start = time.time()
    sr = get_video_sr_as(video_path)
    print(time.time() - start, sr)

    start = time.time()
    sr = get_video_sr_cv(video_path)
    print(time.time() - start, sr)

    start = time.time()
    sr = get_video_sr_ffmpeg(video_path)
    print(time.time() - start, sr)

    start = time.time()
    sr = get_video_sr_ffmpeg_p(video_path)
    print(time.time() - start, sr)

    start = time.time()
    sr = get_video_sr_decord(video_path)
    print(time.time() - start, sr)


if __name__ == "__main__":
    main()
