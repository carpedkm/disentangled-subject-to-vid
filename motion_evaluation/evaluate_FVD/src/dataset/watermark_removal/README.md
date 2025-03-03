# setup

make sure you installed packages below:

```
decord
opencv-python
Pillow
numpy
av
matplotlib
tqdm
```



# Usage

## remove_watermark_video(video_path, shift=None, blur=False, output_dir=None， verbose=True):

removes watermark from one single video

- video_path : the path to the video
- shift : `list` type, shows the pixel shift to the original watermark refence `pics/refer.png`, autodetect if set to `None`
- blur : blur area selection. `False` means no blur, `'low'`  means blur is applied to only the inner area of the watermark, `high` means blur is applied to only the contour of the watermark, and `True` means applying to both high and low area of the watermark
- output_dir : where the output video should be inside.
- verbose: show the progress of the video if set to `True`



## remove_watermark_videos(video_dir, blur=False, output_dir=None, verbose=True, multiprocessing=False): 

entrance for batch video processing

- video_dir : where the input videos are in
- blur : the same to the upper function `remove_watermark_video`
- output_dir : the same to the upper function `remove_watermark_video`
- verbose : the same to the upper function `remove_watermark_video`
- multiprocessing : enable the multiprocessing and increase CPU utilization in order to accelerate



# cmd Usage

To use it in a command line instead of leveraging its function directly, just do

```bash
python video_watermark_removal.py
```

the args are as follows:

- video_dir : where the input videos are in, **required**
- blur : blur area selection, `'low', 'high', 'true', 'false'`only,  **default='high'**
- output_dir: where the output video should be inside. **required**
- verbose: **store_true, default is False**
- multiprocessing: **store_true, default is False**
- save_fail: Since we support resolution of 596 * 336 only, you can set this argument to get a list of failed videos (in the output folder). **store_true, default is False**

# Notice

This code supports shutterstock watermark removal for video whose resolution should be 596 * 336 only. 