from webvid_dataset import WebVid
from torchvision import transforms
import os
from glob import glob
from tqdm import tqdm


def init_transform_dict(input_res=224,
                        center_crop=256,
                        randcrop_scale=(0.5, 1.0),
                        color_jitter=(0, 0, 0),
                        norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225)):
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    tsfm_dict = {
        'train':
        transforms.Compose([
            transforms.RandomResizedCrop(input_res, scale=randcrop_scale),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
            normalize,
            # transforms.Lambda(lambda x: x * 2 - 1)
            # old: transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))
        ]),
        'val':
        transforms.Compose([
            transforms.Resize(center_crop),
            transforms.CenterCrop(center_crop),
            transforms.Resize(input_res),
            normalize,
            # transforms.Lambda(lambda x: x * 2 - 1)
        ]),
        'test':
        transforms.Compose([
            transforms.Resize(center_crop),
            transforms.CenterCrop(center_crop),
            transforms.Resize(input_res),
            normalize,
            # transforms.Lambda(lambda x: x * 2 - 1)
        ])
    }
    return tsfm_dict


vd_info = {
    "type": "TextVideoDataLoader",
    "args": {
        "dataset_name": "WebVid",
        # "data_dir": "/home/v-yukangyang/video_blob",
        "data_dir": "/home/zhiyzh/t2vg_data",
        "shuffle": True,
        "num_workers": 16,
        "batch_size": 448,
        # "split": "train",
        "split": "val",
        "cut": "10M",
        "subsample": 1,
        "text_params": {
            "input": "text"
        },
        "video_params": {
            "input_res": 224,
            "num_frames": 1,  # TBD, if not fix, remove
            "loading": "lax"
        },
        "metadata_folder_name": "webvid10m_meta",
        "first_stage_key": "video",
        "cond_stage_key": "txt",
        "skip_missing_files": False,
    }
}
args = vd_info["args"]

# video_folder = os.path.join(vd_info["args"]["data_dir"], "videos")
# videos = glob(video_folder + "/*.mp4")
# print(len(videos), " videos")

tsfm_params = None if "tsfm_params" not in vd_info.keys() else args["tsfm_params"]
tsfm_split = None if "tsfm_split" not in vd_info.keys() else args["tsfm_split"]
if tsfm_params is None:
    tsfm_params = {}
tsfm_dict = init_transform_dict(**tsfm_params)
if tsfm_split is None:
    tsfm_split = args["split"]
tsfm = tsfm_dict[tsfm_split]
kwargs = {
    key: args[key]
    for key in args.keys() if key in [
        "dataset_name",
        "text_params",
        "video_params",
        "data_dir",
        "metadata_dir",
        "metadata_folder_name",
        "split",
        #  "tsfms", #
        "cut",
        "subsample",
        "sliding_window_stride",
        "reader",
        "first_stage_key",
        "cond_stage_key",
        "skip_missing_files",
    ]
}
kwargs["tsfms"] = tsfm
video_dataset = WebVid(**kwargs)
num = 0
max_num = 10000  #len(video_dataset)
# data = video_dataset[1]
for i in tqdm(range(max_num)):
    data = video_dataset[i]
    import pdb
    pdb.set_trace()  # XXX BREAKPOINT
    num += 1
    # if num == max_num:
    #     break
# data_0 = video_dataset.__getitem__(0)
print("lack num", len(video_dataset.lack_files), len(video_dataset.lack_files) / max_num)
print("trial ends!")
