import os

import pandas as pd

from base_video_dataset import TextVideoDataset


class GettyVid(TextVideoDataset):
    """
    Getty Video dataset.
    Getty videos are stored in a single video folder.
    Getty_vid/
        videos/                                         ($page_dir)
            001f9ecb10fcde945fe743895aa5b49c.mp4        (videoid.mp4)
            ...
            001fe6dec514b57296bf3c0a0b45d822.mp4
            ...
    """
    def _load_metadata(self):
        assert self.metadata_folder_name is not None
        metadata_fp = os.path.join(self.metadata_folder_name, f'results_{self.cut}_{self.split}.csv')
        metadata = pd.read_csv(metadata_fp)

        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        elif self.split == 'val':
            metadata = metadata.sample(1000, random_state=0)  # 15k val is unnecessarily large, downsample.

        # In getty, we have three texts, 'title', 'description', 'tags'
        # Here we only use description as the text input
        metadata['caption'] = metadata['description']
        del metadata['description']
        del metadata['title']
        del metadata['tags']
        self.metadata = metadata
        # TODO: clean final csv so this isn't necessary
        self.metadata.dropna(inplace=True)
        # self.metadata['caption'] = self.metadata['caption'].str[:350]

    def _get_video_path(self, sample):
        # rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        rel_video_fp = sample['videoid']
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        return sample['caption']
