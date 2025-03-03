import pandas as pd

vid_metadata = "D:\\data\\webvid\\results_2M_train_k400_filtered.csv"
metadata = pd.read_csv(vid_metadata, error_bad_lines=False)

print(metadata['name'][200000])
print(len(metadata['name']))