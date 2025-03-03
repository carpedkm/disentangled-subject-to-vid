# FVD evaluation

## Benchmark dataset preparation
* Construct benchmark dataset first.
* File tree
```bash
./[benchmark_dataset_name]/
├── small
│   ├ metadata.jsonl 
│   ├── first_frame
│   ├── first_frame_latent
│   ├── video
│   ├── video_frames
│   ├   ├── [scene_name] (e.g., 2917346)
│   ├   ├   ├── 0.png
│   ├   ├   ├── 1.png
│   ├   ├   ├── ...
│   ├── video_latent
├── medium
│   ├ ...
├── large
│   ├ ...
```

## Synthesized videos preparation
* Synthesize videos and save them as follows
* File tree
```bash
./[Path_to_Synthesized_Videos]/
├── small
│   ├── video_frames
│   ├   ├── [scene_name] (e.g., 2917346)
│   ├   ├   ├── 0.png
│   ├   ├   ├── 1.png
│   ├   ├   ├── ...
├── medium
│   ├ ...
├── large
│   ├ ...
```

## Evaluate FVD by comparing synthesized videos and those of benchmark dataset
```bash
# Modify [Path_to_Benchmark]
# Modify [Path_to_Synthesized_Video]
# Modify [Save_Path]
bash src/evaluation_scripts/evaluate_your_model.sh
```