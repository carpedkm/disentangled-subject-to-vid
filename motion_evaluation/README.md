# Evaluation Protocol for Object Motion Naturalness

## Description
To evaluate model's capability of diverse object motion synthesis, we construct three benchmark video datasets with small, medium, and large object motions, each containing 500 video clips with minimal camera motions to avoid potential bias caused by camera motion.
The datasets are categorized based on the average magnitudes of the optical flow vectors of moving objects: smaller than 20 pixels (Pexels-small), between 20 and 40 pixels (Pexels-medium), and more than 40 pixels (Pexels-large).

To evaluate the video synthesis quality, we synthesize videos and compare these synthesized videos with the datasets.
Then, we evaluate the video synthesis performance of a given method through the Frechet Video Distance (FVD).
</p>

## Benchmark dataset construction
please see [here](obtain_benchmark/README.md).


## FVD evaluation
please see [here](evaluate_FVD/README.md).