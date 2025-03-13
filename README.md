## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Selected Scripts Overview](#scripts-overview)
  - [generate_videos_fsvvd.sh](#generate_videos_fsvvdsh)
  - [align_FSVVD_preprocess_VVD.py](#align_fsvvd_preprocess_vvdpy)
  - [align_FSVVD_resample_ub.py](#align_fsvvd_resample_ubpy)
  - [align_FSVVD_rendering.py](#align_fsvvd_renderingpy)
- [License](#license)

## Introduction

This repository include orignal data process, model training, evaluation, and also include some script to render FoV images, generate FoV-view videos from user's FoV and PCV.
There are two dataset, 8i and FSVVD and they requires different preprocess scripts.

## Prerequisites

- **Python 3.8**
- **ffmpeg**: For video generation from image sequences.
- **Xvfb**: For off-screen rendering.
- **Required Python Libraries**: Install using `pip install -r requirements.txt`
- **data** save 8i and FSVVD PCV and user FoV trajectory data in ../point_cloud_data/



## CellSight Steps
### 1. Preprocess PCV and user FoV trajecory data 
align_FSVVD_preprocess_VVD.py reads FSVVD and preprocess
align_FSVVD_resample_ub.py resamples user behavior data to a consistent frame rate.
preprocess.py preprocees and downsample PCV on 8i and FSVVD data
### 2.1 node_feature_graph.py 
generate node feature base on different trajectory for 8i data
### 2.2 node_feature_graph.FSVVD.py
generate node feature base on different trajectory for FSVVD data
### 3. CellSight_train_eval.py
train and evaluate model by giving parameters
### 4. baseline_loss.py
calculate loss for baselines


## Others
### generate_videos_fsvvd.sh
A bash script used to generate videos from PNG files for multiple users. It automates the process of converting image sequences into MP4 videos using `ffmpeg`.
### align_FSVVD_preprocess_VVD.py
This script preprocesses Volumetric Video Data (VVD) files.
- Removes the alpha channel from images.
- Saves processed data to binary files.
- Forms 300 frames with looping to standardize the length of sequences.

### align_FSVVD_rendering.py
Renders and saves users' FoV images.
- Utilizes off-screen rendering to generate images without a display.
- Saves FoV images for each user for further processing or visualization.
**Note**: Requires setting up a virtual display using Xvfb.
### rendering_pc.py
render 8i data using fov and save to png figures. It can also be used to visualize the ply files using fov.
### baseline_trajectory_prediction.py
TLP, LR baseline
### node_feature_graph.py and node_feature_graph_FSVVD.py is used to generate node feature for baselines and ground truth
~/point_cloud_FoV_Graph/data/{video_name}_VS{voxel_size}_{baseline}/
node_feature for each user after trajectory prediction for {baseline}/
~/point_cloud_FoV_Graph/data/{video_name}_VS{voxel_size}/
node_feature for each user for ground truth trajectory
### baseline_loss.py can generate training/testing data for ours and baselines and evaluate loss(training script can also generate training/testing data for ours)
~/point_cloud_FoV_Graph/data/data/
training/testing data for our model for 8i dataset
~/point_cloud_FoV_Graph/data/fsvvd_raw/
training/testing data for our model for fsvvd_raw dataset

### 8i data
https://plenodb.jpeg.org/pc/8ilabs/
### 8i user FoV data
https://github.com/cwi-dis/6DoF-HMD-UserNavigationData
###  FSVVD data:
https://cuhksz-inml.github.io/user-behavior-in-vv-watching/factsfigures.html
or
https://drive.google.com/drive/folders/1le4dzPzfW975YGL1NkLdo3crym-PrX68?usp=sharing
### FSVVD user FoV data:
https://cuhksz-inml.github.io/full_scene_volumetric_video_dataset/factsfigures.html
### CellSight borrows part of the code structure from GraphGRU
https://github.com/GraphGRU/GraphGRU



