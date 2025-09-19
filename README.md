
# CellSight: Point Cloud Processing and Model Training

---

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Workflow Overview](#workflow-overview)
- [Key Scripts](#key-scripts)
- [Data Sources](#data-sources)
- [License](#license)


## Introduction

CellSight is a comprehensive toolkit for point cloud data processing, model training, and evaluation. It provides utilities for rendering Field of View (FoV) images, generating FoV-view videos, and handling user trajectory data. The repository supports two datasets: **8i** and **FSVVD**, each requiring dedicated preprocessing scripts.


## Prerequisites

- **ffmpeg**: Required for video generation from image sequences.
- **Xvfb**: Enables off-screen rendering (useful for remote/headless servers).
- **Python Libraries**:
  - Install dependencies: `pip install -r requirements.txt`
  - Open3D-related scripts require Python 3.10.14 and `requirements_open3d.txt` (e.g., `node_feature_graph.py`, `align_FSVVD_rendering.py`, `rendering_pc.py`).
  - CellSight model training/evaluation uses Python 3.9.19 and `requirements_model.txt`.
  - Open3D requires a specific version for compatibility; CellSight model training is less restrictive and supports newer libraries.
- **Data Directory**: Store 8i and FSVVD point cloud and user FoV trajectory data in `../point_cloud_data/`.


## Workflow Overview

1. **Preprocessing**
  - `align_FSVVD_preprocess_VVD.py`: Reads and preprocesses FSVVD data.
  - `align_FSVVD_resample_ub.py`: Resamples user behavior data to a consistent frame rate.
  - `preprocess.py`: Processes and downsamples point cloud data for both 8i and FSVVD datasets.
2. **Node Feature Generation**
  - `node_feature_graph.py`: Generates node features based on user trajectories for 8i data.
  - `node_feature_graph_FSVVD.py`: Generates node features for FSVVD data.
3. **Model Training & Evaluation**
  - `CellSight_train_eval.py`: Trains and evaluates models with configurable parameters.
4. **Baseline Evaluation**
  - `baseline_loss.py`: Computes loss metrics for baseline models.


## Key Scripts

- **generate_videos_fsvvd.sh**: Automates video generation from PNG sequences for multiple users using `ffmpeg`.
- **align_FSVVD_preprocess_VVD.py**: Preprocesses Volumetric Video Data (VVD) files by removing alpha channels, saving to binary, and standardizing sequence length (300 frames).
- **align_FSVVD_rendering.py**: Renders and saves users' FoV images via off-screen rendering. *Requires Xvfb for virtual display setup.*
- **rendering_pc.py**: Renders 8i data using FoV and saves as PNG images; also visualizes PLY files.
- **baseline_trajectory_prediction.py**: Implements TLP and LR baseline trajectory prediction.
- **node_feature_graph.py / node_feature_graph_FSVVD.py**: Generates node features for baselines and ground truth.
  - Output directories:
    - `~/point_cloud_FoV_Graph/data/{video_name}_VS{voxel_size}_{baseline}/`: Node features after trajectory prediction for each baseline.
    - `~/point_cloud_FoV_Graph/data/{video_name}_VS{voxel_size}/`: Node features for ground truth trajectories.
- **baseline_loss.py**: Generates training/testing data and evaluates loss for both CellSight and baseline models.
  - Output directories:
    - `~/point_cloud_FoV_Graph/data/data/`: Training/testing data for 8i dataset.
    - `~/point_cloud_FoV_Graph/data/fsvvd_raw/`: Training/testing data for FSVVD dataset.

## Data Sources

- **8i Point Cloud Data**: [8i Labs](https://plenodb.jpeg.org/pc/8ilabs/)
- **8i User FoV Data**: [6DoF-HMD-UserNavigationData](https://github.com/cwi-dis/6DoF-HMD-UserNavigationData)
- **FSVVD Data**:
  - [CUHK Facts & Figures](https://cuhksz-inml.github.io/user-behavior-in-vv-watching/factsfigures.html)
  - [Google Drive](https://drive.google.com/drive/folders/1le4dzPzfW975YGL1NkLdo3crym-PrX68?usp=sharing)
- **FSVVD User FoV Data**: [CUHK Full Scene Volumetric Video Dataset](https://cuhksz-inml.github.io/full_scene_volumetric_video_dataset/factsfigures.html)
- **Reference Implementation**: CellSight borrows part of its code structure from [GraphGRU](https://github.com/GraphGRU/GraphGRU)



