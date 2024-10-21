# Point Cloud FoV Graph

An initial representation of point cloud Field of View (FoV) using various scripts for preprocessing, resampling, rendering, and video generation for multiple users.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Scripts Overview](#scripts-overview)
  - [generate_videos_fsvvd.sh](#generate_videos_fsvvdsh)
  - [align_FSVVD_preprocess_VVD.py](#align_fsvvd_preprocess_vvdpy)
  - [align_FSVVD_resample_ub.py](#align_fsvvd_resample_ubpy)
  - [align_FSVVD_rendering.py](#align_fsvvd_renderingpy)
- [Usage](#usage)
- [Notes](#notes)
- [License](#license)

## Introduction

This repository contains scripts developed for processing point cloud data related to users' Field of View (FoV). The primary goal is to preprocess the data, resample user behavior, render FoV images, and generate videos from PNG files for multiple users.

## Prerequisites

- **Python 3.8**
- **ffmpeg**: For video generation from image sequences.
- **Xvfb**: For off-screen rendering.
- **Required Python Libraries**: Install using `pip install -r requirements.txt`

## Scripts Overview

### generate_videos_fsvvd.sh

A bash script used to generate videos from PNG files for multiple users. It automates the process of converting image sequences into MP4 videos using `ffmpeg`.

### align_FSVVD_preprocess_VVD.py

This script preprocesses Volumetric Video Data (VVD) files.

**Functions**:

- Removes the alpha channel from images.
- Saves processed data to binary files.
- Forms 300 frames with looping to standardize the length of sequences.

### align_FSVVD_resample_ub.py

Resamples user behavior data to a consistent frame rate.

**Purpose**: Adjusts the user behavior data to 60Hz to ensure synchronization across different datasets.

### align_FSVVD_rendering.py

Renders and saves users' FoV images.

**Features**:

- Utilizes off-screen rendering to generate images without a display.
- Saves FoV images for each user for further processing or visualization.

**Note**: Requires setting up a virtual display using Xvfb.