# Powerline Tree Analysis Project

## Overview
This project focuses on analyzing LiDAR point cloud data for powerline infrastructure and tree clearance analysis. The primary purpose is to perform semantic segmentation of LiDAR data to identify powerlines, towers, poles, vegetation, and other infrastructure elements, and then analyze potential tree-encroachment risks near powerlines.

## Key Features

### Data Processing
- **LiDAR Data Support**: Handles `.laz` files using the `laspy` library
- **Semantic Segmentation**: Classifies points into 11 categories (ground, vegetation, building, transmission_wire, distribution_wire, pole, transmission_tower, fence, vehicle, noise, unassigned)
- **Point Cloud Processing**: Uses Open3D for point cloud operations like downsampling, outlier removal, plane segmentation, and clustering

### Tree Clearance Analysis
- **Classification-based Analysis**: Determines clearance distances between conductors and vegetation
- **Export Capabilities**: Generates CSV reports and LAS files with encroachment points
- **Configurable Thresholds**: Allows setting custom clearance distances (default 4.0 meters)

## Key Files and Functions

### eclairDataloader.py
- Implements a PyTorch Dataset class for the ECLAIR dataset
- Handles data loading, preprocessing and augmentation
- Supports train/validation/test splits
- Computes label weights for handling class imbalance

### train.py
- Training script for segmentation models
- Configurable parameters for model training (epochs, batch size, learning rate)

## Dataset
The project uses the ECLAIR dataset which contains:
- Multiple `.laz` files with LiDAR point cloud data
- Labels.json file defining train/validation/test splits
- 11 semantic classes for point cloud segmentation
- Quality control with approved/rejected classifications for tiles

## Building and Running

### Prerequisites
- Python 3.8
- torch 2.9.0+cu126
- laspy 2.6.1
- open3d 0.19.0
- numpy,scipy,pandas


### Running Train
```bash
python train.py --model pointnet2_sem_seg --batch_size 64 --epoch 32
```

## Development Conventions
- Point cloud processing follows standard LiDAR data processing workflows
- Semantic segmentation classes follow the ECLAIR classification scheme
- Uses PyTorch conventions for dataset loading and model training
- Code follows standard Python style guidelines

## Use Cases
1. **Powerline Infrastructure Analysis**: Identifying and classifying powerline components
2. **Tree Clearance Analysis**: Ensuring safe distances between vegetation and power lines
3. **Infrastructure Monitoring**: Tracking changes in powerline environments over time
4. **Risk Assessment**: Identifying potential hazards from vegetation encroachment

## Key Classes and Labels
The semantic segmentation model classifies point clouds into these categories:
- 0: ground
- 1: vegetation
- 2: building
- 3: transmission_wire
- 4: distribution_wire
- 5: pole
- 6: transmission_tower
- 7: fence
- 8: vehicle
- 9: noise
- 10: unassigned