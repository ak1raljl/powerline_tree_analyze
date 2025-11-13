# Powerline Tree Analysis Project

## Overview
This project focuses on analyzing LiDAR point cloud data for powerline infrastructure and tree clearance analysis. The primary purpose is to perform semantic segmentation of LiDAR data to identify powerlines, towers, poles, vegetation, and other infrastructure elements, and then analyze potential tree-encroachment risks near powerlines.

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