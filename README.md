# Powerline Tree Analysis Project

## Overview
This project focuses on analyzing LiDAR point cloud data for powerline infrastructure and tree clearance analysis. The primary purpose is to perform semantic segmentation of LiDAR data to identify powerlines, towers, poles, vegetation, and other infrastructure elements, and then analyze potential tree-encroachment risks near powerlines.

The project uses LiDAR point cloud data and performs semantic segmentation to classify points into categories like ground, vegetation, building, transmission wire, distribution wire, pole, transmission tower, fence, vehicle, noise, and unassigned. It then analyzes clearance between vegetation and power lines to identify potential safety hazards.

## Project Structure
```
├── data/                    # Dataset directory
│   ├── eclair/             # ECLAIR dataset
│   │   ├── labels.json     # Label definitions and split information
│   │   └── pointclouds/    # LiDAR point cloud files (.laz)
│   └── eclairDataloader.py # PyTorch dataset and loader implementation
├── train.py                # Training script for segmentation models
├── test_seg.py             # Tree clearance analysis script
├── test_ladiar.py          # LiDAR visualization and processing script
└── test_dataset.py         # Dataset inspection script
```

## Key Features

### Data Processing
- **LiDAR Data Support**: Handles `.laz` files using the `laspy` library
- **Semantic Segmentation**: Classifies points into 11 categories (ground, vegetation, building, transmission_wire, distribution_wire, pole, transmission_tower, fence, vehicle, noise, unassigned)
- **Point Cloud Processing**: Uses Open3D for point cloud operations like downsampling, outlier removal, plane segmentation, and clustering

### Tree Clearance Analysis
- **Classification-based Analysis**: Determines clearance distances between conductors and vegetation
- **Export Capabilities**: Generates CSV reports and LAS files with encroachment points
- **Configurable Thresholds**: Allows setting custom clearance distances (default 4.0 meters)

### Visualization
- **Open3D Integration**: Visualizes processed point clouds with configurable settings
- **Color Coding**: Different colors for different classified elements

## Key Files and Functions

### eclairDataloader.py
- Implements a PyTorch Dataset class for the ECLAIR dataset
- Handles data loading, preprocessing and augmentation
- Supports train/validation/test splits
- Computes label weights for handling class imbalance

### train.py
- Training script for segmentation models
- Configurable parameters for model training (epochs, batch size, learning rate)

## Technologies Used
- **Python**: Core programming language
- **PyTorch**: Deep learning framework for segmentation model training
- **laspy**: Reading/writing LAS/LAZ point cloud files
- **Open3D**: 3D data processing and visualization
- **NumPy**: Numerical computations
- **SciPy**: Spatial data structures (cKDTree for efficient nearest neighbor searches)
- **Pandas**: Data manipulation for reports

## Dataset
The project uses the ECLAIR dataset which contains:
- Multiple `.laz` files with LiDAR point cloud data
- Labels.json file defining train/validation/test splits
- 11 semantic classes for point cloud segmentation
- Quality control with approved/rejected classifications for tiles

## Building and Running

### Prerequisites
- Python 3.7+
- PyTorch
- laspy
- Open3D
- NumPy, SciPy, Pandas

### Installation
```bash
pip install torch laspy open3d numpy scipy pandas tqdm
```

### Running Analysis
- For tree clearance analysis: `python test_seg.py <input.las|laz> [clearance_m] [out_prefix]`
- For training: `python train.py --model <model_name> --batch_size <size> --epoch <count>`
- For dataset inspection: `python test_dataset.py`

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

This project provides a comprehensive workflow for analyzing powerline infrastructure from LiDAR data, with a focus on identifying potential tree encroachment risks that could affect power delivery and safety.