# Powerline Tree Analysis Project

## Overview
This project focuses on analyzing LiDAR point cloud data for powerline infrastructure and tree clearance analysis. The primary purpose is to perform semantic segmentation of LiDAR data to identify powerlines, towers, poles, vegetation, and other infrastructure elements, and then analyze potential tree-encroachment risks near powerlines.

## Dataset Format

The project supports two data formats:

### 1. LAZ/LAS Format (Original)
- Multiple `.laz` or `.las` files in `data/eclair/pointclouds/`
- Direct loading from LAZ files (slow, single-threaded)
- Suitable for small datasets

### 2. NPY Format (Recommended)
- Pre-converted binary format for faster loading
- Supports multi-threaded data loading (num_workers > 0)
- Stored in `data/eclair/npy_preprocessed/`:
  ```
  data/eclair/npy_preprocessed/
  ├── data/
  │   ├── pointcloud_1_data.npy    # XYZ + RGB (N, 6)
  │   ├── pointcloud_2_data.npy
  │   └── ...
  └── labels/
      ├── pointcloud_1_labels.npy  # Labels (N,)
      ├── pointcloud_2_labels.npy
      └── ...
  ```

## Building and Running

### Prerequisites
- Python 3.8+
- PyTorch 2.9.0+cu126
- laspy 2.6.1 (for LAZ/LAS loading)
- open3d 0.19.0 (for visualization)
- numpy, scipy, pandas

### Quick Start (Recommended: NPY Format)

#### Step 1: Convert LAZ/LAS to NPY (One-time)
```bash
# Convert all data with parallel processing (8 workers)
python convert_laz_to_npy.py --input_dir <dataset_path> --output_dir <output_path> --num_workers 8

# Optional: Enable quality filtering using labels.json
python convert_laz_to_npy.py --input_dir <dataset_path> --output_dir <output_path> --use_quality_filter --num_workers 8
```

#### Step 2: Train Model
```bash
# Basic training
python train_npy.py \
    --model pointnet2_sem_seg \
    --npy_dir data/eclair/npy_preprocessed \
    --batch_size 32 \
    --epoch 50 \
    --num_workers 4 \
    --pin_memory

# High-performance training (recommended for multi-core CPUs)
python train_npy.py \
    --model pointnet2_sem_seg \
    --npy_dir data/eclair/npy_preprocessed \
    --batch_size 64 \
    --epoch 100 \
    --num_workers 8 \
    --pin_memory \
    --learning_rate 0.001 \
    --block_size 20.0

# Resume from checkpoint
python train_npy.py \
    --npy_dir data/eclair/npy_preprocessed \
    --log_dir your_previous_experiment \
    --num_workers 4
```

#### Step 3: Test and Visualize
```bash
# Visualize results (supports both model formats)
python visualize.py \
    --model_path logs/eclair_npy_seg/<timestamp>/checkpoints/best_model.pth \
    --laz_file data/eclair/pointclouds/pointcloud_1.laz \
    --use_dbscan \
    --visualize \
    --export_las \
    --clearance_threshold 4.0

# View LAZ file directly
python view_laz.py data/eclair/pointclouds/pointcloud_1.laz
```

### Alternative: LAZ/LAS Format (Original)
```bash
# Training with direct LAZ loading (slower, single-threaded)
python train.py \
    --model pointnet2_sem_seg \
    --batch_size 32 \
    --epoch 50 \
    --data_root data/eclair/

# Note: train.py does not support num_workers > 0
```

## Key Parameters

### Training Parameters
- `--model`: Model architecture (default: pointnet2_sem_seg)
- `--batch_size`: Batch size (default: 32, recommended: 64 for better GPU utilization)
- `--epoch`: Number of training epochs (default: 50)
- `--learning_rate`: Initial learning rate (default: 0.001)
- `--npoints`: Points per sample (default: 4096)
- `--block_size`: Block size in meters (default: 20.0)

### Performance Parameters (NPY format only)
- `--num_workers`: Number of data loading workers (default: 0, recommended: 4-8)
- `--pin_memory`: Enable pin_memory for faster GPU transfer (recommended)
- `--sample_rate`: Sampling rate for data loading (default: 1.0)

### Data Parameters
- `--npy_dir`: Path to NPY dataset (default: data/eclair/npy_preprocessed)
- `--split_ratio`: Validation split ratio (default: 0.1)
- `--use_quality_filter`: Filter out rejected tiles (convert_laz_to_npy.py only)

## Key Classes and Labels
The semantic segmentation model classifies point clouds into these categories:
- **0: ground** - Ground/terrain
- **1: vegetation** - Trees, shrubs, plants
- **2: building** - Buildings, structures
- **3: transmission_wire** - High-voltage transmission lines
- **4: distribution_wire** - Distribution power lines
- **5: pole** - Utility poles
- **6: transmission_tower** - Large transmission towers
- **7: fence** - Fences, barriers
- **8: vehicle** - Cars, trucks, vehicles
- **9: noise** - Noise points, outliers
- **10: unassigned** - Unclassified points


## Output and Logging

Training logs are saved to:
```
logs/eclair_npy_seg/<timestamp>/
├── checkpoints/
│   ├── best_model.pth          # Best model (highest mIoU)
│   └── epoch_X_model.pth       # Periodic checkpoints
└── log.txt                     # Training log
```

### Validation Output
During training, each epoch shows:
```
Epoch 1 (1/50)
Learning rate: 0.001000
BN momentum: 0.1000
training: 100%|██████████| 4013/4013 [18:23<00:00, 3.64it/s]
Training mean loss: 0.892341
Training accuracy: 0.783456

Epoch 1 validation...
validation: 100%|██████████| 516/516 [02:15<00:00, 3.82it/s]
Validation mean loss: 1.234567
Validation mIoU: 0.654321

Per class IoU:
  0 (ground             ): 0.9382
  1 (vegetation         ): 0.8234
  2 (building           ): 0.8912
  3 (transmission_wire  ): 0.4521
  4 (distribution_wire  ): 0.3876
  5 (pole               ): 0.5643
  6 (transmission_tower ): 0.7821
  7 (fence              ): 0.2345
  8 (vehicle            ): 0.5678
  9 (noise              ): 0.1234
 10 (unassigned         ): N/A

Best model (mIoU: 0.6543)
```

## Troubleshooting

### Common Issues

**RuntimeError: no _data.npy files found**
- Ensure NPY files are in `data/eclair/npy_preprocessed/data/`
- Verify conversion completed successfully: `ls data/eclair/npy_preprocessed/data/ | wc -l`

**CUDA out of memory**
- Reduce batch size: `--batch_size 16`
- Reduce num_workers: `--num_workers 2`
- Disable pin_memory or use smaller block_size

**Slow training**
- Ensure using NPY format (not LAZ)
- Increase num_workers (4-8 recommended)
- Enable pin_memory
- Store data on SSD

**Low GPU utilization**
- Increase batch_size (64-128)
- Increase num_workers
- Check data loading speed: `python test_npy_dataloader.py`

### Model Files
- `best_model.pth`: Best model based on validation mIoU
- `epoch_X_model.pth`: Checkpoint at epoch X (use `--save_freq` to control frequency)

## Citation
If you use this code or dataset, please cite the ECLAIR dataset and PointNet++ paper.

## Dataset
The project uses the ECLAIR dataset which contains:
- Multiple `.laz` or `.las` files with LiDAR point cloud data
- Labels.json file defining train/validation/test splits and quality control
- 11 semantic classes for point cloud segmentation
- Quality control with approved/rejected classifications for tiles

The dataset can be loaded in two formats:
- **LAZ/LAS Format**: Original files, slow loading, single-threaded
- **NPY Format**: Pre-converted binary files, fast loading, multi-threaded (recommended)

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

### Visualize
```bash
# no Clustering
python visualize.py --model_path model.pth --use_dbscan --visualize --export_las
# Clustering all classes
python visualize_dbscan.py --model_path model.pth --use_dbscan --visualize --export_las
# Clustering choosing classes
python visualize.py --model_path model.pth --use_dbscan --dbscan_classes 1 3 4 --visualize
```