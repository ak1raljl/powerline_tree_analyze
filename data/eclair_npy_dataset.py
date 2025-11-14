import os
import json
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

ECLAIR_CLASSES = [
    'ground',              # 0: ground
    'vegetation',          # 1: vegetation
    'building',            # 2: building
    'transmission_wire',   # 3: transmission_wire
    'distribution_wire',   # 4: distribution_wire
    'pole',                # 5: pole
    'transmission_tower',  # 6: transmission_tower
    'fence',               # 7: fence
    'vehicle',             # 8: vehicle
    'noise',               # 9: noise
    'unassigned'           # 10: unassigned
]

class2label = {cls: idx for idx, cls in enumerate(ECLAIR_CLASSES)}
seg_label_to_cat = {i: ECLAIR_CLASSES[i] for i in range(len(ECLAIR_CLASSES))}


class EclairNpyDataset(Dataset):
    def __init__(
            self,
            npy_dir,
            split='train',
            num_points=4096,
            block_size=20.0,
            sample_rate=1.0,
            transform=None,
            split_ratio=0.1,
            use_all_features=True
        ):

        super().__init__()
        self.npy_dir = npy_dir
        self.split = split
        self.num_points = num_points
        self.block_size = block_size
        self.sample_rate = sample_rate
        self.transform = transform
        self.split_ratio = split_ratio
        self.use_all_features = use_all_features

        data_dir = os.path.join(self.npy_dir, 'data')
        all_data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_data.npy')])

        if not all_data_files:
            raise RuntimeError(f"no _data.npy files found in {data_dir}")

        num_val = int(len(all_data_files) * self.split_ratio)
        if self.split == 'train':
            self.data_files = all_data_files[:-num_val] if num_val > 0 else all_data_files
        else:
            self.data_files = all_data_files[-num_val:] if num_val > 0 else []

        print(f"loading {split} data: {len(self.data_files)} files")

        self.scene_paths = []
        self.scene_coord_min = []
        self.scene_coord_max = []
        self.num_points_total = []

        label_weights = np.zeros(len(ECLAIR_CLASSES))

        for data_file in tqdm(self.data_files, desc=f'preprocessing {split} data'):
            data_path = os.path.join(data_dir, data_file)
            self.scene_paths.append(data_path)

            points = np.load(data_path, mmap_mode='r')

            if self.use_all_features and points.shape[1] > 3:
                points_xyz = points[:, :3]
            else:
                points_xyz = points

            coord_min = np.min(points_xyz, axis=0)
            coord_max = np.max(points_xyz, axis=0)

            self.scene_coord_min.append(coord_min)
            self.scene_coord_max.append(coord_max)
            self.num_points_total.append(points.shape[0])

            label_file = data_file.replace('_data.npy', '_labels.npy')
            label_path = os.path.join(self.npy_dir, 'labels', label_file)

            if os.path.exists(label_path):
                labels = np.load(label_path, mmap_mode='r')
                tmp, _ = np.histogram(labels, range(len(ECLAIR_CLASSES) + 1))
                label_weights += tmp

        if label_weights.sum() > 0:
            label_weights = label_weights.astype(np.float32)
            label_weights = label_weights / np.sum(label_weights)
            self.label_weights = np.power(np.max(label_weights) / (label_weights + 1e-6), 1/3.0)
        else:
            self.label_weights = np.ones(len(ECLAIR_CLASSES), dtype=np.float32)

        print("labels weight:", self.label_weights)

        sample_prob = np.array(self.num_points_total) / np.sum(self.num_points_total)
        num_iter = int(np.sum(self.num_points_total) * self.sample_rate / self.num_points)
        scene_idxs = []
        for index in range(len(self.data_files)):
            scene_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.scene_idxs = np.array(scene_idxs)

        print(f"Total {len(self.scene_idxs)} samples in {split} set.")

    def __len__(self):
        return len(self.scene_idxs)

    def __getitem__(self, idx):
        scene_idx = self.scene_idxs[idx]
        data_path = self.scene_paths[scene_idx]

        points = np.load(data_path)

        if self.use_all_features and points.shape[1] >= 6:
            points_xyz = points[:, :3]
            colors = points[:, 3:6]
            points_combined = np.hstack([points_xyz, colors])
        else:
            points_xyz = points
            points_combined = points

        data_file = os.path.basename(data_path)
        label_file = data_file.replace('_data.npy', '_labels.npy')
        label_path = os.path.join(self.npy_dir, 'labels', label_file)

        if os.path.exists(label_path):
            labels = np.load(label_path)
        else:
            labels = np.ones(len(points_xyz), dtype=np.int32) * 10

        N_points = points_xyz.shape[0]

        MIN_POINTS = 32

        if N_points < MIN_POINTS:
            selected_point_idxs = np.random.choice(N_points, self.num_points, replace=True)
            center = np.mean(points_xyz, axis=0)
        elif N_points <= self.num_points:
            selected_point_idxs = np.random.choice(N_points, self.num_points, replace=True)
            center = np.mean(points_xyz, axis=0)
        else:
            center_idx = np.random.choice(N_points)
            center = points_xyz[center_idx]

            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]

            point_idxs = np.where(
                (points_xyz[:, 0] >= block_min[0]) & (points_xyz[:, 0] <= block_max[0]) &
                (points_xyz[:, 1] >= block_min[1]) & (points_xyz[:, 1] <= block_max[1])
            )[0]

            if len(point_idxs) < MIN_POINTS:
                point_idxs = np.arange(N_points)

            if len(point_idxs) >= self.num_points:
                selected_point_idxs = np.random.choice(point_idxs, self.num_points, replace=False)
            else:
                selected_point_idxs = np.random.choice(point_idxs, self.num_points, replace=True)

        selected_points = points_combined[selected_point_idxs]
        current_labels = labels[selected_point_idxs]

        coord_max = self.scene_coord_max[scene_idx]
        current_points = np.zeros((self.num_points, 9))

        current_points[:, 6] = selected_points[:, 0] / coord_max[0]
        current_points[:, 7] = selected_points[:, 1] / coord_max[1]
        current_points[:, 8] = selected_points[:, 2] / coord_max[2]

        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        current_points[:, 0:selected_points.shape[1]] = selected_points

        if self.transform:
            current_points = self.transform(current_points)

        return current_points, current_labels


class EclairNpyDatasetFullCloud(Dataset):
    def __init__(self, npy_dir, use_all_features=True):
        self.npy_dir = npy_dir
        self.use_all_features = use_all_features

        data_dir = os.path.join(self.npy_dir, 'data')
        self.data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_data.npy')])

        if not self.data_files:
            raise RuntimeError(f"no _data.npy files found in {data_dir}")

        print(f"loading full point cloud data: {len(self.data_files)} files")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        data_path = os.path.join(self.npy_dir, 'data', data_file)

        points = np.load(data_path)

        if self.use_all_features and points.shape[1] >= 6:
            points_xyz = points[:, :3]
            colors = points[:, 3:6]
        else:
            points_xyz = points
            colors = None
        
        label_file = data_file.replace('_data.npy', '_labels.npy')
        label_path = os.path.join(self.npy_dir, 'labels', label_file)

        labels = None
        if os.path.exists(label_path):
            labels = np.load(label_path)

        return {
            'points': points_xyz,
            'colors': colors,
            'labels': labels,
            'filename': data_file
        }
