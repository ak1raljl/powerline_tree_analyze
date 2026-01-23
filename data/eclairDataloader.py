import os
import json
import numpy as np
import laspy
from tqdm import tqdm
from torch.utils.data import Dataset

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

class EclairDataset(Dataset):
    def __init__(
            self,
            root_dir,
            split='train',
            num_points=4096,
            block_size=20.0,
            sample_rate=1.0,
            transform=None,
            split_ratio=0.1
        ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.block_size = block_size
        self.sample_rate = sample_rate
        self.transform = transform
        self.split_ratio = split_ratio

        pointcloud_dir = os.path.join(self.root_dir, 'pointclouds')
        all_files = [f for f in os.listdir(pointcloud_dir) if f.endswith(('.las', '.laz'))]

        num_val = int(len(all_files) * self.split_ratio)
        if self.split == 'train':
            self.files = all_files[:-num_val] if num_val > 0 else all_files
        else:
            self.files = all_files[-num_val:] if num_val > 0 else []
        print(f"Loading {split} data: {len(self.files)} files.")

        self.scene_paths = []
        self.scene_coord_min = []
        self.scene_coord_max = []

        label_weights = np.zeros(len(ECLAIR_CLASSES))
        num_points_total = []

        for file_name in tqdm(self.files, desc=f'Preprocessing {split} data'):
            file_path = os.path.join(pointcloud_dir, file_name)
            self.scene_paths.append(file_path)

            las = laspy.read(file_path)
            points_xyz = np.vstack((las.x, las.y, las.z)).transpose()

            if hasattr(las, 'classification'):
                labels_orign = las.classification.astype(np.int32)
                labels = labels_orign - 1
            else:
                labels = np.ones(len(points_xyz), dtype=np.int32) * 10
            
            labels = np.clip(labels, 0, 10)

            tmp, _ = np.histogram(labels, range(12))
            label_weights += tmp

            coprd_min = np.amin(points_xyz, axis=0)
            coprd_max = np.amax(points_xyz, axis=0)

            self.scene_coord_min.append(coprd_min)
            self.scene_coord_max.append(coprd_max)
            num_points_total.append(len(points_xyz))

        label_weights = label_weights.astype(np.float32)
        label_weights = label_weights / np.sum(label_weights)
        self.label_weights = np.power(np.amax(label_weights) / (label_weights + 1e-6), 1/3.0)
        print("Label weights:", self.label_weights)

        sample_prob = num_points_total / np.sum(num_points_total)
        num_iter = int(np.sum(num_points_total) * self.sample_rate / self.num_points)
        scene_idxs = []
        for index in range(len(self.files)):
            scene_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.scene_idxs = np.array(scene_idxs)
        print(f"Total {len(self.scene_idxs)} samples in {split} set.")

    def __len__(self):
        return len(self.scene_idxs)

    def __getitem__(self, idx):
        scene_idx = self.scene_idxs[idx]
        file_path = self.scene_paths[scene_idx]

        las = laspy.read(file_path)
        points_xyz = np.vstack([las.x, las.y, las.z]).transpose()
        N_points = points_xyz.shape[0]

        features = [points_xyz]
        colors = np.vstack([las.red, las.green, las.blue]).transpose()
        colors = colors / 65535.0
        features.append(colors)

        points = np.hstack(features)

        if hasattr(las, 'classification'):
            labels = las.classification.astype(np.int32)
            labels = labels - 1
        else:
            labels = np.ones(N_points, dtype=np.int32) * 10

        labels = np.clip(labels, 0, 10)

        if N_points <= self.num_points:
            point_idxs = np.random.choice(N_points, self.num_points, replace=True)
            center = points[0, :3]
        else:
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]

        
        point_idxs = np.where(
            (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) &
            (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1])
        )[0]

        if point_idxs.size >= self.num_points:
            selected_point_idxs = np.random.choice(point_idxs, self.num_points, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_points, replace=True)
        
        selected_points = points[selected_point_idxs, :]
        coord_max = self.scene_coord_max[scene_idx]
        current_points = np.zeros((self.num_points, 9))
        current_points[:, 6] = selected_points[:, 0] / coord_max[0] 
        current_points[:, 7] = selected_points[:, 1] / coord_max[1]
        current_points[:, 8] = selected_points[:, 2] / coord_max[2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform:
            current_points = self.transform(current_points)
        return current_points, current_labels
