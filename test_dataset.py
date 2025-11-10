import os
import laspy
import numpy as np

dataset_dir = "/home/ak1ra/ladiar/powerline_tree_analyze/data/eclair/pointclouds"
print("number of files:", len(os.listdir(dataset_dir)))

laz_files = [f for f in os.listdir(dataset_dir) if f.endswith('.laz')]
laz_files = sorted(laz_files, key=lambda x: int(x.split('_')[1].split('.')[0]))

file_count = 0
bad_file_count = 0

for file_name in laz_files:
    file_path = os.path.join(dataset_dir, file_name)
    try:
        las = laspy.read(file_path)
        points_xyz = np.vstack((las.x, las.y, las.z)).transpose()
        point_count = len(las.points)
        if point_count <= 8000:
            bad_file_count += 1
            print(f"{file_name:<25} {point_count:<15}")
        file_count += 1
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
print(f"total files processed: {file_count}, bad files: {bad_file_count}")