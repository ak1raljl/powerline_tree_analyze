import os
import numpy as np
import laspy
from tqdm import tqdm
import argparse
import json
from multiprocessing import Pool, cpu_count
from functools import partial

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

def convert_single_file(laz_file_path, output_dir, use_quality_filter=False):
    try:
        las = laspy.read(laz_file_path)
        filename = os.path.basename(laz_file_path)
        file_basename = os.path.splitext(filename)[0]

        if use_quality_filter:
            labels_json_path = os.path.join(os.path.dirname(os.path.dirname(laz_file_path)), 'labels.json')
            if os.path.exists(labels_json_path):
                with open(labels_json_path, 'r') as f:
                    labels_data = json.load(f)
                for tile_info in labels_data:
                    if tile_info.get('tile_name') == filename:
                        if tile_info.get('review_category') == 'rejected':
                            print(f"skip: {filename}")
                            return False
                        break

        points_xyz = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float32)

        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            colors = np.vstack([las.red, las.green, las.blue]).transpose().astype(np.float32)
            colors = colors / 65535.0  # normalize to 0-1
            points = np.hstack([points_xyz, colors])
        else:
            points = points_xyz

        if hasattr(las, 'classification'):
            labels = las.classification.astype(np.int32)
            labels = labels - 1  # convert to 0-10 range
            labels = np.clip(labels, 0, 10)
        else:
            labels = np.ones(len(points_xyz), dtype=np.int32) * 10  # unclassified

        data_dir = os.path.join(output_dir, 'data')
        labels_dir = os.path.join(output_dir, 'labels')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        output_points_path = os.path.join(data_dir, f"{file_basename}_data.npy")
        output_labels_path = os.path.join(labels_dir, f"{file_basename}_labels.npy")

        np.save(output_points_path, points.astype(np.float32))
        np.save(output_labels_path, labels)

        print(f"finish: {filename} -> {points.shape[0]} points")
        return True

    except Exception as e:
        print(f"✗ failed: {laz_file_path}, error: {str(e)}")
        return False

def convert_files_parallel(laz_files, output_dir, use_quality_filter=False, num_workers=None):
    os.makedirs(output_dir, exist_ok=True)

    if num_workers is None:
        num_workers = min(cpu_count(), len(laz_files))

    print(f"using {num_workers} processes for parallel conversion...")

    convert_func = partial(convert_single_file, output_dir=output_dir, use_quality_filter=use_quality_filter)

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(convert_func, laz_files),
            total=len(laz_files),
            desc="Converting LAZ/LAS to NPY"
        ))

    success_count = sum(results)
    print(f"\nfinish: {success_count}/{len(laz_files)} files")

    return success_count

def main():
    parser = argparse.ArgumentParser(description='Convert LAZ/LAS files to NPY format')
    parser.add_argument('--input_dir', type=str, default='data/eclair/pointclouds')
    parser.add_argument('--output_dir', type=str, default='data/eclair/npy_preprocessed')
    parser.add_argument('--use_quality_filter', action='store_true')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--file_list', type=str, nargs='+', default=None)

    args = parser.parse_args()

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Quality filtering: {'Enabled' if args.use_quality_filter else 'Disabled'}")

    # Get LAZ/LAS file list
    if args.file_list:
        laz_files = [os.path.join(args.input_dir, f) for f in args.file_list]
    else:
        laz_files = [os.path.join(args.input_dir, f)
                     for f in os.listdir(args.input_dir)
                     if f.endswith(('.las', '.laz'))]

    if not laz_files:
        print(f"Error: No LAZ/LAS files found in directory {args.input_dir}")
        return

    print(f"Found {len(laz_files)} files")

    os.makedirs(args.output_dir, exist_ok=True)

    success_count = convert_files_parallel(
        laz_files,
        args.output_dir,
        use_quality_filter=args.use_quality_filter,
        num_workers=args.num_workers
    )

    print(f"finish: {success_count}/{len(laz_files)} files")
    print(f"Output directory: {args.output_dir}")

if __name__ == '__main__':
    main()
