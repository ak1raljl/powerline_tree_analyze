import argparse
import os
import sys
import torch
import numpy as np
import laspy
import open3d as o3d
from torch.utils.data import DataLoader
import random
from data.eclairDataloader import EclairDataset, ECLAIR_CLASSES
import importlib
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from models.pointnet2_sem_seg import get_model

# Define colors for each class
def get_class_colors():
    colors = {
        0: [0.5, 0.5, 0.5],       # ground - gray
        1: [0.0, 1.0, 0.0],       # vegetation - green
        2: [1.0, 0.0, 0.0],       # building - red
        3: [1.0, 1.0, 0.0],       # transmission_wire - yellow
        4: [1.0, 0.5, 0.0],       # distribution_wire - orange
        5: [0.5, 0.0, 1.0],       # pole - purple
        6: [0.0, 0.5, 1.0],       # transmission_tower - light blue
        7: [0.7, 0.3, 0.0],       # fence - brown
        8: [1.0, 0.0, 1.0],       # vehicle - magenta
        9: [0.3, 0.3, 0.3],       # noise - dark gray
        10: [0.8, 0.8, 0.8]       # unassigned - light gray
    }
    return colors

def load_model(model_path, num_classes=11):
    classifier = get_model(num_classes).cuda()

    checkpoint = torch.load(model_path, weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    
    return classifier

def run_inference_on_file(model, file_path, num_points=4096, device='cuda', full_cloud=False):
    las = laspy.read(file_path)
    points_xyz = np.vstack((las.x, las.y, las.z)).transpose()
    N_points = points_xyz.shape[0]

    features = [points_xyz]
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        colors = np.vstack([las.red, las.green, las.blue]).transpose()
        colors = colors / 65535.0  # Normalize colors to 0-1 range
        features.append(colors)
    
    if hasattr(las, "classification"):
        gt_labels = las.classification.astype(np.int32)
        gt_labels = gt_labels - 1
        gt_labels = np.clip(gt_labels, 0, 10)
    else:
        gt_labels = None
    
    all_points = np.hstack(features)

    coord_min = np.amin(points_xyz, axis=0)
    coord_max = np.amax(points_xyz, axis=0)
    
    all_predictions = np.zeros(N_points, dtype=np.int32)
    vote_count = np.zeros(N_points, dtype=np.int32)
    
    print(f"Processing entire cloud with {N_points} points using sliding window")
    for start_idx in range(0, N_points, num_points):  # 50% overlap for better coverage
        end_idx = min(start_idx + num_points, N_points)
        batch_size = end_idx - start_idx
        
        selected_points = all_points[start_idx:end_idx]
        corresponding_indices = np.arange(start_idx, end_idx)

        if batch_size < num_points:
            padding_needed = num_points - batch_size
            padding_indices = np.random.choice(batch_size, padding_needed, replace=True)
            selected_points = np.vstack([selected_points, selected_points[padding_indices]])

        center = np.mean(selected_points[:batch_size, :3], axis=0)

        current_points = np.zeros((num_points, 9))
        current_points[:, 6] = selected_points[:, 0] / coord_max[0]  # Normalize coordinates
        current_points[:, 7] = selected_points[:, 1] / coord_max[1]
        current_points[:, 8] = selected_points[:, 2] / coord_max[2]
        selected_points_centered = np.copy(selected_points)
        selected_points_centered[:, 0] = selected_points_centered[:, 0] - center[0]  # Center the batch
        selected_points_centered[:, 1] = selected_points_centered[:, 1] - center[1]
        current_points[:, 0:6] = selected_points_centered
        
        model_input = torch.tensor(current_points[np.newaxis, :, :], dtype=torch.float32).to(device)
        model_input = model_input.transpose(2, 1)  # Transpose for model input: (B, C, N)
        
        with torch.no_grad():
            seg_pred, _ = model(model_input)  # Output shape: (B, N, C)
            pred_choice = seg_pred.contiguous().cpu().data.max(2)[1]  # Get predictions: (B, N)
            pred_choice = pred_choice[0]  # Get the first (and only) batch: (N,)
        
        all_predictions[corresponding_indices] += pred_choice[:batch_size].numpy()
        vote_count[corresponding_indices] += 1
        
        if (start_idx // num_points) % 10 == 0:
            print(f"Processed {end_idx}/{N_points} points ({100*end_idx/N_points:.1f}%)")
    
    non_zero_votes = vote_count > 0
    all_predictions[non_zero_votes] = np.floor_divide(all_predictions[non_zero_votes], vote_count[non_zero_votes])
        
    
    return all_predictions, points_xyz, gt_labels, hasattr(las, 'red')

def export_to_las(predictions, original_file_path, output_file_path):
    original_las = laspy.read(original_file_path)
    
    header = laspy.LasHeader(point_format=original_las.header.point_format, version=original_las.header.version)
    header.x_scale = original_las.header.x_scale
    header.y_scale = original_las.header.y_scale
    header.z_scale = original_las.header.z_scale
    header.x_offset = original_las.header.x_offset
    header.y_offset = original_las.header.y_offset
    header.z_offset = original_las.header.z_offset
    
    header.max = original_las.header.max
    header.min = original_las.header.min
    
    las_out = laspy.LasData(header)

    las_out.x = original_las.x
    las_out.y = original_las.y
    las_out.z = original_las.z
    
    las_out.classification = (predictions + 1).astype(np.uint8)
    
    if hasattr(original_las, 'red'):
        las_out.red = original_las.red
        las_out.green = original_las.green
        las_out.blue = original_las.blue
    if hasattr(original_las, 'intensity'):
        las_out.intensity = original_las.intensity
    if hasattr(original_las, 'return_number'):
        las_out.return_number = original_las.return_number
    if hasattr(original_las, 'number_of_returns'):
        las_out.number_of_returns = original_las.number_of_returns
    if hasattr(original_las, 'scan_direction_flag'):
        las_out.scan_direction_flag = original_las.scan_direction_flag
    if hasattr(original_las, 'edge_of_flight_line'):
        las_out.edge_of_flight_line = original_las.edge_of_flight_line
    if hasattr(original_las, 'classification'):
        # Keep original classification as additional info if needed
        pass  # We're overriding it with our predictions
    if hasattr(original_las, 'synthetic_flag'):
        las_out.synthetic_flag = original_las.synthetic_flag
    if hasattr(original_las, 'key_point_flag'):
        las_out.key_point_flag = original_las.key_point_flag
    if hasattr(original_las, 'withheld_flag'):
        las_out.withheld_flag = original_las.withheld_flag
    if hasattr(original_las, 'scan_angle_rank'):
        las_out.scan_angle_rank = original_las.scan_angle_rank
    if hasattr(original_las, 'user_data'):
        las_out.user_data = original_las.user_data
    if hasattr(original_las, 'point_source_id'):
        las_out.point_source_id = original_las.point_source_id

    las_out.write(output_file_path)
    print(f"Exported segmented point cloud to {output_file_path}")

def visualize_results(points, predictions, title="Point Cloud Segmentation", has_colors=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    colors = get_class_colors()
    point_colors = np.zeros((len(predictions), 3))
    
    for class_id, color in colors.items():
        mask = predictions == class_id
        point_colors[mask] = color
    
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    render_option.point_size = 10.0
    render_option.background_color = np.array([0.1, 0.1, 0.1])

    vis.run()
    vis.destroy_window()

def visualize_comparison_gt(points, predictions, gt_labels, title="Prediction vs Ground Truth"):
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(points)
    
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(points)

    colors = get_class_colors()
    pred_colors = np.zeros((len(predictions), 3))
    gt_colors = np.zeros((len(gt_labels), 3))
    
    for class_id, color in colors.items():
        pred_mask = predictions == class_id
        gt_mask = gt_labels == class_id
        pred_colors[pred_mask] = color
        gt_colors[gt_mask] = color
    
    pcd_pred.colors = o3d.utility.Vector3dVector(pred_colors)
    pcd_gt.colors = o3d.utility.Vector3dVector(gt_colors)
    
    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1400, height=700)

    translation = np.eye(4)
    translation[:3, 3] = [points[:, 0].max() - points[:, 0].min() + 5, 0, 0]  # Shift along X-axis
    pcd_gt.transform(translation)
    
    vis.add_geometry(pcd_pred)
    vis.add_geometry(pcd_gt)
    
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark gray background
    render_option.point_size = 2.0
    
    vis.run()
    vis.destroy_window()


def main(args):
    print("Loading model from checkpoint...")
    model = load_model(args.model_path, num_classes=len(ECLAIR_CLASSES))
    
    pointcloud_dir = os.path.join(args.data_root, 'pointclouds')
    file_list = [f for f in os.listdir(pointcloud_dir) if f.endswith(('.las', '.laz'))]

    file_list = random.sample(file_list, args.num_visualize)

    for i, filename in enumerate(file_list):
        if args.num_visualize and i >= args.num_visualize:
            break
            
        file_path = os.path.join(pointcloud_dir, filename)
        print(f"Processing file {i+1}/{len(file_list)}: {filename}")
        
        predictions, points_xyz, gt_labels, has_colors = run_inference_on_file(
            model, 
            file_path, 
            num_points=args.npoints,
            full_cloud=True
        )
        
        if args.export_las:
            os.makedirs(args.output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(args.output_dir, f"{base_name}_segmented.las")
            
            export_to_las(
                predictions=predictions,
                original_file_path=file_path,
                output_file_path=output_path
            )
        
        if args.visualize and i < args.num_visualize:
            if gt_labels is not None:
                title = f"File {i+1}: {filename} - Prediction vs Ground Truth"
                visualize_comparison_gt(
                    points=points_xyz,
                    predictions=predictions,
                    gt_labels=gt_labels,
                    title=title
                )
            else:
                print(f"Warning: No ground truth labels found in {filename}, skipping comparison visualization")
                title = f"File {i+1}: {filename} - Predictions"
                visualize_results(
                    points=points_xyz,
                    predictions=predictions,
                    title=title,
                    has_colors=has_colors
                )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize model predictions and export to LAS')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='data/eclair/')
    parser.add_argument('--export_las', action='store_true')
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--num_visualize', type=int, default=5)
    parser.add_argument('--npoints', type=int, default=4096)
    
    args = parser.parse_args()
    
    main(args)