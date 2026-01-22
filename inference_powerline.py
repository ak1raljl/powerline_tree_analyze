"""
推理脚本 - 适配 /home/link/data/point_seg/npy_v2 数据格式
用于加载训练好的模型进行预测并可视化结果
"""
import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from models.pointnet2_sem_seg import get_model
from data.powerline_npy_dataset import POWERLINE_CLASSES

# 延迟导入 open3d，只在需要可视化时加载
o3d = None

def lazy_import_open3d():
    global o3d
    if o3d is None:
        import open3d as _o3d
        o3d = _o3d
    return o3d

NUM_CLASSES = len(POWERLINE_CLASSES)

# 为10个类别定义颜色
def get_class_colors():
    colors = {
        0: [0.3, 0.3, 0.3],       # noise_other - dark gray
        1: [0.6, 0.4, 0.2],       # ground - brown
        2: [0.0, 0.8, 0.0],       # vegetation - green
        3: [1.0, 0.0, 0.0],       # building - red
        4: [1.0, 1.0, 0.0],       # wire_connector - yellow
        5: [0.0, 0.5, 1.0],       # bridge - light blue
        6: [1.0, 0.5, 0.0],       # overhead_structure - orange
        7: [1.0, 0.0, 1.0],       # wire - magenta
        8: [0.0, 1.0, 1.0],       # insulator - cyan
        9: [0.5, 0.0, 0.5],       # tower - purple
    }
    return colors


def load_model(model_path, num_classes=NUM_CLASSES):
    """加载训练好的模型"""
    classifier = get_model(num_classes).cuda()
    checkpoint = torch.load(model_path, weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    print(f"Model loaded from {model_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}, mIoU: {checkpoint.get('mIoU', 'N/A')}")
    return classifier


def inference_on_npy(model, data_path, label_path=None, num_points=4096, block_size=20.0, stride=10.0, device='cuda', batch_size=16):
    """
    对单个npy文件进行推理
    使用滑动窗口方式处理整个点云，批量推理加速
    """
    # 加载数据
    points = np.load(data_path)
    points_xyz = points[:, :3]
    colors = points[:, 3:6]  # 已归一化
    
    N_points = points_xyz.shape[0]
    
    # 加载标签(如果存在)
    gt_labels = None
    if label_path and os.path.exists(label_path):
        gt_labels = np.load(label_path)
    
    coord_min = np.min(points_xyz, axis=0)
    coord_max = np.max(points_xyz, axis=0)
    coord_range = coord_max - coord_min + 1e-6
    
    # 存储所有预测和投票计数
    all_predictions = np.zeros((N_points, NUM_CLASSES), dtype=np.float32)
    vote_count = np.zeros(N_points, dtype=np.int32)
    
    # 使用简单的顺序滑动窗口（按点索引）
    num_samples = max(1, int(np.ceil(N_points * 1.5 / num_points)))
    print(f"Processing {N_points} points with {num_samples} samples")
    
    # 预先准备所有采样
    all_samples = []
    all_indices = []
    
    for sample_idx in range(num_samples):
        # 随机选择一个中心点
        center_idx = np.random.randint(0, N_points)
        center = points_xyz[center_idx]
        
        # 找到在当前块内的点
        block_min = center - [block_size / 2.0, block_size / 2.0, float('inf')]
        block_max = center + [block_size / 2.0, block_size / 2.0, float('inf')]
        
        point_idxs = np.where(
            (points_xyz[:, 0] >= block_min[0]) & (points_xyz[:, 0] <= block_max[0]) &
            (points_xyz[:, 1] >= block_min[1]) & (points_xyz[:, 1] <= block_max[1])
        )[0]
        
        if len(point_idxs) < 32:
            point_idxs = np.arange(N_points)
        
        # 采样点
        if len(point_idxs) >= num_points:
            selected_idxs = np.random.choice(point_idxs, num_points, replace=False)
        else:
            selected_idxs = np.random.choice(point_idxs, num_points, replace=True)
        
        selected_xyz = points_xyz[selected_idxs]
        selected_colors = colors[selected_idxs]
        
        # 构建输入特征 (9维)
        current_points = np.zeros((num_points, 9), dtype=np.float32)
        
        # 中心化 xyz
        current_points[:, 0] = selected_xyz[:, 0] - center[0]
        current_points[:, 1] = selected_xyz[:, 1] - center[1]
        current_points[:, 2] = selected_xyz[:, 2] - coord_min[2]
        
        # 颜色
        current_points[:, 3:6] = selected_colors
        
        # 归一化的 xyz
        current_points[:, 6] = (selected_xyz[:, 0] - coord_min[0]) / coord_range[0]
        current_points[:, 7] = (selected_xyz[:, 1] - coord_min[1]) / coord_range[1]
        current_points[:, 8] = (selected_xyz[:, 2] - coord_min[2]) / coord_range[2]
        
        all_samples.append(current_points)
        all_indices.append(selected_idxs)
    
    # 批量推理
    with torch.no_grad():
        for batch_start in tqdm(range(0, num_samples, batch_size), desc="Inference"):
            batch_end = min(batch_start + batch_size, num_samples)
            
            # 构建batch
            batch_points = np.stack(all_samples[batch_start:batch_end], axis=0)
            batch_input = torch.tensor(batch_points, dtype=torch.float32).to(device)
            batch_input = batch_input.transpose(2, 1)  # (B, C, N)
            
            seg_pred, _ = model(batch_input)  # (B, N, C)
            seg_pred = seg_pred.cpu().numpy()
            
            # 聚合预测
            for b in range(batch_end - batch_start):
                sample_idx = batch_start + b
                selected_idxs = all_indices[sample_idx]
                pred = seg_pred[b]  # (N, C)
                
                # 直接累加到对应索引
                np.add.at(all_predictions, selected_idxs, pred)
                np.add.at(vote_count, selected_idxs, 1)
    
    # 计算最终预测
    valid_mask = vote_count > 0
    final_predictions = np.zeros(N_points, dtype=np.int32)
    final_predictions[valid_mask] = np.argmax(all_predictions[valid_mask], axis=1)
    
    # 对没有被采样到的点，使用KDTree快速最近邻
    if not np.all(valid_mask):
        invalid_count = np.sum(~valid_mask)
        print(f"Filling {invalid_count} unsampled points with nearest neighbor predictions")
        
        from scipy.spatial import cKDTree
        
        invalid_indices = np.where(~valid_mask)[0]
        valid_points = points_xyz[valid_mask]
        valid_preds = final_predictions[valid_mask]
        
        # 构建KDTree并查询最近邻
        tree = cKDTree(valid_points)
        _, nearest_indices = tree.query(points_xyz[invalid_indices], k=1)
        final_predictions[invalid_indices] = valid_preds[nearest_indices]
    
    return final_predictions, points_xyz, colors, gt_labels


def calculate_metrics(predictions, gt_labels):
    """计算评估指标"""
    if gt_labels is None:
        return None
    
    metrics = {}
    
    # 整体准确率
    metrics['accuracy'] = np.sum(predictions == gt_labels) / len(gt_labels)
    
    # 每类IoU
    iou_per_class = []
    for c in range(NUM_CLASSES):
        intersection = np.sum((predictions == c) & (gt_labels == c))
        union = np.sum((predictions == c) | (gt_labels == c))
        if union > 0:
            iou = intersection / union
            iou_per_class.append(iou)
        else:
            iou_per_class.append(np.nan)
    
    metrics['iou_per_class'] = iou_per_class
    valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
    metrics['mIoU'] = np.mean(valid_ious) if valid_ious else 0.0
    
    return metrics


def visualize_predictions(points_xyz, predictions, title="Predictions"):
    """可视化预测结果"""
    o3d = lazy_import_open3d()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    
    colors = get_class_colors()
    point_colors = np.zeros((len(predictions), 3))
    
    for class_id, color in colors.items():
        mask = predictions == class_id
        point_colors[mask] = color
    
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1200, height=800)
    vis.add_geometry(pcd)
    
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.1, 0.1, 0.1])
    
    vis.run()
    vis.destroy_window()


def visualize_comparison(points_xyz, predictions, gt_labels, title="Prediction vs Ground Truth"):
    """并排可视化预测和真实标签"""
    o3d = lazy_import_open3d()
    
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(points_xyz)
    
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(points_xyz)
    
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
    
    # 平移GT点云以便并排显示
    x_offset = points_xyz[:, 0].max() - points_xyz[:, 0].min() + 10
    translation = np.eye(4)
    translation[0, 3] = x_offset
    pcd_gt.transform(translation)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1600, height=800)
    vis.add_geometry(pcd_pred)
    vis.add_geometry(pcd_gt)
    
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.1, 0.1, 0.1])
    
    vis.run()
    vis.destroy_window()


def save_predictions_npy(predictions, output_path):
    """保存预测结果为npy文件"""
    np.save(output_path, predictions)
    print(f"Predictions saved to {output_path}")


def main(args):
    print("=" * 60)
    print("Powerline Point Cloud Segmentation Inference")
    print("=" * 60)
    
    # 加载模型
    model = load_model(args.model_path, num_classes=NUM_CLASSES)
    
    # 确定要处理的文件
    data_dir = os.path.join(args.npy_dir, 'data')
    label_dir = os.path.join(args.npy_dir, 'labels')
    
    if args.file:
        # 处理指定文件
        data_files = [args.file]
    else:
        # 处理目录中的所有文件
        data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_data.npy')])
        if args.num_files > 0:
            data_files = data_files[:args.num_files]
    
    print(f"Processing {len(data_files)} files")
    
    # 创建输出目录
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    all_metrics = []
    
    for i, data_file in enumerate(data_files):
        print(f"\n[{i+1}/{len(data_files)}] Processing: {data_file}")
        
        data_path = os.path.join(data_dir, data_file)
        label_file = data_file.replace('_data.npy', '_labels.npy')
        label_path = os.path.join(label_dir, label_file)
        
        if not os.path.exists(label_path):
            label_path = None
        
        # 推理
        predictions, points_xyz, colors, gt_labels = inference_on_npy(
            model=model,
            data_path=data_path,
            label_path=label_path,
            num_points=args.npoints,
            block_size=args.block_size,
            stride=args.stride,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            batch_size=args.batch_size
        )
        
        # 计算指标
        if gt_labels is not None:
            metrics = calculate_metrics(predictions, gt_labels)
            all_metrics.append(metrics)
            print(f"  Accuracy: {metrics['accuracy']:.4f}, mIoU: {metrics['mIoU']:.4f}")
            print("  Per-class IoU:")
            for c, iou in enumerate(metrics['iou_per_class']):
                if not np.isnan(iou):
                    print(f"    {c}: {POWERLINE_CLASSES[c]:20s} IoU: {iou:.4f}")
        
        # 保存预测结果
        if args.output_dir:
            output_file = data_file.replace('_data.npy', '_pred.npy')
            output_path = os.path.join(args.output_dir, output_file)
            save_predictions_npy(predictions, output_path)
        
        # 可视化
        if args.visualize:
            if gt_labels is not None:
                title = f"{data_file} - Prediction (left) vs GT (right)"
                visualize_comparison(points_xyz, predictions, gt_labels, title)
            else:
                title = f"{data_file} - Predictions"
                visualize_predictions(points_xyz, predictions, title)
    
    # 打印汇总指标
    if all_metrics:
        print("\n" + "=" * 60)
        print("Overall Metrics Summary")
        print("=" * 60)
        avg_acc = np.mean([m['accuracy'] for m in all_metrics])
        avg_miou = np.mean([m['mIoU'] for m in all_metrics])
        print(f"Average Accuracy: {avg_acc:.4f}")
        print(f"Average mIoU: {avg_miou:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description='Inference script for Powerline Point Cloud Segmentation')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--npy_dir', type=str, default='/home/link/data/point_seg/npy_v2',
                        help='Path to NPY dataset directory')
    parser.add_argument('--file', type=str, default=None,
                        help='Specific data file to process (e.g., "10-11(N10_N11)_data.npy")')
    parser.add_argument('--num_files', type=int, default=5,
                        help='Number of files to process (0 for all)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save prediction results')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions using Open3D')
    parser.add_argument('--npoints', type=int, default=4096,
                        help='Number of points per sample')
    parser.add_argument('--block_size', type=float, default=20.0,
                        help='Block size in meters')
    parser.add_argument('--stride', type=float, default=10.0,
                        help='Stride for sliding window (smaller = more overlap)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
