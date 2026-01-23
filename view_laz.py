import laspy
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def get_class_colors():
    colors = {
        1: [0.5, 0.5, 0.0],   # Low Vegetation - 绿色
        2: [0.6, 0.4, 0.2],   # Ground - 棕色
        4: [0.0, 0.8, 0.0],   # Medium Vegetation - 绿色
        7: [1.0, 1.0, 0.0],   # Low Point (noise) - 黄色
        12: [1.0, 0.5, 0.0],  # Tower - 橙色
        13: [0.0, 1.0, 1.0],  # Wire - 青色
        16: [0.5, 0.0, 1.0],  # Ground wire - 紫色
        17: [0.0, 1.0, 1.0],  # Wire - 青色
        18: [1.0, 0.0, 1.0],  # Tower - 品红
        27: [0.8, 0.8, 0.8],  # unknown - 浅灰色
    }
    return colors

# preprocess
las = laspy.read("/home/ak1ra/ladiar/powerline_tree_analyze/110kV岗远线#10-#13（110kV远新线#33-#30）.las")  
points = np.vstack((las.x, las.y, las.z)).transpose()  # N×3 

if hasattr(las, 'classification'):
    classifications = np.array(las.classification)
    print(f"classification: {np.unique(classifications)}")
else:
    print("classification: None")

# transform to Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)


colors_map = get_class_colors()
point_colors = np.zeros((len(points), 3))

for class_id, color in colors_map.items():
    mask = classifications == class_id
    point_colors[mask] = color
pcd.colors = o3d.utility.Vector3dVector(point_colors)

voxel_size = 0.5  # m
pcd_down = pcd.voxel_down_sample(voxel_size)
cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=5, std_ratio=1.0)
pcd_clean = pcd_down.select_by_index(ind)
# visualize
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="test", width=1400, height=900)
vis.add_geometry(pcd_clean)

render_option = vis.get_render_option()
render_option.background_color = np.array([0.0, 0.0, 0.0])  # dark background
render_option.point_size = 2.0
# render_option.show_coordinate_frame = True
vis.run()

# plane_model, inliers = pcd_clean.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=1000)
# ground = pcd_clean.select_by_index(inliers)
# non_ground = pcd_clean.select_by_index(inliers, invert=True)


# with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
#     labels = np.array(pcd_clean.cluster_dbscan(eps=1.0, min_points=20, print_progress=True))
# max_label = labels.max()
# print(f"共发现 {max_label+1} 个簇")
# # 为每个簇上色
# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# pcd_clean.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([pcd_clean])
# # 3. 可视化
# o3d.visualization.draw_geometries([pcd_clean], window_name="LiDAR 点云", width=1280, height=720)