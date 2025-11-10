import laspy
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# 1. 读取 LAS 文件
las = laspy.read("/home/ak1ra/ladiar/Pointnet_Pointnet2_pytorch/data/eclair/pointclouds/pointcloud_0.laz")  
points = np.vstack((las.x, las.y, las.z)).transpose()  # N×3 数组

# 2. 转为 Open3D 点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
voxel_size = 0.5  # 米
pcd_down = pcd.voxel_down_sample(voxel_size)
cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=5, std_ratio=1.0)
pcd_clean = pcd_down.select_by_index(ind)
# 3. 可视化
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="test", width=1400, height=900)
vis.add_geometry(pcd_clean)
render_option = vis.get_render_option()
render_option.background_color = np.array([0.0, 0.0, 0.0])  # 黑色背景
render_option.point_size = 2.0
# render_option.show_coordinate_frame = True
vis.run()

plane_model, inliers = pcd_clean.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=1000)
ground = pcd_clean.select_by_index(inliers)
non_ground = pcd_clean.select_by_index(inliers, invert=True)


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