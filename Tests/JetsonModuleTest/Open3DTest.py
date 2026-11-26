import open3d as o3d

file_path = "/home/group340/Desktop/P3/HeliosScanTest/scans/scan_20261120_085219.ply"

# 1. Read the point cloud file
pcd = o3d.io.read_point_cloud(file_path)

# 2. Open the viewer window
o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Viewer")