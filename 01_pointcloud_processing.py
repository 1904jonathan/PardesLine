import open3d as o3d
import numpy as np
import os
import functions as fn

# -------------------------------------------------------------------------
# Output Folder Setup
# -------------------------------------------------------------------------
output_dir = r"C:\PardesLineData"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------------------------------
# Load and Visualize a Sample Point Cloud (Living Room Scene)
# -------------------------------------------------------------------------
dataset_ply = o3d.data.PLYPointCloud()          # Sample point cloud from Open3D dataset
pcd = o3d.io.read_point_cloud(dataset_ply.path) # Load the point cloud
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# Visualize with predefined view parameters for a nicer scene
o3d.visualization.draw_geometries(
    [pcd],
    zoom=0.34,
    front=[0.4257, -0.2125, -0.8795],
    lookat=[2.6172, 2.0475, 1.532],
    up=[-0.0694, -0.9768, 0.2024]
)

# Save the point cloud
o3d.io.write_point_cloud(os.path.join(output_dir, "livingRoom.ply"), pcd)

# -------------------------------------------------------------------------
# Load and Visualize Another Sample Point Cloud (Eagle)
# -------------------------------------------------------------------------
dataset_eagle = o3d.data.EaglePointCloud()
pcd_eagle = o3d.io.read_point_cloud(dataset_eagle.path)

o3d.visualization.draw([pcd_eagle])

# Save the point cloud
o3d.io.write_point_cloud(os.path.join(output_dir, "Eagle.ply"), pcd_eagle)

# -------------------------------------------------------------------------
# Generate and Save a Random Point Cloud
# -------------------------------------------------------------------------
N = 1000
random_points = np.random.rand(N, 3)  # Generate N random XYZ points in [0,1]
pcd_random = fn.points_to_pcd(random_points)

o3d.visualization.draw_geometries([pcd_random])
o3d.io.write_point_cloud(os.path.join(output_dir, "random_points.ply"), pcd_random)

print(f"\n✅ All point clouds have been saved in: {output_dir}")
