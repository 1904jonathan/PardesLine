import open3d as o3d
import numpy as np
import os 

def pcd_o3d(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

folder_path = r"C:\PardesLineData"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
arr = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

o3d.io.write_point_cloud(fr"{folder_path}\livingRoom.ply", pcd)

dataset = o3d.data.EaglePointCloud()
pcd = o3d.io.read_point_cloud(dataset.path)
o3d.visualization.draw([pcd])

o3d.io.write_point_cloud(fr"{folder_path}\Eagle.ply", pcd)

N=1000
arr = np.random.rand(N, 3)
pcd = pcd_o3d(arr)
o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud(fr"{folder_path}\random_points.ply", pcd)


