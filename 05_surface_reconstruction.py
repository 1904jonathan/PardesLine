import numpy as np
import open3d as o3d
import open3d.visualization as o3d_vis
import matplotlib.pyplot as plt
from loguru import logger
from functions_for_surfaceR import (
    load_bunny_point_cloud,
    load_eagle_point_cloud,
    compute_convex_hull,
    reconstruct_alpha_shape,
    reconstruct_ball_pivoting,
    reconstruct_poisson,
    estimate_normals
)


def main():
    logger.info("Starting Surface Reconstruction Pipeline")
    
    # Load point cloud
    pcd = load_bunny_point_cloud(num_points=750)
    o3d_vis.draw_geometries([pcd])
    
    # 1. Convex Hull
    logger.info("Step 1: Convex Hull Computation")
    hull = compute_convex_hull(pcd)
    o3d_vis.draw_geometries([hull])
    
    # 2. Alpha Shape (bigger to smaller alpha)
    logger.info("Step 2: Alpha Shape Reconstruction")
    alpha_values = [0.5, 0.1, 0.05, 0.03, 0.01]  # Bigger to smaller
    alpha_meshes = reconstruct_alpha_shape(pcd, alpha_values=alpha_values)
    
    # 3. Ball Pivoting
    logger.info("Step 3: Ball Pivoting Algorithm")
    dataset = o3d.data.BunnyMesh()
    gt_mesh = o3d.io.read_triangle_mesh(dataset.path)
    gt_mesh.compute_vertex_normals()
    pcd_bp = gt_mesh.sample_points_poisson_disk(3000)
    o3d_vis.draw_geometries([pcd_bp])
    bp_mesh = reconstruct_ball_pivoting(pcd_bp)
    
    # 4. Poisson Reconstruction
    logger.info("Step 4: Poisson Surface Reconstruction")
    pcd_poisson = load_eagle_point_cloud()
    o3d_vis.draw_geometries(
        [pcd_poisson],
        zoom=0.664,
        front=[-0.4761, -0.4698, -0.7434],
        lookat=[1.8900, 3.2596, 0.9284],
        up=[0.2304, -0.8825, 0.4101]
    )
    poisson_mesh, densities = reconstruct_poisson(pcd_poisson)
    
    # 5. Normal Estimation
    logger.info("Step 5: Normal Estimation")
    dataset = o3d.data.BunnyMesh()
    gt_mesh = o3d.io.read_triangle_mesh(dataset.path)
    pcd_normals = gt_mesh.sample_points_poisson_disk(5000)
    pcd_normals = estimate_normals(pcd_normals)
    
    logger.info("Surface Reconstruction Pipeline Completed")


if __name__ == "__main__":
    main()