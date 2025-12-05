"""
Point Cloud Registration Pipeline using RANSAC and ICP.

This script demonstrates a complete pipeline for:
1. Loading the bunny mesh
2. Converting mesh to point cloud with uniform sampling
3. Creating a transformed copy (rotation + translation)
4. Visualizing both point clouds at each step
5. Performing RANSAC-based global registration
6. Refining with ICP-based local registration
7. Computing and displaying the final transformation matrix

All parameters are logged using loguru and visualizations are displayed at each step.
"""

import numpy as np
import open3d as o3d
from loguru import logger

from functions_for_registration import (
    RegistrationConfig,
    load_bunny_mesh,
    mesh_to_point_cloud_uniform,
    create_transformed_copy,
    preprocess_point_cloud,
    execute_global_registration_ransac,
    execute_icp_registration,
    compute_transformation_error,
    visualize_geometries,
    draw_registration_result
)


def main():
    """
    Main pipeline for point cloud registration using RANSAC and ICP.
    """
    logger.info("="*80)
    logger.info("POINT CLOUD REGISTRATION PIPELINE - RANSAC + ICP")
    logger.info("="*80)

    # ============================================================================
    # Step 1: Configuration
    # ============================================================================
    logger.info("\n=== Step 1: Pipeline Configuration ===")

    config = RegistrationConfig(
        num_points=10000,
        rotation_degrees=(30.0, 45.0, 60.0),
        translation=(0.2, 0.3, 0.1),
        voxel_size=0.005,
        ransac_distance_threshold=0.015,
        ransac_n_points=4,
        ransac_max_iterations=100000,
        ransac_confidence=0.999,
        icp_distance_threshold=0.02,
        icp_max_iterations=2000,
        point_size=2.0
    )

    logger.info(f"Configuration loaded:")
    logger.info(f"  Point cloud sampling: {config.num_points} points")
    logger.info(f"  Ground truth rotation: {config.rotation_degrees}° (X, Y, Z)")
    logger.info(f"  Ground truth translation: {config.translation}")
    logger.info(f"  Voxel size: {config.voxel_size}")
    logger.info(f"  RANSAC distance threshold: {config.ransac_distance_threshold}")
    logger.info(f"  RANSAC max iterations: {config.ransac_max_iterations}")
    logger.info(f"  ICP distance threshold: {config.icp_distance_threshold}")
    logger.info(f"  ICP max iterations: {config.icp_max_iterations}")
    logger.info(f"  Visualization point size: {config.point_size}")

    # ============================================================================
    # Step 2: Load Bunny Mesh
    # ============================================================================
    logger.info("\n=== Step 2: Loading Bunny Mesh ===")

    mesh_bunny = load_bunny_mesh()

    # Visualize the original mesh
    logger.info("Visualizing original bunny mesh")
    mesh_bunny.paint_uniform_color([0.7, 0.7, 0.7])  # Gray color
    visualize_geometries([mesh_bunny], "Step 2: Original Bunny Mesh")

    # ============================================================================
    # Step 3: Convert Mesh to Point Cloud (Uniform Sampling)
    # ============================================================================
    logger.info("\n=== Step 3: Mesh to Point Cloud Conversion ===")

    pcd_source = mesh_to_point_cloud_uniform(mesh_bunny, config.num_points)

    # Paint source in yellow (Open3D tutorial standard)
    pcd_source.paint_uniform_color([1, 0.706, 0])  # Yellow color
    logger.info("Source point cloud painted in yellow [1, 0.706, 0]")

    # Visualize source point cloud
    logger.info("Visualizing source point cloud")
    visualize_geometries([pcd_source], "Step 3: Source Point Cloud (Yellow)", config.point_size)

    # ============================================================================
    # Step 4: Create Transformed Copy (Cyan)
    # ============================================================================
    logger.info("\n=== Step 4: Creating Transformed Copy ===")

    pcd_target = create_transformed_copy(
        pcd_source,
        config.rotation_degrees,
        config.translation
    )

    logger.info("Target point cloud painted in cyan [0, 0.651, 0.929]")

    # Store ground truth transformation for error computation
    rx, ry, rz = np.radians(config.rotation_degrees)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    R_y = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    R_z = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    ground_truth_R = R_z @ R_y @ R_x
    ground_truth_t = np.array(config.translation)

    # Create full 4x4 transformation matrix
    ground_truth_T = np.eye(4)
    ground_truth_T[:3, :3] = ground_truth_R
    ground_truth_T[:3, 3] = ground_truth_t

    logger.info(f"Ground truth transformation matrix:")
    logger.info(f"\n{ground_truth_T}")

    # Visualize both point clouds together (before registration)
    logger.info("Visualizing source (yellow) and target (cyan) point clouds")
    visualize_geometries([pcd_source, pcd_target], "Step 4: Source (Yellow) + Target (Cyan)", config.point_size)

    # ============================================================================
    # Step 5: Preprocessing for RANSAC
    # ============================================================================
    logger.info("\n=== Step 5: Preprocessing Point Clouds ===")

    # Preprocess source
    logger.info("Preprocessing source point cloud")
    pcd_source_down, source_fpfh = preprocess_point_cloud(pcd_source, config.voxel_size)
    pcd_source_down.paint_uniform_color([1, 0.706, 0])  # Yellow

    # Preprocess target
    logger.info("Preprocessing target point cloud")
    pcd_target_down, target_fpfh = preprocess_point_cloud(pcd_target, config.voxel_size)
    pcd_target_down.paint_uniform_color([0, 0.651, 0.929])  # Cyan

    logger.info(f"Source downsampled: {len(pcd_source.points)} → {len(pcd_source_down.points)} points")
    logger.info(f"Target downsampled: {len(pcd_target.points)} → {len(pcd_target_down.points)} points")
    logger.info(f"FPFH feature dimension: {source_fpfh.dimension()}")

    # Visualize downsampled point clouds
    logger.info("Visualizing downsampled point clouds")
    visualize_geometries(
        [pcd_source_down, pcd_target_down],
        "Step 5: Downsampled Point Clouds",
        config.point_size * 1.5
    )

    # ============================================================================
    # Step 6: RANSAC Global Registration
    # ============================================================================
    logger.info("\n=== Step 6: RANSAC Global Registration ===")

    ransac_result = execute_global_registration_ransac(
        pcd_source_down,
        pcd_target_down,
        source_fpfh,
        target_fpfh,
        config
    )

    logger.info(f"RANSAC fitness: {ransac_result.fitness:.6f}")
    logger.info(f"RANSAC RMSE: {ransac_result.inlier_rmse:.6f}")
    logger.info(f"RANSAC correspondences: {len(ransac_result.correspondence_set)}")

    # Visualize RANSAC result
    logger.info("Visualizing RANSAC registration result")
    draw_registration_result(
        pcd_source,
        pcd_target,
        ransac_result.transformation,
        "Step 6: RANSAC Registration Result",
        config.point_size
    )

    # Compute RANSAC error
    logger.info("\n--- RANSAC Error Analysis ---")
    ransac_rot_error, ransac_trans_error = compute_transformation_error(
        ransac_result.transformation,
        ground_truth_R,
        ground_truth_t
    )

    # ============================================================================
    # Step 7: ICP Local Refinement
    # ============================================================================
    logger.info("\n=== Step 7: ICP Local Refinement ===")

    icp_result = execute_icp_registration(
        pcd_source,
        pcd_target,
        ransac_result.transformation,
        config
    )

    logger.info(f"ICP fitness: {icp_result.fitness:.6f}")
    logger.info(f"ICP RMSE: {icp_result.inlier_rmse:.6f}")
    logger.info(f"ICP correspondences: {len(icp_result.correspondence_set)}")

    # Visualize ICP result
    logger.info("Visualizing ICP registration result")
    draw_registration_result(
        pcd_source,
        pcd_target,
        icp_result.transformation,
        "Step 7: ICP Registration Result (Final)",
        config.point_size
    )

    # Compute ICP error
    logger.info("\n--- ICP Error Analysis ---")
    icp_rot_error, icp_trans_error = compute_transformation_error(
        icp_result.transformation,
        ground_truth_R,
        ground_truth_t
    )

    # ============================================================================
    # Step 8: Final Results Summary
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("REGISTRATION PIPELINE COMPLETE")
    logger.info("="*80)

    logger.info("\n--- Ground Truth Transformation ---")
    logger.info(f"Rotation (degrees): {config.rotation_degrees}")
    logger.info(f"Translation: {config.translation}")
    logger.info(f"Transformation matrix:")
    logger.info(f"\n{ground_truth_T}")

    logger.info("\n--- RANSAC Results ---")
    logger.info(f"Fitness: {ransac_result.fitness:.6f}")
    logger.info(f"RMSE: {ransac_result.inlier_rmse:.6f}")
    logger.info(f"Rotation error: {ransac_rot_error:.4f} degrees")
    logger.info(f"Translation error: {ransac_trans_error:.6f}")
    logger.info(f"Transformation matrix:")
    logger.info(f"\n{ransac_result.transformation}")

    logger.info("\n--- ICP Results (Final) ---")
    logger.info(f"Fitness: {icp_result.fitness:.6f}")
    logger.info(f"RMSE: {icp_result.inlier_rmse:.6f}")
    logger.info(f"Rotation error: {icp_rot_error:.4f} degrees")
    logger.info(f"Translation error: {icp_trans_error:.6f}")
    logger.info(f"Final transformation matrix:")
    logger.info(f"\n{icp_result.transformation}")

    logger.info("\n--- Improvement (RANSAC → ICP) ---")

    # Calculate improvements with safe division
    if ransac_result.fitness > 0:
        fitness_improvement = ((icp_result.fitness - ransac_result.fitness) / ransac_result.fitness) * 100
    else:
        fitness_improvement = 0.0

    if ransac_result.inlier_rmse > 1e-10:
        rmse_improvement = ((ransac_result.inlier_rmse - icp_result.inlier_rmse) / ransac_result.inlier_rmse) * 100
    else:
        rmse_improvement = 0.0

    if ransac_rot_error > 1e-6:
        rot_error_improvement = ((ransac_rot_error - icp_rot_error) / ransac_rot_error) * 100
    else:
        rot_error_improvement = 0.0

    if ransac_trans_error > 1e-10:
        trans_error_improvement = ((ransac_trans_error - icp_trans_error) / ransac_trans_error) * 100
    else:
        trans_error_improvement = 0.0

    logger.info(f"Fitness improvement: {fitness_improvement:+.2f}%")
    logger.info(f"RMSE improvement: {rmse_improvement:+.2f}%")
    logger.info(f"Rotation error improvement: {rot_error_improvement:+.2f}%")
    logger.info(f"Translation error improvement: {trans_error_improvement:+.2f}%")

    logger.info("\n" + "="*80)
    logger.info("Pipeline finished successfully!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
