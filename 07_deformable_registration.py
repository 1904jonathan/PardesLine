"""
Deformable Point Cloud Registration using Non-Rigid Pressure Deformation and CPD.

This script demonstrates a complete pipeline for:
1. Loading the bunny point cloud from Open3D dataset
2. Downsampling to ~1500 points using voxel downsampling
3. Applying a localized non-rigid pressure deformation with Gaussian falloff
4. Computing deformation field using Coherent Point Drift (CPD)
5. Visualizing original, pressure-deformed, and CPD-registered point clouds
6. Computing and displaying comprehensive deformation metrics

All parameters are logged using loguru and visualizations are displayed at each step.
"""

import numpy as np
import open3d as o3d
from loguru import logger
from pycpd import DeformableRegistration

from functions_for_deformation import (
    DeformationConfig,
    load_bunny_source,
    normalize_points,
    center_points,
    apply_gaussian_deformation,
    compute_deformation_metrics,
    create_colored_point_cloud,
    visualize_geometries
)


def main():
    """
    Main pipeline for deformable point cloud registration using Gaussian deformation and CPD.
    """
    logger.info("=" * 80)
    logger.info("DEFORMABLE POINT CLOUD REGISTRATION - PRESSURE DEFORMATION + CPD")
    logger.info("=" * 80)

    # ============================================================================
    # Step 1: Configuration
    # ============================================================================
    logger.info("\n=== Step 1: Pipeline Configuration ===")

    config = DeformationConfig(
        n_control_points=1,
        displacement_scale=0.004,
        gaussian_beta=0.03,
        cpd_beta=50.0,
        cpd_lambda=0.005,
        cpd_max_iterations=10000,
        point_size=2.0
    )

    logger.info(f"Configuration loaded:")
    logger.info(f"  Max displacement: {config.displacement_scale}")
    logger.info(f"  Pressure radius: {config.gaussian_beta}")
    logger.info(f"  CPD beta (smoothness): {config.cpd_beta}")
    logger.info(f"  CPD lambda (regularization): {config.cpd_lambda}")
    logger.info(f"  CPD max iterations: {config.cpd_max_iterations}")
    logger.info(f"  Visualization point size: {config.point_size}")

    # ============================================================================
    # Step 2: Load and Normalize Bunny Point Cloud
    # ============================================================================
    logger.info("\n=== Step 2: Loading Bunny Point Cloud ===")

    bunny_source = load_bunny_source()
    bunny_normalized = normalize_points(bunny_source)

    logger.info(f"Point cloud normalized to unit scale")
    logger.info(f"Total points: {len(bunny_normalized)}")

    # Visualize normalized source
    pcd_source_vis = create_colored_point_cloud(center_points(bunny_normalized), [0, 0, 1])
    logger.info("Source point cloud painted in blue [0, 0, 1]")
    visualize_geometries([pcd_source_vis], "Step 2: Original Bunny (Blue)", config.point_size)

    # ============================================================================
    # Step 3: Apply Non-Rigid Pressure Deformation
    # ============================================================================
    logger.info("\n=== Step 3: Applying Non-Rigid Pressure Deformation ===")

    bunny_target, gaussian_deformation, control_points, displacements = apply_gaussian_deformation(
        bunny_normalized,
        config.n_control_points,
        config.displacement_scale,
        config.gaussian_beta
    )

    logger.info(f"Pressure deformation applied with Gaussian falloff")
    logger.info(f"Mean deformation magnitude: {np.mean(np.linalg.norm(gaussian_deformation, axis=1)):.6f}")
    logger.info(f"Max deformation magnitude: {np.max(np.linalg.norm(gaussian_deformation, axis=1)):.6f}")

    # Visualize source and Gaussian target
    pcd_source_centered = create_colored_point_cloud(center_points(bunny_normalized), [0, 0, 1])
    pcd_target_centered = create_colored_point_cloud(center_points(bunny_target), [0, 1, 0])

    logger.info("Target point cloud painted in green [0, 1, 0]")
    visualize_geometries(
        [pcd_source_centered, pcd_target_centered],
        "Step 3: Original (Blue) vs Pressure Deformed (Green)",
        config.point_size
    )

    # ============================================================================
    # Step 4: CPD Deformable Registration
    # ============================================================================
    logger.info("\n=== Step 4: Computing CPD Deformation ===")

    logger.info(f"Initializing CPD with beta={config.cpd_beta}, lambda={config.cpd_lambda}")

    reg = DeformableRegistration(
        X=bunny_target,
        Y=bunny_normalized,
        beta=config.cpd_beta,
        lamb=config.cpd_lambda,
        max_iterations=config.cpd_max_iterations
    )

    logger.info("Starting CPD registration...")
    reg.register()
    bunny_cpd_deformed = reg.TY

    logger.info("CPD registration completed successfully")

    # ============================================================================
    # Step 5: Compute Deformation Metrics
    # ============================================================================
    logger.info("\n=== Step 5: Computing Deformation Metrics ===")

    metrics = compute_deformation_metrics(
        bunny_normalized,
        bunny_target,
        bunny_cpd_deformed,
        gaussian_deformation
    )

    logger.info("--- Pressure Deformation Metrics ---")
    logger.info(f"  Mean magnitude: {metrics['gaussian_mean_magnitude']:.6f}")

    logger.info("\n--- CPD Deformation Metrics ---")
    logger.info(f"  Mean magnitude: {metrics['cpd_mean_magnitude']:.6f}")
    logger.info(f"  Max magnitude: {metrics['cpd_max_magnitude']:.6f}")

    logger.info("\n--- Registration Quality ---")
    logger.info(f"  Registration error (CPD to Pressure target): {metrics['registration_error']:.6f}")
    logger.info(f"  Deformation field difference: {metrics['deformation_field_difference']:.6f}")

    # ============================================================================
    # Step 6: Visualization - All Comparisons
    # ============================================================================
    logger.info("\n=== Step 6: Visualizing Registration Results ===")

    pcd_cpd_centered = create_colored_point_cloud(center_points(bunny_cpd_deformed), [1, 0, 0])

    # Visualization 1: Original vs Pressure target
    logger.info("Visualization 1: Original (blue) vs Pressure target (green)")
    visualize_geometries(
        [pcd_source_centered, pcd_target_centered],
        "Step 6.1: Original (Blue) vs Pressure Deformed (Green)",
        config.point_size
    )

    # Visualization 2: Original vs CPD deformed
    logger.info("Visualization 2: Original (blue) vs CPD deformed (red)")
    visualize_geometries(
        [pcd_source_centered, pcd_cpd_centered],
        "Step 6.2: Original (Blue) vs CPD Deformed (Red)",
        config.point_size
    )

    # Visualization 3: Pressure target vs CPD deformed
    logger.info("Visualization 3: Pressure target (green) vs CPD deformed (red)")
    visualize_geometries(
        [pcd_target_centered, pcd_cpd_centered],
        "Step 6.3: Pressure Target (Green) vs CPD Deformed (Red)",
        config.point_size
    )

    # Visualization 4: All three together
    logger.info("Visualization 4: All three point clouds together")
    visualize_geometries(
        [pcd_source_centered, pcd_target_centered, pcd_cpd_centered],
        "Step 6.4: All - Original (Blue), Pressure (Green), CPD (Red)",
        config.point_size
    )

    # ============================================================================
    # Step 7: Final Summary
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("DEFORMABLE REGISTRATION PIPELINE COMPLETE")
    logger.info("=" * 80)

    logger.info("\n--- Configuration Summary ---")
    logger.info(f"Control points: {config.n_control_points}")
    logger.info(f"Displacement scale: {config.displacement_scale}")
    logger.info(f"Gaussian beta: {config.gaussian_beta}")
    logger.info(f"CPD beta: {config.cpd_beta}")
    logger.info(f"CPD lambda: {config.cpd_lambda}")

    logger.info("\n--- Deformation Summary ---")
    logger.info(f"Pressure mean magnitude: {metrics['gaussian_mean_magnitude']:.6f}")
    logger.info(f"CPD mean magnitude: {metrics['cpd_mean_magnitude']:.6f}")
    logger.info(f"CPD max magnitude: {metrics['cpd_max_magnitude']:.6f}")

    logger.info("\n--- Accuracy Summary ---")
    logger.info(f"Registration error: {metrics['registration_error']:.6f}")
    logger.info(f"Deformation field difference: {metrics['deformation_field_difference']:.6f}")

    accuracy_percentage = (1 - metrics['registration_error'] / metrics['gaussian_mean_magnitude']) * 100
    logger.info(f"CPD accuracy: {accuracy_percentage:.2f}%")

    logger.info("\n" + "=" * 80)
    logger.info("Pipeline finished successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
