"""
Point Cloud Segmentation using PointNet with OpenVINO and Open3D.

This script demonstrates a complete pipeline for:
1. Setting up PointNet model with OpenVINO
2. Downloading and loading sample point cloud data (chair)
3. Preprocessing point cloud for inference
4. Running PointNet segmentation to classify each point
5. Visualizing original and segmented point clouds with Open3D
6. Computing and displaying segmentation statistics

The PointNet model performs part segmentation on 3D objects, identifying
different components (e.g., for a chair: back, seat, legs, arms).

All parameters are logged using loguru and visualizations are displayed at each step.
"""

import numpy as np
import open3d as o3d
from loguru import logger
from pathlib import Path

# Fetch notebook_utils module for downloading data
import urllib.request
urllib.request.urlretrieve(
    url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
    filename='notebook_utils.py'
)

from functions_for_segmentation import (
    SegmentationConfig,
    setup_model,
    load_point_cloud_data,
    preprocess_for_inference,
    run_inference,
    compute_segmentation_stats,
    visualize_point_cloud,
    visualize_segmentation,
    download_sample_data,
    save_original_point_cloud,
    save_segmented_point_cloud
)


def main():
    """
    Main pipeline for point cloud segmentation using PointNet.
    """
    logger.info("="*80)
    logger.info("POINT CLOUD SEGMENTATION PIPELINE - PointNet with OpenVINO")
    logger.info("="*80)

    # ============================================================================
    # Step 1: Configuration
    # ============================================================================
    logger.info("\n=== Step 1: Pipeline Configuration ===")

    config = SegmentationConfig(
        model_dir=Path("model"),
        data_dir=Path("data"),
        output_dir=Path(r"C:\PardesLineData"),
        device_name="AUTO",
        classes=['back', 'seat', 'leg', 'arm']
    )

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Configuration loaded:")
    logger.info(f"  Model directory: {config.model_dir}")
    logger.info(f"  Data directory: {config.data_dir}")
    logger.info(f"  Output directory: {config.output_dir}")
    logger.info(f"  Device: {config.device_name}")
    logger.info(f"  Classes: {config.classes}")

    # ============================================================================
    # Step 2: Setup PointNet Model
    # ============================================================================
    logger.info("\n=== Step 2: Setting up PointNet Model ===")

    model, core = setup_model(config)
    logger.info("Model setup complete")

    # ============================================================================
    # Step 3: Download and Load Sample Data
    # ============================================================================
    logger.info("\n=== Step 3: Loading Point Cloud Data ===")

    point_data_path = download_sample_data(config)
    points = load_point_cloud_data(point_data_path)

    logger.info(f"Point cloud shape: {points.shape}")
    logger.info(f"Point cloud min: {points.min(axis=0)}")
    logger.info(f"Point cloud max: {points.max(axis=0)}")

    # ============================================================================
    # Step 4: Visualize Original Point Cloud
    # ============================================================================
    logger.info("\n=== Step 4: Visualizing Original Point Cloud ===")

    visualize_point_cloud(points, window_name="Step 4: Original Chair Point Cloud", point_size=2.0)

    # ============================================================================
    # Step 5: Preprocess for Inference
    # ============================================================================
    logger.info("\n=== Step 5: Preprocessing for Inference ===")

    input_data = preprocess_for_inference(points)
    logger.info(f"Input data shape for model: {input_data.shape}")

    # ============================================================================
    # Step 6: Run PointNet Inference
    # ============================================================================
    logger.info("\n=== Step 6: Running PointNet Inference ===")

    predicted_labels = run_inference(model, core, input_data, device_name=config.device_name)

    logger.info(f"Predictions shape: {predicted_labels.shape}")
    logger.info(f"Number of unique classes found: {len(np.unique(predicted_labels))}")

    # ============================================================================
    # Step 7: Compute Segmentation Statistics
    # ============================================================================
    logger.info("\n=== Step 7: Computing Segmentation Statistics ===")

    stats = compute_segmentation_stats(predicted_labels, config.classes)

    # ============================================================================
    # Step 8: Visualize Segmentation Results
    # ============================================================================
    logger.info("\n=== Step 8: Visualizing Segmentation Results ===")

    visualize_segmentation(
        points,
        predicted_labels,
        config.classes,
        window_name="Step 8: Chair Segmentation Results",
        point_size=2.0
    )

    # ============================================================================
    # Step 9: Save Point Clouds to PLY Files
    # ============================================================================
    logger.info("\n=== Step 9: Saving Point Clouds ===")

    # Save original point cloud
    original_ply_path = config.output_dir / "chair_original.ply"
    save_original_point_cloud(points, original_ply_path)

    # Save segmented point cloud with colors
    segmented_ply_path = config.output_dir / "chair_segmented.ply"
    save_segmented_point_cloud(points, predicted_labels, segmented_ply_path, config.classes)

    # ============================================================================
    # Step 10: Final Summary
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("SEGMENTATION PIPELINE COMPLETE")
    logger.info("="*80)

    logger.info("\n--- Summary ---")
    logger.info(f"Total points processed: {len(points)}")
    logger.info(f"Classes: {config.classes}")
    logger.info(f"Device used: {config.device_name}")

    logger.info("\n--- Segmentation Distribution ---")
    for class_name, class_stats in stats.items():
        logger.info(f"{class_name}:")
        logger.info(f"  Points: {class_stats['count']}")
        logger.info(f"  Percentage: {class_stats['percentage']:.2f}%")

    logger.info("\n--- Saved Files ---")
    logger.info(f"Original point cloud: {original_ply_path}")
    logger.info(f"Segmented point cloud: {segmented_ply_path}")
    logger.info(f"Output directory: {config.output_dir}")

    logger.info("\n" + "="*80)
    logger.info("Pipeline finished successfully!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
