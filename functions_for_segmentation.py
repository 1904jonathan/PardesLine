"""
Helper functions for point cloud segmentation using PointNet with OpenVINO.

This module provides utilities for:
- Loading and preprocessing point cloud data
- Setting up PointNet models with OpenVINO
- Visualizing point clouds and segmentation results with Open3D
- Running inference and processing results
"""

from pathlib import Path
from typing import Union, Tuple
import numpy as np
import open3d as o3d
import openvino as ov
from loguru import logger
from dataclasses import dataclass


@dataclass
class SegmentationConfig:
    """Configuration parameters for PointNet segmentation."""
    model_dir: Path = Path("model")
    model_url: str = "https://storage.googleapis.com/ailia-models/pointnet_pytorch/chair_100.onnx"
    model_name: str = "chair_100.onnx"
    data_url: str = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/pts/chair.pts"
    data_dir: Path = Path("data")
    output_dir: Path = Path(r"C:\PardesLineData")
    device_name: str = "AUTO"
    classes: list = None

    def __post_init__(self):
        if self.classes is None:
            self.classes = ['back', 'seat', 'leg', 'arm']


def setup_model(config: SegmentationConfig) -> Tuple[ov.Model, ov.Core]:
    """
    Download, convert, and load the PointNet ONNX model for OpenVINO.

    Parameters:
        config: SegmentationConfig object with model settings

    Returns:
        Tuple of (model, core) where model is the OpenVINO model and core is the OpenVINO Core instance
    """
    logger.info("Setting up PointNet model")

    # Create model directory
    config.model_dir.mkdir(exist_ok=True)

    # Download model if needed
    from notebook_utils import download_file
    onnx_model_path = config.model_dir / config.model_name

    if not onnx_model_path.exists():
        logger.info(f"Downloading model from {config.model_url}")
        download_file(config.model_url, directory=config.model_dir, show_progress=False)
    else:
        logger.info(f"Model already exists at {onnx_model_path}")

    # Convert to OpenVINO IR format if needed
    ir_model_xml = onnx_model_path.with_suffix(".xml")
    core = ov.Core()

    if not ir_model_xml.exists():
        logger.info("Converting ONNX model to OpenVINO IR format")
        model = ov.convert_model(onnx_model_path)
        ov.save_model(model, ir_model_xml)
        logger.info(f"Model saved to {ir_model_xml}")
    else:
        logger.info(f"Loading existing IR model from {ir_model_xml}")
        model = core.read_model(model=ir_model_xml)

    logger.info(f"Input shape: {model.input(0).partial_shape}")
    logger.info(f"Output shape: {model.output(0).partial_shape}")

    return model, core


def load_point_cloud_data(point_file: Union[str, Path]) -> np.ndarray:
    """
    Load and normalize point cloud data from a .pts file.

    Parameters:
        point_file: Path to .pts file containing XYZ coordinates

    Returns:
        Normalized point cloud as numpy array of shape (N, 3)
    """
    logger.info(f"Loading point cloud from {point_file}")

    point_set = np.loadtxt(point_file).astype(np.float32)
    logger.info(f"Loaded {len(point_set)} points")

    # Normalization: center and scale
    centroid = np.mean(point_set, axis=0)
    point_set = point_set - centroid

    max_distance = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)))
    point_set = point_set / max_distance

    logger.info(f"Point cloud normalized (centered at origin, max distance = 1.0)")

    return point_set


def load_point_cloud_as_pcd(point_file: Union[str, Path]) -> o3d.geometry.PointCloud:
    """
    Load point cloud data and convert to Open3D PointCloud object.

    Parameters:
        point_file: Path to .pts file

    Returns:
        Open3D PointCloud object
    """
    point_set = load_point_cloud_data(point_file)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set)

    return pcd


def preprocess_for_inference(points: np.ndarray) -> np.ndarray:
    """
    Preprocess point cloud for PointNet inference.

    PointNet expects input shape: (batch_size, 3, num_points)

    Parameters:
        points: Point cloud array of shape (N, 3)

    Returns:
        Preprocessed array of shape (1, 3, N)
    """
    # Transpose from (N, 3) to (3, N)
    point = points.transpose(1, 0)

    # Add batch dimension: (3, N) -> (1, 3, N)
    point = np.expand_dims(point, axis=0)

    logger.info(f"Preprocessed point cloud for inference: {point.shape}")

    return point


def run_inference(model: ov.Model, core: ov.Core, input_data: np.ndarray,
                 device_name: str = "AUTO") -> np.ndarray:
    """
    Run PointNet inference on preprocessed point cloud data.

    Parameters:
        model: OpenVINO model
        core: OpenVINO Core instance
        input_data: Preprocessed point cloud of shape (1, 3, N)
        device_name: Device to run inference on (AUTO, CPU, GPU, etc.)

    Returns:
        Segmentation predictions of shape (N,) with class labels for each point
    """
    logger.info(f"Compiling model for device: {device_name}")
    compiled_model = core.compile_model(model=model, device_name=device_name)

    logger.info("Running inference...")
    output_layer = compiled_model.output(0)
    result = compiled_model([input_data])[output_layer]

    # Get predicted class for each point
    # Result shape: (1, N, num_classes) -> (N,)
    predicted_labels = np.argmax(result, axis=2).squeeze()

    logger.info(f"Inference complete. Result shape: {predicted_labels.shape}")
    logger.info(f"Unique labels found: {np.unique(predicted_labels)}")

    return predicted_labels


def compute_segmentation_stats(labels: np.ndarray, class_names: list) -> dict:
    """
    Compute statistics about segmentation results.

    Parameters:
        labels: Predicted class labels for each point
        class_names: List of class names

    Returns:
        Dictionary with statistics for each class
    """
    stats = {}
    total_points = len(labels)

    for i, class_name in enumerate(class_names):
        count = np.sum(labels == i)
        percentage = (count / total_points) * 100
        stats[class_name] = {
            'count': count,
            'percentage': percentage
        }
        logger.info(f"{class_name}: {count} points ({percentage:.1f}%)")

    return stats


def visualize_point_cloud(point_set: np.ndarray, window_name: str = "Point Cloud Visualization",
                         point_size: float = 1.0):
    """
    Visualize point cloud using Open3D.

    Parameters:
        point_set: Point cloud coordinates (N, 3)
        window_name: Window title
        point_size: Size of points in visualization
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])

    logger.info(f"Displaying: {window_name}")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=window_name,
        point_show_normal=False,
        width=800,
        height=600
    )


def visualize_segmentation(point_set: np.ndarray, labels: np.ndarray, class_names: list,
                          window_name: str = "Segmentation Result", point_size: float = 1.0):
    """
    Visualize point cloud with segmentation colors using Open3D.

    Each class is assigned a distinct color for easy identification.

    Parameters:
        point_set: Point cloud coordinates (N, 3)
        labels: Class label for each point (N,)
        class_names: Names of the classes
        window_name: Window title
        point_size: Size of points in visualization
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set)

    # Color map for different parts
    color_map = {
        0: [1.0, 0.0, 0.0],    # red for back
        1: [0.0, 1.0, 0.0],    # green for seat
        2: [0.0, 0.0, 1.0],    # blue for leg
        3: [1.0, 1.0, 0.0],    # yellow for arm
    }

    # Extend color map if we have more classes
    for i in range(4, len(class_names)):
        # Generate random colors for additional classes
        color_map[i] = np.random.rand(3).tolist()

    # Assign colors based on labels
    colors = np.array([color_map.get(label, [0.5, 0.5, 0.5]) for label in labels])
    pcd.colors = o3d.utility.Vector3dVector(colors)

    logger.info(f"Displaying segmentation: {window_name}")
    logger.info(f"Color legend:")
    for i, class_name in enumerate(class_names):
        if i in color_map:
            rgb = [int(c * 255) for c in color_map[i]]
            logger.info(f"  {class_name}: RGB{tuple(rgb)}")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=window_name,
        point_show_normal=False,
        width=800,
        height=600
    )


def download_sample_data(config: SegmentationConfig) -> Path:
    """
    Download sample point cloud data for testing.

    Parameters:
        config: SegmentationConfig with data URLs and directories

    Returns:
        Path to downloaded data file
    """
    from notebook_utils import download_file

    config.data_dir.mkdir(exist_ok=True)

    logger.info(f"Downloading sample data from {config.data_url}")
    point_data = download_file(
        config.data_url,
        directory=str(config.data_dir)
    )

    logger.info(f"Sample data saved to {point_data}")
    return Path(point_data)


def get_color_map() -> dict:
    """
    Get the standard color map for segmentation classes.

    Returns:
        Dictionary mapping class indices to RGB colors (0-1 range)
    """
    color_map = {
        0: [1.0, 0.0, 0.0],    # red for back
        1: [0.0, 1.0, 0.0],    # green for seat
        2: [0.0, 0.0, 1.0],    # blue for leg
        3: [1.0, 1.0, 0.0],    # yellow for arm
    }
    return color_map


def save_segmented_point_cloud(point_set: np.ndarray, labels: np.ndarray,
                               output_path: Union[str, Path],
                               class_names: list = None) -> Path:
    """
    Save segmented point cloud to PLY file with color-coded parts.

    Parameters:
        point_set: Point cloud coordinates (N, 3)
        labels: Class label for each point (N,)
        output_path: Path to save the PLY file
        class_names: Optional list of class names for logging

    Returns:
        Path to saved PLY file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set)

    # Get color map and assign colors
    color_map = get_color_map()

    # Extend color map if we have more classes
    max_label = int(labels.max())
    for i in range(4, max_label + 1):
        color_map[i] = np.random.rand(3).tolist()

    # Assign colors based on labels
    colors = np.array([color_map.get(int(label), [0.5, 0.5, 0.5]) for label in labels])
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save to PLY file
    success = o3d.io.write_point_cloud(str(output_path), pcd)

    if success:
        logger.info(f"Segmented point cloud saved to: {output_path}")
        logger.info(f"File size: {output_path.stat().st_size / 1024:.2f} KB")

        if class_names:
            logger.info("Color mapping:")
            for i, class_name in enumerate(class_names):
                if i in color_map:
                    rgb = [int(c * 255) for c in color_map[i]]
                    logger.info(f"  {class_name}: RGB{tuple(rgb)}")
    else:
        logger.error(f"Failed to save point cloud to {output_path}")

    return output_path


def save_original_point_cloud(point_set: np.ndarray, output_path: Union[str, Path]) -> Path:
    """
    Save original point cloud to PLY file.

    Parameters:
        point_set: Point cloud coordinates (N, 3)
        output_path: Path to save the PLY file

    Returns:
        Path to saved PLY file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # Save to PLY file
    success = o3d.io.write_point_cloud(str(output_path), pcd)

    if success:
        logger.info(f"Original point cloud saved to: {output_path}")
        logger.info(f"File size: {output_path.stat().st_size / 1024:.2f} KB")
    else:
        logger.error(f"Failed to save point cloud to {output_path}")

    return output_path
