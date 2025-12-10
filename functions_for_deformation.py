"""
Helper functions for Gaussian and CPD deformation pipeline.

This module provides utilities for:
- Point cloud normalization and centering
- Gaussian kernel computation
- Deformation field analysis
- Visualization utilities
"""

import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import Tuple
from loguru import logger


@dataclass
class DeformationConfig:
    """Configuration for deformation pipeline."""
    n_control_points: int = 200
    displacement_scale: float = 0.003
    gaussian_beta: float = 0.1
    cpd_beta: float = 100.0
    cpd_lambda: float = 2.0
    cpd_max_iterations: int = 150
    point_size: float = 2.0


def load_bunny_source(filepath: str = r'C:\CODE\pycpd\data\bunny_source.txt') -> np.ndarray:
    """
    Load bunny point cloud from file.

    Args:
        filepath: Path to bunny source file

    Returns:
        Point cloud as numpy array of shape (N, 3)
    """
    logger.info(f"Loading bunny point cloud from: {filepath}")
    points = np.loadtxt(filepath)
    logger.info(f"Loaded {len(points)} points")
    return points


def normalize_points(points: np.ndarray) -> np.ndarray:
    """
    Normalize point cloud to unit scale centered at origin.

    Args:
        points: Input point cloud (N, 3)

    Returns:
        Normalized point cloud (N, 3)
    """
    center = np.mean(points, axis=0)
    scale = np.linalg.norm(points - center)
    normalized = (points - center) / scale

    logger.debug(f"Normalization - Center: {center}, Scale: {scale:.6f}")
    return normalized


def center_points(points: np.ndarray) -> np.ndarray:
    """
    Center point cloud at origin.

    Args:
        points: Input point cloud (N, 3)

    Returns:
        Centered point cloud (N, 3)
    """
    return points - np.mean(points, axis=0)


def gaussian_kernel(x: np.ndarray, y: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """
    Compute Gaussian kernel between two point sets.

    K(x_i, y_j) = exp(-||x_i - y_j||^2 / (2 * beta^2))

    Args:
        x: First point set (N, D)
        y: Second point set (M, D)
        beta: Kernel bandwidth parameter

    Returns:
        Kernel matrix of shape (N, M)
    """
    pairwise_dists = cdist(x, y, 'sqeuclidean')
    return np.exp(-pairwise_dists / (2 * beta ** 2))


def compute_face_normal_at_center(points: np.ndarray) -> np.ndarray:
    """
    Compute a representative face normal near the center of the point cloud.

    Args:
        points: Input point cloud (N, 3)

    Returns:
        Normal vector (3,) pointing outward from a face near center
    """
    # Find center of point cloud
    center = np.mean(points, axis=0)

    # Find points near the center
    distances_to_center = np.linalg.norm(points - center, axis=1)
    near_center_threshold = np.percentile(distances_to_center, 30)
    near_center_mask = distances_to_center < near_center_threshold
    near_center_points = points[near_center_mask]

    logger.debug(f"Found {len(near_center_points)} points near center for normal estimation")

    # Use PCA to find the local surface orientation
    centered = near_center_points - np.mean(near_center_points, axis=0)
    cov_matrix = np.cov(centered.T)
    _, eigenvectors = np.linalg.eigh(cov_matrix)

    # The eigenvector with smallest eigenvalue is the normal direction
    normal = eigenvectors[:, 0]

    # Ensure normal points outward (away from centroid of all points)
    if np.dot(normal, center - np.mean(near_center_points, axis=0)) < 0:
        normal = -normal

    logger.info(f"Computed face normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")

    return normal


def apply_gaussian_deformation(
    points: np.ndarray,
    n_control: int,
    displacement_scale: float,
    beta: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Gaussian process deformation to point cloud.
    Creates a localized "spike" deformation from left to right (X direction),
    centered at the middle height (Y center) of the bunny, with Gaussian propagation.

    Args:
        points: Input point cloud (N, 3)
        n_control: Number of control points (used for intensity control)
        displacement_scale: Scale of the spike displacement
        beta: Gaussian kernel bandwidth for propagation

    Returns:
        Tuple of (deformed_points, deformation_field, control_points, displacements)
    """
    logger.info("Applying Gaussian spike deformation from left to right at Y-center")

    # Find the center of the point cloud
    center = np.mean(points, axis=0)

    # Find the Y (vertical) center, but use left side for X (min X)
    y_center = center[1]
    z_center = center[2]
    x_left = np.min(points[:, 0])

    # Create control point at left side, middle height
    control_point = np.array([x_left, y_center, z_center])
    control_points = control_point.reshape(1, 3)

    logger.info(f"Control point at left side: [{control_point[0]:.4f}, {control_point[1]:.4f}, {control_point[2]:.4f}]")

    # Direction: left to right (positive X direction)
    direction = np.array([1.0, 0.0, 0.0])

    # Scale displacement by displacement_scale and n_control (intensity factor)
    spike_intensity = displacement_scale * np.sqrt(n_control)
    displacements = (direction * spike_intensity).reshape(1, 3)

    logger.info(f"Spike displacement magnitude: {spike_intensity:.6f}")
    logger.info(f"Spike direction: Left to Right (X-axis) [1.0, 0.0, 0.0]")

    # Compute Gaussian kernel for propagation
    K = gaussian_kernel(points, control_points, beta=beta)
    logger.debug(f"Computed Gaussian kernel with beta={beta} for propagation")

    # Interpolate deformation to full point cloud using Gaussian falloff
    deformation_field = K @ displacements

    # Apply deformation
    deformed_points = points + deformation_field

    mean_magnitude = np.mean(np.linalg.norm(deformation_field, axis=1))
    max_magnitude = np.max(np.linalg.norm(deformation_field, axis=1))
    logger.info(f"Gaussian spike deformation applied:")
    logger.info(f"  Mean magnitude: {mean_magnitude:.6f}")
    logger.info(f"  Max magnitude: {max_magnitude:.6f}")

    return deformed_points, deformation_field, control_points, displacements


def compute_deformation_metrics(
    source: np.ndarray,
    target: np.ndarray,
    deformed: np.ndarray,
    gaussian_deformation: np.ndarray
) -> dict:
    """
    Compute comprehensive deformation metrics.

    Args:
        source: Original source points (N, 3)
        target: Target points after Gaussian deformation (N, 3)
        deformed: Points after CPD registration (N, 3)
        gaussian_deformation: Original Gaussian deformation field (N, 3)

    Returns:
        Dictionary containing all metrics
    """
    cpd_deformation = deformed - source

    metrics = {
        'cpd_mean_magnitude': np.mean(np.linalg.norm(cpd_deformation, axis=1)),
        'cpd_max_magnitude': np.max(np.linalg.norm(cpd_deformation, axis=1)),
        'registration_error': np.mean(np.linalg.norm(deformed - target, axis=1)),
        'deformation_field_difference': np.mean(np.linalg.norm(cpd_deformation - gaussian_deformation, axis=1)),
        'gaussian_mean_magnitude': np.mean(np.linalg.norm(gaussian_deformation, axis=1)),
    }

    return metrics


def create_colored_point_cloud(
    points: np.ndarray,
    color: Tuple[float, float, float]
) -> o3d.geometry.PointCloud:
    """
    Create Open3D point cloud with uniform color.

    Args:
        points: Point cloud array (N, 3)
        color: RGB color tuple (values in [0, 1])

    Returns:
        Colored Open3D PointCloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    return pcd


def visualize_geometries(
    geometries: list,
    window_name: str,
    point_size: float = 2.0
) -> None:
    """
    Visualize geometries with custom window name and point size.

    Args:
        geometries: List of Open3D geometries
        window_name: Window title
        point_size: Point rendering size
    """
    logger.info(f"Displaying: {window_name}")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        point_show_normal=False,
        width=1024,
        height=768
    )
