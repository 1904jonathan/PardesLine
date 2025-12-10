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
from scipy.spatial import cKDTree
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


def load_bunny_source(target_points: int = 1500) -> np.ndarray:
    """
    Load bunny point cloud from Open3D and downsample it.

    Args:
        target_points: Target number of points after downsampling (minimum)

    Returns:
        Point cloud as numpy array of shape (N, 3)
    """
    logger.info("Loading bunny point cloud from Open3D dataset")

    # Load bunny from Open3D
    bunny_mesh = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(bunny_mesh.path)

    # Sample points from mesh
    pcd = mesh.sample_points_poisson_disk(number_of_points=target_points)
    logger.info(f"Sampled {len(pcd.points)} points from bunny mesh")

    points = np.asarray(pcd.points)

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


def nonRigidPressureDeformation(
    points: np.ndarray,
    center_point: np.ndarray,
    radius: float = 0.05,
    direction: np.ndarray = np.array([1.0, 0.0, 0.0]),
    max_displacement: float = 0.02
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply a smooth, non-rigid pressure deformation to a point cloud.

    Parameters:
    - points: (N, 3) ndarray of 3D points
    - center_point: (3,) array, center of the pressure
    - radius: float, radius of influence (in same units as point cloud)
    - direction: (3,) array, direction of the pressure (will be normalized)
    - max_displacement: float, maximum displacement at the center

    Returns:
        Tuple of (deformed_points, deformation_field, control_points, displacements)
    """
    logger.info("Applying non-rigid pressure deformation with Gaussian falloff")

    # Normalize direction
    direction = direction / np.linalg.norm(direction)

    logger.info(f"Pressure center: [{center_point[0]:.4f}, {center_point[1]:.4f}, {center_point[2]:.4f}]")
    logger.info(f"Pressure direction: [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]")
    logger.info(f"Radius of influence: {radius:.4f}")
    logger.info(f"Max displacement at center: {max_displacement:.4f}")

    # Initialize deformation field
    deformation_field = np.zeros_like(points)

    # Build spatial index
    tree = cKDTree(points)
    affected_indices = tree.query_ball_point(center_point, r=radius)

    if len(affected_indices) == 0:
        logger.warning("No points within radius of influence!")
        return points.copy(), deformation_field, center_point.reshape(1, 3), (direction * max_displacement).reshape(1, 3)

    logger.info(f"Affecting {len(affected_indices)} points within radius")

    affected_points = points[affected_indices]

    # Compute radial distances from center
    distances = np.linalg.norm(affected_points - center_point, axis=1)

    # Gaussian falloff deformation (smooth and non-rigid)
    sigma = radius / 2.0
    weights = np.exp(- (distances**2) / (2 * sigma**2))  # (M,)
    displacements = (weights[:, np.newaxis]) * max_displacement * direction  # (M, 3)

    # Apply deformation to affected points
    for i, idx in enumerate(affected_indices):
        deformation_field[idx] = displacements[i]

    # Apply deformation
    deformed_points = points + deformation_field

    mean_magnitude = np.mean(np.linalg.norm(deformation_field, axis=1))
    max_magnitude = np.max(np.linalg.norm(deformation_field, axis=1))
    logger.info(f"Pressure deformation applied:")
    logger.info(f"  Mean magnitude: {mean_magnitude:.6f}")
    logger.info(f"  Max magnitude: {max_magnitude:.6f}")

    # Return in same format as previous function
    control_points = center_point.reshape(1, 3)
    displacement_vector = (direction * max_displacement).reshape(1, 3)

    return deformed_points, deformation_field, control_points, displacement_vector


def apply_gaussian_deformation(
    points: np.ndarray,
    displacement_scale: float,
    beta: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply non-rigid pressure deformation to point cloud.
    Creates a localized pressure deformation from left to right (X direction),
    centered at the middle height (Y center) of the bunny.

    This is a wrapper for nonRigidPressureDeformation to maintain API compatibility.

    Args:
        points: Input point cloud (N, 3)
        n_control: Not used (kept for API compatibility)
        displacement_scale: Maximum displacement at pressure center
        beta: Radius of influence for pressure deformation

    Returns:
        Tuple of (deformed_points, deformation_field, control_points, displacements)
    """
    logger.info("Applying pressure deformation from left to right at Y-center")

    # Find the center of the point cloud
    center = np.mean(points, axis=0)

    # Find the Y (vertical) center, but use left side for X (min X)
    y_center = center[1]
    z_center = center[2]
    x_left = np.min(points[:, 0])

    # Create pressure center point at left side, middle height
    center_point = np.array([x_left, y_center, z_center])

    # Direction: left to right (positive X direction)
    direction = np.array([1.0, 0.0, 0.0])

    # Use beta as radius, displacement_scale as max displacement
    radius = beta
    max_displacement = displacement_scale

    return nonRigidPressureDeformation(
        points=points,
        center_point=center_point,
        radius=radius,
        direction=direction,
        max_displacement=max_displacement
    )


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
