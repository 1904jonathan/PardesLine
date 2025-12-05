"""
Helper functions for point cloud registration pipeline.

This module provides utilities for:
- Mesh to point cloud conversion
- Point cloud transformation (rotation, translation)
- RANSAC-based global registration
- ICP-based local refinement
- Visualization utilities
"""

import numpy as np
import open3d as o3d
from typing import Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class RegistrationConfig:
    """Configuration parameters for the point cloud registration pipeline."""

    # Point cloud sampling
    num_points: int = 10000  # Number of points for uniform sampling

    # Transformation parameters
    rotation_degrees: Tuple[float, float, float] = (30.0, 45.0, 60.0)  # Rotation around X, Y, Z axes in degrees
    translation: Tuple[float, float, float] = (0.2, 0.3, 0.1)  # Translation vector (dx, dy, dz)

    # RANSAC parameters
    voxel_size: float = 0.005  # Voxel size for downsampling (5mm)
    ransac_distance_threshold: float = 0.015  # RANSAC inlier threshold (15mm)
    ransac_n_points: int = 4  # Number of points for RANSAC correspondences
    ransac_max_iterations: int = 100000  # Maximum RANSAC iterations
    ransac_confidence: float = 0.999  # RANSAC confidence level

    # ICP parameters
    icp_distance_threshold: float = 0.02  # ICP distance threshold (20mm)
    icp_max_iterations: int = 2000  # Maximum ICP iterations
    icp_relative_fitness: float = 1e-6  # ICP convergence criterion (fitness)
    icp_relative_rmse: float = 1e-6  # ICP convergence criterion (RMSE)

    # Feature parameters
    feature_radius: float = 0.025  # Radius for FPFH feature computation (25mm)
    feature_max_nn: int = 100  # Maximum nearest neighbors for features

    # Visualization
    point_size: float = 2.0  # Point size for visualization


def load_bunny_mesh() -> o3d.geometry.TriangleMesh:
    """
    Load the bunny mesh from Open3D sample datasets.

    Returns:
        Loaded and normalized bunny triangle mesh
    """
    logger.info("Loading bunny mesh from Open3D dataset")
    dataset = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(dataset.path)
    mesh.compute_vertex_normals()

    logger.info(f"Bunny mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    logger.info(f"Mesh bounds: min={np.asarray(mesh.get_min_bound())}, max={np.asarray(mesh.get_max_bound())}")

    return mesh


def mesh_to_point_cloud_uniform(mesh: o3d.geometry.TriangleMesh,
                                 num_points: int) -> o3d.geometry.PointCloud:
    """
    Convert triangle mesh to point cloud using uniform sampling.

    Args:
        mesh: Input triangle mesh
        num_points: Number of points to sample uniformly

    Returns:
        Point cloud with uniform sampling
    """
    logger.info(f"Converting mesh to point cloud with uniform sampling ({num_points} points)")

    # Sample points uniformly from mesh surface
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)

    logger.info(f"Point cloud created: {len(pcd.points)} points")
    logger.debug(f"Point cloud bounds: min={np.asarray(pcd.get_min_bound())}, max={np.asarray(pcd.get_max_bound())}")

    return pcd


def create_transformed_copy(pcd: o3d.geometry.PointCloud,
                           rotation_degrees: Tuple[float, float, float],
                           translation: Tuple[float, float, float]) -> o3d.geometry.PointCloud:
    """
    Create a transformed copy of the point cloud with rotation and translation.

    Args:
        pcd: Input point cloud
        rotation_degrees: Rotation angles around X, Y, Z axes in degrees
        translation: Translation vector (dx, dy, dz)

    Returns:
        Transformed point cloud copy painted in cyan (blue)
    """
    logger.info(f"Creating transformed copy with rotation={rotation_degrees}° and translation={translation}")

    # Create a deep copy
    pcd_transformed = o3d.geometry.PointCloud(pcd)

    # Create rotation matrix from Euler angles (in radians)
    rx, ry, rz = np.radians(rotation_degrees)
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

    # Combined rotation matrix (R = Rz * Ry * Rx)
    R = R_z @ R_y @ R_x

    # Apply rotation
    pcd_transformed.rotate(R, center=(0, 0, 0))

    # Apply translation
    pcd_transformed.translate(translation)

    # Paint in cyan (blue) - Open3D tutorial standard
    pcd_transformed.paint_uniform_color([0, 0.651, 0.929])  # Cyan color

    logger.info(f"Transformation matrix (rotation):")
    logger.info(f"\n{R}")
    logger.debug(f"Transformed point cloud bounds: min={np.asarray(pcd_transformed.get_min_bound())}, max={np.asarray(pcd_transformed.get_max_bound())}")

    return pcd_transformed


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud,
                           voxel_size: float) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """
    Downsample point cloud and compute FPFH features for registration.

    Args:
        pcd: Input point cloud
        voxel_size: Voxel size for downsampling

    Returns:
        Tuple of (downsampled point cloud, FPFH features)
    """
    logger.debug(f"Preprocessing point cloud with voxel_size={voxel_size}")

    # Downsample with voxel grid
    pcd_down = pcd.voxel_down_sample(voxel_size)
    logger.debug(f"Downsampled from {len(pcd.points)} to {len(pcd_down.points)} points")

    # Estimate normals
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    logger.debug(f"Normals estimated with radius={radius_normal}")

    # Compute FPFH features
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    logger.debug(f"FPFH features computed with radius={radius_feature}, feature_dim={fpfh.dimension()}")

    return pcd_down, fpfh


def execute_global_registration_ransac(source: o3d.geometry.PointCloud,
                                       target: o3d.geometry.PointCloud,
                                       source_fpfh: o3d.pipelines.registration.Feature,
                                       target_fpfh: o3d.pipelines.registration.Feature,
                                       config: RegistrationConfig) -> o3d.pipelines.registration.RegistrationResult:
    """
    Execute global registration using RANSAC-based feature matching.

    Args:
        source: Source point cloud (downsampled)
        target: Target point cloud (downsampled)
        source_fpfh: FPFH features of source
        target_fpfh: FPFH features of target
        config: Registration configuration

    Returns:
        RANSAC registration result
    """
    logger.info(f"Starting RANSAC global registration")
    logger.info(f"RANSAC parameters: distance_threshold={config.ransac_distance_threshold}, "
                f"n_points={config.ransac_n_points}, max_iterations={config.ransac_max_iterations}")

    # Execute RANSAC-based registration
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=config.ransac_distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=config.ransac_n_points,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(config.ransac_distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(config.ransac_max_iterations, config.ransac_confidence)
    )

    logger.info(f"RANSAC registration complete:")
    logger.info(f"  Fitness: {result.fitness:.6f}")
    logger.info(f"  RMSE: {result.inlier_rmse:.6f}")
    logger.info(f"  Correspondences: {len(result.correspondence_set)}")
    logger.info(f"  Transformation matrix:")
    logger.info(f"\n{result.transformation}")

    return result


def execute_icp_registration(source: o3d.geometry.PointCloud,
                             target: o3d.geometry.PointCloud,
                             initial_transformation: np.ndarray,
                             config: RegistrationConfig) -> o3d.pipelines.registration.RegistrationResult:
    """
    Execute ICP registration for local refinement.

    Args:
        source: Source point cloud
        target: Target point cloud
        initial_transformation: Initial transformation from RANSAC
        config: Registration configuration

    Returns:
        ICP registration result
    """
    logger.info(f"Starting ICP registration for local refinement")
    logger.info(f"ICP parameters: distance_threshold={config.icp_distance_threshold}, "
                f"max_iterations={config.icp_max_iterations}")

    # Execute point-to-point ICP
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=config.icp_distance_threshold,
        init=initial_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=config.icp_relative_fitness,
            relative_rmse=config.icp_relative_rmse,
            max_iteration=config.icp_max_iterations
        )
    )

    logger.info(f"ICP registration complete:")
    logger.info(f"  Fitness: {result.fitness:.6f}")
    logger.info(f"  RMSE: {result.inlier_rmse:.6f}")
    logger.info(f"  Correspondences: {len(result.correspondence_set)}")
    logger.info(f"  Final transformation matrix:")
    logger.info(f"\n{result.transformation}")

    return result


def compute_transformation_error(estimated_T: np.ndarray,
                                 ground_truth_rotation: np.ndarray,
                                 ground_truth_translation: np.ndarray) -> Tuple[float, float]:
    """
    Compute the error between estimated and ground truth transformation.

    Args:
        estimated_T: Estimated 4x4 transformation matrix
        ground_truth_rotation: Ground truth 3x3 rotation matrix
        ground_truth_translation: Ground truth 3x1 translation vector

    Returns:
        Tuple of (rotation_error_degrees, translation_error)
    """
    # Extract estimated rotation and translation
    estimated_R = estimated_T[:3, :3]
    estimated_t = estimated_T[:3, 3]

    # Compute rotation error using Frobenius norm
    R_error_matrix = estimated_R @ ground_truth_rotation.T

    # Compute trace and clamp to valid range [-1, 1] to avoid numerical errors
    trace_val = np.trace(R_error_matrix)
    cos_theta = (trace_val - 1) / 2

    # Clamp to valid arccos range to handle numerical precision issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    rotation_error_rad = np.arccos(cos_theta)
    rotation_error_deg = np.degrees(rotation_error_rad)

    # Compute translation error using Euclidean distance
    translation_error = np.linalg.norm(estimated_t - ground_truth_translation)

    logger.info(f"Transformation error:")
    logger.info(f"  Rotation error: {rotation_error_deg:.4f} degrees")
    logger.info(f"  Translation error: {translation_error:.6f} units")

    return rotation_error_deg, translation_error


def visualize_geometries(geometries: list,
                        window_name: str,
                        point_size: Optional[float] = None) -> None:
    """
    Visualize geometries in a 3D window.

    Args:
        geometries: List of Open3D geometries to visualize
        window_name: Name of the visualization window
        point_size: Optional point size for rendering
    """
    logger.info(f"Displaying: {window_name}")

    if point_size is not None:
        # Use advanced visualizer for custom point size
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name)

        for geom in geometries:
            vis.add_geometry(geom)

        render_option = vis.get_render_option()
        render_option.point_size = point_size

        vis.run()
        vis.destroy_window()
    else:
        # Use simple visualizer
        o3d.visualization.draw_geometries(geometries, window_name=window_name)


def draw_registration_result(source: o3d.geometry.PointCloud,
                             target: o3d.geometry.PointCloud,
                             transformation: np.ndarray,
                             window_name: str,
                             point_size: float = 2.0) -> None:
    """
    Visualize registration result by applying transformation to source.

    Args:
        source: Source point cloud (will be copied and transformed)
        target: Target point cloud
        transformation: Transformation matrix to apply
        window_name: Name of the visualization window
        point_size: Point size for rendering
    """
    source_temp = o3d.geometry.PointCloud(source)
    source_temp.transform(transformation)

    visualize_geometries([source_temp, target], window_name, point_size)
