import numpy as np
import open3d as o3d
import open3d.visualization as o3d_vis
import matplotlib.pyplot as plt
from loguru import logger


def load_bunny_point_cloud(num_points: int = 750) -> o3d.geometry.PointCloud:
    """Load bunny mesh and sample points from it."""
    logger.info(f"Loading bunny mesh and sampling {num_points} points")
    dataset = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(dataset.path)
    pcd = mesh.sample_points_poisson_disk(num_points)
    logger.debug(f"Point cloud created with {len(pcd.points)} points")
    return pcd


def load_eagle_point_cloud() -> o3d.geometry.PointCloud:
    """Load eagle point cloud dataset."""
    logger.info("Loading eagle point cloud")
    dataset = o3d.data.EaglePointCloud()
    pcd = o3d.io.read_point_cloud(dataset.path)
    logger.debug(f"Eagle point cloud: {len(pcd.points)} points")
    return pcd


def compute_convex_hull(pcd: o3d.geometry.PointCloud, visualize: bool = True) -> o3d.geometry.TriangleMesh:
    """Compute convex hull from point cloud."""
    logger.info("Computing convex hull")
    hull, _ = pcd.compute_convex_hull()
    hull.compute_vertex_normals()
    logger.debug(f"Convex hull: {len(hull.vertices)} vertices, {len(hull.triangles)} triangles")
    
    if visualize:
        hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color([1, 0, 0])
        o3d_vis.draw_geometries([pcd, hull_ls])
    
    return hull


def reconstruct_alpha_shape(
    pcd: o3d.geometry.PointCloud,
    alpha_values: list[float] = None,
    visualize: bool = True
) -> list[o3d.geometry.TriangleMesh]:
    """Reconstruct surface using alpha shapes with decreasing alpha values."""
    if alpha_values is None:
        alpha_values = np.logspace(np.log10(0.5), np.log10(0.01), num=4).tolist()
    
    logger.info(f"Running alpha shape reconstruction with {len(alpha_values)} alpha values")
    
    # Pre-compute tetrahedra for efficiency
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    meshes = []
    
    for alpha in alpha_values:
        logger.debug(f"Processing alpha={alpha:.4f}")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha, tetra_mesh, pt_map
        )
        mesh.compute_vertex_normals()
        meshes.append(mesh)
        
        if visualize:
            o3d_vis.draw_geometries([mesh], mesh_show_back_face=True)
    
    logger.info("Alpha shape reconstruction completed")
    return meshes


def reconstruct_ball_pivoting(
    pcd: o3d.geometry.PointCloud,
    radii: list[float] = None,
    visualize: bool = True
) -> o3d.geometry.TriangleMesh:
    """Reconstruct surface using ball pivoting algorithm."""
    if radii is None:
        radii = [0.005, 0.01, 0.02, 0.04]
    
    logger.info(f"Running ball pivoting with radii: {radii}")
    
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    logger.debug(f"Ball pivoting mesh: {len(rec_mesh.vertices)} vertices, {len(rec_mesh.triangles)} triangles")
    
    if visualize:
        o3d_vis.draw_geometries([pcd, rec_mesh])
    
    return rec_mesh


def reconstruct_poisson(
    pcd: o3d.geometry.PointCloud,
    depth: int = 9,
    density_quantile: float = 0.01,
    visualize: bool = True
) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """Reconstruct surface using Poisson algorithm with density filtering."""
    logger.info(f"Running Poisson reconstruction with depth={depth}")
    
    # Run Poisson reconstruction
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    
    logger.debug(f"Poisson mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    if visualize:
        # Visualize with density colors
        densities = np.asarray(densities)
        density_colors = plt.get_cmap('plasma')(
            (densities - densities.min()) / (densities.max() - densities.min())
        )[:, :3]
        
        density_mesh = o3d.geometry.TriangleMesh()
        density_mesh.vertices = mesh.vertices
        density_mesh.triangles = mesh.triangles
        density_mesh.triangle_normals = mesh.triangle_normals
        density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
        
        o3d_vis.draw_geometries([density_mesh])
    
    # Remove low-density vertices
    logger.debug(f"Removing vertices below {density_quantile:.2%} density quantile")
    vertices_to_remove = densities < np.quantile(densities, density_quantile)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    logger.debug(f"Filtered mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    if visualize:
        o3d_vis.draw_geometries([mesh])
    
    return mesh, densities


def estimate_normals(
    pcd: o3d.geometry.PointCloud,
    k_neighbors: int = 100,
    visualize: bool = True
) -> o3d.geometry.PointCloud:
    """Estimate and orient normals for point cloud."""
    logger.info(f"Estimating normals with k={k_neighbors} neighbors for orientation")
    
    # Invalidate existing normals
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    
    # Estimate normals
    pcd.estimate_normals()
    logger.debug("Normals estimated")
    
    if visualize:
        o3d_vis.draw_geometries([pcd], point_show_normal=True)
    
    # Orient normals consistently
    pcd.orient_normals_consistent_tangent_plane(k_neighbors)
    logger.debug("Normals oriented consistently")
    
    if visualize:
        o3d_vis.draw_geometries([pcd], point_show_normal=True)
    
    return pcd
