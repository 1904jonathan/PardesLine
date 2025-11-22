import numpy as np
import open3d as o3d
from typing import Tuple, Optional
from dataclasses import dataclass
from loguru import logger

@dataclass
class PipelineConfig:
    """Configuration parameters for the SDF pipeline."""
    voxel_size: float = 0.01  # 1 cm voxels
    padding_voxels: int = 3  # Number of voxels for padding
    shell_thickness: float = 0.01  # 5 mm shell thickness
    erosion_iterations: int = 1  # Number of erosion iterations
    colormap: str = 'viridis'  # Colormap for SDF visualization



def load_mesh(file_path: Optional[str] = None) -> o3d.geometry.TriangleMesh:
    """
    Load a triangle mesh from file or use Open3D sample bunny.
    
    Args:
        file_path: Path to mesh file. If None, loads the sample bunny.
    
    Returns:
        Open3D triangle mesh with computed vertex normals.
    """
    if file_path is None:
        data = o3d.data.BunnyMesh()
        mesh = o3d.io.read_triangle_mesh(data.path)
    else:
        mesh = o3d.io.read_triangle_mesh(file_path)
    
    mesh.compute_vertex_normals()
    logger.info(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    return mesh


def create_voxel_grid(mesh: o3d.geometry.TriangleMesh, 
                      config: PipelineConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 3D voxel grid covering the mesh bounding box with padding.
    
    Args:
        mesh: Input triangle mesh
        config: Pipeline configuration
    
    Returns:
        Tuple of (points, grid_shape, origin) where:
            - points: Nx3 array of grid point coordinates
            - grid_shape: (nx, ny, nz) dimensions
            - origin: minimum bound of the grid
    """
    bbox = mesh.get_axis_aligned_bounding_box()
    padding = config.padding_voxels * config.voxel_size
    
    min_bound = bbox.min_bound - padding
    max_bound = bbox.max_bound + padding
    
    # Generate grid vectors
    xs = np.arange(min_bound[0], max_bound[0] + 1e-12, config.voxel_size)
    ys = np.arange(min_bound[1], max_bound[1] + 1e-12, config.voxel_size)
    zs = np.arange(min_bound[2], max_bound[2] + 1e-12, config.voxel_size)
    
    grid_shape = (len(xs), len(ys), len(zs))
    logger.info(f"Grid dimensions: {grid_shape}, total voxels: {np.prod(grid_shape)}")
    
    # Create meshgrid points
    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing='ij')
    points = np.vstack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).T
    
    return points, grid_shape, min_bound


def compute_sdf_open3d(mesh: o3d.geometry.TriangleMesh, 
                       query_points: np.ndarray) -> np.ndarray:
    """
    Compute signed distance field using Open3D's RaycastingScene.
    
    Args:
        mesh: Input triangle mesh
        query_points: Nx3 array of query point coordinates
    
    Returns:
        Signed distance values (negative inside, positive outside)
    """
    # Create raycasting scene
    mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_legacy)
    
    # Compute signed distance
    query_points_tensor = o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)
    signed_distance = scene.compute_signed_distance(query_points_tensor).numpy()
    
    logger.info(f"SDF computed: min={signed_distance.min():.6f}, max={signed_distance.max():.6f}")
    return signed_distance


def compute_occupancy_open3d(mesh: o3d.geometry.TriangleMesh, 
                             query_points: np.ndarray) -> np.ndarray:
    """
    Compute occupancy (inside/outside) using Open3D's RaycastingScene.
    
    Args:
        mesh: Input triangle mesh
        query_points: Nx3 array of query point coordinates
    
    Returns:
        Boolean array indicating if points are inside the mesh
    """
    mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_legacy)
    
    query_points_tensor = o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)
    occupancy = scene.compute_occupancy(query_points_tensor).numpy()
    
    logger.info(f"Occupancy computed: {np.sum(occupancy)} points inside")
    return occupancy.astype(bool)


def create_colored_pointcloud(points: np.ndarray, 
                              sdf_values: np.ndarray,
                              config: PipelineConfig) -> o3d.geometry.PointCloud:
    """
    Create a colored point cloud based on SDF values.
    
    Args:
        points: Nx3 array of point coordinates
        sdf_values: N array of signed distance values
        config: Pipeline configuration
    
    Returns:
        Open3D point cloud with colors mapped to distance
    """
    from matplotlib import cm
    
    abs_dist = np.abs(sdf_values)
    vmin, vmax = 0.0, np.percentile(abs_dist, 99.5)
    norm = np.clip((abs_dist - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
    
    cmap = cm.get_cmap(config.colormap)
    colors = cmap(norm)[:, :3]
    
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)
    
    return pc


def select_shell_points(sdf_values: np.ndarray,
                       occupancy: np.ndarray,
                       shell_thickness: float) -> np.ndarray:
    """
    Select points inside the mesh and within shell thickness outside.
    
    Args:
        sdf_values: Signed distance values
        occupancy: Boolean array of inside/outside
        shell_thickness: Thickness of outer shell to include
    
    Returns:
        Boolean mask of selected points
    """
    outside_shell = (sdf_values > 0.0) & (sdf_values <= shell_thickness)
    selected = occupancy | outside_shell
    
    logger.info(f"Selected points: {np.sum(selected)} / {len(sdf_values)}")
    return selected


def reconstruct_mesh_from_sdf(sdf_grid: np.ndarray,
                              origin: np.ndarray,
                              voxel_size: float) -> o3d.geometry.TriangleMesh:
    """
    Reconstruct mesh from SDF grid using marching cubes.
    
    Args:
        sdf_grid: 3D array of signed distance values
        origin: Grid origin in world coordinates
        voxel_size: Size of each voxel
    
    Returns:
        Reconstructed triangle mesh
    """
    from skimage import measure
    
    # Transpose for marching cubes (expects z, y, x order)
    sdf_transposed = np.transpose(sdf_grid, (2, 1, 0))
    
    # Extract isosurface at level 0
    verts, faces, normals, values = measure.marching_cubes(
        sdf_transposed, 
        level=0.0, 
        spacing=(voxel_size, voxel_size, voxel_size)
    )
    
    # Convert from (z, y, x) back to (x, y, z) and translate to world coords
    verts = verts[:, ::-1] + origin
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()
    
    logger.info(f"Reconstructed mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    return mesh


def apply_binary_erosion(volume: np.ndarray, 
                         iterations: int = 1,
                         connectivity: int = 1) -> np.ndarray:
    """
    Apply binary erosion to a 3D volume.
    
    Args:
        volume: 3D binary volume
        iterations: Number of erosion iterations
        connectivity: Connectivity structure (1=6-neighbors, 2=18, 3=26)
    
    Returns:
        Eroded binary volume
    """
    from scipy.ndimage import generate_binary_structure, binary_erosion
    
    struct = generate_binary_structure(3, connectivity)
    eroded = binary_erosion(volume, structure=struct, iterations=iterations)
    
    logger.info(f"Erosion: {np.sum(volume)} -> {np.sum(eroded)} voxels "
          f"({100*(1-np.sum(eroded)/np.sum(volume)):.1f}% reduction)")
    
    return eroded


def reconstruct_mesh_from_binary(volume: np.ndarray,
                                 origin: np.ndarray,
                                 voxel_size: float) -> o3d.geometry.TriangleMesh:
    """
    Reconstruct mesh from binary volume using marching cubes.
    
    Args:
        volume: 3D binary volume
        origin: Grid origin in world coordinates
        voxel_size: Size of each voxel
    
    Returns:
        Reconstructed triangle mesh
    """
    from skimage import measure
    
    volume_transposed = np.transpose(volume.astype(np.float32), (2, 1, 0))
    
    verts, faces, normals, values = measure.marching_cubes(
        volume_transposed,
        level=0.5,
        spacing=(voxel_size, voxel_size, voxel_size)
    )
    
    verts = verts[:, ::-1] + origin
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()
    
    logger.info(f"Binary reconstruction: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    return mesh


def visualize_step(geometries: list, window_name: str):
    """Visualize geometries in a window."""
    logger.info(f"Displaying: {window_name}")
    o3d.visualization.draw_geometries(geometries, window_name=window_name)