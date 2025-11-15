import open3d as o3d
import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure
from functions import visualize_binary_volume
from loguru import logger

# ----------------------
# 1. Load and normalize mesh
# ----------------------
logger.info("Loading Stanford Bunny mesh...")
dataset = o3d.data.BunnyMesh()
mesh = o3d.io.read_triangle_mesh(dataset.path)
mesh.compute_vertex_normals()

# Normalize to unit cube
scale = 1.0 / np.max(mesh.get_max_bound() - mesh.get_min_bound())
mesh.scale(scale, center=mesh.get_center())
logger.info(f"Mesh normalized with scale factor: {scale:.4f}")

# Visualize normalized mesh
o3d.visualization.draw_geometries([mesh], window_name="Normalized Mesh")

# ----------------------
# 2. Compute axis-aligned bounding box (AABB)
# ----------------------
aabb = mesh.get_axis_aligned_bounding_box()
# oobb = mesh.get_oriented_bounding_box()
aabb.color = (1, 0, 0)  # red
min_b = aabb.get_min_bound()
max_b = aabb.get_max_bound()

o3d.visualization.draw_geometries([mesh, aabb], window_name="Mesh + AABB")

# ----------------------
# 3. Voxelize the mesh
# ----------------------
voxel_size = 0.01
voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)
o3d.visualization.draw_geometries([voxel_grid, aabb], window_name="VoxelGrid + AABB", mesh_show_wireframe=True)

# ----------------------
# 4. Generate full voxel grid covering the bounding box (yellow)
# ----------------------
grid_points = []
dims = np.ceil((max_b - min_b) / voxel_size).astype(int)

for ix in range(dims[0]):
    for iy in range(dims[1]):
        for iz in range(dims[2]):
            center = min_b + np.array([ix, iy, iz]) * voxel_size + voxel_size / 2
            grid_points.append(center)

grid_points = np.array(grid_points)
pcd_grid = o3d.geometry.PointCloud()
pcd_grid.points = o3d.utility.Vector3dVector(grid_points)
pcd_grid.paint_uniform_color([1, 1, 0])  # yellow

o3d.visualization.draw_geometries([pcd_grid, voxel_grid, aabb], window_name="Yellow Grid + Bunny VoxelGrid")

# ----------------------
# 5. Convert voxel grid to binary volume
# ----------------------
voxels = np.array([v.grid_index for v in voxel_grid.get_voxels()])
min_idx = voxels.min(axis=0)
max_idx = voxels.max(axis=0)
nx, ny, nz = max_idx - min_idx + 3  # extra padding
volume = np.zeros((nx, ny, nz), dtype=bool)
volume[(voxels - min_idx)[:,0], (voxels - min_idx)[:,1], (voxels - min_idx)[:,2]] = True

logger.info(f"Binary volume created: shape = {volume.shape}, number of voxels = {np.sum(volume)}")

# ----------------------
# 6. Binary erosion
# ----------------------
# Pad volume to avoid border effects
volume_padded = np.pad(volume, pad_width=1, mode='constant', constant_values=0)

# Define structuring element (6-connected)
struct = generate_binary_structure(rank=3, connectivity=1)

# Perform binary erosion
volume_eroded = binary_erosion(volume_padded, structure=struct)

logger.info(f"After erosion: number of voxels = {np.sum(volume_eroded)}")

# Visualize eroded volume
visualize_binary_volume(volume_eroded, voxel_size=0.01, color=[0,1,0], title="Eroded Volume")
