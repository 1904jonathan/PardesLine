import numpy as np
import open3d as o3d
from loguru import logger
from functions_for_sdf import (
      load_mesh,
      create_voxel_grid,
      compute_sdf_open3d,
      compute_occupancy_open3d,
      select_shell_points,
      create_colored_pointcloud,
      reconstruct_mesh_from_sdf,
      apply_binary_erosion,
      reconstruct_mesh_from_binary,
      visualize_step,
      PipelineConfig
)

def main():
        
    # Configuration
    config = PipelineConfig(
        voxel_size=0.01,
        padding_voxels=10,
        shell_thickness=0.02,
        erosion_iterations=1
    )
    
    # Step 1: Load mesh
    logger.info("\n=== Step 1: Loading Mesh ===")
    mesh_original = load_mesh()
    visualize_step([mesh_original], "Step 1: Original Mesh")
    
    # Step 2: Create voxel grid
    logger.info("\n=== Step 2: Creating Voxel Grid ===")
    points, grid_shape, origin = create_voxel_grid(mesh_original, config)
    
    # Step 3: Compute SDF using Open3D
    logger.info("\n=== Step 3: Computing Signed Distance Field ===")
    sdf_values = compute_sdf_open3d(mesh_original, points)
    sdf_grid = sdf_values.reshape(grid_shape)
    
    # Create colored point cloud for visualization
    pc_sdf = create_colored_pointcloud(points, sdf_values, config)
    visualize_step([pc_sdf, mesh_original], "Step 3: SDF Colored Volume")
    
    # Step 4: Compute occupancy and select shell points
    logger.info("\n=== Step 4: Selecting Shell Points ===")
    occupancy = compute_occupancy_open3d(mesh_original, points)
    selected_mask = select_shell_points(sdf_values, occupancy, config.shell_thickness)
    
    # Visualize selected points
    pc_selected = o3d.geometry.PointCloud()
    pc_selected.points = o3d.utility.Vector3dVector(points[selected_mask])
    colors_selected = np.asarray(pc_sdf.colors)[selected_mask]
    pc_selected.colors = o3d.utility.Vector3dVector(colors_selected)
    visualize_step([pc_selected], "Step 4: Selected Points (Inside + Shell)")
    
    
    # Step 5: Apply erosion
    logger.info("\n=== Step 6: Applying Binary Erosion ===")
    volume_binary = selected_mask.reshape(grid_shape)
    volume_eroded = apply_binary_erosion(volume_binary, config.erosion_iterations)
    
    # reconstruct mesh using marching cubes
    mesh_re = reconstruct_mesh_from_binary(volume_binary, origin, config.voxel_size)
    mesh_re.paint_uniform_color([0, 0, 1])
    
    # Step 6: Reconstruct eroded mesh
    logger.info("\n=== Step 7: Reconstructing Eroded Mesh ===")
    mesh_eroded = reconstruct_mesh_from_binary(volume_eroded, origin, config.voxel_size)
    mesh_eroded.paint_uniform_color([0.2, 0.8, 0.2])
    visualize_step([mesh_eroded], "Step 7: Eroded Mesh")
    
    # Final combined visualization
    logger.info("\n=== Final Visualization ===")
    mesh_original.paint_uniform_color([0.8, 0.8, 0.8])
    visualize_step(
        [mesh_original, mesh_eroded],
        "Complete Pipeline: Original (gray) | Reconstructed (red) | Eroded (green)"
    )
    logger.info("\n=== saving results ===")
    o3d.io.write_triangle_mesh("eroded_mesh.ply", mesh_eroded)
    o3d.io.write_triangle_mesh("mesh_original.ply", mesh_original)
    o3d.io.write_triangle_mesh("mesh_re.ply", mesh_re)


    logger.info("\n=== Pipeline Complete ===")



if __name__ == "__main__":
    main()