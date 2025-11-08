import open3d as o3d
import numpy as np
import os
import functions as fn

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
output_dir = r"C:\PardesLineData"
os.makedirs(output_dir, exist_ok=True)

# Load a sample mesh (Stanford Bunny) from Open3D's sample dataset
dataset = o3d.data.BunnyMesh()
mesh = o3d.io.read_triangle_mesh(dataset.path)

# -------------------------------------------------------------------------
# Basic Mesh Information & Visualization
# -------------------------------------------------------------------------
print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)

# Compute normals for shading
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])

# Paint mesh uniformly (green)
mesh.paint_uniform_color([0, 1, 0])
o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)

# -------------------------------------------------------------------------
# Mesh Subdivision (Upsampling)
# -------------------------------------------------------------------------
mesh_subdiv = mesh.subdivide_midpoint(number_of_iterations=1)
print(f"After subdivision: {len(mesh_subdiv.vertices)} vertices, {len(mesh_subdiv.triangles)} triangles")
o3d.visualization.draw_geometries([mesh_subdiv], mesh_show_wireframe=True)

# -------------------------------------------------------------------------
# Mesh Simplification (Decimation)
# -------------------------------------------------------------------------
mesh_simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=3000)
mesh_simplified.paint_uniform_color([0, 0, 1])
print(f"Simplified mesh: {len(mesh_simplified.vertices)} vertices, {len(mesh_simplified.triangles)} triangles")
o3d.visualization.draw_geometries([mesh_simplified], mesh_show_wireframe=True)

# -------------------------------------------------------------------------
# Convert Mesh Vertices to a Point Cloud
# -------------------------------------------------------------------------
pcd_from_vertices = fn.points_to_pcd(np.asarray(mesh.vertices))
pcd_from_vertices.paint_uniform_color([1, 0, 0])
pcd_from_vertices.estimate_normals()
o3d.visualization.draw_geometries([pcd_from_vertices])

# -------------------------------------------------------------------------
# Sampling Points From Mesh Surface
# -------------------------------------------------------------------------
# Uniform sampling
pcd_uniform = mesh.sample_points_uniformly(number_of_points=2500)
o3d.visualization.draw_geometries([pcd_uniform])

# Poisson-disk sampling (better spatial uniformity)
pcd_poisson = mesh.sample_points_poisson_disk(number_of_points=2000)
pcd_poisson.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([pcd_poisson])

# -------------------------------------------------------------------------
# Save Outputs
# -------------------------------------------------------------------------
o3d.io.write_point_cloud(os.path.join(output_dir, "bunny_2000.ply"), pcd_poisson)
o3d.io.write_triangle_mesh(os.path.join(output_dir, "bunny_simplified.ply"), mesh_simplified)
o3d.io.write_triangle_mesh(os.path.join(output_dir, "bunny_original.ply"), mesh)

print(f"\nSaved results to: {output_dir}")
