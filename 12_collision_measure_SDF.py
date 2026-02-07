import numpy as np
import pyvista as pv
import open3d as o3d

from functions_for_sdf import (
    PipelineConfig,
    create_voxel_grid,
    compute_sdf_open3d,
)

# -----------------------------------------------------------------------------
# 1. Configuration & Ground Truth
# -----------------------------------------------------------------------------
R0, R1 = 0.5, 0.5
PENETRATION_RATIO = 0.30
penetration_gt = PENETRATION_RATIO * R1
center_distance = R0 + R1 - penetration_gt

config = PipelineConfig(voxel_size=0.01, padding_voxels=3)

# -----------------------------------------------------------------------------
# 2. PyVista / VTK Collision (Mesh-based)
# -----------------------------------------------------------------------------
sphere_static_pv = pv.Sphere(radius=R0, center=(0, 0, 0), phi_resolution=50, theta_resolution=50)
sphere_moving_pv = pv.Sphere(radius=R1, center=(center_distance, 0, 0), phi_resolution=50, theta_resolution=50)

collision_data, n_contacts = sphere_static_pv.collision(sphere_moving_pv)
collision_region = sphere_static_pv.extract_cells(collision_data["ContactCells"])

vtk_bbox = collision_region.bounds
pv_collision_width = vtk_bbox[1] - vtk_bbox[0]
pv_bbox_mesh = pv.Box(vtk_bbox)

# -----------------------------------------------------------------------------
# 3. Open3D SDF Collision (Point-based)
# -----------------------------------------------------------------------------
sphere_static_o3d = o3d.geometry.TriangleMesh.create_sphere(radius=R0, resolution=50)
sphere_moving_o3d = o3d.geometry.TriangleMesh.create_sphere(radius=R1, resolution=50)
sphere_moving_o3d.translate((center_distance, 0, 0))

# Sample points on the moving sphere to check against static sphere SDF
pcd = sphere_moving_o3d.sample_points_uniformly(number_of_points=50000)
moving_points = np.asarray(pcd.points)

sdf_on_moving = compute_sdf_open3d(sphere_static_o3d, moving_points)
penetrating_points = moving_points[sdf_on_moving < 0]

if penetrating_points.size == 0:
    raise RuntimeError("SDF collision failed: no penetration detected.")

sdf_bbox_min = penetrating_points.min(axis=0)
sdf_bbox_max = penetrating_points.max(axis=0)
sdf_collision_width = sdf_bbox_max[0] - sdf_bbox_min[0]

o3d_bbox_mesh = pv.Box([sdf_bbox_min[0], sdf_bbox_max[0], 
                        sdf_bbox_min[1], sdf_bbox_max[1], 
                        sdf_bbox_min[2], sdf_bbox_max[2]])

# -----------------------------------------------------------------------------
# 4. Accuracy Analysis (Updated for Penetration Depth)
# -----------------------------------------------------------------------------

# SDF Penetration Depth is the absolute minimum (most negative) SDF value
sdf_penetration_depth = abs(sdf_on_moving.min())

# PyVista doesn't give depth easily, so we use the BBox width 
# (Note: For spheres, BBox width of intersection != Penetration depth)
pv_measured_width = vtk_bbox[1] - vtk_bbox[0]

print("\n" + "="*65)
print(f"{'COLLISION ACCURACY ANALYSIS':^65}")
print("="*65)
print(f"{'Metric':<25} | {'Value':<15} | {'GT':<15}")
print("-" * 65)
# This is the value you were looking for!
print(f"{'Open3D Max Depth (SDF)':<25} | {sdf_penetration_depth:<15.6f} | {penetration_gt:<15.6f}")
print(f"{'Open3D BBox Width':<25} | {sdf_collision_width:<15.6f} | {'N/A'}")
print(f"{'PyVista BBox Width':<25} | {pv_measured_width:<15.6f} | {'N/A'}")
print("-" * 65)

# Calculate error based on actual penetration depth
final_error = abs(sdf_penetration_depth - penetration_gt)
print(f"Final SDF Accuracy Error: {final_error:.6f}")
print("="*65)



# -----------------------------------------------------------------------------
# 5. Visualization
# -----------------------------------------------------------------------------
plotter = pv.Plotter(title="Collision Bounding Box Comparison")

plotter.add_mesh(sphere_static_pv, color="cyan", opacity=0.2, label="Static Sphere")
plotter.add_mesh(sphere_moving_pv, color="magenta", opacity=0.2, label="Moving Sphere")

# PyVista BBox - Red
plotter.add_mesh(pv_bbox_mesh, color="red", style="wireframe", line_width=4, label="BBox PyVista")

# Open3D BBox - Green (scaled slightly for visibility)
plotter.add_mesh(o3d_bbox_mesh.scale(1.01), color="lime", style="wireframe", line_width=8, label="BBox Open3D")

plotter.add_points(penetrating_points, color="yellow", point_size=2, label="SDF Penetration Points")

plotter.add_legend()
plotter.show()