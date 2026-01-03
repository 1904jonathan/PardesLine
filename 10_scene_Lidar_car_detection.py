#!/usr/bin/env python3
"""
nuScenes 3D Visualization Tool
Displays LiDAR point clouds with 3D bounding boxes for all scenes
"""

import open3d as o3d
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

# Initialize nuScenes dataset
nusc = NuScenes(version='v1.0-mini', dataroot=".", verbose=False)


def visualize_scene(scene_idx=0):
    """
    Visualize a single scene with LiDAR point cloud and 3D bounding boxes

    Args:
        scene_idx: Index of the scene to visualize
    """
    scene = nusc.scene[scene_idx]
    sample = nusc.get('sample', scene['first_sample_token'])

    # Load LiDAR point cloud
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    pc = LidarPointCloud.from_file(nusc.dataroot + '/' + lidar_data['filename'])
    points = pc.points[:3, :].T

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Color points by height for better visualization
    colors = np.zeros((points.shape[0], 3))
    z_norm = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min() + 1e-6)
    colors[:, 0] = z_norm * 0.3  # Red
    colors[:, 1] = z_norm * 0.8  # Green
    colors[:, 2] = (1 - z_norm) * 0.6  # Blue
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Get bounding boxes from nuScenes (already transformed to LiDAR frame)
    _, boxes_nusc, _ = nusc.get_sample_data(lidar_token)

    # Convert to Open3D bounding boxes with 90° rotation correction
    boxes_o3d = []
    car_count = 0

    for box_nusc in boxes_nusc:
        if 'car' in box_nusc.name.lower():
            # Apply 90° rotation around Z-axis to align boxes with cars
            rotation_correction = np.array([
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]
            ])

            corrected_rotation = box_nusc.rotation_matrix @ rotation_correction

            bbox = o3d.geometry.OrientedBoundingBox(
                center=box_nusc.center,
                R=corrected_rotation,
                extent=box_nusc.wlh
            )
            bbox.color = [0, 1, 0]  # Green for cars
            boxes_o3d.append(bbox)
            car_count += 1

    # Setup visualization
    scene_name = scene['name']
    window_title = f"Scene {scene_idx}/{len(nusc.scene)-1}: {scene_name} - {car_count} cars"

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title, width=1920, height=1080)

    # Add geometries
    vis.add_geometry(pcd)
    for box in boxes_o3d:
        vis.add_geometry(box)

    # Configure rendering options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.02, 0.02, 0.02])  # Dark background
    opt.point_size = 2.0
    opt.line_width = 8.0
    opt.show_coordinate_frame = True

    # Set camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.4)
    ctr.set_front([0.3, -0.8, -0.5])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 0, 1])

    # Display info
    print(f"\n{'='*70}")
    print(f"Scene {scene_idx}/{len(nusc.scene)-1}: {scene_name}")
    print(f"{'='*70}")
    print(f"  Vehicles detected: {car_count}")
    print(f"  LiDAR points: {len(points):,}")
    print(f"\n  Controls:")
    print(f"    • Left mouse: Rotate view")
    print(f"    • Scroll: Zoom")
    print(f"    • Q or ESC: Close window")
    print(f"{'='*70}\n")

    vis.run()
    vis.destroy_window()


def main():
    """Main function to iterate through all scenes"""
    print("\n" + "="*70)
    print("nuScenes 3D Viewer - Professional Edition")
    print("="*70)
    print(f"Dataset: v1.0-mini")
    print(f"Total scenes: {len(nusc.scene)}")
    print("="*70 + "\n")

    # Iterate through all scenes
    for scene_idx in range(len(nusc.scene)):
        visualize_scene(scene_idx)

        # Ask to continue (except for last scene)
        if scene_idx < len(nusc.scene) - 1:
            response = input("\nPress ENTER for next scene, or 'q' to quit: ").lower().strip()
            if response == 'q':
                print("\nExiting viewer.")
                break

    print("\n" + "="*70)
    print("Visualization complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
