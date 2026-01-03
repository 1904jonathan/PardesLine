#!/usr/bin/env python3
"""
nuScenes Temporal Viewer - With 2D/3D Alignment Verification
Projects 3D bounding boxes onto camera image to verify alignment
Shows both 3D LiDAR and 2D camera projections side by side
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points
from PIL import Image
import time

# Initialize nuScenes dataset
nusc = NuScenes(version='v1.0-mini', dataroot=".", verbose=False)


def load_camera_image(sample, camera_channel='CAM_FRONT'):
    """Load camera image for a given sample"""
    cam_token = sample['data'][camera_channel]
    cam_data = nusc.get('sample_data', cam_token)
    img_path = nusc.dataroot + '/' + cam_data['filename']
    img = Image.open(img_path)
    return np.array(img)


def render_boxes_on_image(ax, sample, camera_channel='CAM_FRONT'):
    """
    Render 2D bounding boxes on camera image using nuScenes utilities

    Args:
        ax: matplotlib axis
        sample: nuScenes sample
        camera_channel: Camera channel name
    """
    # Load image
    cam_img = load_camera_image(sample, camera_channel)
    img_height, img_width = cam_img.shape[:2]

    ax.imshow(cam_img)

    # Fix the axis limits to prevent resizing
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)  # Inverted Y-axis for image coordinates

    # Get camera data
    cam_token = sample['data'][camera_channel]
    cam_data = nusc.get('sample_data', cam_token)

    # Get calibration
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

    # Get all boxes in the sample
    _, boxes, camera_intrinsic = nusc.get_sample_data(cam_token)

    car_count = 0

    # Draw each box
    for box in boxes:
        if 'car' not in box.name.lower():
            continue

        car_count += 1

        # Get box corners in 3D
        corners_3d = box.corners()

        # Project to 2D image plane
        corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True)[:2, :]

        # Check if box is in front of camera
        if not np.all(corners_3d[2, :] > 0):
            continue

        # Define the edges to draw
        def draw_rect(selected_corners, color, linewidth=2):
            """Draw a rectangle defined by corners"""
            prev = selected_corners[-1]
            for corner in selected_corners:
                ax.plot([prev[0], corner[0]], [prev[1], corner[1]],
                       color=color, linewidth=linewidth, alpha=0.8)
                prev = corner

        # Draw front face (corners 0,1,2,3)
        corners_front = corners_2d[:, [0, 1, 2, 3]].T
        draw_rect(corners_front, 'lime', linewidth=2)

        # Draw back face (corners 4,5,6,7)
        corners_back = corners_2d[:, [4, 5, 6, 7]].T
        draw_rect(corners_back, 'cyan', linewidth=1)

        # Draw connecting lines
        for i in range(4):
            ax.plot([corners_2d[0, i], corners_2d[0, i+4]],
                   [corners_2d[1, i], corners_2d[1, i+4]],
                   'y-', linewidth=1, alpha=0.6)

    return car_count


def create_bbox_vertices(center, extent, rotation_matrix):
    """Create vertices for a 3D bounding box"""
    w, l, h = extent

    # For cars: length along X, width along Y, height along Z
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]

    corners = np.vstack([x_corners, y_corners, z_corners])
    corners = rotation_matrix @ corners
    corners = corners.T + center

    return corners


def draw_bbox_3d(ax, center, extent, rotation_matrix, color='green', alpha=0.25):
    """Draw a 3D bounding box on matplotlib axis"""
    vertices = create_bbox_vertices(center, extent, rotation_matrix)

    # Define edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Top face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Bottom face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]

    # Draw edges
    for edge in edges:
        points = vertices[edge]
        ax.plot3D(*points.T, color=color, linewidth=2, alpha=alpha+0.5)

    # Define faces
    faces = [
        [0, 1, 2, 3],  # Top
        [4, 5, 6, 7],  # Bottom
        [0, 1, 5, 4],  # Front
        [2, 3, 7, 6],  # Back
        [0, 3, 7, 4],  # Left
        [1, 2, 6, 5]   # Right
    ]

    # Draw faces
    face_collection = [[vertices[j] for j in face] for face in faces]
    poly = Poly3DCollection(face_collection, alpha=alpha, facecolor=color, edgecolor=color)
    ax.add_collection3d(poly)


def visualize_scene_temporal(scene_idx=0, fps=2, max_points=10000):
    """
    Visualize a scene with 3D LiDAR and 2D camera with projected boxes

    Args:
        scene_idx: Index of the scene to visualize
        fps: Frames per second for animation
        max_points: Maximum number of LiDAR points to display
    """
    scene = nusc.scene[scene_idx]
    scene_name = scene['name']

    print(f"\n{'='*70}")
    print(f"Scene {scene_idx + 1}/{len(nusc.scene)}: {scene_name}")
    print(f"{'='*70}")
    print(f"  Total samples: {scene['nbr_samples']}")
    print(f"  Description: {scene['description']}")
    print(f"\n  Visualization:")
    print(f"    • LEFT: 3D LiDAR point cloud with bounding boxes")
    print(f"    • RIGHT: Camera image with projected 2D boxes")
    print(f"    • Colors: Lime=front face, Cyan=back face, Yellow=connecting lines")
    print(f"{'='*70}\n")

    # Get all samples
    sample_token = scene['first_sample_token']
    samples = []
    while sample_token:
        sample = nusc.get('sample', sample_token)
        samples.append(sample)
        sample_token = sample['next']

    # Create figure with 3D and 2D subplots
    fig = plt.figure(figsize=(18, 8))
    ax_3d = fig.add_subplot(121, projection='3d')
    ax_cam = fig.add_subplot(122)

    fig.canvas.manager.set_window_title(f"nuScenes Aligned Viewer - {scene_name}")
    plt.ion()
    plt.show()

    frame_delay = 1.0 / fps

    try:
        for frame_idx, sample in enumerate(samples):
            start_time = time.time()

            print(f"\r  Frame {frame_idx + 1}/{len(samples)}", end='', flush=True)

            # ========== 3D LIDAR VIEW ==========
            # Load LiDAR point cloud
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = nusc.get('sample_data', lidar_token)
            pc = LidarPointCloud.from_file(nusc.dataroot + '/' + lidar_data['filename'])
            points = pc.points[:3, :].T

            # Downsample
            if len(points) > max_points:
                indices = np.random.choice(len(points), max_points, replace=False)
                points_display = points[indices]
            else:
                points_display = points

            # Clear and plot 3D
            ax_3d.clear()

            # Color by height
            z_norm = (points_display[:, 2] - points_display[:, 2].min()) / \
                     (points_display[:, 2].max() - points_display[:, 2].min() + 1e-6)
            colors = np.zeros((len(points_display), 3))
            colors[:, 0] = z_norm * 0.3
            colors[:, 1] = z_norm * 0.8
            colors[:, 2] = (1 - z_norm) * 0.6

            ax_3d.scatter(points_display[:, 0], points_display[:, 1], points_display[:, 2],
                         c=colors, s=1, alpha=0.6)

            # Draw 3D bounding boxes
            _, boxes_nusc, _ = nusc.get_sample_data(lidar_token)
            car_count_3d = 0

            for box_nusc in boxes_nusc:
                if 'car' in box_nusc.name.lower():
                    rotation_correction = np.array([
                        [0, -1, 0],
                        [1, 0, 0],
                        [0, 0, 1]
                    ])
                    corrected_rotation = box_nusc.rotation_matrix @ rotation_correction
                    draw_bbox_3d(ax_3d, box_nusc.center, box_nusc.wlh,
                               corrected_rotation, color='lime', alpha=0.2)
                    car_count_3d += 1

            # Configure 3D plot
            ax_3d.set_xlabel('X (m)', fontsize=10)
            ax_3d.set_ylabel('Y (m)', fontsize=10)
            ax_3d.set_zlabel('Z (m)', fontsize=10)
            ax_3d.set_title(f'LiDAR 3D - Frame {frame_idx + 1}/{len(samples)}',
                           fontsize=12, fontweight='bold', color='navy')
            ax_3d.set_xlim([-50, 50])
            ax_3d.set_ylim([-50, 50])
            ax_3d.set_zlim([-5, 5])
            ax_3d.view_init(elev=20, azim=-60)
            ax_3d.set_facecolor((0.02, 0.02, 0.02))
            ax_3d.grid(True, alpha=0.3)

            # ========== 2D CAMERA VIEW ==========
            ax_cam.clear()
            car_count_2d = render_boxes_on_image(ax_cam, sample, 'CAM_FRONT')
            ax_cam.axis('off')
            ax_cam.set_aspect('equal')  # Maintain aspect ratio

            # Add info
            timestamp_ms = sample['timestamp'] / 1000
            info_text = (
                f"Front Camera with 3D→2D Projection\n"
                f"Frame: {frame_idx + 1}/{len(samples)} | "
                f"Cars (3D): {car_count_3d} | Cars (2D): {car_count_2d}\n"
                f"LiDAR Points: {len(points):,} | Time: {timestamp_ms:.1f}ms"
            )
            ax_cam.set_title(info_text, fontsize=11, color='darkgreen',
                           fontweight='bold', pad=10)

            # Update display
            plt.draw()
            plt.pause(0.01)

            if not plt.fignum_exists(fig.number):
                print("\n  Window closed")
                break

            # Frame rate control
            elapsed = time.time() - start_time
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)

    except KeyboardInterrupt:
        print("\n  Interrupted")
    finally:
        print()
        plt.close(fig)
        print(f"\n  Scene {scene_name} completed!")


def main():
    """Main function"""
    print("\n" + "="*70)
    print("nuScenes Aligned Viewer - 2D/3D Synchronization Check")
    print("="*70)
    print(f"Dataset: v1.0-mini")
    print(f"Total scenes: {len(nusc.scene)}")
    print(f"")
    print(f"This viewer projects 3D boxes onto the 2D camera image")
    print(f"to verify that LiDAR and camera are properly aligned.")
    print("="*70 + "\n")

    # Iterate through scenes
    for scene_idx in range(len(nusc.scene)):
        try:
            visualize_scene_temporal(scene_idx, fps=2, max_points=10000)

            if scene_idx < len(nusc.scene) - 1:
                response = input("\nPress ENTER for next scene, or 'q' to quit: ").lower().strip()
                if response == 'q':
                    print("\nExiting viewer.")
                    break
        except KeyboardInterrupt:
            print("\n\nExiting viewer.")
            break

    print("\n" + "="*70)
    print("Visualization complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
