import open3d as o3d
from pycpd import DeformableRegistration
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import partial


def visualize_iteration(iteration, error, X, Y, ax):
    """Callback function to visualize registration iterations using matplotlib.

    Args:
        iteration: Current iteration number
        error: Current registration error
        X: Target point cloud (Nx3)
        Y: Source point cloud at current iteration (Nx3)
        ax: Matplotlib 3D axis object
    """
    plt.cla()
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='red', label='Target', alpha=0.6)
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color='blue', label='Source', alpha=0.6)
    ax.text2D(0.87, 0.92, 'Iteration: {:d}'.format(iteration),
              horizontalalignment='center', verticalalignment='center',
              transform=ax.transAxes, fontsize='x-large')
    ax.text2D(0.87, 0.87, 'Error: {:.6f}'.format(error),
              horizontalalignment='center', verticalalignment='center',
              transform=ax.transAxes, fontsize='large')
    ax.legend(loc='upper left', fontsize='x-large')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.draw()
    plt.pause(0.001)


def create_point_cloud(points, color):
    """Create an Open3D point cloud with a specified color.

    Args:
        points: Nx3 numpy array of point coordinates
        color: RGB color as [r, g, b] with values in [0, 1]

    Returns:
        o3d.geometry.PointCloud object
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    return pcd


def visualize_registration(X, Y, Y_registered, title_before="Before Registration", title_after="After Registration"):
    """Visualize point clouds before and after registration using Open3D.

    Args:
        X: Target point cloud (Nx3)
        Y: Source point cloud before registration (Nx3)
        Y_registered: Source point cloud after registration (Nx3)
        title_before: Window title for before visualization
        title_after: Window title for after visualization
    """
    # Create point clouds for before registration
    target_before = create_point_cloud(X, [1, 0, 0])  # Red for target
    source_before = create_point_cloud(Y, [0, 0, 1])  # Blue for source

    # Create point clouds for after registration
    target_after = create_point_cloud(X, [1, 0, 0])   # Red for target
    source_after = create_point_cloud(Y_registered, [0, 1, 0])  # Green for registered source

    # Visualize before registration
    print("Showing point clouds BEFORE registration...")
    print("  Red: Target point cloud")
    print("  Blue: Source point cloud")
    o3d.visualization.draw_geometries(
        [target_before, source_before],
        window_name=title_before,
        width=800,
        height=600
    )

    # Visualize after registration
    print("\nShowing point clouds AFTER registration...")
    print("  Red: Target point cloud")
    print("  Green: Registered source point cloud")
    o3d.visualization.draw_geometries(
        [target_after, source_after],
        window_name=title_after,
        width=800,
        height=600
    )


def main():
    # Load fish data
    fish_target = np.loadtxt('data/fish_target.txt')
    X1 = np.zeros((fish_target.shape[0], fish_target.shape[1] + 1))
    X1[:, :-1] = fish_target
    X2 = np.ones((fish_target.shape[0], fish_target.shape[1] + 1))
    X2[:, :-1] = fish_target
    X = np.vstack((X1, X2))

    fish_source = np.loadtxt('data/fish_source.txt')
    Y1 = np.zeros((fish_source.shape[0], fish_source.shape[1] + 1))
    Y1[:, :-1] = fish_source
    Y2 = np.ones((fish_source.shape[0], fish_source.shape[1] + 1))
    Y2[:, :-1] = fish_source
    Y = np.vstack((Y1, Y2))

    # Step 1: Visualize BEFORE registration with Open3D
    print("=" * 60)
    print("STEP 1: Visualizing BEFORE registration (Open3D)")
    print("=" * 60)
    target_before = create_point_cloud(X, [1, 0, 0])  # Red for target
    source_before = create_point_cloud(Y, [0, 0, 1])  # Blue for source
    print("  Red: Target point cloud")
    print("  Blue: Source point cloud")
    print("\nClose the window to continue...")
    o3d.visualization.draw_geometries(
        [target_before, source_before],
        window_name="Before Registration (Open3D)",
        width=800,
        height=600
    )

    # Step 2: Perform registration WITH matplotlib visualization
    print("\n" + "=" * 60)
    print("STEP 2: Running registration with iterations (Matplotlib 3D)")
    print("=" * 60)
    print("Performing deformable registration...")
    print("Watch the matplotlib window for iteration updates...")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    callback = partial(visualize_iteration, ax=ax)

    reg = DeformableRegistration(**{'X': X, 'Y': Y})
    reg.register(callback)

    print("\nRegistration complete!")
    print(f"Target points: {X.shape[0]}")
    print(f"Source points: {Y.shape[0]}")

    plt.close()  # Close matplotlib window

    # Step 3: Get transformed points and visualize AFTER with Open3D
    Y_registered = reg.transform_point_cloud(Y)

    print("\n" + "=" * 60)
    print("STEP 3: Visualizing AFTER registration (Open3D)")
    print("=" * 60)
    target_after = create_point_cloud(X, [1, 0, 0])   # Red for target
    source_after = create_point_cloud(Y_registered, [0, 1, 0])  # Green for registered
    print("  Red: Target point cloud")
    print("  Green: Registered source point cloud")
    print("\nClose the window to exit...")
    o3d.visualization.draw_geometries(
        [target_after, source_after],
        window_name="After Registration (Open3D)",
        width=800,
        height=600
    )

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
