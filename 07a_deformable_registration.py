import open3d as o3d
from pycpd import DeformableRegistration
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import partial
from functions_for_deformation import (
    visualize_iteration,
    create_point_cloud,
    visualize_registration,
)


def main():
    # Load fish data
    fish_target = np.loadtxt('data/fish_target.txt')
    fish_source = np.loadtxt('data/fish_source.txt')

    # Step 0: Visualize original 2D fish data with Matplotlib
    print("=" * 60)
    print("STEP 0: Visualizing original 2D fish data (Matplotlib)")
    print("=" * 60)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot source fish
    ax1.scatter(fish_source[:, 0], fish_source[:, 1], color='blue', s=20, alpha=0.6)
    ax1.set_title('Source Fish (2D)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # Plot target fish
    ax2.scatter(fish_target[:, 0], fish_target[:, 1], color='red', s=20, alpha=0.6)
    ax2.set_title('Target Fish (2D)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    print(f"Target fish points: {fish_target.shape[0]}")
    print(f"Source fish points: {fish_source.shape[0]}")
    print("\nClose the matplotlib window to continue...")
    plt.show()




    X1 = np.zeros((fish_target.shape[0], fish_target.shape[1] + 1))
    X1[:, :-1] = fish_target
    X2 = np.ones((fish_target.shape[0], fish_target.shape[1] + 1))
    X2[:, :-1] = fish_target
    X = np.vstack((X1, X2))

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
