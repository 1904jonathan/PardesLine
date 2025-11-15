import open3d as o3d
import numpy as np

def points_to_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Convert an (N, 3) NumPy array of XYZ coordinates into an Open3D PointCloud.

    Parameters
    ----------
    points : np.ndarray
        Array of 3D points with shape (N, 3)

    Returns
    -------
    o3d.geometry.PointCloud
        Open3D point cloud object
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def visualize_binary_volume(volume, voxel_size=0.05, color=[1,0,1], title="Binary Volume"):
    """
    Build and visualize a voxel grid in Open3D from a numpy binary volume.

    Parameters
    ----------
    volume : np.ndarray
        3D numpy array (binary: 0 background, 1 foreground), shape (Z,Y,X)
    voxel_size : float
        Size of each voxel cube in visualization
    color : list of float
        RGB color of the voxels
    title : str
        Window title for Open3D visualization
    """
    assert volume.ndim == 3, "Volume must be 3D"

    # Extract coordinates of occupied voxels
    zyx = np.argwhere(volume > 0)
    xyz = zyx[:, [2,1,0]].astype(float)  # convert to x,y,z

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.paint_uniform_color(color)

    # Visualize point cloud
    o3d.visualization.draw_geometries([pcd], window_name=f"{title} - PointCloud")

    return 
