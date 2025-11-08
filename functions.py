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
