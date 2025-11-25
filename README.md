# PardesLine – 3D Computer Vision

PardesLine is a Python repository demonstrating **essential 3D mesh and point cloud workflows** using [Open3D](http://www.open3d.org/). Perfect for learning and experimenting with 3D data in **computer vision, robotics, VR/AR, and machine learning**.

---

## Modules Overview

### Core Scripts

- **`01_pointcloud_processing.py`** – Load, visualize, and manipulate point clouds. Convert meshes to point clouds, sample points using Uniform and Poisson-disk sampling, visualize in 3D, and save outputs.

- **`02_mesh_processing.py`** – Comprehensive mesh operations including loading, visualization, coloring, vertex normal computation, mesh subdivision, and simplification.

- **`03_mesh_pcd_to_volume.py`** – Convert 3D meshes and point clouds to voxel grids (volumetric representations). Includes AABB/OBB computation, voxelization, point cloud to volume conversion, and binary volume visualization with erosion operations.

- **`04_SDF.py`** – Advanced pipeline for Signed Distance Field (SDF) computation and mesh reconstruction. Computes SDFs from meshes, creates shell representations, applies morphological operations (erosion), and reconstructs meshes from SDF volumes. Uses a configurable pipeline with logging.

### Helper Modules

- **`functions.py`** – Utility functions for point cloud operations (NumPy array to Open3D PointCloud conversion) and binary volume visualization.

- **`functions_for_sdf.py`** – Complete SDF pipeline utilities with:
  - Mesh loading and normalization
  - Voxel grid creation
  - SDF computation (Open3D backend)
  - Occupancy grid generation
  - Binary morphological operations (erosion)
  - Point cloud generation from SDF/occupancy
  - Mesh reconstruction
  - Visualization utilities
  - `PipelineConfig` dataclass for configuration management

---

## Installation

```bash
git clone https://github.com/1904jonathan/PardesLine.git
cd PardesLine
pip install -r requirements.txt
```

### Requirements

- `open3d` – 3D geometry processing
- `numpy` – Numerical computing
- `scipy` – Scientific computing (morphological operations)
- `scikit-image` – Image processing utilities
- `matplotlib` – Plotting and visualization
- `loguru` – Enhanced logging

---

## Usage

Run any module to explore its workflow:

```bash
python 01_pointcloud_processing.py
python 02_mesh_processing.py
python 03_mesh_pcd_to_volume.py
python 04_SDF.py
```

**Output Directory:** Processed meshes and point clouds are saved in `C:\PardesLineData` by default.

---

## Workflow Overview

```
Input: Mesh (e.g., Stanford Bunny)
  ↓
[01_pointcloud_processing] → Point Cloud operations
  ↓
[02_mesh_processing] → Mesh operations (color, subdivide, simplify)
  ↓
[03_mesh_pcd_to_volume] → Voxel Grid / Volume conversion
  ↓
[04_SDF] → SDF Computation & Mesh Reconstruction
  ↓
Output: Processed models + visualizations
```

---

## Applications

- 3D scanning & reconstruction
- Robotics perception & SLAM
- Medical imaging & surgical navigation
- VR/AR asset preparation
- Machine learning on point clouds (PointNet, etc.)
- Signed distance fields for neural implicit representations (NeRF, DeepSDF)

---

## Key Features

✓ Full pipeline from mesh to volumetric representations  
✓ SDF computation and mesh reconstruction  
✓ Voxelization with configurable resolution  
✓ Binary morphological operations (erosion)  
✓ Interactive 3D visualization using Open3D  
✓ Comprehensive logging with Loguru  
✓ Modular, extensible design  


## Author

[1904jonathan](https://github.com/1904jonathan)
