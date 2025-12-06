# PardesLine – 3D Computer Vision

PardesLine is a Python repository demonstrating **essential 3D mesh and point cloud workflows** using [Open3D](http://www.open3d.org/). Perfect for learning and experimenting with 3D data in **computer vision, robotics, VR/AR, and machine learning**.

---

## Modules Overview

### Core Scripts

- **`01_pointcloud_processing.py`** – Load, visualize, and manipulate point clouds. Convert meshes to point clouds, sample points using Uniform and Poisson-disk sampling, visualize in 3D, and save outputs.

- **`02_mesh_processing.py`** – Comprehensive mesh operations including loading, visualization, coloring, vertex normal computation, mesh subdivision, and simplification.

- **`03_mesh_pcd_to_volume.py`** – Convert 3D meshes and point clouds to voxel grids (volumetric representations). Includes AABB/OBB computation, voxelization, point cloud to volume conversion, and binary volume visualization with erosion operations.

- **`04_SDF.py`** – Advanced pipeline for Signed Distance Field (SDF) computation and mesh reconstruction. Computes SDFs from meshes, creates shell representations, applies morphological operations (erosion), and reconstructs meshes from SDF volumes. Uses a configurable pipeline with logging.

- **`05_surface_reconstruction.py`** – Surface reconstruction pipeline demonstrating multiple algorithms: Convex Hull computation, Alpha Shapes with varying alpha values, Ball Pivoting Algorithm (BPA), and Poisson Surface Reconstruction with density-based filtering. Includes normal estimation and orientation techniques.

- **`06_point_cloud_registration.py`** – Complete point cloud registration pipeline using RANSAC and ICP algorithms. Demonstrates mesh-to-point cloud conversion with uniform sampling, geometric transformations (rotation + translation), FPFH feature extraction, global registration with RANSAC, and local refinement with ICP. Includes comprehensive error analysis and transformation matrix recovery with yellow/cyan visualization.

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

- **`functions_for_surfaceR.py`** – Surface reconstruction utilities with:
  - Point cloud loading (bunny, eagle datasets)
  - Convex Hull computation
  - Alpha Shape reconstruction with TetraMesh optimization
  - Ball Pivoting Algorithm (BPA) reconstruction
  - Poisson Surface Reconstruction with density visualization and filtering
  - Normal estimation and consistent orientation

- **`functions_for_registration.py`** – Point cloud registration utilities with:
  - Bunny mesh loading from Open3D datasets
  - Mesh-to-point cloud uniform sampling
  - Geometric transformations (rotation + translation)
  - FPFH feature computation for registration
  - RANSAC global registration with feature matching
  - ICP local refinement (point-to-point)
  - Transformation error analysis (rotation, translation)
  - Visualization utilities with custom point sizes
  - `RegistrationConfig` dataclass for parameter management

### Mathematical Demonstrations

- **`mathDemo/convex_hull_2d_demo.py`** – Educational 2D convex hull computation using Graham Scan algorithm. Demonstrates:
  - Anchor point selection (lowest y-coordinate)
  - Polar angle sorting with `atan2`
  - Cross product for turn detection (CCW/CW)
  - Stack-based hull construction
  - Step-by-step visualization with matplotlib
  - Mathematical formulas and explanations

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
python 05_surface_reconstruction.py
python 06_point_cloud_registration.py
python mathDemo/convex_hull_2d_demo.py
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
[05_surface_reconstruction] → Surface Reconstruction (Convex Hull, Alpha Shapes, BPA, Poisson)
  ↓
[06_point_cloud_registration] → RANSAC + ICP Registration & Alignment
  ↓
Output: Processed models + visualizations

Math Demonstrations:
[mathDemo/convex_hull_2d_demo] → 2D Convex Hull (Graham Scan)
```

---

## Applications

- 3D scanning & reconstruction
- Robotics perception & SLAM (point cloud registration for localization)
- Medical imaging & surgical navigation (registration of pre-op/intra-op scans)
- VR/AR asset preparation and alignment
- Machine learning on point clouds (PointNet, etc.)
- Signed distance fields for neural implicit representations (NeRF, DeepSDF)
- Surface reconstruction from point cloud scans
- Multi-view 3D reconstruction (camera pose estimation with RANSAC)
- Object tracking and 6DOF pose estimation
- Educational demonstrations of computational geometry algorithms

---

## Key Features

✓ Full pipeline from mesh to volumetric representations
✓ SDF computation and mesh reconstruction
✓ Surface reconstruction (Convex Hull, Alpha Shapes, BPA, Poisson)
✓ Point cloud registration with RANSAC and ICP algorithms
✓ FPFH feature-based matching for robust alignment
✓ Geometric transformations (rotation, translation, scaling)
✓ Voxelization with configurable resolution
✓ Binary morphological operations (erosion)
✓ Normal estimation and orientation
✓ Interactive 3D visualization using Open3D
✓ Comprehensive logging with Loguru
✓ Modular, extensible design with dataclass configurations
✓ Educational mathematical demonstrations (Graham Scan, cross products, polar angles)


## Author

[1904jonathan](https://github.com/1904jonathan)
