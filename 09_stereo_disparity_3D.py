"""
Stereo 3D Reconstruction Pipeline

Professional-grade stereo vision module for 3D reconstruction from disparity maps.
This module provides a complete pipeline for:
- Camera calibration using chessboard patterns
- Stereo image rectification and undistortion
- Disparity map computation using block matching
- 3D point cloud reconstruction
- Visualization and export to PLY format

Author: Computer Vision Expert
Date: 2025-12-24
"""

from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass
import logging
import sys

import numpy as np
import cv2
import open3d as o3d
from loguru import logger

@dataclass
class StereoConfig:
    """Configuration parameters for stereo 3D reconstruction pipeline."""

    # Calibration parameters
    calibration_pattern: str = 'stereoData\\cal\\cal*.png'
    pattern_size: Tuple[int, int] = (9, 6)
    square_size_mm: float = 1.0  # Real-world chessboard square size

    # Stereo image paths
    left_image_path: str = 'stereoData\\left1.png'
    right_image_path: str = 'stereoData\\right1.png'

    # Stereo matching parameters
    num_disparities: int = 64  # Must be divisible by 16
    block_size: int = 15  # Must be odd number >= 3

    # Disparity filtering
    min_disparity: int = 20
    max_disparity: int = 220

    # Stereo baseline (mm) - critical for accurate depth
    baseline_mm: float = 60.0

    # Output configuration
    output_dir: Path = Path('output')
    output_filename: str = 'reconstruction_3d.ply'

    # Visualization flags
    show_calibration_images: bool = False
    show_undistorted_images: bool = False
    show_disparity: bool = True
    show_point_cloud: bool = True


class CameraCalibrator:
    """Handles camera calibration using chessboard patterns."""

    def __init__(self, config: StereoConfig):
        self.config = config
        self.K: Optional[np.ndarray] = None
        self.dist: Optional[np.ndarray] = None
        self.K_optimal: Optional[np.ndarray] = None

    def calibrate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform camera calibration from chessboard images.

        Returns:
            Tuple of (K, dist, K_optimal):
                - K: Camera intrinsic matrix (3x3)
                - dist: Distortion coefficients (1x5 or more)
                - K_optimal: Optimal camera matrix for undistortion

        Raises:
            FileNotFoundError: If no calibration images found
            ValueError: If insufficient valid patterns detected
        """
        logger.info("=" * 70)
        logger.info("CAMERA CALIBRATION")
        logger.info("=" * 70)
        logger.info(f"Calibration pattern: {self.config.calibration_pattern}")
        logger.info(f"Chessboard size: {self.config.pattern_size}")
        logger.info(f"Square size: {self.config.square_size_mm} mm")

        # Find calibration images
        import glob
        images = sorted(glob.glob(self.config.calibration_pattern))

        if not images:
            raise FileNotFoundError(
                f"No calibration images found at: {self.config.calibration_pattern}"
            )

        logger.info(f"Found {len(images)} calibration images")

        # Prepare object points
        objp = np.zeros(
            (self.config.pattern_size[0] * self.config.pattern_size[1], 3),
            np.float32
        )
        objp[:, :2] = np.mgrid[
            0:self.config.pattern_size[0],
            0:self.config.pattern_size[1]
        ].T.reshape(-1, 2)
        objp *= self.config.square_size_mm

        # Storage for detected points
        objpoints = []
        imgpoints = []

        # Corner refinement criteria
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        # Detect chessboard corners
        logger.info("Detecting chessboard corners...")
        valid_count = 0
        image_size = None

        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Failed to load: {Path(img_path).name}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if image_size is None:
                image_size = (gray.shape[1], gray.shape[0])

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(
                gray,
                self.config.pattern_size,
                None
            )

            if ret:
                # Refine corner locations
                corners_refined = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    criteria
                )

                objpoints.append(objp)
                imgpoints.append(corners_refined)
                valid_count += 1
                logger.info(f"  ✓ Detected: {Path(img_path).name}")

                # Optionally visualize
                if self.config.show_calibration_images:
                    img_corners = cv2.drawChessboardCorners(
                        img.copy(),
                        self.config.pattern_size,
                        corners_refined,
                        ret
                    )
                    cv2.imshow('Calibration', img_corners)
                    cv2.waitKey(100)
            else:
                logger.warning(f"  ✗ Not found: {Path(img_path).name}")

        if self.config.show_calibration_images:
            cv2.destroyAllWindows()

        # Validate sufficient detections
        if valid_count < 3:
            raise ValueError(
                f"Insufficient valid patterns: {valid_count}/3 minimum required"
            )

        logger.info(f"Successfully detected {valid_count}/{len(images)} patterns")

        # Perform calibration
        logger.info("Computing calibration parameters...")
        ret, K, dist, _, _ = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            image_size,
            None,
            None
        )

        # Compute optimal camera matrix
        K_optimal, _ = cv2.getOptimalNewCameraMatrix(
            K,
            dist,
            image_size,
            1,
            image_size
        )

        # Store results
        self.K = K
        self.dist = dist
        self.K_optimal = K_optimal

        # Log calibration results
        logger.info("Calibration completed successfully!")
        logger.info(f"Reprojection error: {ret:.4f} pixels")
        logger.info(f"Focal lengths: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")
        logger.info(f"Principal point: cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
        logger.info(f"Distortion coefficients: {dist.ravel()}")

        return K, dist, K_optimal


class StereoReconstructor:
    """Handles stereo matching and 3D reconstruction."""

    def __init__(self, config: StereoConfig, K: np.ndarray, dist: np.ndarray, K_optimal: np.ndarray):
        self.config = config
        self.K = K
        self.dist = dist
        self.K_optimal = K_optimal

    def load_and_undistort(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load stereo image pair and apply lens distortion correction.

        Returns:
            Tuple of (imgL_undist, imgR_undist, grayL, grayR)

        Raises:
            FileNotFoundError: If stereo images cannot be loaded
            ValueError: If image dimensions don't match
        """
        logger.info("=" * 70)
        logger.info("LOADING AND UNDISTORTING STEREO IMAGES")
        logger.info("=" * 70)
        logger.info(f"Left image: {self.config.left_image_path}")
        logger.info(f"Right image: {self.config.right_image_path}")

        # Load images
        imgL = cv2.imread(self.config.left_image_path)
        imgR = cv2.imread(self.config.right_image_path)

        if imgL is None or imgR is None:
            raise FileNotFoundError(
                "Failed to load stereo images. Check paths:\n"
                f"  Left: {self.config.left_image_path}\n"
                f"  Right: {self.config.right_image_path}"
            )

        # Validate dimensions
        if imgL.shape != imgR.shape:
            raise ValueError(
                f"Image dimension mismatch: "
                f"Left {imgL.shape} vs Right {imgR.shape}"
            )

        logger.info(f"Image dimensions: {imgL.shape[1]}x{imgL.shape[0]}")

        # Undistort images
        logger.info("Applying lens distortion correction...")
        imgL_undist = cv2.undistort(imgL, self.K, self.dist, None, self.K_optimal)
        imgR_undist = cv2.undistort(imgR, self.K, self.dist, None, self.K_optimal)

        # Convert to grayscale for stereo matching
        grayL = cv2.cvtColor(imgL_undist, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR_undist, cv2.COLOR_BGR2GRAY)

        logger.info("Undistortion complete")

        # Optionally display
        if self.config.show_undistorted_images:
            cv2.imshow('Left Undistorted', imgL_undist)
            cv2.imshow('Right Undistorted', imgR_undist)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return imgL_undist, imgR_undist, grayL, grayR

    def compute_disparity(self, grayL: np.ndarray, grayR: np.ndarray) -> np.ndarray:
        """
        Compute stereo disparity map using block matching.

        Args:
            grayL: Left grayscale image
            grayR: Right grayscale image

        Returns:
            Disparity map (uint8, normalized to 0-255 with filtering)
        """
        logger.info("=" * 70)
        logger.info("COMPUTING DISPARITY MAP")
        logger.info("=" * 70)
        logger.info(f"Algorithm: StereoBM (Block Matching)")
        logger.info(f"Number of disparities: {self.config.num_disparities}")
        logger.info(f"Block size: {self.config.block_size}")
        logger.info(f"Disparity range: [{self.config.min_disparity}, {self.config.max_disparity}]")

        # Create stereo matcher
        stereo = cv2.StereoBM_create(
            numDisparities=self.config.num_disparities,
            blockSize=self.config.block_size
        )

        # Compute raw disparity
        disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

        # Normalize to 0-255 range
        disp_min = disp[disp > 0].min() if np.any(disp > 0) else 0
        disp_max = disp.max()

        if disp_max > disp_min:
            disp_normalized = 255 * (disp - disp_min) / (disp_max - disp_min)
        else:
            disp_normalized = np.zeros_like(disp)

        disp8 = np.uint8(disp_normalized)

        # Apply threshold filtering
        disp8[(disp8 < self.config.min_disparity) | (disp8 > self.config.max_disparity)] = 0

        # Calculate statistics
        valid_pixels = np.sum(disp8 > 0)
        total_pixels = disp8.shape[0] * disp8.shape[1]
        valid_percentage = 100 * valid_pixels / total_pixels

        logger.info(f"Disparity range: [{disp_min:.2f}, {disp_max:.2f}]")
        logger.info(f"Valid pixels: {valid_pixels}/{total_pixels} ({valid_percentage:.1f}%)")

        return disp8

    def reconstruct_3d(self, disp8: np.ndarray, imgL: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Reconstruct 3D point cloud from disparity map.

        Args:
            disp8: Normalized disparity map (uint8, 0-255)
            imgL: Left color image for texture mapping

        Returns:
            Open3D point cloud with RGB colors

        Raises:
            ValueError: If no valid 3D points generated
        """
        logger.info("=" * 70)
        logger.info("3D POINT CLOUD RECONSTRUCTION")
        logger.info("=" * 70)

        # Extract intrinsic parameters
        fx = self.K_optimal[0, 0]
        fy = self.K_optimal[1, 1]
        cx = self.K_optimal[0, 2]
        cy = self.K_optimal[1, 2]

        logger.info(f"Focal lengths: fx={fx:.2f}, fy={fy:.2f} pixels")
        logger.info(f"Principal point: cx={cx:.2f}, cy={cy:.2f} pixels")
        logger.info(f"Baseline: {self.config.baseline_mm:.2f} mm")

        # Create mask for valid disparities
        mask = disp8 > 0
        valid_count = np.sum(mask)

        if valid_count == 0:
            raise ValueError("No valid disparities found. Cannot reconstruct 3D.")

        # Compute depth using disparity-depth relationship
        # Depth (Z) = (focal_length * baseline) / disparity
        # Using normalized disparity, we apply a scaling factor
        depth = np.zeros_like(disp8, dtype=np.float32)
        depth[mask] = (fx * self.config.baseline_mm) / (disp8[mask].astype(np.float32) + 1e-6)

        # Generate 3D coordinates
        h, w = depth.shape
        v_coords, u_coords = np.indices((h, w))

        # Convert pixel coordinates to 3D
        X = (u_coords - cx) * depth / fx
        Y = (v_coords - cy) * depth / fy
        Z = depth

        # Extract valid 3D points
        points_3d = np.stack([X[mask], Y[mask], Z[mask]], axis=-1)

        # Extract RGB colors (convert BGR to RGB)
        colors_bgr = imgL[mask].astype(np.float32) / 255.0
        colors_rgb = colors_bgr[:, ::-1]

        logger.info(f"Generated {len(points_3d)} 3D points")

        # Compute bounding box
        bbox_min = points_3d.min(axis=0)
        bbox_max = points_3d.max(axis=0)
        bbox_size = bbox_max - bbox_min

        logger.info(f"Bounding box (mm):")
        logger.info(f"  X: [{bbox_min[0]:.2f}, {bbox_max[0]:.2f}] (size: {bbox_size[0]:.2f})")
        logger.info(f"  Y: [{bbox_min[1]:.2f}, {bbox_max[1]:.2f}] (size: {bbox_size[1]:.2f})")
        logger.info(f"  Z: [{bbox_min[2]:.2f}, {bbox_max[2]:.2f}] (size: {bbox_size[2]:.2f})")

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

        logger.info("Point cloud reconstruction complete")

        return pcd


class Visualizer:
    """Handles visualization of results."""

    @staticmethod
    def show_disparity(disp8: np.ndarray, imgL: np.ndarray):
        """Display disparity map and masked left image."""
        logger.info("Displaying disparity results...")

        # Create color-mapped disparity
        disp_color = cv2.applyColorMap(disp8, cv2.COLORMAP_JET)

        # Create masked left image
        imgL_masked = imgL.copy()
        imgL_masked[disp8 == 0] = 0

        # Display
        cv2.imshow('Disparity Map (Grayscale)', disp8)
        cv2.imshow('Disparity Map (Color)', disp_color)
        cv2.imshow('Left Image (Masked)', imgL_masked)

        logger.info("Press any key to close disparity windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def show_point_cloud(pcd: o3d.geometry.PointCloud):
        """Visualize 3D point cloud using Open3D viewer."""
        logger.info("Opening 3D point cloud viewer...")
        logger.info("Controls:")
        logger.info("  - Mouse: Rotate/Pan/Zoom")
        logger.info("  - +/-: Increase/Decrease point size")
        logger.info("  - ESC: Close window")


        # Visualize
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="3D Point Cloud Reconstruction",
            width=1280,
            height=720,
            zoom=0.5,
            front=[0, 0, -1],
            lookat=[0, 0, 0],
            up=[0, -1, 0]
        )


def main():
    """Main execution pipeline for stereo 3D reconstruction."""

    try:
        logger.info("╔" + "═" * 68 + "╗")
        logger.info("║" + " " * 15 + "STEREO 3D RECONSTRUCTION PIPELINE" + " " * 20 + "║")
        logger.info("╚" + "═" * 68 + "╝")
        logger.info("")

        # Initialize configuration
        config = StereoConfig()

        # Create output directory
        config.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {config.output_dir}")
        logger.info("")

        # Step 1: Camera Calibration
        calibrator = CameraCalibrator(config)
        K, dist, K_optimal = calibrator.calibrate()
        logger.info("")

        # Step 2: Stereo Reconstruction
        reconstructor = StereoReconstructor(config, K, dist, K_optimal)

        # Load and undistort images
        imgL_undist, _, grayL, grayR = reconstructor.load_and_undistort()
        logger.info("")

        # Compute disparity map
        disp8 = reconstructor.compute_disparity(grayL, grayR)
        logger.info("")

        # Display disparity (optional)
        if config.show_disparity:
            Visualizer.show_disparity(disp8, imgL_undist)

        # Reconstruct 3D point cloud
        pcd = reconstructor.reconstruct_3d(disp8, imgL_undist)
        logger.info("")

        # Save point cloud
        output_path = config.output_dir / config.output_filename
        logger.info(f"Saving point cloud to: {output_path}")
        success = o3d.io.write_point_cloud(str(output_path), pcd)

        if success:
            file_size_kb = output_path.stat().st_size / 1024
            logger.info(f"Point cloud saved successfully ({file_size_kb:.2f} KB)")
        else:
            logger.error(f"Failed to save point cloud to {output_path}")
        logger.info("")

        # Visualize point cloud (optional)
        if config.show_point_cloud:
            Visualizer.show_point_cloud(pcd)

        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Value error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
