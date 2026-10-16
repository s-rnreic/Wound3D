"""
Pointcloud Processing Workflow
===============================

Complete pipeline for pointcloud processing:
1. Load COLMAP dense model
2. Apply Camera Ray Intersection reduction (remove distant outliers)
3. Perform FPS sampling for uniform downsampling
4. Save the processed pointcloud
"""

import os
import numpy as np
import fpsample
import open3d as o3d
from pathlib import Path
from typing import Dict, Tuple

from wound3d.legacy.utils.read_write_model import Camera, Image, Point3D, read_model, qvec2rotmat


def extract_camera_poses(images: Dict[int, Image]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract camera positions and viewing directions from COLMAP images.
    """
    camera_positions = []
    camera_directions = []
    
    for image in images.values():
        R = qvec2rotmat(image.qvec)
        camera_center = -R.T @ image.tvec
        camera_positions.append(camera_center)
        camera_forward = R.T @ np.array([0, 0, -1])
        camera_directions.append(camera_forward)
    
    return np.array(camera_positions), np.array(camera_directions)


def find_ray_intersection_point(camera_positions: np.ndarray, 
                                camera_directions: np.ndarray) -> np.ndarray:
    """
    Find the point closest to all camera rays using least squares.
    """
    directions = camera_directions / np.linalg.norm(camera_directions, axis=1, keepdims=True)
    
    A = np.zeros((3, 3))
    b = np.zeros(3)
    
    for i in range(len(camera_positions)):
        O = camera_positions[i]
        D = directions[i]
        I_minus_DDT = np.eye(3) - np.outer(D, D)
        A += I_minus_DDT
        b += I_minus_DDT @ O
    
    return np.linalg.solve(A, b)


def radial_filter_pointcloud_o3d(pcd: o3d.geometry.PointCloud,
                                 center_point: np.ndarray,
                                 percentile: float) -> Tuple[o3d.geometry.PointCloud, np.ndarray, float]:
    """
    Filter Open3D pointcloud by keeping points within distance percentile from center.
    
    Args:
        pcd: Open3D PointCloud object
        center_point: Center point for radial filtering
        percentile: Keep points within this percentile (e.g., 0.75 keeps closest 75%)
        
    Returns:
        Tuple of (filtered_pcd, kept_indices, threshold)
    """
    coords = np.asarray(pcd.points)
    
    distances = np.linalg.norm(coords - center_point, axis=1)
    threshold = np.percentile(distances, percentile * 100)
    
    keep_mask = distances <= threshold
    kept_indices = np.where(keep_mask)[0]
    
    # Select points using indices
    filtered_pcd = pcd.select_by_index(kept_indices.tolist())
    
    return filtered_pcd, kept_indices, threshold


def radial_filter_pointcloud(points3D: Dict[int, Point3D],
                             center_point: np.ndarray,
                             percentile: float) -> Tuple[Dict[int, Point3D], float]:
    """
    Filter pointcloud by keeping points within distance percentile from center.
    """
    point_ids = list(points3D.keys())
    coords = np.array([points3D[pid].xyz for pid in point_ids])
    
    distances = np.linalg.norm(coords - center_point, axis=1)
    threshold = np.percentile(distances, percentile * 100)
    
    keep_mask = distances <= threshold
    filtered_points = {
        pid: points3D[pid]
        for pid, keep in zip(point_ids, keep_mask)
        if keep
    }
    
    return filtered_points, threshold


def save_points_to_ply(points_dict: Dict[int, Point3D], output_path: Path, normals: np.ndarray = None):
    """
    Save points to a PLY file in COLMAP binary format.
    
    Args:
        points_dict: Dictionary of point_id -> Point3D objects
        output_path: Path to save PLY file
        normals: Optional (N, 3) array of normals. If None, zeros are used.
    """
    import struct
    
    coords = np.array([p.xyz for p in points_dict.values()], dtype=np.float32)
    colors = np.array([p.rgb for p in points_dict.values()], dtype=np.uint8)
    
    # Use provided normals or zeros
    if normals is None:
        normals = np.zeros((len(coords), 3), dtype=np.float32)
    else:
        normals = normals.astype(np.float32)
    
    num_points = len(coords)
    
    with open(output_path, 'wb') as f:
        # Write header
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {num_points}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode('ascii'))
        
        # Write binary data
        for i in range(num_points):
            # Pack: x, y, z (float), nx, ny, nz (float), r, g, b (uchar)
            vertex_data = struct.pack(
                '<ffffff BBB',
                coords[i, 0], coords[i, 1], coords[i, 2],
                normals[i, 0], normals[i, 1], normals[i, 2],
                colors[i, 0], colors[i, 1], colors[i, 2]
            )
            f.write(vertex_data)
    
    print(f"Saved {num_points} points to: {output_path}")


def save_pcd_to_ply(pcd: o3d.geometry.PointCloud, output_path: Path):
    """
    Save Open3D pointcloud to a PLY file in COLMAP binary format.
    
    Args:
        pcd: Open3D PointCloud object (with points, colors, normals)
        output_path: Path to save PLY file
    """
    import struct
    
    coords = np.asarray(pcd.points).astype(np.float32)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else np.zeros((len(coords), 3), dtype=np.uint8)
    normals = np.asarray(pcd.normals).astype(np.float32) if pcd.has_normals() else np.zeros((len(coords), 3), dtype=np.float32)
    
    num_points = len(coords)
    
    with open(output_path, 'wb') as f:
        # Write header
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {num_points}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode('ascii'))
        
        # Write binary data
        for i in range(num_points):
            vertex_data = struct.pack(
                '<ffffff BBB',
                coords[i, 0], coords[i, 1], coords[i, 2],
                normals[i, 0], normals[i, 1], normals[i, 2],
                colors[i, 0], colors[i, 1], colors[i, 2]
            )
            f.write(vertex_data)
    
    print(f"Saved {num_points} points to: {output_path}")


def process_pointcloud(dense_model_path: Path,
                       output_path: Path,
                       radial_percentile: float = 0.75,
                       target_points: int = 100000,
                       SOR_std=None,
                       SOR_n=None
                       ):
    """
    Complete pointcloud processing workflow.
    
    Loads fused.ply from dense model (preserves normals) and camera poses.
    
    Args:
        dense_model_path: Path to COLMAP dense model directory (contains fused.ply, cameras.bin, images.bin)
        output_path: Path to save the processed PLY file
        radial_percentile: Keep this fraction of points (closest to camera focus)
        target_points: Target number of points after FPS sampling
    """
    print("=" * 60)
    print("Pointcloud Processing Workflow")
    print("=" * 60)
    
    # Step 1: Load fused.ply from COLMAP dense model (preserves normals)
    print("\n--- Step 1: Loading fused.ply from Dense Model ---")
    fused_ply_path = dense_model_path / "fused.ply"
    if not fused_ply_path.exists():
        raise FileNotFoundError(f"fused.ply not found at: {fused_ply_path}")
    
    pcd = o3d.io.read_point_cloud(str(fused_ply_path))
    print(f"  Loaded {len(pcd.points)} points from fused.ply")
    print(f"  Has normals: {pcd.has_normals()}")
    print(f"  Has colors: {pcd.has_colors()}")
    
    # Step 1b: Load camera poses from dense model
    print("\n--- Step 1b: Loading Camera Poses ---")
    cameras, images, _ = read_model(dense_model_path, ext=".bin")
    print(f"  Loaded {len(images)} camera poses")
    
    # Step 2: Extract camera poses and find intersection point
    print("\n--- Step 2: Finding Camera Focus Point ---")
    camera_positions, camera_directions = extract_camera_poses(images)
    intersection_point = find_ray_intersection_point(camera_positions, camera_directions)
    print(f"  Camera focus point: {intersection_point}")
    
    # Step 3: Apply radial reduction (using Open3D version to preserve normals)
    print(f"\n--- Step 3: Radial Reduction (keep {radial_percentile*100:.0f}%) ---")
    original_count = len(pcd.points)
    filtered_pcd, kept_indices, threshold = radial_filter_pointcloud_o3d(
        pcd, intersection_point, radial_percentile
    )
    print(f"  Threshold distance: {threshold:.4f}")
    print(f"  Points after radial filter: {len(filtered_pcd.points)}")
    print(f"  Removed: {original_count - len(filtered_pcd.points)} points")
    
    # Step 4: FPS Sampling (preserving normals)
    print(f"\n--- Step 4: FPS Sampling (target: {target_points} points) ---")
    
    # Extract coordinates for FPS
    coords = np.asarray(filtered_pcd.points)
    
    # Adjust target if we have fewer points
    actual_target = min(target_points, len(coords))
    if actual_target < target_points:
        print(f"  Warning: Only {len(coords)} points available, sampling all")
    
    # Run FPS sampling
    print(f"  Running bucket_fps_kdtree_sampling...")
    fps_indices = fpsample.bucket_fps_kdtree_sampling(coords, actual_target)
    
    # Select points by FPS indices (preserves all attributes including normals)
    sampled_pcd = filtered_pcd.select_by_index(fps_indices)
    print(f"  Points after FPS: {len(sampled_pcd.points)}")
    print(f"  Normals preserved: {sampled_pcd.has_normals()}")
    
    # Step 4: Apply SOR to processed pointcloud
    if SOR_n and SOR_std is not None:
        sampled_pcd, _ = sampled_pcd.remove_statistical_outlier(
            nb_neighbors=int(SOR_n), 
            std_ratio=float(SOR_std)
        )
        print(f" Points after SOR: {len(sampled_pcd.points)}")
        #print(f" numer of removed points: {len()}")
    
    # Step 5: Save output (using COLMAP binary format with normals)
    print(f"\n--- Step 5: Saving Output ---")
    save_pcd_to_ply(sampled_pcd, output_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)
    print(f"  Original points:     {original_count}")
    print(f"  After radial filter: {len(filtered_pcd.points)}")
    print(f"  After FPS sampling:  {len(sampled_pcd.points)}")
    print(f"  Total reduction:     {100 * (1 - len(sampled_pcd.points)/original_count):.1f}%")
    print(f"  Output saved to:     {output_path}")
    
    return sampled_pcd


def execute_dense_postprocessing(
    dense_model_path: Path,
    output_path: Path = None,
    radial_percentile: float = 0.75,
    target_points: int = 100000,
    SOR_std: float = None,
    SOR_n: int = None,
    use_pointcleannet: bool = False,
    pointcleannet_denoise: bool = True,
    pointcleannet_outliers: bool = False,
    denoise_iterations: int = 1,
    outlier_threshold: float = 0.5,
    n_neighbours: int = 100
):
    """
    Execute complete dense postprocessing workflow.
    
    Combines process_pointcloud with optional PointCleanNet denoising/outlier removal.
    
    Args:
        dense_model_path: Path to COLMAP dense model directory (contains fused.ply)
        output_path: Path to save final PLY file. If None, saves to dense_model_path/processed_pointcloud.ply
        radial_percentile: Keep this fraction of points (closest to camera focus)
        target_points: Target number of points after FPS sampling
        SOR_std: Standard deviation for Statistical Outlier Removal (None to skip)
        SOR_n: Number of neighbors for SOR (None to skip)
        use_pointcleannet: Whether to apply PointCleanNet processing
        pointcleannet_denoise: Apply PointCleanNet denoising (requires use_pointcleannet=True)
        pointcleannet_outliers: Apply PointCleanNet outlier removal (requires use_pointcleannet=True)
        denoise_iterations: Number of denoising iterations for PointCleanNet
        outlier_threshold: Threshold for PointCleanNet outlier classification (0-1)
        n_neighbours: Number of neighbors for PointCleanNet smoothing
        
    Returns:
        Path to the final output file
    """
    import time
    start_time = time.time()
    
    dense_model_path = Path(dense_model_path)
    
    # Set default output path
    if output_path is None:
        output_path = dense_model_path / "processed_pointcloud.ply"
    else:
        output_path = Path(output_path)
    
    print("=" * 60)
    print("DENSE POSTPROCESSING WORKFLOW")
    print("=" * 60)
    print(f"  Input: {dense_model_path}")
    print(f"  Output: {output_path}")
    print(f"  PointCleanNet: {'Enabled' if use_pointcleannet else 'Disabled'}")
    if use_pointcleannet:
        print(f"    - Denoise: {pointcleannet_denoise} ({denoise_iterations} iterations)")
        print(f"    - Outlier removal: {pointcleannet_outliers} (threshold: {outlier_threshold})")
    print("=" * 60)
    
    # Step 1: Run process_pointcloud (radial filter + FPS + optional SOR)
    intermediate_path = output_path if not use_pointcleannet else dense_model_path / "intermediate_processed.ply"
    
    process_pointcloud(
        dense_model_path=dense_model_path,
        output_path=intermediate_path,
        radial_percentile=radial_percentile,
        target_points=target_points,
        SOR_std=SOR_std,
        SOR_n=SOR_n
    )
    
    # Step 2: Optional PointCleanNet processing
    if use_pointcleannet:
        print("\n" + "=" * 60)
        print("POINTCLEANNET PROCESSING")
        print("=" * 60)
        
        try:
            from wound3d.legacy.modules.dense_postprocessing.PointCleanNet import (
                denoise_pointcloud, 
                remove_outliers, 
                clean_pointcloud
            )
        except ImportError:
            # Try relative import
            from PointCleanNet import denoise_pointcloud, remove_outliers, clean_pointcloud
        
        current_input = intermediate_path
        
        if pointcleannet_outliers and pointcleannet_denoise:
            # Run combined cleaning
            print("\n--- Running PointCleanNet: Outlier Removal + Denoising ---")
            clean_pointcloud(
                input_ply=str(current_input),
                output_ply=str(output_path),
                denoise=True,
                remove_outliers_flag=True,
                num_iterations=denoise_iterations,
                outlier_threshold=outlier_threshold
            )
        elif pointcleannet_outliers:
            # Run outlier removal only
            print("\n--- Running PointCleanNet: Outlier Removal ---")
            remove_outliers(
                input_ply=str(current_input),
                output_ply=str(output_path),
                threshold=outlier_threshold
            )
        elif pointcleannet_denoise:
            # Run denoising only
            print("\n--- Running PointCleanNet: Denoising ---")
            denoise_pointcloud(
                input_ply=str(current_input),
                output_ply=str(output_path),
                num_iterations=denoise_iterations,
                n_neighbours=n_neighbours
            )
        
        # Clean up intermediate file
        if intermediate_path != output_path and intermediate_path.exists():
            intermediate_path.unlink()
            print(f"  Cleaned up intermediate file: {intermediate_path}")
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("DENSE POSTPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Total time: {elapsed_time:.2f} seconds")
    print(f"  Output saved to: {output_path}")
    
    return output_path, elapsed_time
