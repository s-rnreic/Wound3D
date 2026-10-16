"""
Sparse point cloud post-processing module for COLMAP/GLOMAP reconstruction.

This module provides functionality to:
1. Filter 3D points based on track length and reprojection error
2. Update image references to match filtered points
3. Run bundle adjustment on the filtered reconstruction
4. Visualize the changes with rerun
"""

import os
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import sys

# Add workspace root to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_workspace_root = os.path.abspath(os.path.join(_current_dir, '../../..'))
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

from wound3d.legacy.utils.read_write_model import Point3D, Image, Camera, read_model, write_model


def filter_points3D(
    points3D: Dict[int, Point3D],
    images: Dict[int, Image],
    max_reprojection_error: float = 1.0,
    min_track_length: int = 3
) -> Tuple[Dict[int, Point3D], Dict[int, Image]]:
    """
    Filter 3D points based on reprojection error and track length.
    
    Args:
        points3D: Dictionary of Point3D objects from COLMAP model
        images: Dictionary of Image objects from COLMAP model
        max_reprojection_error: Maximum allowed reprojection error (in pixels)
        min_track_length: Minimum number of observations required for a point
        
    Returns:
        Tuple of (filtered_points3D, updated_images)
    """
    
    print(f"Filtering points with constraints:")
    print(f"  - Max reprojection error: {max_reprojection_error}")
    print(f"  - Min track length: {min_track_length}")
    
    initial_count = len(points3D)
    
    # Filter points based on criteria
    filtered_points3D = {}
    removed_point_ids = set()
    
    for point_id, point in points3D.items():
        track_length = len(point.image_ids)
        
        # Check both criteria
        if point.error <= max_reprojection_error and track_length >= min_track_length:
            filtered_points3D[point_id] = point
        else:
            removed_point_ids.add(point_id)
    
    # Update images to remove references to filtered points
    updated_images = {}
    for img_id, image in images.items():
        # Create masks for valid points remaining in this image
        valid_masks = [
            point_id not in removed_point_ids 
            for point_id in image.point3D_ids
        ]
        
        # Keep only valid 2D-3D correspondences
        updated_xys = image.xys[valid_masks]
        updated_point3D_ids = image.point3D_ids[valid_masks]
        
        # Create updated image
        updated_images[img_id] = Image(
            id=image.id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=image.name,
            xys=updated_xys,
            point3D_ids=updated_point3D_ids
        )
    
    final_count = len(filtered_points3D)
    removed_count = initial_count - final_count
    
    print(f"\nFiltering results:")
    print(f"  - Points removed: {removed_count} / {initial_count} ({100*removed_count/initial_count:.2f}%)")
    print(f"  - Points remaining: {final_count}")
    
    return filtered_points3D, updated_images


def get_point_statistics(points3D: Dict[int, Point3D]) -> Dict:
    """
    Calculate statistics about the 3D point cloud.
    
    Args:
        points3D: Dictionary of Point3D objects
        
    Returns:
        Dictionary with statistics
    """
    
    if not points3D:
        return {
            'count': 0,
            'mean_error': 0,
            'median_error': 0,
            'max_error': 0,
            'min_error': 0,
            'mean_track_length': 0,
            'median_track_length': 0,
            'max_track_length': 0,
            'min_track_length': 0,
        }
    
    errors = np.array([point.error for point in points3D.values()])
    track_lengths = np.array([len(point.image_ids) for point in points3D.values()])
    
    stats = {
        'count': len(points3D),
        'mean_error': float(np.mean(errors)),
        'median_error': float(np.median(errors)),
        'max_error': float(np.max(errors)),
        'min_error': float(np.min(errors)),
        'mean_track_length': float(np.mean(track_lengths)),
        'median_track_length': float(np.median(track_lengths)),
        'max_track_length': int(np.max(track_lengths)),
        'min_track_length': int(np.min(track_lengths)),
    }
    
    return stats


def print_point_statistics(points3D: Dict[int, Point3D], label: str = "Point Cloud"):
    """Print formatted statistics about the point cloud."""
    
    stats = get_point_statistics(points3D)
    
    print(f"\n{label} Statistics:")
    print(f"  Count: {stats['count']}")
    print(f"  Reprojection Error - Mean: {stats['mean_error']:.4f}, Median: {stats['median_error']:.4f}, "
          f"Max: {stats['max_error']:.4f}, Min: {stats['min_error']:.4f}")
    print(f"  Track Length - Mean: {stats['mean_track_length']:.2f}, Median: {stats['median_track_length']:.1f}, "
          f"Max: {stats['max_track_length']}, Min: {stats['min_track_length']}")


def run_bundle_adjustment(
    workspace_path: str,
    sparse_path: str,
    ini_path: str,
    database_path: str,
    camera_model: str = "PINHOLE"
) -> float:
    """
    Run COLMAP bundle adjustment on the filtered model.
    
    Args:
        workspace_path: Path to the workspace
        sparse_path: Path to sparse reconstruction folder
        ini_path: Path to project.ini file
        database_path: Path to database.db file
        camera_model: Camera model to use (PINHOLE, RADIAL, etc.)
        
    Returns:
        Elapsed time for the operation
    """
    
    import time
    start_time = time.time()
    
    # Create bundle_adjuster command
    # Reference: https://colmap.github.io/cli.html#bundle-adjuster
    ba_command = (
        f"colmap bundle_adjuster "
        f"--input_path \"{sparse_path}/0\" "
        f"--output_path \"{sparse_path}/0\" "
        f"--BundleAdjustment.refine_principal_point=true "
        f"--BundleAdjustment.refine_focal_length=true "
        f"--BundleAdjustment.refine_extra_params=true"
    )
    
    print(f"\nRunning Bundle Adjustment...")
    print(f"Command: {ba_command}")
    
    result = subprocess.run(ba_command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Bundle adjustment failed!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError("Bundle adjustment failed")
    
    elapsed_time = time.time() - start_time
    print(f"Bundle adjustment completed in {elapsed_time:.2f} seconds")
    
    return elapsed_time


def apply_sparse_filtering(
    workspace_path: str,
    sparse_path: str,
    ini_path: str,
    database_path: str,
    max_reprojection_error: float = 1.0,
    min_track_length: int = 3,
    run_ba: bool = True,
    ba_camera_model: str = "PINHOLE"
) -> Tuple[Dict, Dict, Dict, float]:
    """
    Complete workflow: filter points and optionally run bundle adjustment.
    
    Args:
        workspace_path: Path to workspace
        sparse_path: Path to sparse reconstruction
        ini_path: Path to project.ini
        database_path: Path to database.db
        max_reprojection_error: Max reprojection error threshold
        min_track_length: Min track length threshold
        run_ba: Whether to run bundle adjustment after filtering
        ba_camera_model: Camera model for bundle adjustment
        
    Returns:
        Tuple of (cameras, images, points3D, total_elapsed_time)
    """
    
    import time
    total_start_time = time.time()
    
    # Read current model
    print("Reading current sparse reconstruction...")
    model_path = Path(sparse_path) / "0"
    cameras, images, points3D = read_model(model_path, ext=".bin")
    
    # Print before statistics
    print_point_statistics(points3D, "Before Filtering")
    
    # Apply filtering
    print("\nApplying point filtering...")
    filtered_points3D, updated_images = filter_points3D(
        points3D,
        images,
        max_reprojection_error=max_reprojection_error,
        min_track_length=min_track_length
    )
    
    # Print after filtering statistics
    print_point_statistics(filtered_points3D, "After Filtering")
    
    # Write filtered model
    print("\nWriting filtered model to disk...")
    write_model(cameras, updated_images, filtered_points3D, str(model_path), ext=".bin")
    print(f"Filtered model written to {model_path}")
    
    total_elapsed_time = time.time() - total_start_time
    
    # Run bundle adjustment if requested
    if run_ba:
        ba_elapsed_time = run_bundle_adjustment(
            workspace_path,
            sparse_path,
            ini_path,
            database_path,
            camera_model=ba_camera_model
        )
        total_elapsed_time += ba_elapsed_time
        
        # Re-read the model after BA
        print("\nRe-reading model after bundle adjustment...")
        cameras, updated_images, filtered_points3D = read_model(model_path, ext=".bin")
        print_point_statistics(filtered_points3D, "After Bundle Adjustment")
    
    total_elapsed_time = time.time() - total_start_time
    
    return cameras, updated_images, filtered_points3D, total_elapsed_time


def create_comparison_visualization(
    before_points3D: Dict[int, Point3D],
    after_points3D: Dict[int, Point3D],
    rr_module
) -> None:
    """
    Create a comparison visualization using rerun showing before/after filtering.
    
    Args:
        before_points3D: Points before filtering
        after_points3D: Points after filtering
        rr_module: Rerun module (import rerun as rr)
    """
    
    # Before filtering visualization
    before_xyz = np.array([point.xyz for point in before_points3D.values()])
    before_rgb = np.array([point.rgb for point in before_points3D.values()])
    
    rr_module.log(
        "comparison/before_filtering",
        rr_module.Points3D(before_xyz, colors=before_rgb),
        static=True
    )
    
    # After filtering visualization
    after_xyz = np.array([point.xyz for point in after_points3D.values()])
    after_rgb = np.array([point.rgb for point in after_points3D.values()])
    
    rr_module.log(
        "comparison/after_filtering",
        rr_module.Points3D(after_xyz, colors=after_rgb),
        static=True
    )
    
    # Create a difference visualization (points that were removed)
    removed_ids = set(before_points3D.keys()) - set(after_points3D.keys())
    if removed_ids:
        removed_xyz = np.array([before_points3D[pid].xyz for pid in removed_ids])
        removed_colors = [[255, 0, 0] for _ in removed_ids]  # Red for removed
        
        rr_module.log(
            "comparison/removed_points",
            rr_module.Points3D(removed_xyz, colors=removed_colors),
            static=True
        )
        
        print(f"\nRemoved points visualization created: {len(removed_ids)} points shown in red")
