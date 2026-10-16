"""
Camera-Based Pointcloud Reduction
===================================

This script:
1. Loads a COLMAP dense pointcloud with camera poses
2. Extracts camera positions and viewing directions
3. Calculates the intersection point of all camera rays (least squares)
4. Uses this point as center for radial reduction at multiple percentiles
5. Visualizes everything in Rerun

This approach is better than center-of-mass for objects captured in a typical
camera setup, as it finds the point the cameras are "looking at".
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import rerun as rr

from wound3d.legacy.utils.read_write_model import Camera, Image, Point3D, read_model, qvec2rotmat


def extract_camera_poses(images: Dict[int, Image]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract camera positions and viewing directions from COLMAP images.
    
    Args:
        images: Dictionary of image_id -> Image objects
        
    Returns:
        Tuple of (camera_positions, camera_directions)
        - camera_positions: (N, 3) array of camera centers in world coordinates
        - camera_directions: (N, 3) array of camera forward vectors (viewing directions)
    """
    camera_positions = []
    camera_directions = []
    
    for image in images.values():
        # Convert quaternion to rotation matrix
        R = qvec2rotmat(image.qvec)
        
        # Camera center in world coordinates: C = -R^T * t
        camera_center = -R.T @ image.tvec
        camera_positions.append(camera_center)
        
        # Camera viewing direction (forward vector in world coordinates)
        # In camera space, forward is typically -Z axis (0, 0, -1)
        # Transform to world space: R^T @ (0, 0, -1)
        camera_forward = R.T @ np.array([0, 0, -1])
        camera_directions.append(camera_forward)
    
    return np.array(camera_positions), np.array(camera_directions)


def find_ray_intersection_point(camera_positions: np.ndarray, 
                                camera_directions: np.ndarray) -> np.ndarray:
    """
    Find the point closest to all camera rays using least squares.
    
    This solves the problem: find point P that minimizes the sum of squared
    distances to all camera rays.
    
    For a ray from origin O with direction D, the closest point on the ray to P is:
    Q = O + ((P - O) · D) * D
    
    We minimize: sum ||P - Q||^2 for all rays
    
    Args:
        camera_positions: (N, 3) camera centers
        camera_directions: (N, 3) normalized camera viewing directions
        
    Returns:
        (3,) intersection point
    """
    # Normalize directions
    directions = camera_directions / np.linalg.norm(camera_directions, axis=1, keepdims=True)
    
    # Build linear system: A * P = b
    # For each ray i: (I - D_i * D_i^T) * P = (I - D_i * D_i^T) * O_i
    A = np.zeros((3, 3))
    b = np.zeros(3)
    
    for i in range(len(camera_positions)):
        O = camera_positions[i]
        D = directions[i]
        
        # Projection matrix onto plane perpendicular to D
        I_minus_DDT = np.eye(3) - np.outer(D, D)
        
        A += I_minus_DDT
        b += I_minus_DDT @ O
    
    # Solve for intersection point
    intersection_point = np.linalg.solve(A, b)
    
    return intersection_point


def radial_filter_pointcloud(points3D: Dict[int, Point3D],
                             center_point: np.ndarray,
                             percentile: float) -> Dict[int, Point3D]:
    """
    Filter pointcloud by keeping points within distance percentile from center.
    
    Args:
        points3D: Dictionary of point_id -> Point3D objects
        center_point: Center point for radial filtering
        percentile: Keep points within this percentile (e.g., 0.9 keeps closest 90%)
        
    Returns:
        Filtered points dictionary
    """
    point_ids = list(points3D.keys())
    coords = np.array([points3D[pid].xyz for pid in point_ids])
    
    # Calculate distances from center
    distances = np.linalg.norm(coords - center_point, axis=1)
    
    # Find threshold
    threshold = np.percentile(distances, percentile * 100)
    
    # Filter points
    keep_mask = distances <= threshold
    filtered_points = {
        pid: points3D[pid]
        for pid, keep in zip(point_ids, keep_mask)
        if keep
    }
    
    return filtered_points, threshold


def visualize_camera_reduction(points3D: Dict[int, Point3D],
                               camera_positions: np.ndarray,
                               camera_directions: np.ndarray,
                               intersection_point: np.ndarray,
                               filtered_points_list: List[Tuple[Dict[int, Point3D], float, float]]):
    """
    Visualize the camera-based reduction in Rerun.
    
    Args:
        points3D: Original points dictionary
        camera_positions: (N, 3) camera centers
        camera_directions: (N, 3) camera viewing directions
        intersection_point: Computed ray intersection point
        filtered_points_list: List of (filtered_points, percentile, threshold) tuples
    """
    # Initialize Rerun
    rr.init("Camera_Ray_Intersection_Reduction", spawn=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    
    # Original pointcloud
    original_coords = np.array([p.xyz for p in points3D.values()])
    original_colors = np.array([p.rgb for p in points3D.values()])
    
    rr.log("/original/points", rr.Points3D(
        original_coords,
        colors=original_colors,
        radii=0.001
    ), static=True)
    
    # Camera positions
    rr.log("/cameras/positions", rr.Points3D(
        camera_positions,
        colors=[255, 0, 255],  # Magenta
        radii=0.005
    ), static=True)
    
    # Camera rays (viewing directions)
    ray_length = 0.3  # Adjust based on scene scale
    for i, (pos, direction) in enumerate(zip(camera_positions, camera_directions)):
        ray_end = pos + direction * ray_length
        rr.log(f"/cameras/rays/{i}", rr.LineStrips3D(
            [np.array([pos, ray_end])],
            colors=[255, 165, 0]  # Orange
        ), static=True)
    
    # Intersection point
    rr.log("/intersection_point", rr.Points3D(
        [intersection_point],
        colors=[0, 255, 0],  # Green
        radii=0.015
    ), static=True)
    
    # Filtered pointclouds at different percentiles
    for filtered_points, percentile, threshold in filtered_points_list:
        percentile_pct = int(percentile * 100)
        
        # Kept points (with original colors)
        filtered_coords = np.array([p.xyz for p in filtered_points.values()])
        filtered_colors = np.array([p.rgb for p in filtered_points.values()])
        
        rr.log(f"/filtered/{percentile_pct}pct/kept_points", rr.Points3D(
            filtered_coords,
            colors=filtered_colors,
            radii=0.001
        ), static=True)
        
        # Removed/deleted points (in red)
        original_ids = set(points3D.keys())
        filtered_ids = set(filtered_points.keys())
        removed_ids = original_ids - filtered_ids
        
        if len(removed_ids) > 0:
            removed_coords = np.array([points3D[pid].xyz for pid in removed_ids])
            rr.log(f"/filtered/{percentile_pct}pct/removed_points", rr.Points3D(
                removed_coords,
                colors=[255, 0, 0],  # Red
                radii=0.0008
            ), static=True)
    
    print("\nRerun visualization launched!")
    
    # Find ray intersection point
    print("\nCalculating camera ray intersection point...")
    intersection_point = find_ray_intersection_point(camera_positions, camera_directions)
    print(f"  Intersection point: {intersection_point}")
    
    # Perform radial filtering at multiple percentiles
    percentiles = [0.90, 0.80, 0.70]  # Keep 90%, 80%, 70% (remove top 10%, 20%, 30%)
    filtered_results = []
    
    print("\nPerforming radial filtering...")
    for percentile in percentiles:
        filtered_points, threshold = radial_filter_pointcloud(points3D, intersection_point, percentile)
        
        reduction_pct = (1 - percentile) * 100
        kept_pct = percentile * 100
        
        print(f"\n  Remove top {reduction_pct:.0f}% (keep {kept_pct:.0f}%):")
        print(f"    Threshold distance: {threshold:.4f}")
        print(f"    Original points: {len(points3D)}")
        print(f"    Filtered points: {len(filtered_points)}")
        print(f"    Actual reduction: {100 * (1 - len(filtered_points)/len(points3D)):.1f}%")
        
        filtered_results.append((filtered_points, percentile, threshold))
    
    # Visualize
    print("\nLaunching Rerun visualization...")
    visualize_camera_reduction(
        points3D,
        camera_positions,
        camera_directions,
        intersection_point,
        filtered_results
    )
    
    print("\n" + "=" * 60)
    print("Complete! Check Rerun viewer.")
    print("Toggle between different percentile filters in the viewer.")
    print("=" * 60)
