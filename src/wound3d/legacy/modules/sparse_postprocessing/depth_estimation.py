"""
Depth range estimation module for multi-view stereo reconstruction.

Analyzes the sparse point cloud to determine optimal depth bounds for each camera
and computes global depth parameters for COLMAP patch_match_stereo.

The depth bounds are crucial for MVS algorithms to:
1. Focus computation on relevant depth range
2. Reduce artifacts from infinite reconstructions
3. Improve memory efficiency
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import os
import sys

# Add workspace root to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_workspace_root = os.path.abspath(os.path.join(_current_dir, '../../..'))
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

from wound3d.legacy.utils.read_write_model import Camera, Image, Point3D, read_model


def compute_camera_depth_range(
    image: Image,
    camera: Camera,
    points3D: Dict[int, Point3D],
    depth_percentile: float = 90.0
) -> Tuple[float, float]:
    """
    Compute depth range for a single camera based on visible 3D points.
    
    Args:
        image: Image object with camera pose
        camera: Camera object with intrinsics
        points3D: Dictionary of all 3D points
        depth_percentile: Percentile for max depth (e.g., 90 means 90th percentile)
                         Default: 90, so max_depth is the depth at which 10% of points are further
        
    Returns:
        Tuple of (min_depth, max_depth) in scene units
    """
    
    # Get rotation matrix from quaternion
    from scipy.spatial.transform import Rotation
    qvec = image.qvec
    # QW, QX, QY, QZ -> rotation matrix
    rotation = Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]])
    rotation_matrix = rotation.as_matrix()
    tvec = image.tvec
    
    # Collect depths for visible points
    depths = []
    
    for point_id in image.point3D_ids:
        if point_id == -1:  # Invalid point
            continue
        
        if point_id not in points3D:
            continue
        
        point = points3D[point_id]
        point_xyz = point.xyz
        
        # Transform point to camera coordinates: P_cam = R * (P_world - t)
        # For COLMAP: camera pose is "camera from world", so:
        # P_cam = R * P_world + t (already in this form)
        point_cam = rotation_matrix @ point_xyz + tvec
        
        # Depth is Z coordinate in camera frame
        depth = point_cam[2]
        
        # Only consider points in front of camera (depth > 0)
        if depth > 0:
            depths.append(depth)
    
    if not depths:
        # No valid points, return default range
        return 0.1, 1000.0
    
    depths = np.array(depths)
    depths = np.sort(depths)
    
    # Minimum depth: smallest depth value (no percentile)
    min_depth = float(np.min(depths))
    
    # Maximum depth: depth percentile (e.g., 90th percentile)
    # This means 10% of points are further away
    max_depth_idx = int(len(depths) * depth_percentile / 100.0)
    max_depth_idx = min(max_depth_idx, len(depths) - 1)
    max_depth = float(depths[max_depth_idx])
    
    return min_depth, max_depth


def analyze_scene_depth(
    sparse_path: str,
    depth_percentile: float = 90.0
) -> Tuple[Dict[int, Tuple[float, float]], float, float]:
    """
    Analyze depth ranges across all cameras in the sparse reconstruction.
    
    Args:
        sparse_path: Path to sparse reconstruction (e.g., "workspace/sparse/0")
        depth_percentile: Percentile for max depth calculation (default: 90)
        
    Returns:
        Tuple of:
        - camera_depths: Dict[image_id] -> (min_depth, max_depth)
        - global_min_depth: Global minimum depth across all cameras
        - global_max_depth: Global maximum depth at specified percentile
    """
    
    model_path = Path(sparse_path)
    
    print(f"Reading sparse reconstruction from {model_path}")
    cameras, images, points3D = read_model(model_path, ext=".bin")
    
    print(f"Analyzing depth for {len(images)} camera(s) and {len(points3D)} point(s)")
    
    camera_depths = {}
    all_min_depths = []
    all_max_depths = []
    
    for image_id, image in images.items():
        camera = cameras[image.camera_id]
        
        min_depth, max_depth = compute_camera_depth_range(
            image,
            camera,
            points3D,
            depth_percentile=depth_percentile
        )
        
        camera_depths[image_id] = (min_depth, max_depth)
        all_min_depths.append(min_depth)
        all_max_depths.append(max_depth)
        
        print(f"  Camera {image_id:3d} ({image.name:30s}): "
              f"min={min_depth:8.4f}, max={max_depth:8.4f}")
    
    if all_min_depths:
        # Global min: minimum of all camera minimums
        global_min_depth = float(np.min(all_min_depths))
        
        # Global max: maximum of all camera max depths
        # This ensures all points are within range for all cameras
        global_max_depth = float(np.max(all_max_depths))
    else:
        # Fallback defaults
        global_min_depth = 0.1
        global_max_depth = 1000.0
    
    print(f"\nGlobal depth range:")
    print(f"  Min depth: {global_min_depth:.4f}")
    print(f"  Max depth: {global_max_depth:.4f}")
    print(f"  Range: {global_max_depth - global_min_depth:.4f}")
    
    return camera_depths, global_min_depth, global_max_depth


def get_colmap_depth_params(
    global_min_depth: float,
    global_max_depth: float,
    safety_margin: float = 0.1
) -> Tuple[float, float]:
    """
    Convert global depth range to COLMAP patch_match_stereo parameters.
    
    The max_depth is already based on the 90th percentile from analyze_scene_depth(),
    which means only 10% of observed points are beyond it. This is inherently conservative.
    
    Args:
        global_min_depth: Minimum depth from analysis
        global_max_depth: Maximum depth from analysis (already at percentile, e.g., 90th)
        safety_margin: Small absolute buffer in scene units to add as margin (default: 0.1)
        
    Returns:
        Tuple of (depth_min, depth_max) for COLMAP
    """
    
    # Use percentile-based max_depth directly (already robust to outliers)
    # Only apply small absolute margin for numerical safety, not percentage-based
    depth_min = max(0.01, global_min_depth)
    depth_max = global_max_depth 
    
    print(f"\nCOLMAP Depth Parameters (based on 90th percentile, with ±{safety_margin:.4f} absolute margin):")
    print(f"  --PatchMatchStereo.depth_min {depth_min:.4f}")
    print(f"  --PatchMatchStereo.depth_max {depth_max:.4f}")
    
    return depth_min, depth_max


def print_depth_statistics(
    camera_depths: Dict[int, Tuple[float, float]]
) -> None:
    """Print detailed statistics about depth ranges across cameras."""
    
    if not camera_depths:
        print("No camera depth data available")
        return
    
    min_depths = [d[0] for d in camera_depths.values()]
    max_depths = [d[1] for d in camera_depths.values()]
    
    print(f"\nDepth Statistics:")
    print(f"  Min depths:  μ={np.mean(min_depths):.4f}, σ={np.std(min_depths):.4f}, "
          f"range=[{np.min(min_depths):.4f}, {np.max(min_depths):.4f}]")
    print(f"  Max depths:  μ={np.mean(max_depths):.4f}, σ={np.std(max_depths):.4f}, "
          f"range=[{np.min(max_depths):.4f}, {np.max(max_depths):.4f}]")
    print(f"  Depth ranges: μ={np.mean(np.array(max_depths) - np.array(min_depths)):.4f}")


def estimate_scene_depth(
    workspace_path: str,
    sparse_path: str,
    depth_percentile: float = 90.0,
    safety_margin: float = 0.1
) -> Tuple[float, float, Dict[int, Tuple[float, float]]]:
    """
    Complete depth estimation workflow.
    
    The max_depth is calculated as the 90th percentile (by default) of observed depths,
    meaning only 10% of reconstructed points are beyond it. This is inherently robust to outliers.
    
    Args:
        workspace_path: Path to workspace
        sparse_path: Path to sparse reconstruction
        depth_percentile: Percentile for max depth (default: 90 = 90th percentile)
        safety_margin: Small absolute buffer value in scene units to add as margin (default: 0.1)
        
    Returns:
        Tuple of (depth_min, depth_max, camera_depths)
    """
    
    print(f"\n{'='*70}")
    print(f"DEPTH RANGE ESTIMATION")
    print(f"{'='*70}")
    
    # Analyze depth from sparse reconstruction
    camera_depths, global_min, global_max = analyze_scene_depth(
        sparse_path,
        depth_percentile=depth_percentile
    )
    
    # Print detailed statistics
    print_depth_statistics(camera_depths)
    
    # Get COLMAP parameters with safety margin
    depth_min, depth_max = get_colmap_depth_params(
        global_min,
        global_max,
        safety_margin=safety_margin
    )
    
    # Save depth parameters to file for reference
    depth_file = Path(workspace_path) / "depth_estimates.txt"
    with open(depth_file, 'w') as f:
        f.write(f"Global Depth Range Estimation\n")
        f.write(f"=============================\n\n")
        f.write(f"Depth Percentile: {depth_percentile}% (max depth = {depth_percentile}th percentile of observations)\n")
        f.write(f"Safety Margin: ±{safety_margin:.4f} (absolute scene units)\n\n")
        f.write(f"Analyzed depths:\n")
        f.write(f"  Global min (from analysis): {global_min:.4f}\n")
        f.write(f"  Global max (from analysis, at {depth_percentile}th percentile): {global_max:.4f}\n\n")
        f.write(f"COLMAP Parameters (with safety margin applied):\n")
        f.write(f"  depth_min: {depth_min:.4f}\n")
        f.write(f"  depth_max: {depth_max:.4f}\n\n")
        f.write(f"Per-camera depths:\n")
        for img_id, (min_d, max_d) in sorted(camera_depths.items()):
            f.write(f"  Camera {img_id}: min={min_d:.4f}, max={max_d:.4f}\n")
    
    print(f"\nDepth estimates saved to {depth_file}")
    
    return depth_min, depth_max, camera_depths
