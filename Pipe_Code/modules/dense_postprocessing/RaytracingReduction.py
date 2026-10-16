"""
Dense Pointcloud Reduction using Wound Mask Projection
======================================================

This module implements adaptive density filtering for dense point clouds based on
wound mask projection from multiple camera viewpoints. The workflow includes:

1. Load wound masks and camera positions from COLMAP model
2. Project wound masks onto 3D pointcloud to identify wound region points
3. Apply spherical density falloff from wound region
4. Perform adaptive density filtering with configurable wound density

Key Features:
    - Mask-based 3D point classification (wound vs. non-wound)
    - Spherical density falloff from wound region center
    - Multi-camera consensus for robust wound region identification
    - Configurable density parameters for wound and background regions
    - Visualization with Rerun (optional)

Author: COLMAP Pipeline
Date: February 2026
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
import cv2
from collections import defaultdict
from scipy.spatial import cKDTree

# Import required modules
from Pipe_Code.utils.read_write_model import Camera, Image, Point3D, read_model, qvec2rotmat


class WoundMaskProjector:
    """
    Projects 2D wound masks onto 3D pointcloud to identify wound region points.
    
    This class handles the projection of wound masks from multiple camera viewpoints
    onto a 3D point cloud, marking points that fall within the wound region.
    """
    
    def __init__(self):
        """Initialize the WoundMaskProjector."""
        # COLMAP data
        self.cameras = None
        self.images = None
        self.points3D = None
        
        # Mask data
        self.masks = {}  # image_name -> mask_array
        
        # Classification results
        self.wound_point_indices = set()  # Points inside wound region
        self.wound_point_votes = {}  # point_idx -> num_cameras_seeing_as_wound
        self.wound_point_depths = {}  # point_idx -> min_depth_from_cameras
        
        # Processing state
        self.model_loaded = False
        self.masks_loaded = False
        self.projection_completed = False
        
    def load_colmap_model(self, model_path: Path):
        """
        Load COLMAP sparse or dense model.
        
        Args:
            model_path: Path to COLMAP model directory (containing cameras.bin, images.bin, points3D.bin)
        """
        model_path = Path(model_path)
        print(f"Loading COLMAP model from: {model_path}")
        
        self.cameras, self.images, self.points3D = read_model(model_path, ext=".bin")
        self.model_loaded = True
        
        print(f"Loaded {len(self.cameras)} cameras, {len(self.images)} images, {len(self.points3D)} points")
    
    def load_masks(self, mask_dir: Path):
        """
        Load wound masks for all images.
        
        Args:
            mask_dir: Directory containing mask files
        """
        mask_dir = Path(mask_dir)
        print(f"Loading masks from: {mask_dir}")
        
        masks_loaded = 0
        for img_id, img in self.images.items():
            # Try different mask naming conventions
            mask_name = f"{img.name}.png"
            mask_path = mask_dir / mask_name
            
            if not mask_path.exists():
                # Try replacing extension
                mask_name = Path(img.name).stem + ".png"
                mask_path = mask_dir / mask_name
            
            if mask_path.exists():
                try:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        self.masks[img.name] = mask
                        masks_loaded += 1
                except Exception as e:
                    print(f"Error loading mask {mask_path}: {e}")
        
        self.masks_loaded = masks_loaded > 0
        print(f"Loaded {masks_loaded}/{len(self.images)} masks")
    
    def project_points_to_image(self, points_3d: np.ndarray, image: Image, 
                                camera: Camera) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project 3D points onto image plane of a specific camera.
        
        Args:
            points_3d: Array of 3D points (N, 3)
            image: COLMAP Image object
            camera: COLMAP Camera object
            
        Returns:
            Tuple of (u_coords, v_coords, depths)
        """
        # Get camera parameters
        R = qvec2rotmat(image.qvec)
        t = image.tvec.reshape(3, 1)
        
        # Parse intrinsics
        if camera.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = camera.params[0]
            cx, cy = camera.params[1], camera.params[2]
        elif camera.model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
            fx, fy = camera.params[0], camera.params[1]
            cx, cy = camera.params[2], camera.params[3]
        else:
            raise ValueError(f"Unsupported camera model: {camera.model}")
        
        # Transform points to camera coordinates
        points_cam = (R @ points_3d.T + t).T
        
        # Extract depths
        depths = points_cam[:, 2]
        
        # Project to image plane
        u = (fx * points_cam[:, 0] / points_cam[:, 2]) + cx
        v = (fy * points_cam[:, 1] / points_cam[:, 2]) + cy
        
        return u, v, depths
    
    def project_masks_to_pointcloud(self, min_votes: int = 2, 
                                    keep_closest: bool = True) -> Dict[str, Any]:
        """
        Project all wound masks onto the 3D pointcloud.
        
        For each camera with a mask:
        1. Project all 3D points onto the image
        2. Check which points fall inside the wound mask
        3. Track vote count and minimum depth for each point
        
        Args:
            min_votes: Minimum number of cameras that must see a point as "wound"
            keep_closest: If True, use minimum depth across cameras
            
        Returns:
            Dict with projection results
        """
        if not self.model_loaded:
            raise ValueError("COLMAP model must be loaded first")
        if not self.masks_loaded:
            raise ValueError("Masks must be loaded first")
        
        print(f"Projecting masks onto pointcloud...")
        
        # Get all point coordinates
        point_coords = np.array([p.xyz for p in self.points3D.values()])
        point_ids = list(self.points3D.keys())
        
        # Reset tracking
        self.wound_point_votes = {}
        self.wound_point_depths = {}
        
        # Process each image with a mask
        images_processed = 0
        for img_name, mask in self.masks.items():
            # Find corresponding image
            img = None
            for image in self.images.values():
                if image.name == img_name:
                    img = image
                    break
            
            if img is None:
                print(f"Warning: Image {img_name} not found in COLMAP model")
                continue
            
            camera = self.cameras[img.camera_id]
            
            # Project points to this camera
            u, v, depths = self.project_points_to_image(point_coords, img, camera)
            
            # Filter points that are in front of camera and within image bounds
            valid_mask = (
                (depths > 0) &
                (u >= 0) & (u < camera.width) &
                (v >= 0) & (v < camera.height)
            )
            
            # Check which valid points fall inside the wound mask
            for i, point_id in enumerate(point_ids):
                if not valid_mask[i]:
                    continue
                
                # Sample mask at projected location
                ui, vi = int(u[i]), int(v[i])
                mask_value = mask[vi, ui]
                
                # If point is inside wound region (mask value > 127)
                if mask_value > 127:
                    # Update vote count
                    self.wound_point_votes[point_id] = self.wound_point_votes.get(point_id, 0) + 1
                    
                    # Update minimum depth
                    if point_id not in self.wound_point_depths:
                        self.wound_point_depths[point_id] = depths[i]
                    elif keep_closest:
                        self.wound_point_depths[point_id] = min(
                            self.wound_point_depths[point_id], depths[i]
                        )
            
            images_processed += 1
        
        # Filter by minimum votes
        self.wound_point_indices = {
            idx for idx, votes in self.wound_point_votes.items()
            if votes >= min_votes
        }
        
        self.projection_completed = True
        
        print(f"Processed {images_processed} images")
        print(f"Found {len(self.wound_point_indices)} wound points (with >={min_votes} votes)")
        
        # Compute vote distribution
        vote_distribution = defaultdict(int)
        for votes in self.wound_point_votes.values():
            vote_distribution[votes] += 1
        
        return {
            'total_points': len(self.points3D),
            'wound_points': len(self.wound_point_indices),
            'images_processed': images_processed,
            'vote_distribution': dict(vote_distribution),
            'min_votes_threshold': min_votes
        }
    
    def get_wound_point_coordinates(self) -> np.ndarray:
        """
        Get 3D coordinates of all wound points.
        
        Returns:
            Array of wound point coordinates (N, 3)
        """
        if not self.projection_completed:
            raise ValueError("Mask projection must be completed first")
        
        wound_coords = np.array([
            self.points3D[idx].xyz for idx in self.wound_point_indices
        ])
        
        return wound_coords
    
    def get_wound_center(self) -> np.ndarray:
        """
        Compute the center of the wound region.
        
        Returns:
            3D coordinates of wound center
        """
        wound_coords = self.get_wound_point_coordinates()
        return np.median(wound_coords, axis=0)


class AdaptiveDensityFilter:
    """
    Voxel-based adaptive density filtering with local wound preservation.
    
    This class implements density-based point cloud filtering using adaptive voxels:
    - Creates adaptive voxel grid (smaller near wounds, larger far away)
    - Classifies voxels as wound/non-wound based on point content and adjacency
    - Applies density reduction based on distance from nearest wound voxel
    - Supports multiple wounds independently
    """
    
    def __init__(self):
        """Initialize the AdaptiveDensityFilter."""
        self.voxel_grid = {}  # voxel_key -> list of point indices
        self.voxel_sizes = {}  # voxel_key -> voxel size
        self.wound_voxels = set()  # Set of voxel keys containing wound points
        self.voxel_centers = {}  # voxel_key -> voxel center coordinates
        
    def _get_voxel_size(self, point: np.ndarray, wound_coords: np.ndarray, 
                       base_size: float, max_size: float) -> float:
        """
        Compute adaptive voxel size based on distance to nearest wound point.
        
        Args:
            point: Point coordinates (3,)
            wound_coords: All wound point coordinates (N, 3)
            base_size: Minimum voxel size (near wounds)
            max_size: Maximum voxel size (far from wounds)
            
        Returns:
            Voxel size for this point
        """
        # Find distance to nearest wound point
        distances = np.linalg.norm(wound_coords - point, axis=1)
        min_distance = np.min(distances)
        
        # Linear interpolation: base_size at distance 0, max_size at distance > threshold
        threshold_distance = 0.5  # Distance at which voxel size reaches max
        
        if min_distance <= threshold_distance:
            size = base_size + (max_size - base_size) * (min_distance / threshold_distance)
        else:
            size = max_size
        
        return size
    
    def _get_voxel_key(self, point: np.ndarray, voxel_size: float) -> Tuple[int, int, int]:
        """
        Get voxel key (grid coordinates) for a point.
        
        Args:
            point: Point coordinates (3,)
            voxel_size: Size of voxel
            
        Returns:
            Tuple of (x, y, z) voxel grid coordinates
        """
        return tuple((point / voxel_size).astype(int))
    
    def _get_voxel_neighbors(self, voxel_key: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """
        Get all 26 neighboring voxels (3x3x3 neighborhood, excluding center).
        
        Args:
            voxel_key: Voxel grid coordinates (x, y, z)
            
        Returns:
            List of neighboring voxel keys
        """
        x, y, z = voxel_key
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbors.append((x + dx, y + dy, z + dz))
        return neighbors
    
    def build_adaptive_voxel_grid(self, point_coords: np.ndarray, wound_coords: np.ndarray,
                                 base_size: float = 0.005, max_size: float = 0.02) -> None:
        """
        Build adaptive voxel grid where voxel size depends on distance to wounds.
        
        Args:
            point_coords: All point coordinates (N, 3)
            wound_coords: Wound point coordinates (M, 3)
            base_size: Minimum voxel size near wounds (meters)
            max_size: Maximum voxel size far from wounds (meters)
        """
        print(f"Building adaptive voxel grid (base_size={base_size}, max_size={max_size})...")
        
        # For simplicity, use uniform voxel size but apply adaptive density
        # (True adaptive sizing requires hierarchical grid, which is more complex)
        # We'll use base_size as the universal size and implement adaptivity through density
        voxel_size = base_size
        
        # Assign points to voxels
        for i, point in enumerate(point_coords):
            voxel_key = self._get_voxel_key(point, voxel_size)
            
            if voxel_key not in self.voxel_grid:
                self.voxel_grid[voxel_key] = []
                self.voxel_sizes[voxel_key] = voxel_size
                # Compute voxel center
                self.voxel_centers[voxel_key] = (np.array(voxel_key) + 0.5) * voxel_size
            
            self.voxel_grid[voxel_key].append(i)
        
        # Identify wound voxels (voxels containing wound points)
        for wound_point in wound_coords:
            voxel_key = self._get_voxel_key(wound_point, voxel_size)
            if voxel_key in self.voxel_grid:
                self.wound_voxels.add(voxel_key)
        
        # Expand wound voxels to include neighbors
        expanded_wound_voxels = set(self.wound_voxels)
        for wound_voxel in self.wound_voxels:
            for neighbor in self._get_voxel_neighbors(wound_voxel):
                if neighbor in self.voxel_grid:
                    expanded_wound_voxels.add(neighbor)
        
        self.wound_voxels = expanded_wound_voxels
        
        print(f"  Created {len(self.voxel_grid)} voxels")
        print(f"  Wound voxels: {len(self.wound_voxels)}")
        print(f"  Non-wound voxels: {len(self.voxel_grid) - len(self.wound_voxels)}")
    
    def filter_pointcloud(self, point_coords: np.ndarray, 
                         wound_point_indices: Set[int],
                         density_wound: float = 1.0,
                         density_min: float = 0.05,
                         falloff_radius_multiplier: float = 3.0,
                         falloff_type: str = "linear",
                         base_voxel_size: float = 0.005,
                         seed: int = 42) -> np.ndarray:
        """
        Apply voxel-based adaptive density filtering to the point cloud.
        
        The filtering process:
        1. Build adaptive voxel grid
        2. Classify voxels as wound/non-wound (with adjacency expansion)
        3. For each voxel, compute distance to nearest wound voxel
        4. Apply retention probability based on distance with falloff
        5. Randomly sample points within each voxel according to probability
        
        Args:
            point_coords: All point coordinates (N, 3)
            wound_point_indices: Set of indices belonging to wound region
            density_wound: Retention ratio for wound voxels (1.0 = keep all)
            density_min: Minimum retention ratio for far voxels
            falloff_radius_multiplier: How far the falloff extends (in voxel units)
            falloff_type: Type of falloff function ("linear", "exp", "squared", "step")
                         - "linear": Linear gradient from wound to far
                         - "exp": Exponential decay (fast near wound, slow far)
                         - "squared": Quadratic decay (slow near wound, fast far)
                         - "step": No falloff, everything outside wound at density_min
            base_voxel_size: Base size of voxels (meters)
            seed: Random seed for reproducibility
            
        Returns:
            Array of indices to keep
        """
        # Build voxel grid
        wound_indices_list = list(wound_point_indices)
        wound_coords = point_coords[wound_indices_list]
        
        if len(wound_coords) == 0:
            print("Warning: No wound points found, keeping all points")
            return np.arange(len(point_coords))
        
        self.build_adaptive_voxel_grid(point_coords, wound_coords, base_size=base_voxel_size)
        
        # Compute distance from each voxel center to nearest wound voxel center (KD-TREE)
        print("Computing voxel distances to wounds (using KD-tree)...")
        
        # Convert to arrays for KD-tree
        all_voxel_keys = list(self.voxel_grid.keys())
        wound_voxel_keys = list(self.wound_voxels)
        
        # Create array of wound voxel centers (M, 3)
        wound_centers = np.array([self.voxel_centers[k] for k in wound_voxel_keys])
        
        # Build KD-tree from wound voxel centers
        wound_tree = cKDTree(wound_centers)
        
        # Query nearest wound voxel for each voxel (vectorized, memory-efficient)
        all_centers = np.array([self.voxel_centers[k] for k in all_voxel_keys])
        min_distances, _ = wound_tree.query(all_centers)
        
        # Build dictionary
        voxel_distances = {k: float(d) for k, d in zip(all_voxel_keys, min_distances)}
        
        # Calculate max distance for normalization
        max_distance = float(np.max(min_distances)) if len(min_distances) > 0 else 1.0
        falloff_distance = max_distance * falloff_radius_multiplier
        
        # Calculate retention probability for each voxel based on distance to nearest wound
        voxel_retention_probs = {}
        for voxel_key, distance in voxel_distances.items():
            if distance == 0.0:
                # Wound voxel
                retention_prob = density_wound
            elif falloff_type == "step":
                # Step function: everything outside wound at density_min
                retention_prob = density_min
            elif distance >= falloff_distance:
                # Beyond falloff distance
                retention_prob = density_min
            else:
                # In falloff zone
                normalized_distance = distance / falloff_distance  # 0 at wound, 1 at falloff edge
                
                if falloff_type == "linear":
                    # Linear falloff
                    falloff_factor = normalized_distance
                elif falloff_type == "exp":
                    # Exponential decay
                    falloff_factor = 1.0 - np.exp(-3.0 * normalized_distance)
                elif falloff_type == "squared":
                    # Quadratic falloff
                    falloff_factor = normalized_distance ** 2
                else:
                    raise ValueError(f"Unknown falloff type: {falloff_type}")
                
                # Interpolate between density_wound and density_min
                retention_prob = density_wound + (density_min - density_wound) * falloff_factor
            
            voxel_retention_probs[voxel_key] = retention_prob
        
        # Apply random sampling within each voxel based on retention probability
        rng = np.random.default_rng(seed)
        kept_indices = []
        
        for voxel_key, point_indices in self.voxel_grid.items():
            retention_prob = voxel_retention_probs[voxel_key]
            
            # Sample points within this voxel
            for point_idx in point_indices:
                if rng.random() < retention_prob:
                    kept_indices.append(point_idx)
        
        kept_indices = np.array(kept_indices, dtype=int)
        
        print(f"\nDensity filtering results:")
        print(f"  Original points: {len(point_coords)}")
        print(f"  Kept points: {len(kept_indices)}")
        print(f"  Reduction: {100 * (1 - len(kept_indices)/len(point_coords)):.1f}%")
        
        # Count wound points kept
        kept_set = set(kept_indices)
        wound_points_kept = len(kept_set.intersection(wound_point_indices))
        print(f"  Wound points kept: {wound_points_kept}/{len(wound_point_indices)}")
        
        return kept_indices


def reduce_dense_pointcloud(dense_model_path: Path,
                            mask_dir: Path,
                            output_path: Path = None,
                            density_wound: float = 0.8,
                            density_min: float = 0.05,
                            min_votes: int = 2,
                            falloff_radius_multiplier: float = 3.0,
                            falloff_type: str = "linear",
                            base_voxel_size: float = 0.005,
                            remove_distant_outliers: bool = True,
                            outlier_percentile: float = 0.1,
                            visualize: bool = False) -> Dict[str, Any]:
    """
    Main workflow for dense pointcloud reduction based on wound masks using voxel-based filtering.
    
    This function:
    1. Projects 2D wound masks onto 3D pointcloud
    2. (Optional) Removes distant outliers while protecting wound points
    3. Builds adaptive voxel grid
    4. Applies density reduction based on voxel distance to wounds
    
    Args:
        dense_model_path: Path to COLMAP dense model directory
        mask_dir: Directory containing wound masks
        output_path: Optional path to save reduced pointcloud (as .ply or COLMAP format)
        density_wound: Retention ratio for wound voxels (0.0-1.0)
        density_min: Minimum retention ratio for far voxels (0.0-1.0)
        min_votes: Minimum cameras that must see a point as "wound"
        falloff_radius_multiplier: Distance multiplier for density falloff
        falloff_type: Type of falloff function ("linear", "exp", "squared", "step")
                      - "linear": Linear gradient from wound to far
                      - "exp": Exponential decay (fast near wound, slow far)
                      - "squared": Quadratic decay (slow near wound, fast far)
                      - "step": No falloff, everything outside wound at density_min
        base_voxel_size: Base size of voxels in meters (default: 0.005 = 5mm)
        remove_distant_outliers: If True, remove farthest points from wound center (default: True)
        outlier_percentile: Fraction of farthest points to remove (default: 0.1 = 10%)
        visualize: If True, launch Rerun visualization
        
    Returns:
        Dict with processing results and statistics
    """
    print("=" * 60)
    print("Dense Pointcloud Reduction with Wound Mask Projection")
    print("=" * 60)
    
    # Step 1: Project wound masks onto pointcloud
    print("\n--- Step 1: Projecting Wound Masks ---")
    projector = WoundMaskProjector()
    projector.load_colmap_model(dense_model_path)
    projector.load_masks(mask_dir)
    
    projection_stats = projector.project_masks_to_pointcloud(
        min_votes=min_votes,
        keep_closest=True
    )
    
    # Get all point coordinates
    point_coords = np.array([p.xyz for p in projector.points3D.values()])
    point_ids = list(projector.points3D.keys())
    
    # Step 2: Remove distant outliers (optional)
    outlier_removal_stats = None
    if remove_distant_outliers:
        print("\n--- Step 2: Removing Distant Outliers ---")
        
        # Get wound center
        wound_coords = projector.get_wound_point_coordinates()
        wound_center = np.median(wound_coords, axis=0)
        
        # Calculate distances from all points to wound center
        distances = np.linalg.norm(point_coords - wound_center, axis=1)
        
        # Find threshold distance (top outlier_percentile%)
        distance_threshold = np.percentile(distances, (1 - outlier_percentile) * 100)
        
        # Identify outliers (farthest points)
        is_outlier = distances > distance_threshold
        
        # Don't remove wound points
        is_wound = np.array([pid in projector.wound_point_indices for pid in point_ids])
        can_remove = is_outlier & ~is_wound
        
        # Filter out the outliers
        keep_mask = ~can_remove
        kept_point_ids = [pid for i, pid in enumerate(point_ids) if keep_mask[i]]
        
        num_removed = np.sum(can_remove)
        num_protected = np.sum(is_outlier & is_wound)
        
        print(f"  Wound center: {wound_center}")
        print(f"  Distance threshold (top {outlier_percentile*100:.1f}%): {distance_threshold:.4f}")
        print(f"  Points beyond threshold: {np.sum(is_outlier)}")
        print(f"  Removed (non-wound): {num_removed}")
        print(f"  Protected (wound): {num_protected}")
        print(f"  Remaining points: {len(kept_point_ids)}/{len(point_ids)}")
        
        # Update points3D dictionary
        projector.points3D = {pid: projector.points3D[pid] for pid in kept_point_ids}
        
        # Update coordinates and IDs
        point_coords = np.array([p.xyz for p in projector.points3D.values()])
        point_ids = list(projector.points3D.keys())
        
        outlier_removal_stats = {
            'removed': num_removed,
            'protected': num_protected,
            'remaining': len(kept_point_ids),
            'distance_threshold': float(distance_threshold)
        }
    
    # Step 3: Apply voxel-based adaptive density filtering
    print("\n--- Step 3: Voxel-Based Adaptive Density Filtering ---")
    filter_engine = AdaptiveDensityFilter()
    
    # Convert wound point indices to array indices
    wound_array_indices = {
        i for i, pid in enumerate(point_ids) if pid in projector.wound_point_indices
    }
    
    kept_indices = filter_engine.filter_pointcloud(
        point_coords,
        wound_array_indices,
        density_wound=density_wound,
        density_min=density_min,
        falloff_radius_multiplier=falloff_radius_multiplier,
        falloff_type=falloff_type,
        base_voxel_size=base_voxel_size
    )
    
    # Step 4: Extract filtered points
    print("\n--- Step 4: Extracting Filtered Points ---")
    filtered_point_ids = [point_ids[i] for i in kept_indices]
    filtered_points = {pid: projector.points3D[pid] for pid in filtered_point_ids}
    
    # Step 5: Save output (optional)
    if output_path is not None:
        print(f"\n--- Step 5: Saving Output to {output_path} ---")
        output_path = Path(output_path)
        
        if output_path.suffix == '.ply':
            save_points_to_ply(filtered_points, output_path)
        else:
            # Save as COLMAP format
            output_path.mkdir(parents=True, exist_ok=True)
            # Note: Would need to implement COLMAP binary writing
            print("Saving to COLMAP format not yet implemented")
    
    # Step 6: Visualization (optional)
    if visualize:
        print("\n--- Step 6: Launching Visualization ---")
        # Compute wound center for visualization
        wound_indices_list = list(wound_array_indices)
        wound_coords_viz = point_coords[wound_indices_list]
        wound_center = np.median(wound_coords_viz, axis=0)
        
        # Compute wound radius as 95th percentile
        distances_to_center = np.linalg.norm(wound_coords_viz - wound_center, axis=1)
        wound_radius = np.percentile(distances_to_center, 95)
        
        visualize_reduction(
            point_coords,
            wound_array_indices,
            kept_indices,
            wound_center,
            wound_radius,
            voxel_grid=filter_engine.voxel_grid,
            wound_voxels=filter_engine.wound_voxels,
            voxel_centers=filter_engine.voxel_centers,
            voxel_size=base_voxel_size
        )
    
    # Compile results
    wound_indices_list = list(wound_array_indices)
    wound_coords_result = point_coords[wound_indices_list]
    wound_center_result = np.median(wound_coords_result, axis=0)
    
    results = {
        'projection_stats': projection_stats,
        'outlier_removal_stats': outlier_removal_stats,
        'filtering_stats': {
            'original_points': len(point_coords),
            'filtered_points': len(kept_indices),
            'reduction_percentage': 100 * (1 - len(kept_indices)/len(point_coords)),
            'wound_center': wound_center_result.tolist(),
            'num_voxels': len(filter_engine.voxel_grid),
            'num_wound_voxels': len(filter_engine.wound_voxels),
        },
        'parameters': {
            'density_wound': density_wound,
            'density_min': density_min,
            'min_votes': min_votes,
            'falloff_radius_multiplier': falloff_radius_multiplier,
            'falloff_type': falloff_type,
            'remove_distant_outliers': remove_distant_outliers,
            'outlier_percentile': outlier_percentile,
        },
        'filtered_points': filtered_points
    }
    
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    
    return results


def save_points_to_ply(points_dict: Dict[int, Point3D], output_path: Path):
    """
    Save points to a PLY file.
    
    Args:
        points_dict: Dictionary of point_id -> Point3D
        output_path: Path to save PLY file
    """
    try:
        import open3d as o3d
        
        # Extract coordinates and colors
        coords = np.array([p.xyz for p in points_dict.values()])
        colors = np.array([p.rgb for p in points_dict.values()])
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        
        if colors is not None and len(colors) > 0:
            colors_normalized = colors.astype(np.float64)
            if colors_normalized.max() > 1.0:
                colors_normalized = colors_normalized / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
        
        o3d.io.write_point_cloud(str(output_path), pcd)
        print(f"Saved {len(coords)} points to: {output_path}")
        
    except ImportError:
        # Fallback to manual PLY writing
        _write_ply_manual(points_dict, output_path)


def _write_ply_manual(points_dict: Dict[int, Point3D], output_path: Path):
    """Manually write points to PLY file without Open3D."""
    coords = np.array([p.xyz for p in points_dict.values()])
    colors = np.array([p.rgb for p in points_dict.values()])
    
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(coords)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(len(coords)):
            r, g, b = colors[i].astype(int)
            f.write(f"{coords[i, 0]} {coords[i, 1]} {coords[i, 2]} {r} {g} {b}\n")
    
    print(f"Saved {len(coords)} points to: {output_path}")


def visualize_reduction(point_coords: np.ndarray,
                       wound_point_indices: Set[int],
                       kept_indices: np.ndarray,
                       wound_center: np.ndarray,
                       wound_radius: float,
                       voxel_grid: Dict = None,
                       wound_voxels: Set = None,
                       voxel_centers: Dict = None,
                       voxel_size: float = 0.005):
    """
    Visualize the reduction process using Rerun.
    
    Args:
        point_coords: All point coordinates
        wound_point_indices: Indices of wound points
        kept_indices: Indices of points that were kept
        wound_center: Center of wound region
        wound_radius: Radius of wound region
        voxel_grid: Optional voxel grid dictionary {voxel_key -> [point_indices]}
        wound_voxels: Optional set of wound voxel keys
        voxel_centers: Optional dictionary {voxel_key -> center}
        voxel_size: Size of voxels (for bounding box visualization)
    """
    try:
        import rerun as rr
    except ImportError:
        print("Rerun not available. Install with: pip install rerun-sdk")
        return
    
    try:
        rr.init("Dense_Pointcloud_Reduction", spawn=True)
    except RuntimeError as e:
        print(f"Could not launch Rerun viewer: {e}")
        return
    
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    
    # Log all points (original)
    rr.log("/original/all_points", rr.Points3D(
        point_coords,
        colors=[128, 128, 128],
        radii=0.001
    ), static=True)
    
    # Log wound points
    wound_indices_list = list(wound_point_indices)
    wound_coords = point_coords[wound_indices_list]
    rr.log("/original/wound_points", rr.Points3D(
        wound_coords,
        colors=[255, 0, 0],
        radii=0.002
    ), static=True)
    
    # Log wound center and radius sphere
    rr.log("/wound_region/center", rr.Points3D(
        [wound_center],
        colors=[0, 255, 0],
        radii=0.01
    ), static=True)
    
    # Log kept points
    kept_coords = point_coords[kept_indices]
    rr.log("/filtered/kept_points", rr.Points3D(
        kept_coords,
        colors=[100, 150, 255],
        radii=0.001
    ), static=True)
    
    # Log removed points
    all_indices = set(range(len(point_coords)))
    removed_indices = all_indices - set(kept_indices)
    removed_coords = point_coords[list(removed_indices)]
    rr.log("/filtered/removed_points", rr.Points3D(
        removed_coords,
        colors=[255, 100, 100],
        radii=0.0005
    ), static=True)
    
    # Log voxel bounding boxes if provided
    if voxel_grid is not None and voxel_centers is not None:
        print(f"Visualizing {len(voxel_grid)} voxels...")
        
        # Visualize wound voxels
        if wound_voxels is not None and len(wound_voxels) > 0:
            wound_voxel_centers = np.array([voxel_centers[k] for k in wound_voxels if k in voxel_centers])
            if len(wound_voxel_centers) > 0:
                # Create bounding boxes for wound voxels
                half_size = voxel_size / 2.0
                for center in wound_voxel_centers:
                    rr.log("/voxels/wound_voxels", rr.Boxes3D(
                        sizes=[voxel_size, voxel_size, voxel_size],
                        centers=[center],
                        colors=[255, 0, 0, 100]  # Semi-transparent red
                    ))
        
        # Sample non-wound voxels for visualization (to avoid clutter)
        non_wound_voxels = [k for k in voxel_grid.keys() if k not in (wound_voxels or set())]
        sample_size = min(1000, len(non_wound_voxels))  # Show max 1000 non-wound voxels
        if len(non_wound_voxels) > sample_size:
            import random
            sampled_voxels = random.sample(non_wound_voxels, sample_size)
        else:
            sampled_voxels = non_wound_voxels
        
        if len(sampled_voxels) > 0:
            sampled_centers = np.array([voxel_centers[k] for k in sampled_voxels])
            rr.log("/voxels/non_wound_voxels", rr.Boxes3D(
                sizes=[[voxel_size, voxel_size, voxel_size]] * len(sampled_centers),
                centers=sampled_centers,
                colors=[100, 100, 255, 50],
            ), static=True)

    print("Rerun visualization launched")