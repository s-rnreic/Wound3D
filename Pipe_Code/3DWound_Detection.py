"""
3D Wound Edge Detection - Clean Structured Version

This module provides a clean, structured approach to 3D wound segmentation from COLMAP dense point clouds.
It uses a modular design with separate classes for cameras (Spectator) and the overall model (3DWoundModel).

Classes:
    DenseVoxel: Superpixel-like voxel structure for local 3D point cloud analysis
    Spectator: Handles individual camera information, images, masks, and visibility
    WoundModel3D: Main class for loading dense point clouds and managing 3D wound segmentation

Author: Your Name
Date: November 2025
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
import cv2

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from CODE_Evaluation_Viewer_Align.Transformation import *
from read_write_model import read_model, qvec2rotmat
from CODE_Evaluation_Viewer_Align.read_write_dense import read_array


class DenseVoxel:
    """
    A superpixel-like voxel structure for local 3D point cloud analysis.
    
    This class represents a volumetric region containing multiple 3D points and
    computes local statistical properties like mean normal vectors, color statistics,
    and geometric properties for analysis and segmentation.
    """
    
    def __init__(self, voxel_id: int, center: np.ndarray = None):
        """
        Initialize a DenseVoxel.
        
        Args:
            voxel_id: Unique identifier for this voxel
            center: Initial center position (3D coordinates)
        """
        self.voxel_id = voxel_id
        self.center = center if center is not None else np.zeros(3)
        
        # Point data
        self.point_indices = []  # Indices of points belonging to this voxel
        self.points_3d = None    # Actual 3D coordinates (N, 3)
        self.points_colors = None # Point colors (N, 3)
        self.points_normals = None # Point normals (N, 3) - if available
        
        # Computed properties
        self.mean_position = None
        self.mean_normal = None
        self.normal_deviation = None  # Angular deviation in radians
        self.mean_color = None
        self.color_deviation = None   # RGB standard deviation
        self.volume = 0.0
        self.density = 0.0  # points per unit volume
        
        # Geometric properties
        self.bounding_box_min = None
        self.bounding_box_max = None
        self.principal_axes = None  # PCA axes for shape analysis
        
        # Convergence tracking
        self.iteration_centers = []  # Track center movement during optimization
        self.energy = float('inf')   # Optimization energy
        
        # Validity flags
        self.properties_computed = False
        self.is_valid = True
    
    def add_points(self, point_indices: List[int], points_3d: np.ndarray, 
                   points_colors: np.ndarray = None, points_normals: np.ndarray = None):
        """
        Add points to this voxel.
        
        Args:
            point_indices: List of point indices in the global point cloud
            points_3d: 3D coordinates of points (N, 3)
            points_colors: RGB colors of points (N, 3)
            points_normals: Normal vectors of points (N, 3)
        """
        self.point_indices = point_indices
        self.points_3d = points_3d.copy() if points_3d is not None else None
        self.points_colors = points_colors.copy() if points_colors is not None else None
        self.points_normals = points_normals.copy() if points_normals is not None else None
        
        # Reset computed properties
        self.properties_computed = False
        
        # Update validity
        self.is_valid = len(self.point_indices) > 0
    
    def compute_properties(self):
        """Compute all statistical and geometric properties of the voxel."""
        if not self.is_valid or self.points_3d is None or len(self.points_3d) == 0:
            return
        
        # Basic statistics
        self.mean_position = np.mean(self.points_3d, axis=0)
        
        # Bounding box
        self.bounding_box_min = np.min(self.points_3d, axis=0)
        self.bounding_box_max = np.max(self.points_3d, axis=0)
        
        # Volume approximation (bounding box volume)
        box_dims = self.bounding_box_max - self.bounding_box_min
        self.volume = np.prod(box_dims) if np.all(box_dims > 0) else 1e-6
        self.density = len(self.points_3d) / self.volume
        
        # Color statistics
        if self.points_colors is not None:
            self.mean_color = np.mean(self.points_colors, axis=0)
            self.color_deviation = np.std(self.points_colors, axis=0)
        
        # Normal vector statistics
        if self.points_normals is not None:
            self._compute_normal_statistics()
        
        # Principal component analysis for shape
        self._compute_principal_axes()
        
        self.properties_computed = True
    
    def _compute_normal_statistics(self):
        """Compute normal vector statistics including angular deviations."""
        if self.points_normals is None or len(self.points_normals) == 0:
            return
        
        # Compute mean normal (normalize to unit vector)
        mean_normal_raw = np.mean(self.points_normals, axis=0)
        norm = np.linalg.norm(mean_normal_raw)
        if norm > 1e-6:
            self.mean_normal = mean_normal_raw / norm
        else:
            self.mean_normal = np.array([0, 0, 1])  # Default up vector
        
        # Compute angular deviations
        if len(self.points_normals) > 1:
            # Calculate angle between each normal and mean normal
            dot_products = np.dot(self.points_normals, self.mean_normal)
            # Clamp to [-1, 1] to avoid numerical issues with arccos
            dot_products = np.clip(dot_products, -1.0, 1.0)
            angles = np.arccos(np.abs(dot_products))  # abs for undirected normals
            
            # Statistical measures of angular deviation
            self.normal_deviation = {
                'mean_angle': np.mean(angles),
                'std_angle': np.std(angles),
                'max_angle': np.max(angles),
                'angle_variance': np.var(angles)
            }
        else:
            self.normal_deviation = {
                'mean_angle': 0.0,
                'std_angle': 0.0, 
                'max_angle': 0.0,
                'angle_variance': 0.0
            }
    
    def _compute_principal_axes(self):
        """Compute principal axes using PCA for shape analysis."""
        if self.points_3d is None or len(self.points_3d) < 3:
            return
        
        # Center the points
        centered_points = self.points_3d - self.mean_position
        
        # Compute covariance matrix
        cov_matrix = np.cov(centered_points.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue magnitude (largest first)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        self.principal_axes = {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'shape_ratios': eigenvalues / (eigenvalues[0] + 1e-6)  # Normalized shape
        }
    
    def get_energy(self, points_3d: np.ndarray, points_colors: np.ndarray = None, 
                   w_spatial: float = 1.0, w_color: float = 0.5) -> float:
        """
        Compute energy function for voxel optimization.
        
        Args:
            points_3d: Candidate points for this voxel
            points_colors: Candidate point colors
            w_spatial: Weight for spatial distance term
            w_color: Weight for color similarity term
            
        Returns:
            Energy value (lower is better)
        """
        if len(points_3d) == 0:
            return float('inf')
        
        # Spatial energy: variance from center
        spatial_distances = np.linalg.norm(points_3d - self.center, axis=1)
        spatial_energy = np.var(spatial_distances)
        
        total_energy = w_spatial * spatial_energy
        
        # Color energy: color variance
        if points_colors is not None and w_color > 0:
            color_variance = np.mean(np.var(points_colors, axis=0))
            total_energy += w_color * color_variance
        
        return total_energy
    
    def update_center(self, new_center: np.ndarray):
        """Update voxel center and track for convergence."""
        self.iteration_centers.append(self.center.copy())
        self.center = new_center.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of voxel properties."""
        summary = {
            'voxel_id': self.voxel_id,
            'num_points': len(self.point_indices),
            'center': self.center.tolist() if self.center is not None else None,
            'mean_position': self.mean_position.tolist() if self.mean_position is not None else None,
            'volume': self.volume,
            'density': self.density,
            'is_valid': self.is_valid,
            'properties_computed': self.properties_computed
        }
        
        if self.mean_color is not None:
            summary['mean_color'] = self.mean_color.tolist()
            summary['color_deviation'] = self.color_deviation.tolist()
        
        if self.mean_normal is not None:
            summary['mean_normal'] = self.mean_normal.tolist()
            summary['normal_deviation'] = self.normal_deviation
        
        if self.principal_axes is not None:
            summary['shape_ratios'] = self.principal_axes['shape_ratios'].tolist()
        
        return summary


class Spectator:
    """
    A class representing a camera viewpoint in the 3D scene.
    
    The Spectator holds all information related to a specific camera position:
    - Camera intrinsics and extrinsics
    - Image data and corresponding wound mask
    - Visibility information for 3D points
    - Filtering statistics
    """
    
    def __init__(self, image_id: int, image_name: str, camera_id: int):
        """
        Initialize a new Spectator (camera viewpoint).
        
        Args:
            image_id: COLMAP image ID
            image_name: Name of the image file
            camera_id: COLMAP camera ID
        """
        # Basic identifiers
        self.image_id = image_id
        self.image_name = image_name
        self.camera_id = camera_id
        
        # Camera parameters (will be set by load_camera_data)
        self.rotation_matrix = None  # 3x3 rotation matrix
        self.translation_vector = None  # 3x1 translation vector
        self.intrinsics = None  # Camera intrinsics dict with fx, fy, cx, cy
        self.image_width = None
        self.image_height = None
        
        # Image and mask data
        self.image_data = None  # Original image (optional)
        self.mask_data = None  # Wound mask array
        self.mask_loaded = False
        
        # Depth map data
        self.depth_map = None  # Depth map array
        self.depth_map_loaded = False
        
        # Point visibility tracking
        self.visible_point_indices = []  # List of 3D point indices visible through mask
        self.filtered_out_indices = {}  # Dict: filter_name -> list of filtered indices
        self.depth_statistics = {}  # Statistical information about point depths
        
        # Processing flags
        self.camera_data_loaded = False
        self.points_processed = False
    
    def load_camera_data(self, colmap_image, colmap_camera):
        """
        Load camera intrinsics and extrinsics from COLMAP data.
        
        Args:
            colmap_image: COLMAP image object
            colmap_camera: COLMAP camera object
        """
        # Extract camera pose
        quat = colmap_image.qvec
        tvec = colmap_image.tvec
        self.rotation_matrix = qvec2rotmat(quat)
        self.translation_vector = tvec.reshape(3, 1)
        
        # Extract camera intrinsics
        self.intrinsics = self._parse_camera_intrinsics(colmap_camera)
        self.image_width = colmap_camera.width
        self.image_height = colmap_camera.height
        
        self.camera_data_loaded = True
        print(f"Loaded camera data for {self.image_name}")
    
    def _parse_camera_intrinsics(self, cam) -> Dict[str, float]:
        """Parse camera intrinsics from COLMAP camera model."""
        if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = cam.params[0]
            cx, cy = cam.params[1], cam.params[2]
        elif cam.model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
            fx, fy = cam.params[0], cam.params[1]
            cx, cy = cam.params[2], cam.params[3]
        else:
            raise ValueError(f"Unsupported camera model: {cam.model}")
        
        return {
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            'model': cam.model, 'width': cam.width, 'height': cam.height
        }
    
    def load_mask(self, mask_dir: Path) -> bool:
        """
        Load wound mask for this camera view.
        
        Args:
            mask_dir: Directory containing mask files
            
        Returns:
            bool: True if mask was loaded successfully
        """
        mask_dir = Path(mask_dir)
        
        # Try different mask file naming conventions
        possible_names = [
            # exact filename with an extra .png appended (e.g. "image.jpg.png")
            f"{self.image_name}.png",
            # explicit replacements for common cases
            self.image_name.replace('.jpg', '.jpg.png'),
            self.image_name.replace('.png', '.png.png'),
            # other common mask naming schemes
        ]
        
        for mask_name in possible_names:
            mask_path = mask_dir / mask_name
            if mask_path.exists():
                try:
                    self.mask_data = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if self.mask_data is not None:
                        self.mask_loaded = True
                        print(f"Loaded mask for {self.image_name}: {mask_path}")
                        return True
                except Exception as e:
                    print(f"Error loading mask {mask_path}: {e}")
        
        print(f"Warning: No mask found for {self.image_name}")
        return False
    
    def load_depth_map(self, depth_dir: Path) -> bool:
        """
        Load depth map for this camera view.
        
        Args:
            depth_dir: Directory containing depth map files
            
        Returns:
            bool: True if depth map was loaded successfully
        """
        depth_dir = Path(depth_dir)
        
        # Try different depth map file naming conventions
        base_name = Path(self.image_name).name
        possible_names = [
            f"{base_name}.geometric.bin",
            f"{base_name}.photometric.bin"
        ]
        
        for depth_name in possible_names:
            depth_path = depth_dir / depth_name
            if depth_path.exists():
                try:
                    # Use the read_array function from your existing imports
                    self.depth_map = read_array(str(depth_path))
                    if self.depth_map is not None:
                        self.depth_map_loaded = True
                        print(f"Loaded depth map for {self.image_name}: {depth_path}")
                        return True
                except Exception as e:
                    print(f"Error loading depth map {depth_path}: {e}")
        
        print(f"Warning: No depth map found for {self.image_name}")
        return False
    
    def project_points_to_image(self, points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project 3D points to this camera's image plane.
        
        Args:
            points_3d: 3D points array of shape (N, 3)
            
        Returns:
            Tuple of (u_coords, v_coords, depths) where depths are z-coordinates in camera frame
        """
        if not self.camera_data_loaded:
            raise ValueError("Camera data must be loaded before projecting points")
        
        # Transform points to camera coordinate system
        points_cam = self.rotation_matrix @ points_3d.T + self.translation_vector  # (3, N)
        x_cam, y_cam, z_cam = points_cam
        
        # Project to image coordinates
        fx, fy, cx, cy = self.intrinsics['fx'], self.intrinsics['fy'], self.intrinsics['cx'], self.intrinsics['cy']
        u = fx * x_cam / z_cam + cx
        v = fy * y_cam / z_cam + cy
        
        return u, v, z_cam
    
    def filter_points_through_mask(self, points_3d: np.ndarray, point_colors: np.ndarray = None, 
                                 use_depth_agreement: bool = True) -> Dict[str, Any]:
        """
        Filter 3D points through this camera's mask with multiple filtering steps.
        
        Args:
            points_3d: 3D points array of shape (N, 3)
            point_colors: Optional point colors array of shape (N, 3)
            use_depth_agreement: Whether to use depth agreement filtering (requires loaded depth map)
            
        Returns:
            Dict containing filtering results and statistics
        """
        if not self.camera_data_loaded:
            raise ValueError("Camera data must be loaded before filtering points")
        if not self.mask_loaded:
            raise ValueError("Mask must be loaded before filtering points")
        
        image_constraints_filtering = True
        mask_filtering = True
        depth_filtering = True
        depth_filtering = True
        statistical_filtering = False
        
        
        # Initialize tracking
        n_points = len(points_3d)
        all_indices = np.arange(n_points)
        valid_mask = np.ones(n_points, dtype=bool)
        
        # Project points to image --> transformes 3D points to 2D image coordinates (u,v) and depths
        u, v, depths = self.project_points_to_image(points_3d)
        
        #BEGINNING OF FILTERING STEPS
        # Initialize filtering results
        filtering_results = {
            'total_points': n_points,
            'visible_indices': [],
            'filtered_steps': {},
            'depth_stats': {},
            'processing_successful': True
        }
        
        
        if image_constraints_filtering:
            # Step 1: Filter points behind camera
            behind_camera = depths <= 0
            behind_indices = all_indices[behind_camera & valid_mask]
            valid_mask &= ~behind_camera
            filtering_results['filtered_steps']['behind_camera'] = behind_indices.tolist()
            
            # Step 2: Filter points outside image bounds
            out_of_bounds = ((u < 0) | (u >= self.image_width) | 
                            (v < 0) | (v >= self.image_height))
            out_bounds_indices = all_indices[out_of_bounds & valid_mask]
            valid_mask &= ~out_of_bounds
            filtering_results['filtered_steps']['out_of_bounds'] = out_bounds_indices.tolist()
        
        if mask_filtering:    
            # Step 3: Filter points outside mask
            if np.any(valid_mask):
                valid_u = u[valid_mask].astype(int)
                valid_v = v[valid_mask].astype(int)
                valid_indices_array = all_indices[valid_mask]
                
                # Check mask values (assuming mask > 0 means wound region)
                mask_values = self.mask_data[valid_v, valid_u]
                outside_mask = mask_values == 0
                outside_mask_indices = valid_indices_array[outside_mask]
                
                # Update valid mask
                temp_valid = valid_mask.copy()
                temp_valid[valid_indices_array[outside_mask]] = False
                valid_mask = temp_valid
                
                filtering_results['filtered_steps']['outside_mask'] = outside_mask_indices.tolist()
            else:
                filtering_results['filtered_steps']['outside_mask'] = []
        
        if depth_filtering:
            # Step 4: Depth agreement filtering (if depth map available)
            if use_depth_agreement and self.depth_map_loaded and np.any(valid_mask):
                try:
                    depth_agreement_indices = self._apply_depth_agreement_filter(
                        all_indices[valid_mask], u[valid_mask], v[valid_mask], depths[valid_mask]
                    )
                    # Update valid mask
                    temp_valid = valid_mask.copy()
                    temp_valid[depth_agreement_indices] = False
                    valid_mask = temp_valid
                    filtering_results['filtered_steps']['depth_agreement'] = depth_agreement_indices.tolist()
                except Exception as e:
                    print(f"Warning: Depth agreement filtering failed for {self.image_name}: {e}")
                    filtering_results['filtered_steps']['depth_agreement'] = []
            else:
                filtering_results['filtered_steps']['depth_agreement'] = []
        
        if statistical_filtering:    
            # Step 5: Statistical depth filtering
            if np.any(valid_mask):
                valid_depths = depths[valid_mask]
                valid_indices_array = all_indices[valid_mask]
                
                # Remove statistical outliers (beyond 2 standard deviations)
                depth_mean = np.mean(valid_depths)
                depth_std = np.std(valid_depths)
                depth_outliers = np.abs(valid_depths - depth_mean) > depth_std
                outlier_indices = valid_indices_array[depth_outliers]
                
                # Update valid mask
                temp_valid = valid_mask.copy()
                temp_valid[outlier_indices] = False
                valid_mask = temp_valid
                
                filtering_results['filtered_steps']['statistical_depth'] = outlier_indices.tolist()
                
                # Calculate depth statistics
                final_depths = depths[valid_mask] if np.any(valid_mask) else []
                filtering_results['depth_stats'] = {
                    'mean_depth': float(np.mean(final_depths)) if len(final_depths) > 0 else 0.0,
                    'median_depth': float(np.median(final_depths)) if len(final_depths) > 0 else 0.0,
                    'std_depth': float(np.std(final_depths)) if len(final_depths) > 0 else 0.0,
                    'min_depth': float(np.min(final_depths)) if len(final_depths) > 0 else 0.0,
                    'max_depth': float(np.max(final_depths)) if len(final_depths) > 0 else 0.0,
                    'count': len(final_depths)
                }
            else:
                filtering_results['filtered_steps']['statistical_depth'] = []
                filtering_results['depth_stats'] = {'count': 0}
        
        # Final visible indices
        filtering_results['visible_indices'] = all_indices[valid_mask].tolist()
        
        # Store results in this spectator
        self.visible_point_indices = filtering_results['visible_indices']
        self.filtered_out_indices = filtering_results['filtered_steps']
        self.depth_statistics = filtering_results['depth_stats']
        self.points_processed = True
        
    
    def _apply_depth_agreement_filter(self, point_indices: np.ndarray,
                                    u_coords: np.ndarray, v_coords: np.ndarray, 
                                    point_depths: np.ndarray, abs_tol: float = 0.02, 
                                    rel_tol: float = 0.1) -> np.ndarray:
        """Apply depth agreement filtering using loaded depth map."""
        try:
            if not self.depth_map_loaded or self.depth_map is None:
                print("No depth map loaded for depth agreement filtering")
                return np.array([])
            
            # Sample depth map at projected coordinates
            u_int = np.clip(u_coords.astype(int), 0, self.depth_map.shape[1] - 1)
            v_int = np.clip(v_coords.astype(int), 0, self.depth_map.shape[0] - 1)
            depth_map_values = self.depth_map[v_int, u_int]
            
            # Check depth agreement
            depth_diff = np.abs(point_depths - depth_map_values)
            relative_error = depth_diff / (point_depths + 1e-6)
            
            # Points that disagree with depth map
            disagree_mask = (depth_diff > abs_tol) & (relative_error > rel_tol)
            return point_indices[disagree_mask]
            
        except Exception as e:
            print(f"Error in depth agreement filtering: {e}")
            return np.array([])
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for this spectator."""
        summary = {
            'image_name': self.image_name,
            'image_id': self.image_id,
            'camera_data_loaded': self.camera_data_loaded,
            'mask_loaded': self.mask_loaded,
            'depth_map_loaded': self.depth_map_loaded,
            'points_processed': self.points_processed,
            'visible_points_count': len(self.visible_point_indices),
            'depth_stats': self.depth_statistics
        }
        
        if self.points_processed:
            total_filtered = sum(len(indices) for indices in self.filtered_out_indices.values())
            summary['total_filtered_points'] = total_filtered
            summary['filtering_breakdown'] = {
                step: len(indices) for step, indices in self.filtered_out_indices.items()
            }
        
        return summary


class WoundModel3D:
    """
    Main class for 3D wound segmentation from COLMAP dense point clouds.
    
    This class manages the overall workflow:
    - Loading dense point clouds from COLMAP
    - Managing multiple camera viewpoints (Spectators)
    - Coordinating 3D point filtering across all cameras
    - Tracking point visibility statistics
    - Providing analysis and visualization functions
    """
    
    def __init__(self):
        """Initialize empty WoundModel3D."""
        # Dense point cloud data
        self.dense_points = None  # (N, 3) array of 3D coordinates
        self.dense_colors = None  # (N, 3) array of RGB colors
        self.dense_normals = None  # (N, 3) array of normal vectors (optional)
        
        # Camera and image management
        self.spectators = {}  # Dict: image_name -> Spectator object
        # self.cameras = {}  # COLMAP cameras dict
        # self.images = {}  # COLMAP images dict
        
        # Global point tracking
        self.point_visibility_count = {}  # point_index -> number of cameras seeing it
        self.wound_points_global = set()  # All points visible through at least one mask
        self.all_filtered_indices = set()  # All points filtered out by at least one step

        
        # Model state
        self.model_loaded = False
        self.masks_loaded = False
        self.points_processed = False
        
        # Configuration
        self.filtering_config = {
            'depth_agreement_abs_tol': 0.02,
            'depth_agreement_rel_tol': 0.1,
            'statistical_depth_std': 2.0
        }
    
    def load_colmap_model(self, model_path: Path, fused_ply_path: Path = None):
        """
        Load dense point cloud and camera data from COLMAP output.
        
        Args:
            model_path: Path to COLMAP sparse model directory
            fused_ply_path: Optional path to fused.ply file from COLMAP
        """
        model_path = Path(model_path)
        print(f"Loading COLMAP model from: {model_path}")
        
        # Load COLMAP (cameras, images, dense points - focus on dense pointcloud but sparse also possible)
        if fused_ply_path and Path(fused_ply_path).exists():
            print(f"Loading dense points from PLY: {fused_ply_path}")
            self.dense_points, self.dense_colors, self.dense_normals = self._load_ply_file(Path(fused_ply_path))
            self.cameras, self.images , _ = read_model(model_path)
        else:
            self.cameras, self.images, dense_points = read_model(model_path)
            self.dense_colors = np.array([p.rgb for p in dense_points.values()])
            self.dense_points = np.array([p.xyz for p in dense_points.values()])
            
        
        # Create Spectator objects for each image
        self._create_spectators()
        
        self.model_loaded = True
        print(f"Loaded {len(self.dense_points)} dense points and {len(self.spectators)} cameras")
    
    def load_ply_only(self, ply_path: Path):
        """
        Load dense point cloud from PLY file only (no camera information).
        
        Args:
            ply_path: Path to PLY file containing dense points
        """
        ply_path = Path(ply_path)
        print(f"Loading points from PLY: {ply_path}")
        
        self.dense_points, self.dense_colors = self._load_ply_file(ply_path)
        
        # Initialize empty camera structures
        self.cameras = {}
        self.images = {}
        self.spectators = {}
        
        self.model_loaded = True
        print(f"Loaded {len(self.dense_points)} points from PLY")
    
    def _load_ply_file(self, ply_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load points and colors from PLY file."""
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(ply_path))
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            normals = np.asarray(pcd.normals)
            if colors.shape[0] == 0:  # No colors in file
                colors = np.ones((len(points), 3)) * 128  # Default gray
            elif colors.max() <= 1.0:  # Colors in [0,1] range
                colors = (colors * 255).astype(np.uint8)
            return points, colors, normals
        except ImportError:
            return self._parse_ply_manual(ply_path)
    
    def _parse_ply_manual(self, ply_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Manual PLY file parsing for basic point clouds."""
        with open(ply_path, 'rb') as f:
            # This is a simplified parser - in production you'd want more robust parsing
            lines = f.read().decode('utf-8').split('\n')
            
            # Find number of vertices
            vertex_count = 0
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('end_header'):
                    header_end = i + 1
                    break
            
            # Parse vertex data (assuming x y z r g b format)
            points = []
            colors = []
            for i in range(header_end, header_end + vertex_count):
                if i < len(lines) and lines[i].strip():
                    parts = lines[i].split()
                    if len(parts) >= 6:
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
            
            return np.array(points), np.array(colors)
    
    def _create_spectators(self):
        """Create Spectator objects for each COLMAP image."""
        self.spectators = {}
        
        for img_id, img in self.images.items():
            spectator = Spectator(img_id, img.name, img.camera_id)
            
            # Load camera data
            camera = self.cameras[img.camera_id]
            spectator.load_camera_data(img, camera)
            
            self.spectators[img.name] = spectator
        
        print(f"Created {len(self.spectators)} spectators")
    
    def load_masks(self, mask_dir: Path):
        """
        Load wound masks for all spectators.
        
        Args:
            mask_dir: Directory containing mask files
        """
        mask_dir = Path(mask_dir)
        print(f"Loading masks from: {mask_dir}")
        
        masks_loaded = 0
        for spectator in self.spectators.values():
            if spectator.load_mask(mask_dir):
                masks_loaded += 1
        
        self.masks_loaded = masks_loaded > 0
        print(f"Loaded masks for {masks_loaded}/{len(self.spectators)} cameras")
    
    def load_depth_maps(self, depth_dir: Path):
        """
        Load depth maps for all spectators.
        
        Args:
            depth_dir: Directory containing depth map files
        """
        depth_dir = Path(depth_dir)
        print(f"Loading depth maps from: {depth_dir}")
        
        depth_maps_loaded = 0
        for spectator in self.spectators.values():
            if spectator.load_depth_map(depth_dir):
                depth_maps_loaded += 1
        
        print(f"Loaded depth maps for {depth_maps_loaded}/{len(self.spectators)} cameras")
    
    def process_wound_segmentation(self, use_depth_agreement: bool = True):
        """
        Process 3D wound segmentation by filtering points through all camera masks.
        
        Args:
            use_depth_agreement: Whether to use depth agreement filtering (requires loaded depth maps)
        """
        if not self.model_loaded:
            raise ValueError("Model must be loaded before processing")
        if not self.masks_loaded:
            raise ValueError("Masks must be loaded before processing")
        
        print(f"Processing wound segmentation with {len(self.spectators)} cameras...")
        
        # Reset global tracking
        self.point_visibility_count = {}
        self.wound_points_global = set()
        self.all_filtered_indices = set()
        
        # Process each spectator
        for spectator_name, spectator in self.spectators.items():
            if not spectator.mask_loaded:
                print(f"Skipping {spectator_name}: no mask loaded")
                continue
            
            print(f"Processing {spectator_name}...")
            
            # Filter points through this spectator's mask
            spectator.filter_points_through_mask(
                self.dense_points, self.dense_colors, use_depth_agreement
            )
            
            # Update global tracking
            visible_indices = spectator.visible_point_indices
            self.wound_points_global.update(visible_indices)
            
            
            
            # Update visibility count
            for point_idx in visible_indices:
                self.point_visibility_count[point_idx] = self.point_visibility_count.get(point_idx, 0) + 1
            
            # Track filtered indices
            for step_name, filtered_indices in spectator.filtered_out_indices.items():
                self.all_filtered_indices.update(filtered_indices)
        print(f"{len(self.wound_points_global)} points visible after Single View filtering")
        self.points_processed=True
        
    
    def filter_by_visibility(self, min_visibility_count: int = 2) -> Set[int]:
        """
        Filter points by minimum visibility count across cameras.
        
        Args:
            min_visibility_count: Minimum number of cameras that must see a point
            
        Returns:
            Set of point indices that meet the visibility threshold
        """
        if not self.points_processed:
            raise ValueError("Points must be processed before visibility filtering")
        
        high_visibility_points = {
            point_idx for point_idx, count in self.point_visibility_count.items()
            if count >= min_visibility_count
        }
        
        print(f"Visibility filtering: {len(high_visibility_points)}/{len(self.wound_points_global)} points "
              f"seen by >= {min_visibility_count} cameras")
        
        return high_visibility_points
    
    def filter_by_color_similarity(self, reference_indices: Set[int] = None, 
                                 color_threshold: float = 50.0) -> Set[int]:
        """
        Filter points by color similarity to reference points.
        
        Args:
            reference_indices: Set of reference point indices (if None, use high-visibility points)
            color_threshold: Maximum RGB distance for similarity
            
        Returns:
            Set of point indices that pass color similarity test
        """
        if not self.points_processed:
            raise ValueError("Points must be processed before color filtering")
        
        if reference_indices is None:
            reference_indices = self.filter_by_visibility(min_visibility_count=2)
        
        if not reference_indices:
            print("Warning: No reference points for color similarity filtering")
            return self.wound_points_global
        
        # Calculate reference color statistics
        reference_colors = self.dense_colors[list(reference_indices)]
        median_color = np.median(reference_colors, axis=0)
        
        # Test all visible points
        all_visible_list = list(self.wound_points_global)
        all_visible_colors = self.dense_colors[all_visible_list]
        
        # Calculate color distances
        color_distances = np.linalg.norm(all_visible_colors - median_color, axis=1)
        similar_mask = color_distances <= color_threshold
        
        similar_indices = {all_visible_list[i] for i in range(len(all_visible_list)) if similar_mask[i]}
        
        print(f"Color similarity filtering: {len(similar_indices)}/{len(self.wound_points_global)} points "
              f"within {color_threshold} RGB distance")
        
        return similar_indices
    
    def get_wound_points(self, min_visibility: int = 2, color_similarity: bool = True, 
                        color_threshold: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get final wound points after all filtering steps.
        
        Args:
            min_visibility: Minimum number of cameras that must see a point
            color_similarity: Whether to apply color similarity filtering
            color_threshold: RGB distance threshold for color similarity
            
        Returns:
            Tuple of (wound_points, wound_colors) arrays
        """
        if not self.points_processed:
            raise ValueError("Points must be processed before getting wound points")
        
        # Start with visibility filtering
        filtered_indices = self.filter_by_visibility(min_visibility)
        
        # Apply color similarity filtering if requested
        # if color_similarity and len(filtered_indices) > 0:
        #     high_vis_indices = self.filter_by_visibility(min_visibility_count=max(3, min_visibility))
        #     filtered_indices = self.filter_by_color_similarity(high_vis_indices, color_threshold)
        
        if not filtered_indices:
            print("Warning: No points passed all filtering steps")
            return np.array([]), np.array([])
        
        # Extract final points and colors
        filtered_list = list(filtered_indices)
        self.wound_points_global.clear()
        self.wound_points_global.update(filtered_indices)
        #self.wound_points=[wound_points,wound_colors]
        
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the wound segmentation process."""
        stats = {
            'model_loaded': self.model_loaded,
            'masks_loaded': self.masks_loaded,
            'points_processed': self.points_processed,
            'total_dense_points': len(self.dense_points) if self.dense_points is not None else 0,
            'total_spectators': len(self.spectators),
            'spectators_with_masks': sum(1 for s in self.spectators.values() if s.mask_loaded)
        }
        
        if self.points_processed:
            stats.update({
                'visible_points_count': len(self.wound_points_global),
                'filtered_points_count': len(self.all_filtered_indices),
                'visibility_distribution': self._get_visibility_distribution(),
                'per_camera_stats': {name: spec.get_summary() for name, spec in self.spectators.items()}
            })
        
        return stats
    
    def _get_visibility_distribution(self) -> Dict[str, int]:
        """Get distribution of point visibility counts."""
        if not self.point_visibility_count:
            return {}
        
        counts = list(self.point_visibility_count.values())
        distribution = {}
        
        for i in range(1, max(counts) + 1):
            distribution[f'{i}_cameras'] = sum(1 for c in counts if c == i)
        
        return distribution
    
    def create_dense_voxels(self, target_points_per_voxel: int = 100, 
                           max_iterations: int = 50, 
                           convergence_threshold: float = 0.001,
                           algorithm: str = "kmeans_adaptive") -> List[DenseVoxel]:
        """
        Divide the dense point cloud into adaptive voxels for local analysis.
        
        This method implements an iterative clustering approach that creates voxels
        containing approximately the target number of points while optimizing for
        spatial and color coherence.
        
        Args:
            target_points_per_voxel: Target number of points per voxel
            max_iterations: Maximum number of optimization iterations
            convergence_threshold: Convergence threshold for center movement
            algorithm: Clustering algorithm ('kmeans_adaptive', 'region_growing', 'hierarchical')
            
        Returns:
            List of DenseVoxel objects with computed properties
        """
        if not self.model_loaded or self.dense_points is None:
            raise ValueError("Dense point cloud must be loaded before voxelization")
        
        print(f"Creating dense voxels with target {target_points_per_voxel} points per voxel...")
        print(f"Algorithm: {algorithm}, Max iterations: {max_iterations}")
        
        # Estimate normals if not available
        if self.dense_normals is None:
            print("Estimating normals for voxel analysis...")
            self.dense_normals = self._estimate_point_normals()
        
        if algorithm == "kmeans_adaptive":
            voxels = self._create_voxels_kmeans_adaptive(target_points_per_voxel, max_iterations, convergence_threshold)
        elif algorithm == "region_growing":
            voxels = self._create_voxels_region_growing(target_points_per_voxel)
        elif algorithm == "hierarchical":
            voxels = self._create_voxels_hierarchical(target_points_per_voxel)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Compute properties for all voxels
        print("Computing voxel properties...")
        for voxel in voxels:
            voxel.compute_properties()
        
        valid_voxels = [v for v in voxels if v.is_valid and len(v.point_indices) > 0]
        print(f"Created {len(valid_voxels)} valid voxels (avg {np.mean([len(v.point_indices) for v in valid_voxels]):.1f} points/voxel)")
        
        return valid_voxels
    
    def _estimate_point_normals(self, k_neighbors: int = 15) -> np.ndarray:
        """Estimate normal vectors for dense points using local PCA."""
        try:
            from sklearn.neighbors import NearestNeighbors
            
            # Find k nearest neighbors for each point
            nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree').fit(self.dense_points)
            distances, indices = nbrs.kneighbors(self.dense_points)
            
            normals = np.zeros_like(self.dense_points)
            
            for i in range(len(self.dense_points)):
                # Get neighborhood points
                neighbor_indices = indices[i]
                neighbors = self.dense_points[neighbor_indices]
                
                # Center the neighborhood
                centroid = np.mean(neighbors, axis=0)
                centered = neighbors - centroid
                
                # PCA to find normal (smallest eigenvector)
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                
                # Normal is eigenvector with smallest eigenvalue
                normal = eigenvectors[:, 0]
                
                # Ensure consistent orientation (could be improved with more context)
                if normal[2] < 0:  # Prefer upward-pointing normals
                    normal *= -1
                
                normals[i] = normal
            
            return normals
            
        except ImportError:
            print("Warning: sklearn not available, using simplified normal estimation")
            return self._estimate_normals_simple()
    
    def _create_voxels_kmeans_adaptive(self, target_points: int, max_iter: int, threshold: float) -> List[DenseVoxel]:
        """Create voxels using adaptive K-means clustering."""
        n_points = len(self.dense_points)
        
        # Limit initial clusters to prevent memory issues
        max_initial_clusters = min(10000, n_points // target_points)  # Cap at 10k clusters
        initial_k = max(100, max_initial_clusters)  # Minimum 100 clusters
        
        print(f"Starting adaptive K-means with {initial_k} initial clusters for {n_points} points...")
        
        # Use fast initialization for all datasets to avoid memory issues
        print("Using fast random initialization...")
        centers = self._initialize_kmeans_centers_fast(initial_k)
        
        # Adaptive optimization
        for iteration in range(max_iter):
            print(f"Iteration {iteration + 1}/{max_iter}...")
            
            # Assign points to nearest centers
            assignments = self._assign_points_to_centers(centers)
            
            # Update centers and split/merge if needed
            new_centers, centers_moved = self._update_centers_adaptive(centers, assignments, target_points, threshold)
            
            if not centers_moved:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            centers = new_centers
        
        # Create final voxels
        print("Creating final voxel assignments...")
        final_assignments = self._assign_points_to_centers(centers)
        voxels = self._create_voxels_from_assignments(centers, final_assignments)
        
        return voxels
    
    def _initialize_kmeans_centers(self, k: int) -> np.ndarray:
        """Initialize K-means centers using K-means++ algorithm."""
        n_points = len(self.dense_points)
        centers = np.zeros((k, 3))
        
        # First center: random point
        centers[0] = self.dense_points[np.random.randint(n_points)]
        
        # Subsequent centers: weighted by distance to existing centers
        for i in range(1, k):
            distances = np.array([min([np.linalg.norm(point - center)**2 
                                     for center in centers[:i]]) 
                                for point in self.dense_points])
            probs = distances / distances.sum()
            centers[i] = self.dense_points[np.random.choice(n_points, p=probs)]
        
        return centers
    
    def _initialize_kmeans_centers_fast(self, k: int) -> np.ndarray:
        """Fast initialization for large datasets using random sampling."""
        n_points = len(self.dense_points)
        
        # For very large datasets, use random sampling
        if n_points > 500000:
            # Sample a subset of points for initialization
            sample_size = min(50000, n_points // 2)
            sample_indices = np.random.choice(n_points, sample_size, replace=False)
            sample_points = self.dense_points[sample_indices]
            
            # Initialize centers from sample
            center_indices = np.random.choice(sample_size, k, replace=False)
            centers = sample_points[center_indices].copy()
        else:
            # Use regular random initialization
            center_indices = np.random.choice(n_points, k, replace=False)
            centers = self.dense_points[center_indices].copy()
        
        return centers
    
    def _assign_points_to_centers(self, centers: np.ndarray) -> np.ndarray:
        """Assign each point to the nearest center (memory efficient version)."""
        n_points = len(self.dense_points)
        assignments = np.zeros(n_points, dtype=int)
        
        # Process in chunks to avoid memory issues
        chunk_size = min(10000, n_points // 4)  # Process in smaller chunks
        
        for start_idx in range(0, n_points, chunk_size):
            end_idx = min(start_idx + chunk_size, n_points)
            chunk_points = self.dense_points[start_idx:end_idx]
            
            # Compute distances for this chunk
            distances = np.linalg.norm(chunk_points[:, np.newaxis] - centers, axis=2)
            assignments[start_idx:end_idx] = np.argmin(distances, axis=1)
        
        return assignments
    
    def _update_centers_adaptive(self, centers: np.ndarray, assignments: np.ndarray, 
                               target_points: int, threshold: float) -> Tuple[np.ndarray, bool]:
        """Update centers and adapt cluster count based on target points per cluster."""
        new_centers = []
        centers_moved = False
        
        for i in range(len(centers)):
            cluster_points = self.dense_points[assignments == i]
            
            if len(cluster_points) == 0:
                continue  # Skip empty clusters
            
            # Compute new center
            new_center = np.mean(cluster_points, axis=0)
            
            # Check for movement
            if np.linalg.norm(new_center - centers[i]) > threshold:
                centers_moved = True
            
            # Adaptive splitting/merging logic
            if len(cluster_points) > target_points * 1.5:
                # Split large cluster
                sub_centers = self._split_cluster(cluster_points, 2)
                new_centers.extend(sub_centers)
            elif len(cluster_points) < target_points * 0.5:
                # Keep small cluster (merging would be more complex)
                new_centers.append(new_center)
            else:
                new_centers.append(new_center)
        
        return np.array(new_centers), centers_moved
    
    def _split_cluster(self, points: np.ndarray, n_splits: int) -> List[np.ndarray]:
        """Split a cluster into sub-clusters."""
        if len(points) < n_splits:
            return [np.mean(points, axis=0)]
        
        # Simple splitting using K-means
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init=10)
            kmeans.fit(points)
            return kmeans.cluster_centers_
        except:
            # Fallback: random splitting
            indices = np.random.choice(len(points), n_splits, replace=False)
            return points[indices]
    
    def _create_voxels_region_growing(self, target_points: int) -> List[DenseVoxel]:
        """Create voxels using region growing algorithm."""
        print("Region growing algorithm not yet implemented, falling back to K-means")
        return self._create_voxels_kmeans_adaptive(target_points, 20, 0.001)
    
    def _create_voxels_hierarchical(self, target_points: int) -> List[DenseVoxel]:
        """Create voxels using hierarchical clustering."""
        print("Hierarchical algorithm not yet implemented, falling back to K-means")
        return self._create_voxels_kmeans_adaptive(target_points, 20, 0.001)
    
    def _create_voxels_from_assignments(self, centers: np.ndarray, assignments: np.ndarray) -> List[DenseVoxel]:
        """Create DenseVoxel objects from cluster assignments."""
        voxels = []
        
        for i, center in enumerate(centers):
            cluster_mask = assignments == i
            cluster_indices = np.where(cluster_mask)[0].tolist()
            
            if len(cluster_indices) == 0:
                continue
            
            # Create voxel
            voxel = DenseVoxel(voxel_id=i, center=center)
            
            # Add points
            voxel.add_points(
                point_indices=cluster_indices,
                points_3d=self.dense_points[cluster_indices],
                points_colors=self.dense_colors[cluster_indices] if self.dense_colors is not None else None,
                points_normals=self.dense_normals[cluster_indices] if self.dense_normals is not None else None
            )
            
            voxels.append(voxel)
        
        return voxels
    
    def analyze_voxel_properties(self, voxels: List[DenseVoxel]) -> Dict[str, Any]:
        """Analyze statistical properties across all voxels."""
        if not voxels:
            return {}
        
        # Collect properties
        point_counts = [len(v.point_indices) for v in voxels if v.is_valid]
        volumes = [v.volume for v in voxels if v.is_valid and v.properties_computed]
        densities = [v.density for v in voxels if v.is_valid and v.properties_computed]
        
        analysis = {
            'total_voxels': len(voxels),
            'valid_voxels': len([v for v in voxels if v.is_valid]),
            'point_distribution': {
                'min_points': min(point_counts) if point_counts else 0,
                'max_points': max(point_counts) if point_counts else 0,
                'mean_points': np.mean(point_counts) if point_counts else 0,
                'std_points': np.std(point_counts) if point_counts else 0
            }
        }
        
        if volumes:
            analysis['volume_distribution'] = {
                'min_volume': min(volumes),
                'max_volume': max(volumes),
                'mean_volume': np.mean(volumes),
                'std_volume': np.std(volumes)
            }
        
        if densities:
            analysis['density_distribution'] = {
                'min_density': min(densities),
                'max_density': max(densities),
                'mean_density': np.mean(densities),
                'std_density': np.std(densities)
            }
        
        # Normal vector analysis
        normal_deviations = []
        color_variances = []
        
        for voxel in voxels:
            if voxel.is_valid and voxel.properties_computed:
                if voxel.normal_deviation:
                    normal_deviations.append(voxel.normal_deviation['std_angle'])
                if voxel.color_deviation is not None:
                    color_variances.append(np.mean(voxel.color_deviation))
        
        if normal_deviations:
            analysis['normal_coherence'] = {
                'mean_deviation': np.mean(normal_deviations),
                'std_deviation': np.std(normal_deviations)
            }
        
        if color_variances:
            analysis['color_coherence'] = {
                'mean_variance': np.mean(color_variances),
                'std_variance': np.std(color_variances)
            }
        
        return analysis
    
    def visualize_with_rerun(self, session_name: str = "WoundModel3D_Visualization"):
        """
        Visualize the wound segmentation process using Rerun.
        
        Creates a comprehensive visualization with:
        - Time series showing each spectator's view
        - 3D model with projected wound points
        - Camera positions and orientations
        - Filtering statistics per spectator
        - Voxel visualization if available
        
        Args:
            session_name: Name for the Rerun session
        """
        if not self.model_loaded:
            raise ValueError("Model must be loaded before visualization")
        
        has_voxels = hasattr(self, '_voxels') and self._voxels
        print(f"Visualization mode: {'Points processed' if self.points_processed else 'Basic model'}, {'Voxels available' if has_voxels else 'No voxels'}")
        
        try:
            import rerun as rr
            import rerun.blueprint as rrb
        except ImportError:
            print("Error: Rerun not installed. Install with: pip install rerun-sdk")
            return
        
        # Initialize Rerun session (use web viewer if spawn fails)
        try:
            rr.init(session_name, spawn=True)
        except RuntimeError as e:
            if "Failed to find Rerun Viewer executable" in str(e):
                print("Rerun viewer not found, using web viewer...")
                rr.init(session_name)

            else:
                raise e
        
        # Set up coordinate system (right-handed, Y-downs for 3D graphics)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        
        # Create blueprint with multiple tabs
        blueprint = rrb.Blueprint(
            rrb.Tabs(
                # Tab 1: 3D Overview
                rrb.Tabs(name="3D Overview", contents=[
                    rrb.Grid(
                        rrb.Tabs(
                            name="Full Scene", contents=[rrb.Spatial3DView(origin="/world", contents=["+ $origin/**","- /world/normals/**"], name="Full 3D Scene"),rrb.Spatial3DView(origin="/world/normals", name="3D Normals")]),
                        rrb.Spatial3DView(origin="/spectators", name="Camera Views"),
                    )
                ]),
                # Tab 2: Spectator Time Series
                rrb.Tabs(name="Spectator Views", contents=[
                    rrb.Grid(
                        rrb.Spatial3DView(origin="/timeseries/world", name="3D Scene by Time"),
                        rrb.Grid(
                            rrb.Spatial2DView(origin="/timeseries/camera_view", name="Camera Image + Mask"),
                            rrb.Spatial3DView(origin="/timeseries/filtered", name="Filtered Points by Step"),
                        ),
                        rrb.Tabs(name="Spectator Info", contents=[
                            rrb.BarChartView(origin="/timeseries/wound_color/red", name="Barchart of Red Channel"),
                            rrb.BarChartView(origin="/timeseries/wound_color/green", name="Barchart of Green Channel"),
                            rrb.BarChartView(origin="/timeseries/wound_color/blue", name="Barchart of Blue Channel"),
                            ]
                        )
                    )
                ]),
                # Tab 3: Voxel Analysis
                rrb.Tabs(name="Voxel Analysis", contents=[
                    rrb.Grid(
                        rrb.Spatial3DView(origin="/voxels", name="Voxel Visualization", contents=["+ $origin/**"]),
                        rrb.Grid(
                            rrb.Spatial3DView(origin="/voxels/normals", name="Voxel Normals & Deviation"),
                            rrb.TextDocumentView(origin="/voxels/statistics", name="Voxel Statistics"),
                        ),
                    )
                ]),
                # Tab 4: Filtering Statistics
                rrb.Tabs(name="Filtering Stats", contents=[
                    rrb.Grid(
                        rrb.TimeSeriesView(origin="/stats/filtering", name="Points Filtered per Spectator"),
                        rrb.TimeSeriesView(origin="/stats/visible", name="Visible Points per Spectator"),
                        rrb.BarChartView(origin="/stats/summary", name="Filter Step Summary"),
                    )
                ]),
            )
        )
        rr.send_blueprint(blueprint)
        
        # Log static 3D scene data
        self._log_static_scene()
        
        # Log time series data for each spectator (only if points processed)
        if self.points_processed and len(self.spectators) > 0:
            self._log_spectator_timeseries()
        
        # Log filtering statistics (only if points processed)
        if self.points_processed and len(self.spectators) > 0:
            self._log_filtering_statistics()
        
        # Log voxel visualization if voxels exist
        self._log_voxel_visualization()
        
        print(f"Rerun visualization '{session_name}' started successfully!")
        print("Navigate through time to see different spectator views")
    
    def _log_static_scene(self):
        """Log static 3D scene elements."""
        import rerun as rr
        
        # Log complete dense point cloud
        rr.log("/world/dense_pointcloud", rr.Points3D(
            positions=self.dense_points,
            colors=self.dense_colors,
            radii=0.001
        ), static=True)
        
        # Log all visible wound points in red (if available)
        if hasattr(self, 'wound_points_global') and self.wound_points_global:
            visible_indices = list(self.wound_points_global)
            visible_points = self.dense_points[visible_indices]
            wound_colors = [[255, 0, 0]] * len(visible_points)  # Red for wound points
            
            rr.log("/world/wound_points", rr.Points3D(
                positions=visible_points,
                colors=wound_colors,
                radii=0.002
            ), static=True)
        
        #Log all points normals if available
        if self.dense_normals is not None:
            rr.log("/world/normals", 
                    rr.Arrows3D(vectors=self.dense_normals,
                                origins=self.dense_points,
                                colors=[[100, 100, 255]] * len(self.dense_points))
                    , static=True)
        
        # Log camera positions and orientations
        for spectator_name, spectator in self.spectators.items():
            if spectator.camera_data_loaded:
                # Camera position (translation)
                cam_pos = -spectator.rotation_matrix.T @ spectator.translation_vector.flatten()
                
                # Camera coordinate frame
                R = spectator.rotation_matrix.T  # World to camera rotation
                axes_length = 0.1
                
                # Create coordinate axes in world coordinates
                x_axis = cam_pos + R[:, 0] * axes_length  # Red axis
                y_axis = cam_pos + R[:, 1] * axes_length  # Green axis  
                z_axis = cam_pos + R[:, 2] * axes_length  # Blue axis (optical axis)
                
                rr.log(f"/world/cameras/{spectator_name}/position", rr.Points3D(
                    positions=[cam_pos],
                    colors=[[255, 255, 0]],  # Yellow
                    radii=0.005
                ), static=True)
                
                # Log camera axes
                rr.log(f"/world/cameras/{spectator_name}/axes", rr.LineStrips3D(
                    strips=[
                        [cam_pos, x_axis],  # X-axis
                        [cam_pos, y_axis],  # Y-axis
                        [cam_pos, z_axis]   # Z-axis (optical)
                    ],
                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                    radii=0.002
                ), static=True)
    
    def _log_spectator_timeseries(self):
        """Log time series data for each spectator."""
        import rerun as rr
        
        spectator_list = list(self.spectators.items())
        
        for time_idx, (spectator_name, spectator) in enumerate(spectator_list):
            if not spectator.points_processed:
                continue
            
            # Set timeline for this spectator
            rr.set_time("spectator", sequence=time_idx)
            #rr.set_time("wall_clock", sequence=time_idx * 0.5)  # 0.5 second intervals
            
            # Log current spectator info
            rr.log("/timeseries/current_spectator", rr.TextLog(
                f"Spectator: {spectator_name}\nVisible Points: {len(spectator.visible_point_indices)}"
            ))
            
            # Log 3D scene with current spectator's visible points highlighted
            if spectator.visible_point_indices:
                visible_points = self.dense_points[spectator.visible_point_indices]
                rr.log("/timeseries/world/current_wound_points", rr.Points3D(
                    positions=visible_points,
                    colors=[[255, 100, 100]] * len(visible_points),  # Light red
                    radii=0.003
                ))
            
            # Log dense background (faded)
            rr.log("/timeseries/world/dense_background", rr.Points3D(
                positions=self.dense_points,
                colors=self.dense_colors,  # Faded
                radii=0.0005
            ))
            
            # Log current camera position
            if spectator.camera_data_loaded:
                cam_pos = -spectator.rotation_matrix.T @ spectator.translation_vector.flatten()
                rr.log("/timeseries/world/current_camera", rr.Points3D(
                    positions=[cam_pos],
                    colors=[[255, 255, 0]],
                    radii=0.008
                ))
                
                # Log camera frustum (simplified)
                if spectator.intrinsics:
                    fx, fy, cx, cy = (spectator.intrinsics['fx'], spectator.intrinsics['fy'], 
                                    spectator.intrinsics['cx'], spectator.intrinsics['cy'])
                    w, h = spectator.image_width, spectator.image_height
                    
                    # Create frustum corners in camera coordinates
                    depth = 0.5  # Frustum depth
                    corners_cam = np.array([
                        [-cx/fx * depth, -cy/fy * depth, depth],      # Top-left
                        [(w-cx)/fx * depth, -cy/fy * depth, depth],   # Top-right
                        [(w-cx)/fx * depth, (h-cy)/fy * depth, depth], # Bottom-right
                        [-cx/fx * depth, (h-cy)/fy * depth, depth],   # Bottom-left
                    ])
                    
                    # Transform to world coordinates
                    R_inv = spectator.rotation_matrix.T
                    corners_world = (R_inv @ corners_cam.T).T + cam_pos
                    
                    # Log frustum lines
                    frustum_lines = [
                        [cam_pos, corners_world[0]],  # To top-left
                        [cam_pos, corners_world[1]],  # To top-right
                        [cam_pos, corners_world[2]],  # To bottom-right
                        [cam_pos, corners_world[3]],  # To bottom-left
                        [corners_world[0], corners_world[1]],  # Top edge
                        [corners_world[1], corners_world[2]],  # Right edge
                        [corners_world[2], corners_world[3]],  # Bottom edge
                        [corners_world[3], corners_world[0]],  # Left edge
                    ]
                    
                    rr.log("/timeseries/world/camera_frustum", rr.LineStrips3D(
                        strips=frustum_lines,
                        colors=[[255, 255, 0]] * len(frustum_lines),
                        radii=0.001
                    ))
            
            # Log camera view with image and mask overlay
            self._log_camera_view_with_mask(spectator, time_idx)
            
            # Log color histograms for visible wound points
            self._log_wound_color_histograms(spectator)
            
            # Log filtered points visualization by step
            self._log_filtered_points_by_step(spectator)
    
    def _log_camera_view_with_mask(self, spectator, time_idx):
        """Log camera view with underlying image and transparent mask overlay."""
        import rerun as rr
        
        if not spectator.mask_loaded:
            return
        
        # Try to load the original image
        image_loaded = False
        base_path = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_WoundDetection\Ferse1\dense\images")
        
        # Try different image paths
        possible_image_paths = [
            base_path / spectator.image_name,
            base_path / spectator.image_name.replace('.bin', '.jpg'),
            base_path / spectator.image_name.replace('.bin', '.png'),
            base_path / f"{Path(spectator.image_name).stem}.jpg",
            base_path / f"{Path(spectator.image_name).stem}.png"
        ]
        
        original_image = None
        for img_path in possible_image_paths:
            if img_path.exists():
                try:
                    import cv2
                    original_image = cv2.imread(str(img_path))
                    if original_image is not None:
                        # Convert BGR to RGB
                        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                        image_loaded = True
                        break
                except Exception as e:
                    continue
        
        if image_loaded and original_image is not None:
            # Resize mask to match image if needed
            if original_image.shape[:2] != spectator.mask_data.shape:
                mask_resized = cv2.resize(spectator.mask_data, 
                                        (original_image.shape[1], original_image.shape[0]))
            else:
                mask_resized = spectator.mask_data
            
            # Create mask overlay with transparency
            mask_overlay = np.zeros_like(original_image)
            mask_overlay[mask_resized > 0] = [255, 0, 0]  # Red for wound areas
            
            # Blend original image with mask (75% image, 25% mask where mask is true)
            alpha = 0.25
            blended_image = original_image.copy()
            mask_areas = mask_resized > 0
            blended_image[mask_areas] = (
                (1 - alpha) * original_image[mask_areas] + 
                alpha * mask_overlay[mask_areas]
            ).astype(np.uint8)
            
            # Log the blended image
            rr.log("/timeseries/camera_view/image_with_mask", rr.Image(blended_image))
            
            # Also log separate components
            rr.log("/timeseries/camera_view/original_image", rr.Image(original_image))
            
        else:
            # Fallback: just show mask as before
            mask_rgb = np.stack([spectator.mask_data] * 3, axis=-1)
            mask_rgb = (mask_rgb > 0).astype(np.uint8) * 255
            rr.log("/timeseries/camera_view/mask_only", rr.Image(mask_rgb))
        
        # Log mask statistics
        mask_pixels = np.sum(spectator.mask_data > 0)
        total_pixels = spectator.mask_data.shape[0] * spectator.mask_data.shape[1]
        mask_percentage = (mask_pixels / total_pixels) * 100
        
        rr.log("/timeseries/camera_view/info", rr.TextLog(
            f"Spectator: {spectator.image_name}\n"
            f"Mask Coverage: {mask_percentage:.1f}%\n"
            f"Mask Pixels: {mask_pixels:,}\n"
            f"Image Size: {spectator.image_width}×{spectator.image_height}\n"
            f"Image Loaded: {'Yes' if image_loaded else 'No'}"
        ))
    
    def _log_filtered_points_by_step(self, spectator):
        """Log filtered points visualization by filtering step."""
        import rerun as rr
        
        if not spectator.points_processed:
            return
        
        # Define colors for each filter step
        filter_colors = {
            'behind_camera': [100, 100, 100],      # Gray
            'out_of_bounds': [128, 0, 128],        # Purple  
            'outside_mask': [255, 165, 0],         # Orange
            'depth_agreement': [255, 0, 255],      # Magenta
            'statistical_depth': [255, 255, 0],    # Yellow
        }
        
        # Log points filtered by each step
        for step_name, filtered_indices in spectator.filtered_out_indices.items():
            if filtered_indices:
                filtered_points = self.dense_points[filtered_indices]
                step_color = filter_colors.get(step_name, [128, 128, 128])
                
                rr.log(f"/timeseries/filtered/{step_name}", rr.Points3D(
                    positions=filtered_points,
                    colors=[step_color] * len(filtered_points),
                    radii=0.002
                ))
            else:
                # Log empty points for consistent timeline
                rr.log(f"/timeseries/filtered/{step_name}", rr.Points3D(
                    positions=np.empty((0, 3)),
                    colors=[],
                    radii=[]
                ))
        
        # Log legend information
        legend_text = "Filter Step Colors:\n"
        for step_name, color in filter_colors.items():
            count = len(spectator.filtered_out_indices.get(step_name, []))
            legend_text += f"● {step_name}: {count} points (RGB: {color})\n"
        
        rr.log("/timeseries/filtered/legend", rr.TextLog(legend_text))
    
    def _log_wound_color_histograms(self, spectator):
        """Log color histograms for each RGB channel of visible wound points."""
        import rerun as rr
        
        if not spectator.points_processed or not spectator.visible_point_indices:
            # Log empty histograms for consistent timeline
            rr.log("/timeseries/wound_color/red", rr.BarChart([0]))
            rr.log("/timeseries/wound_color/green", rr.BarChart([0]))
            rr.log("/timeseries/wound_color/blue", rr.BarChart([0]))
            return
        
        # Get colors of visible wound points (after all filtering)
        visible_colors = self.dense_colors[spectator.visible_point_indices]
        
        # Create histograms for each channel (0-255 range, 256 bins)
        bins = np.arange(0, 257)  # 0 to 256 for bin edges
        bin_centers = bins[:-1]  # 0 to 255 for abscissa
        
        # Compute histograms
        red_hist, _ = np.histogram(visible_colors[:, 0], bins=bins)
        green_hist, _ = np.histogram(visible_colors[:, 1], bins=bins)
        blue_hist, _ = np.histogram(visible_colors[:, 2], bins=bins)
        
        # Log histograms with abscissa to show 0-255 range
        rr.log("/timeseries/wound_color/red", rr.BarChart(
            values=red_hist,
            abscissa=bin_centers
        ))
        
        rr.log("/timeseries/wound_color/green", rr.BarChart(
            values=green_hist,
            abscissa=bin_centers
        ))
        
        rr.log("/timeseries/wound_color/blue", rr.BarChart(
            values=blue_hist,
            abscissa=bin_centers
        ))
    
    def _log_filtering_statistics(self):
        """Log filtering statistics as time series and summaries."""
        import rerun as rr
        
        # Prepare data for time series
        spectator_names = []
        visible_counts = []
        filter_step_counts = {
            'behind_camera': [],
            'out_of_bounds': [],
            'outside_mask': [],
            'depth_agreement': [],
            'statistical_depth': []
        }
        
        for time_idx, (spectator_name, spectator) in enumerate(self.spectators.items()):
            if not spectator.points_processed:
                continue
            
            spectator_names.append(spectator_name)
            visible_counts.append(len(spectator.visible_point_indices))
            
            # Collect filter step counts
            for step_name in filter_step_counts.keys():
                count = len(spectator.filtered_out_indices.get(step_name, []))
                filter_step_counts[step_name].append(count)
            
            # Log time series data
            rr.set_time_sequence("spectator", time_idx)
            
            # Visible points count
            rr.log("/stats/visible/count", rr.Scalars(len(spectator.visible_point_indices)))
            
            # Filtered points by step
            total_filtered = sum(len(indices) for indices in spectator.filtered_out_indices.values())
            rr.log("/stats/filtering/total_filtered", rr.Scalars(total_filtered))
            
            for step_name, indices_list in spectator.filtered_out_indices.items():
                rr.log(f"/stats/filtering/{step_name}", rr.Scalars(len(indices_list)))
        
        # Log summary statistics
        if spectator_names:
            # Overall summary
            total_spectators = len(spectator_names)
            total_visible = sum(visible_counts)
            avg_visible = np.mean(visible_counts) if visible_counts else 0
            
            summary_text = f"""
            Processing Summary:
            - Total Spectators: {total_spectators}
            - Total Visible Points: {total_visible:,}
            - Average Visible per Camera: {avg_visible:.1f}
            - Unique Visible Points: {len(self.wound_points_global):,}
            - Max Visibility Count: {max(self.point_visibility_count.values()) if self.point_visibility_count else 0}
            """
            
            rr.log("/stats/summary/text", rr.TextDocument(summary_text), static=True)
            
            # Filter step summary as bar chart
            step_totals = {step: sum(counts) for step, counts in filter_step_counts.items()}
            if any(step_totals.values()):
                step_names = list(step_totals.keys())
                step_values = list(step_totals.values())
                
                rr.log("/stats/summary/filter_steps", rr.BarChart(
                    values=step_values,
                ), static=True)
    
    def _log_voxel_visualization(self):
        """Log voxel visualization with bounding boxes, normals, and statistics."""
        import rerun as rr
        
        # Check if we have any voxels to visualize
        if not hasattr(self, '_voxels') or not self._voxels:
            print("No voxels available for visualization. Run create_dense_voxels() first.")
            return
        
        print(f"Logging {len(self._voxels)} voxels for visualization...")
        
        # Log voxel centers as points
        voxel_centers = []
        voxel_colors = []
        voxel_radii = []
        
        # Collect voxel normals and statistics
        voxel_normals = []
        normal_origins = []
        normal_colors = []
        deviation_circles = []
        
        statistics_text = "Voxel Analysis Summary\n" + "="*50 + "\n\n"
        
        for i, voxel in enumerate(self._voxels):
            if not voxel.is_valid or len(voxel.point_indices) == 0:
                continue
            
            # Voxel center visualization
            voxel_centers.append(voxel.center)
            
            # Color voxels by point count (blue to red gradient)
            point_count = len(voxel.point_indices)
            max_points = max(len(v.point_indices) for v in self._voxels if v.is_valid)
            color_intensity = min(255, int(255 * (point_count / max_points)))
            voxel_colors.append([255 - color_intensity, 0, color_intensity])  # Blue to red
            
            # Radius based on voxel size
            if voxel.properties_computed and voxel.volume > 0:
                radius = (voxel.volume * 3 / (4 * np.pi)) ** (1/3)  # Equivalent sphere radius
                voxel_radii.append(min(0.01, max(0.002, radius)))
            else:
                voxel_radii.append(0.005)
            
            # Log individual voxel bounding box
            if len(voxel.point_indices) > 0:
                voxel_points = self.dense_points[voxel.point_indices]
                bbox_min = np.min(voxel_points, axis=0)
                bbox_max = np.max(voxel_points, axis=0)
                self._log_voxel_bounding_box(i, bbox_min, bbox_max, voxel.center)
            
            # Log voxel points
            if len(voxel.point_indices) > 0:
                voxel_points = self.dense_points[voxel.point_indices]
                voxel_point_colors = self.dense_colors[voxel.point_indices] if self.dense_colors is not None else None
                
                rr.log(f"/voxels/voxel_{i}/points", rr.Points3D(
                    positions=voxel_points,
                    colors=voxel_point_colors if voxel_point_colors is not None else [[100, 100, 100]] * len(voxel_points),
                    radii=0.001
                ), static=True)
            
            # Normal vector visualization
            if voxel.properties_computed and voxel.mean_normal is not None:
                normal_length = 0.02  # Fixed length for visibility
                voxel_normals.append(voxel.mean_normal * normal_length)
                normal_origins.append(voxel.center)
                
                # Color normals by deviation (green = low deviation, red = high deviation)  
                if voxel.normal_deviation and 'std_angle' in voxel.normal_deviation:
                    deviation_angle = voxel.normal_deviation['std_angle']
                    max_deviation = np.pi / 4  # 45 degrees max for color scaling
                    deviation_ratio = min(1.0, deviation_angle / max_deviation)
                    normal_color = [int(255 * deviation_ratio), int(255 * (1 - deviation_ratio)), 0]
                    normal_colors.append(normal_color)
                    
                    # Create deviation circle around normal vector
                    self._create_deviation_circle(voxel.center, voxel.mean_normal, deviation_angle, i)
                else:
                    normal_colors.append([0, 255, 0])  # Default green
            
            # Add to statistics text
            if voxel.properties_computed:
                statistics_text += f"Voxel {i}:\n"
                statistics_text += f"  Points: {len(voxel.point_indices)}\n"
                statistics_text += f"  Volume: {voxel.volume:.6f}\n" if voxel.volume else "  Volume: N/A\n"
                statistics_text += f"  Density: {voxel.density:.2f}\n" if voxel.density else "  Density: N/A\n"
                statistics_text += f"  Center: ({voxel.center[0]:.3f}, {voxel.center[1]:.3f}, {voxel.center[2]:.3f})\n"
                
                if voxel.mean_normal is not None:
                    statistics_text += f"  Normal: ({voxel.mean_normal[0]:.3f}, {voxel.mean_normal[1]:.3f}, {voxel.mean_normal[2]:.3f})\n"
                
                if voxel.normal_deviation and 'std_angle' in voxel.normal_deviation:
                    deviation_deg = np.degrees(voxel.normal_deviation['std_angle'])
                    statistics_text += f"  Normal Deviation: {deviation_deg:.1f}°\n"
                
                if voxel.mean_color is not None:
                    color_str = f"RGB({voxel.mean_color[0]:.0f}, {voxel.mean_color[1]:.0f}, {voxel.mean_color[2]:.0f})"
                    statistics_text += f"  Mean Color: {color_str}\n"
                
                statistics_text += "\n"
        
        # Log all voxel centers
        if voxel_centers:
            rr.log("/voxels/centers", rr.Points3D(
                positions=voxel_centers,
                colors=voxel_colors,
                radii=voxel_radii
            ), static=True)
        
        # Log all voxel normals
        if voxel_normals and normal_origins:
            rr.log("/voxels/normals/arrows", rr.Arrows3D(
                vectors=voxel_normals,
                origins=normal_origins,
                colors=normal_colors
            ), static=True)
        
        # Log statistics
        rr.log("/voxels/statistics", rr.TextDocument(statistics_text), static=True)
        
        # Log summary information
        valid_voxel_count = len([v for v in self._voxels if v.is_valid])
        avg_points_per_voxel = np.mean([len(v.point_indices) for v in self._voxels if v.is_valid and len(v.point_indices) > 0])
        
        summary_info = f"""
Voxelization Summary:
- Total Voxels: {len(self._voxels)}
- Valid Voxels: {valid_voxel_count}
- Average Points per Voxel: {avg_points_per_voxel:.1f}
- Color Legend:
  • Voxel Centers: Blue (few points) → Red (many points)
  • Normal Vectors: Green (low deviation) → Red (high deviation)
  • Deviation Circles: Show angular standard deviation
        """
        
        rr.log("/voxels/info", rr.TextDocument(summary_info), static=True)
    
    def _log_voxel_bounding_box(self, voxel_id: int, bbox_min: np.ndarray, bbox_max: np.ndarray, center: np.ndarray):
        """Log a bounding box wireframe for a voxel."""
        import rerun as rr
        
        # Create bounding box corners
        corners = np.array([
            [bbox_min[0], bbox_min[1], bbox_min[2]],  # 0: min corner
            [bbox_max[0], bbox_min[1], bbox_min[2]],  # 1: +x
            [bbox_max[0], bbox_max[1], bbox_min[2]],  # 2: +x+y
            [bbox_min[0], bbox_max[1], bbox_min[2]],  # 3: +y
            [bbox_min[0], bbox_min[1], bbox_max[2]],  # 4: +z
            [bbox_max[0], bbox_min[1], bbox_max[2]],  # 5: +x+z
            [bbox_max[0], bbox_max[1], bbox_max[2]],  # 6: +x+y+z (max corner)
            [bbox_min[0], bbox_max[1], bbox_max[2]],  # 7: +y+z
        ])
        
        # Define wireframe edges
        edges = [
            # Bottom face (z = min)
            [corners[0], corners[1]], [corners[1], corners[2]], 
            [corners[2], corners[3]], [corners[3], corners[0]],
            # Top face (z = max)
            [corners[4], corners[5]], [corners[5], corners[6]], 
            [corners[6], corners[7]], [corners[7], corners[4]],
            # Vertical edges
            [corners[0], corners[4]], [corners[1], corners[5]], 
            [corners[2], corners[6]], [corners[3], corners[7]],
        ]
        
        # Log bounding box wireframe
        rr.log(f"/voxels/voxel_{voxel_id}/bbox", rr.LineStrips3D(
            strips=edges,
            colors=[[128, 128, 128]] * len(edges),  # Gray wireframe
            radii=0.0005
        ), static=True)
    
    def _create_deviation_circle(self, center: np.ndarray, normal: np.ndarray, deviation_angle: float, voxel_id: int):
        """Create a circle around the normal vector to show angular deviation."""
        import rerun as rr
        
        if deviation_angle < 0.01:  # Less than ~0.5 degrees, too small to visualize
            return
        
        # Create two perpendicular vectors to the normal
        if abs(normal[2]) < 0.9:  # Normal is not too close to z-axis
            v1 = np.cross(normal, [0, 0, 1])
        else:  # Normal is close to z-axis, use x-axis instead
            v1 = np.cross(normal, [1, 0, 0])
        
        v1 = v1 / np.linalg.norm(v1)  # Normalize
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)  # Normalize
        
        # Create circle points
        num_points = 24  # Circle resolution
        angles = np.linspace(0, 2 * np.pi, num_points)
        
        # Circle radius based on deviation angle and distance from center
        circle_distance = 0.015  # Distance from voxel center
        circle_center = center + normal * circle_distance
        circle_radius = circle_distance * np.tan(deviation_angle)
        
        circle_points = []
        for angle in angles:
            point = (circle_center + 
                    circle_radius * np.cos(angle) * v1 + 
                    circle_radius * np.sin(angle) * v2)
            circle_points.append(point)
        
        # Close the circle
        circle_points.append(circle_points[0])
        
        # Color based on deviation magnitude (green to red)
        max_deviation = np.pi / 4  # 45 degrees
        deviation_ratio = min(1.0, deviation_angle / max_deviation)
        circle_color = [int(255 * deviation_ratio), int(255 * (1 - deviation_ratio)), 0]
        
        # Log deviation circle
        rr.log(f"/voxels/normals/deviation_circle_{voxel_id}", rr.LineStrips3D(
            strips=[circle_points],
            colors=[circle_color],
            radii=0.0008
        ), static=True)



def main():
    """
    Example usage of the WoundModel3D and Spectator classes.
    
    This demonstrates the typical workflow:
    1. Load COLMAP model and dense points
    2. Load wound masks
    3. Process 3D wound segmentation
    4. Apply additional filtering
    5. Save results
    """
    
    # Example paths (adjust these to your data)
    model_path = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_WoundDetection\Ferse1\dense")  # Contains cameras.bin, images.bin, points3D.bin
    fused_ply_path = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_WoundDetection\Ferse1\dense\fused.ply")  # Dense point cloud
    mask_dir = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_WoundDetection\Ferse1\dense\images_woundmask")  # Directory with mask images
    depth_dir = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_WoundDetection\Ferse1\dense\stereo\depth_maps")  # Optional depth maps
    output_path = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_WoundDetection\Ferse1\dense")  # Output file
    
    try:
        # Initialize the model
        wound_model = WoundModel3D()
        
        # Load COLMAP data
        print("=== Loading COLMAP Model ===")
        wound_model.load_colmap_model(model_path, fused_ply_path)
        
        # Create voxels and store them for visualization
        # print("\n=== Creating Dense Voxels ===")
        # # Use larger target for this big dataset (664k points)
        # voxels = wound_model.create_dense_voxels(target_points_per_voxel=1000, max_iterations=10, algorithm="kmeans_adaptive")
        # wound_model._voxels = voxels  # Store for visualization
        
        
        #Load masks
        print("\n=== Loading Masks ===")
        wound_model.load_masks(mask_dir)
        
        # Load depth maps
        print("\n=== Loading Depth Maps ===")
        wound_model.load_depth_maps(depth_dir)
        
        #Process wound segmentation
        print("\n=== Processing Wound Segmentation ===")
        wound_model.process_wound_segmentation(use_depth_agreement=True)
        
        # Get statistics
        print("\n=== Statistics ===")
        stats = wound_model.get_statistics()
        print(f"Total dense points: {stats['total_dense_points']}")
        print(f"Visible points: {stats['visible_points_count']}")
        print(f"Cameras with masks: {stats['spectators_with_masks']}")
        
        if 'visibility_distribution' in stats:
            print("Visibility distribution:")
            for cameras, count in stats['visibility_distribution'].items():
                print(f"  {cameras}: {count} points")
        
        # Get final wound points with filtering
        # print("\n=== Final Filtering ===")
        wound_model.get_wound_points(
            min_visibility=3,  # Points must be seen by at least 4 cameras
            color_similarity=True,  # Apply color similarity filtering
            color_threshold=50.0  # RGB distance threshold
        )
        
        
        # Launch Rerun visualization
        print("\n=== Launching Rerun Visualization ===")
        wound_model.visualize_with_rerun("WoundModel3D_Main")
        
        print("\n=== Processing Complete ===")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the example with your actual data paths
    print("=== Running example with your data structure ===")
    #example_usage_with_your_data()
    
    print("\n=== Running generic example ===")
    main()
