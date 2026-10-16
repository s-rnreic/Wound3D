"""
World Center Calculation and Hole Property Visualization

This module provides tools for analyzing point cloud geometry to identify
mathematical properties that characterize holes/wounds in 3D reconstructions.

Key Features:
    1. Center of gravity calculation relative to camera positions
    2. Height profile computation relative to reference plane
    3. Curvature analysis (mean curvature, Gaussian curvature)
    4. Gradient magnitude for edge detection
    5. Energy-based hole characterization

Author: COLMAP Pipeline
Date: January 2026
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree, Delaunay
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import warnings

# Add paths for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_pipe_code_dir = os.path.abspath(os.path.join(_current_dir, '..', '..'))
_workspace_root = os.path.abspath(os.path.join(_current_dir, '..', '..', '..'))

if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)
if _pipe_code_dir not in sys.path:
    sys.path.insert(0, _pipe_code_dir)


class PointCloudGeometryAnalyzer:
    """
    Analyzes geometric properties of point clouds for hole/wound identification.
    
    This class computes and visualizes mathematical descriptors that can
    characterize depressions, holes, or wounds in 3D surface reconstructions.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        # Point cloud data
        self.points = None  # (N, 3) array
        self.colors = None  # (N, 3) array
        self.normals = None  # (N, 3) array
        
        # Camera data
        self.camera_positions = []  # List of camera center positions
        self.camera_orientations = []  # List of camera rotation matrices
        
        # Computed reference frame
        self.center_of_gravity = None
        self.reference_plane_normal = None  # Usually z-axis after alignment
        self.reference_plane_origin = None
        
        # Height field (gridded)
        self.height_grid = None
        self.grid_x = None
        self.grid_y = None
        self.grid_resolution = 0.001  # 1mm default
        
        # Computed properties
        self.height_values = None  # Per-point height relative to plane
        self.mean_curvature = None
        self.gaussian_curvature = None
        self.gradient_magnitude = None
        self.laplacian = None
        self.hole_energy = None
        
        # Grid-based properties (internal)
        self._grad_mag_grid = None
        self._mean_curv_grid = None
        self._laplacian_grid = None
        self._energy_grid = None
        self._dx_grid = None
        self._dy_grid = None
        self._x_2d = None
        self._y_2d = None
        self._u_axis = None
        self._v_axis = None
        
        # Normal-based center of gravity (internal)
        self._voxel_centers_normals = None
        self._voxel_normals = None
        self._voxel_counts_normals = None
        
        # Reference plane normals from different approaches
        self._ref_normal_geometric = None  # From camera positions to geometric center
        self._ref_normal_orientation = None  # From actual camera orientations
        self._ref_normal_pointcloud = None  # From point cloud surface normals
        
        # Voxel-based properties (for sparse visualization)
        self.voxel_centers = None
        self.voxel_properties = {}
        self._voxel_size = None  # Store voxel size for reuse
        self._voxel_point_mapping = None  # Maps each point to its voxel index
    
    def load_point_cloud(self, points: np.ndarray, colors: np.ndarray = None,
                        normals: np.ndarray = None):
        """
        Load point cloud data.
        
        Args:
            points: (N, 3) array of 3D coordinates
            colors: Optional (N, 3) array of RGB colors
            normals: Optional (N, 3) array of normal vectors
        """
        self.points = np.asarray(points)
        self.colors = np.asarray(colors) if colors is not None else None
        self.normals = np.asarray(normals) if normals is not None else None
        
        print(f"Loaded {len(self.points)} points")
        
        # Estimate normals if not provided
        if self.normals is None:
            self._estimate_normals()
    
    def load_from_ply(self, ply_path: Path):
        """Load point cloud from PLY file using Open3D."""
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(ply_path))
            self.points = np.asarray(pcd.points)
            self.colors = np.asarray(pcd.colors) if pcd.has_colors() else None
            self.normals = np.asarray(pcd.normals) if pcd.has_normals() else None
            
            if self.normals is None or len(self.normals) == 0:
                self._estimate_normals()
            
            print(f"Loaded {len(self.points)} points from {ply_path}")
        except Exception as e:
            print(f"Error loading PLY: {e}")
    
    def set_camera_positions(self, positions: List[np.ndarray], 
                            orientations: List[np.ndarray] = None):
        """
        Set camera positions for center of gravity calculation.
        
        Args:
            positions: List of camera center positions in world coordinates
            orientations: Optional list of 3x3 rotation matrices
        """
        self.camera_positions = [np.asarray(p) for p in positions]
        self.camera_orientations = orientations if orientations else []
        print(f"Set {len(self.camera_positions)} camera positions")
    
    def _estimate_normals(self, k_neighbors: int = 30):
        """Estimate normals using PCA on local neighborhoods."""
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(k_neighbors))
            pcd.orient_normals_consistent_tangent_plane(k_neighbors)
            self.normals = np.asarray(pcd.normals)
            print(f"Estimated normals for {len(self.normals)} points")
        except Exception as e:
            print(f"Normal estimation failed: {e}")
            # Fallback: simple normal estimation
            self.normals = self._estimate_normals_manual(k_neighbors)
    
    def _estimate_normals_manual(self, k: int = 30) -> np.ndarray:
        """Manual normal estimation using PCA."""
        tree = KDTree(self.points)
        normals = np.zeros_like(self.points)
        
        for i in range(len(self.points)):
            _, indices = tree.query(self.points[i], k=k)
            neighbors = self.points[indices]
            centered = neighbors - neighbors.mean(axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normals[i] = eigenvectors[:, 0]  # Smallest eigenvalue = normal
        
        return normals
    
    # =========================================================================
    # 1. CENTER OF GRAVITY CALCULATION
    # =========================================================================
    
    def compute_center_of_gravity(self, use_cameras: bool = True, 
                                     use_orientations: bool = True) -> np.ndarray:
        """
        Compute center of gravity considering camera arrangement.
        
        For a top-down camera setup (half circles around object), the 
        center of gravity is computed as the geometric center weighted
        by camera view directions.
        
        This method computes reference plane normals using up to 3 approaches:
        1. Geometric: Direction from cameras to geometric center (fallback)
        2. Orientation: Actual camera viewing directions from rotation matrices
        3. Point cloud: Surface normals (computed separately via compute_center_of_gravity_from_normals)
        
        Args:
            use_cameras: If True, weight by camera positions
            use_orientations: If True and orientations available, use actual 
                             camera viewing directions instead of geometric
            
        Returns:
            Center of gravity as (3,) array
        """
        if self.points is None:
            raise ValueError("No points loaded")
        
        # Basic geometric center
        geometric_center = np.mean(self.points, axis=0)
        
        if use_cameras and len(self.camera_positions) > 0:
            camera_array = np.array(self.camera_positions)
            
            # === Approach 1: Geometric (camera positions to center) ===
            # Estimate viewing direction as: geometric_center - camera_position
            geometric_viewing_dirs = geometric_center - camera_array
            geometric_viewing_dirs = geometric_viewing_dirs / (
                np.linalg.norm(geometric_viewing_dirs, axis=1, keepdims=True) + 1e-8
            )
            mean_geometric_dir = np.mean(geometric_viewing_dirs, axis=0)
            mean_geometric_dir /= np.linalg.norm(mean_geometric_dir) + 1e-8
            self._ref_normal_geometric = -mean_geometric_dir
            
            # === Approach 2: Orientation (actual camera viewing directions) ===
            if use_orientations and len(self.camera_orientations) == len(self.camera_positions):
                # Camera viewing direction is -Z axis of the camera coordinate system
                # For rotation matrix R, the viewing direction in world coords is R @ [0, 0, 1]
                # (or -R[:, 2] depending on convention - COLMAP uses +Z forward after rotation)
                orientation_viewing_dirs = []
                for R in self.camera_orientations:
                    R = np.asarray(R)
                    # COLMAP convention: camera looks along +Z in camera space
                    # In world space: viewing_dir = R.T @ [0, 0, 1] = R.T[:, 2]
                    # But we want the direction the camera is looking, which is the
                    # third column of R transposed (world-to-camera becomes camera-to-world)
                    viewing_dir = R.T @ np.array([0, 0, 1])
                    viewing_dir /= np.linalg.norm(viewing_dir) + 1e-8
                    orientation_viewing_dirs.append(viewing_dir)
                
                orientation_viewing_dirs = np.array(orientation_viewing_dirs)
                mean_orientation_dir = np.mean(orientation_viewing_dirs, axis=0)
                mean_orientation_dir /= np.linalg.norm(mean_orientation_dir) + 1e-8
                self._ref_normal_orientation = -mean_orientation_dir
                
                # Use orientation-based viewing directions for refinement
                actual_viewing_dirs = orientation_viewing_dirs
                self.reference_plane_normal = self._ref_normal_orientation
                print("  Using actual camera orientations for viewing directions")
            else:
                # Fall back to geometric approach
                actual_viewing_dirs = geometric_viewing_dirs
                self.reference_plane_normal = self._ref_normal_geometric
                self._ref_normal_orientation = None
                if use_orientations and len(self.camera_orientations) != len(self.camera_positions):
                    print(f"  Warning: {len(self.camera_orientations)} orientations for "
                          f"{len(self.camera_positions)} cameras - using geometric fallback")
                else:
                    print("  Using geometric viewing directions (no orientations provided)")
            
            # Refine center: find the point closest to all camera rays
            self.center_of_gravity = self._refine_center_from_cameras(
                geometric_center, camera_array, actual_viewing_dirs
            )
        else:
            self.center_of_gravity = geometric_center
            self.reference_plane_normal = np.array([0, 0, 1])  # Default: Z-up
            self._ref_normal_geometric = self.reference_plane_normal.copy()
            self._ref_normal_orientation = None
        
        self.reference_plane_origin = self.center_of_gravity.copy()
        
        print(f"Center of gravity: {self.center_of_gravity}")
        print(f"Reference plane normal: {self.reference_plane_normal}")
        
        return self.center_of_gravity
    
    def _refine_center_from_cameras(self, initial_center: np.ndarray,
                                    camera_positions: np.ndarray,
                                    viewing_directions: np.ndarray = None) -> np.ndarray:
        """
        Refine center by finding point that minimizes distance to camera rays.
        
        For half-circle camera arrangements, the center is where all viewing
        rays approximately intersect.
        
        Args:
            initial_center: Initial guess for center (geometric center)
            camera_positions: (N, 3) array of camera positions
            viewing_directions: Optional (N, 3) array of viewing directions.
                               If None, computed as initial_center - camera_positions.
        """
        rays_origins = camera_positions
        
        if viewing_directions is not None:
            rays_directions = viewing_directions
        else:
            # Fallback: viewing rays from each camera toward initial center
            rays_directions = initial_center - camera_positions
            rays_directions = rays_directions / (np.linalg.norm(rays_directions, axis=1, keepdims=True) + 1e-8)
        
        # Find point minimizing sum of squared distances to all rays
        # Using least squares: minimize ||A*x - b||^2
        n = len(rays_origins)
        A = np.zeros((3 * n, 3))
        b = np.zeros(3 * n)
        
        for i, (origin, direction) in enumerate(zip(rays_origins, rays_directions)):
            # Project onto plane perpendicular to ray
            I = np.eye(3)
            P = I - np.outer(direction, direction)
            A[3*i:3*i+3, :] = P
            b[3*i:3*i+3] = P @ origin
        
        # Solve least squares
        center, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        return center
    
    def visualize_center_of_gravity(self, ray_length: float = None, 
                                    camera_size: float = None,
                                    save_path: Path = None):
        """
        Visualize the center of gravity computation with cameras and viewing rays.
        
        Shows:
        - Camera positions as pyramids/markers with proper size
        - Viewing direction arrows from each camera
        - Intersection point (center of gravity)
        - Point cloud for context
        
        Args:
            ray_length: Length of viewing rays (auto-computed if None)
            camera_size: Size of camera markers (auto-computed if None)
            save_path: Optional path to save the figure
        """
        if self.center_of_gravity is None:
            self.compute_center_of_gravity()
        
        if len(self.camera_positions) == 0:
            print("No camera positions set. Cannot visualize camera arrangement.")
            return
        
        camera_array = np.array(self.camera_positions)
        
        # Auto-compute sizes based on scene scale
        if ray_length is None:
            # Ray length = distance from cameras to center
            distances = np.linalg.norm(camera_array - self.center_of_gravity, axis=1)
            ray_length = np.mean(distances) * 1.2
        
        if camera_size is None:
            # Camera size proportional to scene
            scene_scale = np.linalg.norm(camera_array.max(axis=0) - camera_array.min(axis=0))
            camera_size = scene_scale * 0.03
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        
        # Main 3D view
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        self._plot_cameras_and_rays(ax1, camera_array, ray_length, camera_size, 
                                    title='Camera Arrangement & Center of Gravity')
        
        # Top view (XY plane)
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        self._plot_cameras_and_rays(ax2, camera_array, ray_length, camera_size,
                                    title='Top View (looking down Z)')
        ax2.view_init(elev=90, azim=0)
        
        # Side view (XZ plane)
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        self._plot_cameras_and_rays(ax3, camera_array, ray_length, camera_size,
                                    title='Side View (looking along Y)')
        ax3.view_init(elev=0, azim=0)
        
        # Front view (YZ plane)
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        self._plot_cameras_and_rays(ax4, camera_array, ray_length, camera_size,
                                    title='Front View (looking along X)')
        ax4.view_init(elev=0, azim=90)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved center of gravity visualization to {save_path}")
        
        plt.show()
    
    def _plot_cameras_and_rays(self, ax, camera_array: np.ndarray, 
                               ray_length: float, camera_size: float,
                               title: str = ''):
        """Helper to plot cameras, viewing rays, and intersection point."""
        n_cameras = len(camera_array)
        
        # Subsample point cloud for context
        if self.points is not None:
            max_points = 1000
            if len(self.points) > max_points:
                indices = np.random.choice(len(self.points), max_points, replace=False)
                pts = self.points[indices]
            else:
                pts = self.points
            
            # Plot point cloud faintly
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                      c='gray', s=0.5, alpha=0.2, label='Point Cloud')
        
        # Compute viewing directions toward center
        viewing_dirs = self.center_of_gravity - camera_array
        viewing_dirs = viewing_dirs / (np.linalg.norm(viewing_dirs, axis=1, keepdims=True) + 1e-8)
        
        # Color cameras by index for identification
        colors = plt.cm.hsv(np.linspace(0, 0.9, n_cameras))
        
        # Plot cameras as markers
        ax.scatter(camera_array[:, 0], camera_array[:, 1], camera_array[:, 2],
                  c=colors, s=100, marker='^', edgecolors='black', linewidth=0.5,
                  label='Cameras', zorder=5)
        
        # Plot viewing direction rays
        for i, (cam_pos, view_dir, color) in enumerate(zip(camera_array, viewing_dirs, colors)):
            # Ray from camera toward center (and beyond)
            ray_end = cam_pos + view_dir * ray_length
            
            ax.plot([cam_pos[0], ray_end[0]], 
                   [cam_pos[1], ray_end[1]], 
                   [cam_pos[2], ray_end[2]],
                   color=color, alpha=0.6, linewidth=1)
            
            # Add arrow head using quiver (shorter arrow for visibility)
            arrow_scale = ray_length * 0.15
            ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                     view_dir[0] * arrow_scale, 
                     view_dir[1] * arrow_scale, 
                     view_dir[2] * arrow_scale,
                     color=color, arrow_length_ratio=0.3, linewidth=1.5)
        
        # Plot center of gravity (intersection point)
        ax.scatter([self.center_of_gravity[0]], 
                  [self.center_of_gravity[1]], 
                  [self.center_of_gravity[2]],
                  c='red', s=200, marker='*', edgecolors='darkred', 
                  linewidth=1, label='Center of Gravity', zorder=10)
        
        # Plot reference plane normal
        normal_length = ray_length * 0.3
        normal_end = self.center_of_gravity + self.reference_plane_normal * normal_length
        ax.quiver(self.center_of_gravity[0], self.center_of_gravity[1], self.center_of_gravity[2],
                 self.reference_plane_normal[0] * normal_length,
                 self.reference_plane_normal[1] * normal_length,
                 self.reference_plane_normal[2] * normal_length,
                 color='cyan', arrow_length_ratio=0.2, linewidth=3,
                 label='Plane Normal')
        
        # Add geometric center for comparison
        if self.points is not None:
            geo_center = np.mean(self.points, axis=0)
            ax.scatter([geo_center[0]], [geo_center[1]], [geo_center[2]],
                      c='orange', s=100, marker='o', edgecolors='darkorange',
                      linewidth=1, label='Geometric Center', zorder=9)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend(loc='upper left', fontsize=8)
        
        # Equal aspect ratio
        self._set_axes_equal(ax)
    
    def _set_axes_equal(self, ax):
        """Set equal aspect ratio for 3D plot."""
        if self.points is not None:
            pts = self.points
        else:
            pts = np.array(self.camera_positions)
        
        max_range = np.array([
            pts[:, 0].max() - pts[:, 0].min(),
            pts[:, 1].max() - pts[:, 1].min(),
            pts[:, 2].max() - pts[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (pts[:, 0].max() + pts[:, 0].min()) * 0.5
        mid_y = (pts[:, 1].max() + pts[:, 1].min()) * 0.5
        mid_z = (pts[:, 2].max() + pts[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    def compute_center_of_gravity_from_normals(self, voxel_size: float = 0.01,
                                                use_median: bool = False,
                                                min_points_per_voxel: int = 5) -> np.ndarray:
        """
        Compute center of gravity using point cloud normals.
        
        This alternative method uses the surface normals to determine the
        reference plane orientation. Normals are computed per voxel to reduce
        noise and computation.
        
        Args:
            voxel_size: Size of voxels for normal computation (in same units as points).
                       Larger values = fewer voxels, faster but less detail.
            use_median: If True, use median of normals (robust to outliers).
                       If False, use mean (default).
            min_points_per_voxel: Minimum points required for a valid voxel.
            
        Returns:
            Center of gravity as (3,) array
        """
        if self.points is None:
            raise ValueError("No points loaded")
        
        print(f"Computing center of gravity from normals (voxel_size={voxel_size})...")
        
        # Compute normals if not available
        if self.normals is None:
            print("  Estimating normals...")
            self._estimate_normals()
        
        # Voxelize the point cloud
        print("  Voxelizing point cloud...")
        voxel_indices = np.floor(self.points / voxel_size).astype(int)
        unique_voxels, inverse_indices = np.unique(
            voxel_indices, axis=0, return_inverse=True
        )
        
        n_voxels = len(unique_voxels)
        print(f"  Created {n_voxels} voxels")
        
        # Compute per-voxel centers and normals
        voxel_centers = []
        voxel_normals = []
        voxel_counts = []
        
        for i in range(n_voxels):
            mask = inverse_indices == i
            count = np.sum(mask)
            
            if count < min_points_per_voxel:
                continue
            
            pts = self.points[mask]
            nrms = self.normals[mask]
            
            # Compute voxel center
            center = pts.mean(axis=0)
            voxel_centers.append(center)
            voxel_counts.append(count)
            
            # Compute voxel normal (average of point normals)
            # First, ensure normals are consistently oriented
            mean_normal = nrms.mean(axis=0)
            mean_normal_norm = np.linalg.norm(mean_normal)
            
            if mean_normal_norm > 1e-8:
                mean_normal /= mean_normal_norm
            else:
                # Fallback: use PCA to estimate normal
                centered = pts - center
                if len(centered) >= 3:
                    cov = np.cov(centered.T)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    mean_normal = eigenvectors[:, 0]  # Smallest eigenvalue
                else:
                    mean_normal = np.array([0, 0, 1])
            
            voxel_normals.append(mean_normal)
        
        self._voxel_centers_normals = np.array(voxel_centers)
        self._voxel_normals = np.array(voxel_normals)
        self._voxel_counts_normals = np.array(voxel_counts)
        
        print(f"  Valid voxels: {len(voxel_centers)}")
        
        # Compute reference plane normal from voxel normals
        if use_median:
            # Median is more robust to outliers
            # For 3D vectors, compute component-wise median
            ref_normal = np.median(self._voxel_normals, axis=0)
        else:
            # Weighted mean by voxel point count
            weights = np.array(voxel_counts).astype(float)
            weights /= weights.sum()
            ref_normal = np.average(self._voxel_normals, axis=0, weights=weights)
        
        # Normalize
        ref_normal /= np.linalg.norm(ref_normal) + 1e-8
        
        # Ensure normal points "up" (toward cameras if available, else +Z)
        if len(self.camera_positions) > 0:
            camera_center = np.mean(self.camera_positions, axis=0)
            geo_center = np.mean(self.points, axis=0)
            up_direction = camera_center - geo_center
            up_direction /= np.linalg.norm(up_direction) + 1e-8
            
            if np.dot(ref_normal, up_direction) < 0:
                ref_normal = -ref_normal
        else:
            if ref_normal[2] < 0:
                ref_normal = -ref_normal
        
        self.reference_plane_normal = ref_normal
        
        # Store as point cloud-based reference normal
        self._ref_normal_pointcloud = ref_normal.copy()
        
        # Center of gravity is geometric center of point cloud
        self.center_of_gravity = np.mean(self.points, axis=0)
        self.reference_plane_origin = self.center_of_gravity.copy()
        
        print(f"Center of gravity: {self.center_of_gravity}")
        print(f"Reference plane normal (from {len(voxel_normals)} voxels): {self.reference_plane_normal}")
        
        return self.center_of_gravity
    
    def visualize_normals_center_of_gravity_rerun(self, 
                                                   session_name: str = "NormalsCenterOfGravity",
                                                   normal_scale: float = None,
                                                   max_voxels_display: int = 10000):
        """
        Visualize the normal-based center of gravity computation using Rerun.
        
        Shows:
        - Point cloud with colors
        - Voxel centers as points
        - Voxel normals as arrows
        - Reference plane normal (mean/median of all normals)
        - Center of gravity
        
        Args:
            session_name: Name for Rerun session
            normal_scale: Scale for normal arrows (auto-computed if None)
            max_voxels_display: Maximum voxels to display for performance
        """
        try:
            import rerun as rr
        except ImportError:
            print("Rerun not available. Install with: pip install rerun-sdk")
            return False
        
        try:
            rr.init(session_name, spawn=True)
        except RuntimeError as e:
            print(f"Could not launch Rerun: {e}")
            return False
        
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        
        # Ensure normals center of gravity is computed
        if not hasattr(self, '_voxel_centers_normals') or self._voxel_centers_normals is None:
            print("Computing center of gravity from normals first...")
            self.compute_center_of_gravity_from_normals()
        
        # Auto-compute normal scale
        if normal_scale is None:
            scene_scale = np.linalg.norm(self.points.max(axis=0) - self.points.min(axis=0))
            normal_scale = scene_scale * 0.02
        
        # === Log static elements ===
        
        # Point cloud
        if self.colors is not None:
            colors = self.colors
            if colors.max() > 1:
                colors = colors / 255.0
        else:
            colors = np.ones((len(self.points), 3)) * 0.6
        
        # Subsample point cloud for performance
        max_points = 50000
        if len(self.points) > max_points:
            indices = np.random.choice(len(self.points), max_points, replace=False)
            pts_display = self.points[indices]
            colors_display = colors[indices]
        else:
            pts_display = self.points
            colors_display = colors
        
        rr.log("scene/pointcloud", rr.Points3D(
            positions=pts_display,
            colors=colors_display,
            radii=0.0005
        ), static=True)
        
        # Voxel centers
        voxel_centers = self._voxel_centers_normals
        voxel_normals = self._voxel_normals
        voxel_counts = self._voxel_counts_normals
        
        # Subsample voxels if too many
        if len(voxel_centers) > max_voxels_display:
            # Prioritize voxels with more points
            priority_indices = np.argsort(voxel_counts)[::-1][:max_voxels_display]
            voxel_centers_display = voxel_centers[priority_indices]
            voxel_normals_display = voxel_normals[priority_indices]
            voxel_counts_display = voxel_counts[priority_indices]
        else:
            voxel_centers_display = voxel_centers
            voxel_normals_display = voxel_normals
            voxel_counts_display = voxel_counts
        
        # Color voxels by how aligned they are with reference normal
        alignment = np.abs(np.dot(voxel_normals_display, self.reference_plane_normal))
        # Colormap: blue (aligned) to red (perpendicular)
        voxel_colors = np.zeros((len(alignment), 3))
        voxel_colors[:, 0] = 1 - alignment  # Red channel
        voxel_colors[:, 2] = alignment       # Blue channel
        
        rr.log("scene/voxels/centers", rr.Points3D(
            positions=voxel_centers_display,
            colors=voxel_colors,
            radii=0.003
        ), static=True)
        
        # Voxel normals as arrows
        rr.log("scene/voxels/normals", rr.Arrows3D(
            origins=voxel_centers_display,
            vectors=voxel_normals_display * normal_scale,
            colors=voxel_colors
        ), static=True)
        
        # Center of gravity
        rr.log("scene/center_of_gravity", rr.Points3D(
            positions=[self.center_of_gravity],
            colors=[[1.0, 1.0, 0.0]],  # Yellow
            radii=0.015
        ), static=True)
        
        # Reference plane normal (large arrow from center)
        ref_normal_scale = normal_scale * 5
        rr.log("scene/reference_normal", rr.Arrows3D(
            origins=[self.center_of_gravity],
            vectors=[self.reference_plane_normal * ref_normal_scale],
            colors=[[0.0, 1.0, 1.0]]  # Cyan
        ), static=True)
        
        # Camera positions if available
        if len(self.camera_positions) > 0:
            rr.log("scene/cameras", rr.Points3D(
                positions=self.camera_positions,
                colors=[[0.0, 0.5, 1.0]] * len(self.camera_positions),
                radii=0.008
            ), static=True)
        
        # === Log info text ===
        info_text = f"""Normal-Based Center of Gravity Analysis
========================================
Total points: {len(self.points)}
Total voxels: {len(voxel_centers)}
Displayed voxels: {len(voxel_centers_display)}

Center of Gravity: [{self.center_of_gravity[0]:.4f}, {self.center_of_gravity[1]:.4f}, {self.center_of_gravity[2]:.4f}]
Reference Normal: [{self.reference_plane_normal[0]:.4f}, {self.reference_plane_normal[1]:.4f}, {self.reference_plane_normal[2]:.4f}]

Color Legend:
- Blue voxels: Normals aligned with reference
- Red voxels: Normals perpendicular to reference
- Yellow point: Center of gravity
- Cyan arrow: Reference plane normal
"""
        rr.log("info", rr.TextDocument(info_text), static=True)
        
        # === Timeseries: Show normal distribution ===
        # Step through and highlight voxels by alignment
        n_steps = 5
        thresholds = np.linspace(0.2, 1.0, n_steps)
        
        for step, threshold in enumerate(thresholds):
            rr.set_time("alignment_filter", sequence=step)
            
            mask = alignment >= threshold
            if mask.any():
                filtered_centers = voxel_centers_display[mask]
                filtered_normals = voxel_normals_display[mask]
                filtered_colors = voxel_colors[mask]
                
                rr.log("filtered/voxels", rr.Points3D(
                    positions=filtered_centers,
                    colors=filtered_colors,
                    radii=0.004
                ))
                
                rr.log("filtered/normals", rr.Arrows3D(
                    origins=filtered_centers,
                    vectors=filtered_normals * normal_scale,
                    colors=filtered_colors
                ))
                
                rr.log("filtered/info", rr.TextDocument(
                    f"Alignment threshold: ≥ {threshold:.2f}\n"
                    f"Voxels shown: {mask.sum()} / {len(alignment)}"
                ))
        
        print(f"Rerun visualization launched: {session_name}")
        return True

    def visualize_reference_plane_normals_rerun(self, 
                                                 session_name: str = "ReferencePlaneNormals",
                                                 arrow_scale: float = None):
        """
        Visualize the 3 different approaches for calculating the reference plane normal.
        
        Shows:
        - Point cloud for context
        - Camera positions with their actual viewing directions
        - 3 reference plane normals as arrows:
          1. Geometric: From camera positions toward geometric center (yellow)
          2. Orientation: From actual camera orientations (green)
          3. Point cloud: From surface normals (cyan)
        - Comparison information
        
        Args:
            session_name: Name for Rerun session
            arrow_scale: Scale for normal arrows (auto-computed if None)
        """
        try:
            import rerun as rr
        except ImportError:
            print("Rerun not available. Install with: pip install rerun-sdk")
            return False
        
        try:
            rr.init(session_name, spawn=True)
        except RuntimeError as e:
            print(f"Could not launch Rerun: {e}")
            return False
        
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        
        # Ensure we have computed something
        if self.center_of_gravity is None:
            print("Computing center of gravity first...")
            self.compute_center_of_gravity()
        
        # Auto-compute arrow scale based on scene size
        if arrow_scale is None:
            scene_scale = np.linalg.norm(self.points.max(axis=0) - self.points.min(axis=0))
            arrow_scale = scene_scale * 0.15
        
        # === Log point cloud ===
        if self.colors is not None:
            colors = self.colors
            if colors.max() > 1:
                colors = colors / 255.0
        else:
            colors = np.ones((len(self.points), 3)) * 0.6
        
        # Subsample for performance
        max_points = 50000
        if len(self.points) > max_points:
            indices = np.random.choice(len(self.points), max_points, replace=False)
            pts_display = self.points[indices]
            colors_display = colors[indices]
        else:
            pts_display = self.points
            colors_display = colors
        
        rr.log("scene/pointcloud", rr.Points3D(
            positions=pts_display,
            colors=colors_display,
            radii=0.0005
        ), static=True)
        
        # === Log center of gravity ===
        rr.log("scene/center_of_gravity", rr.Points3D(
            positions=[self.center_of_gravity],
            colors=[[1.0, 0.0, 0.0]],  # Red
            radii=0.015
        ), static=True)
        
        # === Log camera positions and viewing directions ===
        if len(self.camera_positions) > 0:
            camera_array = np.array(self.camera_positions)
            
            # Camera positions
            rr.log("scene/cameras/positions", rr.Points3D(
                positions=camera_array,
                colors=[[0.0, 0.5, 1.0]] * len(camera_array),
                radii=0.008
            ), static=True)
            
            # Geometric viewing directions (toward center)
            geometric_dirs = self.center_of_gravity - camera_array
            geometric_dirs = geometric_dirs / (np.linalg.norm(geometric_dirs, axis=1, keepdims=True) + 1e-8)
            
            rr.log("scene/cameras/viewing_geometric", rr.Arrows3D(
                origins=camera_array,
                vectors=geometric_dirs * arrow_scale * 0.5,
                colors=[[1.0, 0.8, 0.0, 0.5]] * len(camera_array)  # Yellow, semi-transparent
            ), static=True)
            
            # Actual orientation-based viewing directions (if available)
            if len(self.camera_orientations) == len(self.camera_positions):
                orientation_dirs = []
                for R in self.camera_orientations:
                    R = np.asarray(R)
                    # COLMAP convention: viewing direction is R.T @ [0, 0, 1]
                    viewing_dir = R.T @ np.array([0, 0, 1])
                    viewing_dir /= np.linalg.norm(viewing_dir) + 1e-8
                    orientation_dirs.append(viewing_dir)
                
                orientation_dirs = np.array(orientation_dirs)
                
                rr.log("scene/cameras/viewing_orientation", rr.Arrows3D(
                    origins=camera_array,
                    vectors=orientation_dirs * arrow_scale * 0.5,
                    colors=[[0.0, 1.0, 0.3, 0.7]] * len(camera_array)  # Green
                ), static=True)
        
        # === Log the 3 reference plane normals ===
        normals_info = []
        
        # 1. Geometric approach (yellow)
        if self._ref_normal_geometric is not None:
            rr.log("reference_normals/geometric", rr.Arrows3D(
                origins=[self.center_of_gravity],
                vectors=[self._ref_normal_geometric * arrow_scale],
                colors=[[1.0, 0.8, 0.0]]  # Yellow
            ), static=True)
            rr.log("reference_normals/geometric/label", rr.Points3D(
                positions=[self.center_of_gravity + self._ref_normal_geometric * arrow_scale * 1.1],
                colors=[[1.0, 0.8, 0.0]],
                radii=0.003
            ), static=True)
            normals_info.append(f"Geometric (yellow): [{self._ref_normal_geometric[0]:.4f}, "
                               f"{self._ref_normal_geometric[1]:.4f}, {self._ref_normal_geometric[2]:.4f}]")
        
        # 2. Orientation approach (green)
        if self._ref_normal_orientation is not None:
            rr.log("reference_normals/orientation", rr.Arrows3D(
                origins=[self.center_of_gravity],
                vectors=[self._ref_normal_orientation * arrow_scale],
                colors=[[0.0, 1.0, 0.3]]  # Green
            ), static=True)
            rr.log("reference_normals/orientation/label", rr.Points3D(
                positions=[self.center_of_gravity + self._ref_normal_orientation * arrow_scale * 1.1],
                colors=[[0.0, 1.0, 0.3]],
                radii=0.003
            ), static=True)
            normals_info.append(f"Orientation (green): [{self._ref_normal_orientation[0]:.4f}, "
                               f"{self._ref_normal_orientation[1]:.4f}, {self._ref_normal_orientation[2]:.4f}]")
        
        # 3. Point cloud approach (cyan)
        if self._ref_normal_pointcloud is not None:
            rr.log("reference_normals/pointcloud", rr.Arrows3D(
                origins=[self.center_of_gravity],
                vectors=[self._ref_normal_pointcloud * arrow_scale],
                colors=[[0.0, 1.0, 1.0]]  # Cyan
            ), static=True)
            rr.log("reference_normals/pointcloud/label", rr.Points3D(
                positions=[self.center_of_gravity + self._ref_normal_pointcloud * arrow_scale * 1.1],
                colors=[[0.0, 1.0, 1.0]],
                radii=0.003
            ), static=True)
            normals_info.append(f"Point Cloud (cyan): [{self._ref_normal_pointcloud[0]:.4f}, "
                               f"{self._ref_normal_pointcloud[1]:.4f}, {self._ref_normal_pointcloud[2]:.4f}]")
        
        # === Compute and display angles between normals ===
        angles_text = "\nAngles between approaches:"
        
        if self._ref_normal_geometric is not None and self._ref_normal_orientation is not None:
            dot = np.clip(np.dot(self._ref_normal_geometric, self._ref_normal_orientation), -1, 1)
            angle_deg = np.degrees(np.arccos(dot))
            angles_text += f"\n  Geometric ↔ Orientation: {angle_deg:.2f}°"
        
        if self._ref_normal_geometric is not None and self._ref_normal_pointcloud is not None:
            dot = np.clip(np.dot(self._ref_normal_geometric, self._ref_normal_pointcloud), -1, 1)
            angle_deg = np.degrees(np.arccos(dot))
            angles_text += f"\n  Geometric ↔ Point Cloud: {angle_deg:.2f}°"
        
        if self._ref_normal_orientation is not None and self._ref_normal_pointcloud is not None:
            dot = np.clip(np.dot(self._ref_normal_orientation, self._ref_normal_pointcloud), -1, 1)
            angle_deg = np.degrees(np.arccos(dot))
            angles_text += f"\n  Orientation ↔ Point Cloud: {angle_deg:.2f}°"
        
        # === Log info text ===
        info_text = f"""Reference Plane Normal Comparison
========================================

Center of Gravity: [{self.center_of_gravity[0]:.4f}, {self.center_of_gravity[1]:.4f}, {self.center_of_gravity[2]:.4f}]

Reference Plane Normals (3 approaches):

1. Geometric Approach (Yellow Arrow)
   Direction from camera positions toward geometric center.
   Best for: Simple setups, fallback method.
   
2. Orientation Approach (Green Arrow)
   Actual camera viewing directions from rotation matrices.
   Best for: Accurate camera data, precise setups.
   
3. Point Cloud Approach (Cyan Arrow)
   Surface normals from the reconstructed point cloud.
   Best for: Dense reconstructions, surface-aware analysis.

{chr(10).join(normals_info)}
{angles_text}

Camera Viewing Directions:
- Yellow arrows: Geometric (toward center)
- Green arrows: Actual orientation (from rotation matrices)

Number of cameras: {len(self.camera_positions)}
Number of orientations: {len(self.camera_orientations)}
"""
        rr.log("info", rr.TextDocument(info_text), static=True)
        
        print(f"Rerun visualization launched: {session_name}")
        print(f"  Available approaches: {len([n for n in [self._ref_normal_geometric, self._ref_normal_orientation, self._ref_normal_pointcloud] if n is not None])}")
        return True

    # =========================================================================
    # 2. HEIGHT PROFILE CALCULATION
    # =========================================================================
    
    def compute_height_profile(self, grid_resolution: float = None) -> np.ndarray:
        """
        Compute height profile relative to the reference plane.
        
        Height is measured as signed distance from each point to the
        reference plane (defined by center of gravity and plane normal).
        
        Args:
            grid_resolution: Resolution for height grid (in same units as points)
            
        Returns:
            Per-point height values
        """
        if self.center_of_gravity is None:
            self.compute_center_of_gravity()
        
        if grid_resolution:
            self.grid_resolution = grid_resolution
        
        # Compute signed distance to reference plane
        # h = (p - origin) · normal
        vectors_to_plane = self.points - self.reference_plane_origin
        self.height_values = np.dot(vectors_to_plane, self.reference_plane_normal)
        
        print(f"Height range: [{self.height_values.min():.4f}, {self.height_values.max():.4f}]")
        
        # Create gridded height field for gradient/curvature computation
        self._create_height_grid()
        
        return self.height_values
    
    def _create_height_grid(self):
        """Create a regular grid of height values for differential analysis."""
        # Project points onto reference plane
        # Create local 2D coordinate system on the plane
        
        # Find two orthogonal axes on the plane
        n = self.reference_plane_normal
        if abs(n[2]) < 0.9:
            u = np.cross(n, [0, 0, 1])
        else:
            u = np.cross(n, [1, 0, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        
        # Project points to 2D plane coordinates
        centered = self.points - self.reference_plane_origin
        x_2d = np.dot(centered, u)
        y_2d = np.dot(centered, v)
        
        # Create regular grid
        margin = 0.02  # 2cm margin
        x_min, x_max = x_2d.min() - margin, x_2d.max() + margin
        y_min, y_max = y_2d.min() - margin, y_2d.max() + margin
        
        nx = int((x_max - x_min) / self.grid_resolution) + 1
        ny = int((y_max - y_min) / self.grid_resolution) + 1
        
        # Limit grid size
        max_grid = 500
        if nx > max_grid or ny > max_grid:
            scale = max(nx, ny) / max_grid
            self.grid_resolution *= scale
            nx = min(nx, max_grid)
            ny = min(ny, max_grid)
        
        self.grid_x = np.linspace(x_min, x_max, nx)
        self.grid_y = np.linspace(y_min, y_max, ny)
        
        # Interpolate height values onto grid
        grid_xx, grid_yy = np.meshgrid(self.grid_x, self.grid_y)
        
        try:
            self.height_grid = griddata(
                (x_2d, y_2d), self.height_values,
                (grid_xx, grid_yy), method='linear', fill_value=np.nan
            )
        except Exception as e:
            print(f"Grid interpolation warning: {e}")
            self.height_grid = griddata(
                (x_2d, y_2d), self.height_values,
                (grid_xx, grid_yy), method='nearest'
            )
        
        # Store 2D coordinates for later use
        self._x_2d = x_2d
        self._y_2d = y_2d
        self._u_axis = u
        self._v_axis = v
        
        print(f"Created height grid: {nx}x{ny} at resolution {self.grid_resolution:.4f}")
    
    # =========================================================================
    # 3. CURVATURE AND GRADIENT COMPUTATIONS
    # =========================================================================
    
    def compute_gradient_magnitude(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient magnitude of the height field.
        
        E_g(x,y) = ||∇h(x,y)||
        
        This highlights areas with steep slopes (potential hole edges).
        
        Returns:
            Tuple of (grid gradient magnitude, per-point gradient magnitude)
        """
        if self.height_grid is None:
            self.compute_height_profile()
        
        # Compute gradients on grid
        h = self.height_grid.copy()
        h = np.nan_to_num(h, nan=np.nanmean(h))
        
        # Smooth slightly to reduce noise
        h_smooth = gaussian_filter(h, sigma=1.0)
        
        # Gradient using central differences
        dy, dx = np.gradient(h_smooth, self.grid_resolution)
        
        # Gradient magnitude
        grad_mag_grid = np.sqrt(dx**2 + dy**2)
        
        # Interpolate back to original points
        grid_xx, grid_yy = np.meshgrid(self.grid_x, self.grid_y)
        self.gradient_magnitude = griddata(
            (grid_xx.ravel(), grid_yy.ravel()), grad_mag_grid.ravel(),
            (self._x_2d, self._y_2d), method='linear', fill_value=0
        )
        
        # Store grid gradients
        self._dx_grid = dx
        self._dy_grid = dy
        self._grad_mag_grid = grad_mag_grid
        
        print(f"Gradient magnitude range: [{self.gradient_magnitude.min():.4f}, {self.gradient_magnitude.max():.4f}]")
        
        return grad_mag_grid, self.gradient_magnitude
    
    def compute_mean_curvature(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean curvature of the height field.
        
        For a height field h(x,y), mean curvature is:
        H = ∇ · (∇h / sqrt(1 + |∇h|²))
        
        Expanded:
        H = ((1 + h_x²)h_yy - 2h_x h_y h_xy + (1 + h_y²)h_xx) / (1 + h_x² + h_y²)^(3/2)
        
        Positive curvature = convex (peak), Negative = concave (hole/valley)
        
        Returns:
            Tuple of (grid mean curvature, per-point mean curvature)
        """
        if self.height_grid is None:
            self.compute_height_profile()
        
        h = self.height_grid.copy()
        h = np.nan_to_num(h, nan=np.nanmean(h))
        
        # Smooth to reduce noise before second derivatives
        h_smooth = gaussian_filter(h, sigma=2.0)
        
        dx = self.grid_resolution
        
        # First derivatives
        h_y, h_x = np.gradient(h_smooth, dx)
        
        # Second derivatives
        h_yy, h_xy = np.gradient(h_y, dx)
        _, h_xx = np.gradient(h_x, dx)
        
        # Mean curvature formula
        numerator = (1 + h_x**2) * h_yy - 2 * h_x * h_y * h_xy + (1 + h_y**2) * h_xx
        denominator = (1 + h_x**2 + h_y**2) ** 1.5
        
        mean_curv_grid = numerator / (denominator + 1e-10)
        
        # Clamp extreme values
        percentile = 99
        vmin, vmax = np.nanpercentile(mean_curv_grid, [100-percentile, percentile])
        mean_curv_grid = np.clip(mean_curv_grid, vmin, vmax)
        
        # Interpolate back to original points
        grid_xx, grid_yy = np.meshgrid(self.grid_x, self.grid_y)
        self.mean_curvature = griddata(
            (grid_xx.ravel(), grid_yy.ravel()), mean_curv_grid.ravel(),
            (self._x_2d, self._y_2d), method='linear', fill_value=0
        )
        
        self._mean_curv_grid = mean_curv_grid
        
        print(f"Mean curvature range: [{self.mean_curvature.min():.4f}, {self.mean_curvature.max():.4f}]")
        print("  Negative values indicate concave regions (holes/valleys)")
        
        return mean_curv_grid, self.mean_curvature
    
    def compute_laplacian(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Laplacian of height field.
        
        ∇²h = h_xx + h_yy
        
        The Laplacian is simpler than mean curvature but still identifies
        concave regions. Negative Laplacian = local maximum (convex),
        Positive Laplacian = local minimum (concave/hole).
        
        Returns:
            Tuple of (grid Laplacian, per-point Laplacian)
        """
        if self.height_grid is None:
            self.compute_height_profile()
        
        h = self.height_grid.copy()
        h = np.nan_to_num(h, nan=np.nanmean(h))
        
        h_smooth = gaussian_filter(h, sigma=2.0)
        
        dx = self.grid_resolution
        
        # Second derivatives
        h_y, h_x = np.gradient(h_smooth, dx)
        h_yy, _ = np.gradient(h_y, dx)
        _, h_xx = np.gradient(h_x, dx)
        
        laplacian_grid = h_xx + h_yy
        
        # Interpolate to points
        grid_xx, grid_yy = np.meshgrid(self.grid_x, self.grid_y)
        self.laplacian = griddata(
            (grid_xx.ravel(), grid_yy.ravel()), laplacian_grid.ravel(),
            (self._x_2d, self._y_2d), method='linear', fill_value=0
        )
        
        self._laplacian_grid = laplacian_grid
        
        print(f"Laplacian range: [{self.laplacian.min():.4f}, {self.laplacian.max():.4f}]")
        print("  Positive values indicate concave regions (holes)")
        
        return laplacian_grid, self.laplacian
    
    def compute_hole_energy(self, alpha: float = 1.0, beta: float = 1.0, 
                           gamma: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute energy-based hole characterization.
        
        E(u) = ∫∫ [α|∇u| + β|h - μ_u|² + γ|∇h|²u] dxdy
        
        Where:
        - α|∇u|: Edge length term (penalizes boundary length)
        - β|h - μ_u|²: Depth consistency (deviation from local mean)
        - γ|∇h|²u: Slope penalty weighted by region indicator
        
        Here we compute a simplified per-point version.
        
        Args:
            alpha: Weight for edge/gradient term
            beta: Weight for depth consistency term
            gamma: Weight for slope penalty term
            
        Returns:
            Tuple of (grid energy, per-point energy)
        """
        if self.height_grid is None:
            self.compute_height_profile()
        if self._grad_mag_grid is None:
            self.compute_gradient_magnitude()
        
        h = self.height_grid.copy()
        h = np.nan_to_num(h, nan=np.nanmean(h))
        
        # Local mean (smoothed height)
        h_mean = gaussian_filter(h, sigma=5.0)
        
        # Components
        edge_term = self._grad_mag_grid  # |∇h|
        depth_deviation = (h - h_mean) ** 2  # (h - μ)²
        slope_term = self._grad_mag_grid ** 2  # |∇h|²
        
        # Create indicator for potential hole region (negative depth deviation)
        # Points below local mean are potential hole regions
        u_indicator = np.maximum(0, h_mean - h)  # Positive where h < h_mean
        u_indicator = u_indicator / (u_indicator.max() + 1e-10)
        
        # Combined energy (lower in flat regions, higher at hole boundaries)
        energy_grid = alpha * edge_term + beta * depth_deviation + gamma * slope_term * u_indicator
        
        # Normalize
        energy_grid = energy_grid / (energy_grid.max() + 1e-10)
        
        # Interpolate to points
        grid_xx, grid_yy = np.meshgrid(self.grid_x, self.grid_y)
        hole_energy = griddata(
            (grid_xx.ravel(), grid_yy.ravel()), energy_grid.ravel(),
            (self._x_2d, self._y_2d), method='linear', fill_value=0
        )
        
        self._energy_grid = energy_grid
        self.hole_energy = hole_energy
        
        print(f"Hole energy range: [{hole_energy.min():.4f}, {hole_energy.max():.4f}]")
        
        return energy_grid, hole_energy
    
    # =========================================================================
    # 4. VOXEL-BASED ANALYSIS (for sparse visualization)
    # =========================================================================
    
    def compute_voxel_properties(self, voxel_size: float = 0.01, 
                                 min_points_per_voxel: int = 5) -> Dict[str, np.ndarray]:
        """
        Compute properties per voxel for sparse/minimalistic visualization.
        
        Only aggregates properties that have been computed. Height is minimum.
        
        Args:
            voxel_size: Size of each voxel
            min_points_per_voxel: Minimum points to consider voxel valid
            
        Returns:
            Dictionary of voxel properties
        """
        if self.points is None:
            raise ValueError("No points loaded")
        
        # Height is minimum requirement
        if self.height_values is None:
            self.compute_height_profile()
        
        # Store voxel size for later use
        self._voxel_size = voxel_size
        
        # Voxelize
        voxel_indices = np.floor(self.points / voxel_size).astype(int)
        unique_voxels, inverse_indices = np.unique(
            voxel_indices, axis=0, return_inverse=True
        )
        
        n_voxels = len(unique_voxels)
        
        # Build mapping from original voxel index to filtered voxel index
        # (after removing voxels with too few points)
        valid_voxel_indices = []
        
        # Compute per-voxel properties (only for available properties)
        voxel_centers = []
        voxel_heights = []
        voxel_gradients = [] if self.gradient_magnitude is not None else None
        voxel_curvatures = [] if self.mean_curvature is not None else None
        voxel_laplacians = [] if self.laplacian is not None else None
        voxel_energies = [] if (hasattr(self, 'hole_energy') and self.hole_energy is not None) else None
        voxel_normals = [] if self.normals is not None else None
        voxel_counts = []
        
        for i in range(n_voxels):
            mask = inverse_indices == i
            count = np.sum(mask)
            
            if count < min_points_per_voxel:
                continue
            
            valid_voxel_indices.append(i)
            pts = self.points[mask]
            
            voxel_centers.append(pts.mean(axis=0))
            voxel_heights.append(self.height_values[mask].mean())
            voxel_counts.append(count)
            
            if voxel_gradients is not None:
                voxel_gradients.append(self.gradient_magnitude[mask].mean())
            if voxel_curvatures is not None:
                voxel_curvatures.append(self.mean_curvature[mask].mean())
            if voxel_laplacians is not None:
                voxel_laplacians.append(self.laplacian[mask].mean())
            if voxel_energies is not None:
                voxel_energies.append(self.hole_energy[mask].mean())
            if voxel_normals is not None:
                voxel_normals.append(self.normals[mask].mean(axis=0))
        
        self.voxel_centers = np.array(voxel_centers)
        self.voxel_properties = {
            'height': np.array(voxel_heights),
            'count': np.array(voxel_counts),
        }
        
        # Create mapping from points to filtered voxel indices
        # -1 means point belongs to a voxel that was filtered out
        valid_voxel_set = set(valid_voxel_indices)
        old_to_new_voxel = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_voxel_indices)}
        self._voxel_point_mapping = np.array([
            old_to_new_voxel.get(inverse_indices[i], -1) 
            for i in range(len(self.points))
        ])
        
        # Add optional properties
        if voxel_gradients is not None:
            self.voxel_properties['gradient'] = np.array(voxel_gradients)
        if voxel_curvatures is not None:
            self.voxel_properties['curvature'] = np.array(voxel_curvatures)
        if voxel_laplacians is not None:
            self.voxel_properties['laplacian'] = np.array(voxel_laplacians)
        if voxel_energies is not None:
            self.voxel_properties['energy'] = np.array(voxel_energies)
        if voxel_normals is not None:
            self.voxel_properties['normals'] = np.array(voxel_normals)
        
        available = list(self.voxel_properties.keys())
        print(f"Created {len(self.voxel_centers)} voxels (size={voxel_size})")
        print(f"  Available voxel properties: {available}")
        
        return self.voxel_properties
    
    # =========================================================================
    # 5. VISUALIZATION
    # =========================================================================
    
    def visualize_all_properties(self, save_path: Path = None, 
                                 show_voxels: bool = True,
                                 voxel_size: float = 0.01):
        """
        Create comprehensive visualization of all computed properties.
        
        Only visualizes properties that have been computed. Call the
        individual compute_* methods before this to include them.
        
        Args:
            save_path: Optional path to save figure
            show_voxels: If True, show voxel-based visualizations
            voxel_size: Voxel size for sparse visualization
        """
        # Check what's available - height is minimum requirement
        if self.height_values is None:
            if self.points is None:
                raise ValueError("No points loaded. Call load_point_cloud or load_from_ply first.")
            print("Computing height profile (minimum requirement)...")
            self.compute_height_profile()
        
        if show_voxels and self.voxel_centers is None:
            self.compute_voxel_properties(voxel_size)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # ===== Row 1: Grid-based visualizations =====
        
        if self.height_grid is not None:
        # 1.1 Height field
            ax1 = fig.add_subplot(3, 4, 1)
            im1 = ax1.imshow(self.height_grid, cmap='RdBu_r', origin='lower',
                            extent=[self.grid_x[0], self.grid_x[-1], 
                                self.grid_y[0], self.grid_y[-1]])
            ax1.set_title('Height Field h(x,y)\n(blue=low, red=high)')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            plt.colorbar(im1, ax=ax1, label='Height')
        
        
        # 1.2 Gradient magnitude
        if self._grad_mag_grid is not None:
            ax2 = fig.add_subplot(3, 4, 2)
            im2 = ax2.imshow(self._grad_mag_grid, cmap='hot', origin='lower',
                            extent=[self.grid_x[0], self.grid_x[-1], 
                                self.grid_y[0], self.grid_y[-1]])
            ax2.set_title('Gradient Magnitude ||∇h||\n(bright=steep slope)')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            plt.colorbar(im2, ax=ax2, label='|∇h|')
            
        # 1.3 Mean curvature
        if self._mean_curv_grid is not None:
            ax3 = fig.add_subplot(3, 4, 3)
            vmax = np.nanpercentile(np.abs(self._mean_curv_grid), 95)
            im3 = ax3.imshow(self._mean_curv_grid, cmap='coolwarm', origin='lower',
                            extent=[self.grid_x[0], self.grid_x[-1], 
                                self.grid_y[0], self.grid_y[-1]],
                            vmin=-vmax, vmax=vmax)
            ax3.set_title('Mean Curvature H\n(blue=concave/hole, red=convex)')
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
            plt.colorbar(im3, ax=ax3, label='H')
        
        # 1.4 Laplacian
        if self._laplacian_grid is not None:
            ax4 = fig.add_subplot(3, 4, 4)
            vmax = np.nanpercentile(np.abs(self._laplacian_grid), 95)
            im4 = ax4.imshow(self._laplacian_grid, cmap='coolwarm', origin='lower',
                        extent=[self.grid_x[0], self.grid_x[-1], 
                               self.grid_y[0], self.grid_y[-1]],
                        vmin=-vmax, vmax=vmax)
            ax4.set_title('Laplacian ∇²h\n(red=concave/hole, blue=convex)')
            ax4.set_xlabel('x')
            ax4.set_ylabel('y')
            plt.colorbar(im4, ax=ax4, label='∇²h')
        
        # ===== Row 2: 3D point cloud visualizations =====
        
        # 2.1 Height-colored point cloud
        if self.height_values is not None:
            ax5 = fig.add_subplot(3, 4, 5, projection='3d')
            self._plot_3d_points(ax5, self.height_values, 'RdBu_r', 
                                'Height on Point Cloud')
            
        # 2.2 Gradient-colored point cloud
        if self.gradient_magnitude is not None:
            ax6 = fig.add_subplot(3, 4, 6, projection='3d')
            self._plot_3d_points(ax6, self.gradient_magnitude, 'hot',
                                'Gradient Magnitude')
            
        # 2.3 Curvature-colored point cloud
        if self.mean_curvature is not None:
            ax7 = fig.add_subplot(3, 4, 7, projection='3d')
            self._plot_3d_points(ax7, self.mean_curvature, 'coolwarm',
                                'Mean Curvature', symmetric=True)
            
        # 2.4 Energy-colored point cloud
        if self.hole_energy is not None:
            ax8 = fig.add_subplot(3, 4, 8, projection='3d')
            self._plot_3d_points(ax8, self.hole_energy, 'viridis',
                                'Hole Energy')
        
        # ===== Row 3: Voxel-based visualizations =====
        
        if show_voxels and self.voxel_centers is not None and len(self.voxel_centers) > 0:
            # 3.1 Voxel height (with gradient arrows if available)
            ax9 = fig.add_subplot(3, 4, 9, projection='3d')
            self._plot_voxels_with_arrows(ax9, 'height', 'Voxel Height')
            
            # 3.2 Voxel curvature (if available)
            if 'curvature' in self.voxel_properties:
                ax10 = fig.add_subplot(3, 4, 10, projection='3d')
                self._plot_voxels_with_property(ax10, 'curvature', 'coolwarm',
                                               'Voxel Mean Curvature', symmetric=True)
            
            # 3.3 Voxel laplacian (if available)
            if 'laplacian' in self.voxel_properties:
                ax11 = fig.add_subplot(3, 4, 11, projection='3d')
                self._plot_voxels_with_property(ax11, 'laplacian', 'coolwarm',
                                               'Voxel Laplacian', symmetric=True)
            
            # 3.4 Hole detection summary (if we have enough properties)
            if self._can_compute_hole_score():
                ax12 = fig.add_subplot(3, 4, 12, projection='3d')
                self._plot_hole_detection_summary(ax12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
        
        plt.show()
    
    def _plot_3d_points(self, ax, values, cmap, title, symmetric=False, 
                        max_points=1000):
        """Helper to plot 3D point cloud colored by values."""
        # Subsample if too many points
        if len(self.points) > max_points:
            indices = np.random.choice(len(self.points), max_points, replace=False)
        else:
            indices = np.arange(len(self.points))
        
        pts = self.points[indices]
        vals = values[indices]
        
        if symmetric:
            vmax = np.percentile(np.abs(vals), 95)
            vmin = -vmax
        else:
            vmin, vmax = np.percentile(vals, [5, 95])
        
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=vals, cmap=cmap, s=1, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.colorbar(sc, ax=ax, shrink=0.5)
    
    def _plot_voxels_with_property(self, ax, property_name, cmap, title, 
                                   symmetric=False):
        """Helper to plot voxels colored by a property."""
        centers = self.voxel_centers
        values = self.voxel_properties[property_name]
        
        if symmetric:
            vmax = np.percentile(np.abs(values), 95)
            vmin = -vmax
        else:
            vmin, vmax = np.percentile(values, [5, 95])
        
        sc = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                       c=values, cmap=cmap, s=20, vmin=vmin, vmax=vmax,
                       alpha=0.8, edgecolors='k', linewidth=0.2)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.colorbar(sc, ax=ax, shrink=0.5)
    
    def _plot_voxels_with_arrows(self, ax, property_name, title):
        """Plot voxels with gradient direction arrows (if gradient available)."""
        centers = self.voxel_centers
        values = self.voxel_properties[property_name]
        
        # Color by property
        vmin, vmax = np.percentile(values, [5, 95])
        sc = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                       c=values, cmap='RdBu_r', s=15, vmin=vmin, vmax=vmax,
                       alpha=0.6)
        
        # Add arrows for high-gradient voxels only (if gradient available)
        if 'gradient' not in self.voxel_properties:
            ax.set_title(title)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.colorbar(sc, ax=ax, shrink=0.5)
            return
        
        gradients = self.voxel_properties['gradient']
        high_grad_threshold = np.percentile(gradients, 80)
        high_grad_mask = gradients > high_grad_threshold
        
        if 'normals' in self.voxel_properties and high_grad_mask.any():
            normals = self.voxel_properties['normals'][high_grad_mask]
            arrow_centers = centers[high_grad_mask]
            
            # Subsample arrows for clarity
            max_arrows = 50
            if len(arrow_centers) > max_arrows:
                idx = np.random.choice(len(arrow_centers), max_arrows, replace=False)
                arrow_centers = arrow_centers[idx]
                normals = normals[idx]
            
            # Scale arrows
            arrow_scale = 0.01
            ax.quiver(arrow_centers[:, 0], arrow_centers[:, 1], arrow_centers[:, 2],
                     normals[:, 0], normals[:, 1], normals[:, 2],
                     length=arrow_scale, color='green', arrow_length_ratio=0.3,
                     linewidth=1)
        
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.colorbar(sc, ax=ax, shrink=0.5)
    
    def _can_compute_hole_score(self) -> bool:
        """Check if we have enough properties to compute hole score."""
        # Need at least height and one other property
        if self.voxel_properties is None:
            return False
        required = ['height']
        optional = ['curvature', 'gradient', 'laplacian']
        has_required = all(k in self.voxel_properties for k in required)
        has_optional = any(k in self.voxel_properties for k in optional)
        return has_required and has_optional
    
    def _plot_hole_detection_summary(self, ax):
        """Plot summary highlighting detected hole region using available properties."""
        centers = self.voxel_centers
        props = self.voxel_properties
        
        # Normalize helper
        def normalize(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-10)
        
        # Build hole score from available properties
        score_components = []
        weights = []
        
        # Height is always available (low height = high hole score)
        height = props['height']
        height_norm = 1 - normalize(height)
        score_components.append(height_norm)
        weights.append(0.4)  # Base weight for height
        
        # Add curvature if available (low/negative curvature = concave = high score)
        if 'curvature' in props:
            curvature = props['curvature']
            curv_norm = 1 - normalize(curvature)
            score_components.append(curv_norm)
            weights.append(0.25)
        
        # Add gradient if available (high gradient = edge = high score)
        if 'gradient' in props:
            gradient = props['gradient']
            grad_norm = normalize(gradient)
            score_components.append(grad_norm)
            weights.append(0.15)
        
        # Add laplacian if available (positive laplacian = concave = high score)
        if 'laplacian' in props:
            laplacian = props['laplacian']
            lapl_norm = normalize(np.maximum(0, laplacian))
            score_components.append(lapl_norm)
            weights.append(0.2)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Compute weighted hole score
        hole_score = sum(w * c for w, c in zip(weights, score_components))
        
        # Threshold for hole detection
        threshold = np.percentile(hole_score, 70)
        hole_mask = hole_score > threshold
        
        # Plot all voxels faintly
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                  c='gray', s=5, alpha=0.2)
        
        # Highlight hole region
        if hole_mask.any():
            hole_centers = centers[hole_mask]
            hole_scores = hole_score[hole_mask]
            
            sc = ax.scatter(hole_centers[:, 0], hole_centers[:, 1], hole_centers[:, 2],
                           c=hole_scores, cmap='Reds', s=30, alpha=0.9,
                           edgecolors='darkred', linewidth=0.3)
            plt.colorbar(sc, ax=ax, shrink=0.5, label='Hole Score')
        
        ax.set_title(f'Hole Detection Summary\n({hole_mask.sum()} voxels detected)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    def visualize_with_rerun(self, session_name: str = "HolePropertyAnalysis"):
        """
        Visualize properties in Rerun with interactive exploration.
        
        Only visualizes properties that have been computed.
        
        Args:
            session_name: Name for Rerun session
        """
        try:
            import rerun as rr
        except ImportError:
            print("Rerun not available. Install with: pip install rerun-sdk")
            return
        
        try:
            rr.init(session_name, spawn=True)
        except RuntimeError as e:
            print(f"Could not launch Rerun: {e}")
            return
        
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        
        # Ensure minimum requirement (height)
        if self.height_values is None:
            if self.points is None:
                print("No points loaded. Call load_point_cloud or load_from_ply first.")
                return
            self.compute_height_profile()
        
        # Compute voxel properties if not done yet
        if self.voxel_centers is None:
            self.compute_voxel_properties()
        
        # Static scene elements
        self._log_rerun_static(rr)
        
        # Property timeseries (step through available properties)
        self._log_rerun_properties_timeseries(rr)
        
        print(f"Rerun visualization launched: {session_name}")
    
    def _log_rerun_static(self, rr):
        """Log static scene elements to Rerun."""
        # Base point cloud
        if self.colors is not None:
            colors = self.colors / 255.0 if self.colors.max() > 1 else self.colors
        else:
            colors = np.ones((len(self.points), 3)) * 0.5
        
        rr.log("scene/pointcloud", rr.Points3D(
            positions=self.points,
            colors=colors,
            radii=0.0005
        ), static=True)
        
        # Center of gravity
        if self.center_of_gravity is not None:
            rr.log("scene/center_of_gravity", rr.Points3D(
                positions=[self.center_of_gravity],
                colors=[[1.0, 1.0, 0.0]],
                radii=0.01
            ), static=True)
            
            # Reference plane normal
            arrow_end = self.center_of_gravity + self.reference_plane_normal * 0.05
            rr.log("scene/plane_normal", rr.Arrows3D(
                origins=[self.center_of_gravity],
                vectors=[self.reference_plane_normal * 0.05],
                colors=[[0.0, 1.0, 1.0]]
            ), static=True)
        
        # Camera positions
        if self.camera_positions:
            rr.log("scene/cameras", rr.Points3D(
                positions=self.camera_positions,
                colors=[[0.0, 0.5, 1.0]] * len(self.camera_positions),
                radii=0.005
            ), static=True)
    
    def _log_colorbar(self, rr, name: str, title: str, values: np.ndarray, cmap_name: str):
        """
        Log a colorbar/scale visualization for a property.
        
        Creates a visual scale showing the mapping from values to colors.
        """
        import io
        import base64
        
        # Create colorbar figure
        fig, ax = plt.subplots(figsize=(1.5, 6))
        
        # Get colormap
        cmap = cm.get_cmap(cmap_name)
        
        # Determine value range (handling symmetric colormaps for curvature)
        if name in ['curvature', 'laplacian']:
            vmax = np.percentile(np.abs(values), 95)
            vmin = -vmax
        else:
            vmin, vmax = np.percentile(values, [5, 95])
        
        # Create colorbar gradient
        gradient = np.linspace(0, 1, 256).reshape(-1, 1)
        ax.imshow(gradient, aspect='auto', cmap=cmap, origin='lower', extent=[0, 1, vmin, vmax])
        
        # Labels
        ax.set_ylabel(title)
        ax.set_xticks([])
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        
        # Add tick labels at key percentiles
        tick_values = np.linspace(vmin, vmax, 7)
        ax.set_yticks(tick_values)
        ax.set_yticklabels([f'{v:.4f}' for v in tick_values])
        
        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        plt.close(fig)
        
        # Log as image
        from PIL import Image
        img = Image.open(buf)
        img_array = np.array(img)
        
        rr.log(f"colorbar/{name}", rr.Image(img_array), static=True)
        
        # Also log a text-based scale for easier reading
        scale_text = f"""{title} Scale
========================================
Colormap: {cmap_name}

Value Range (5th-95th percentile):
  Min: {vmin:.6f}
  Max: {vmax:.6f}

Actual Range:
  Min: {values.min():.6f}
  Max: {values.max():.6f}

Color Mapping:
"""
        # Add sample values with their approximate color description
        sample_vals = np.linspace(vmin, vmax, 9)
        sample_norms = (sample_vals - vmin) / (vmax - vmin + 1e-10)
        sample_colors = cmap(sample_norms)
        
        for val, norm, col in zip(sample_vals, sample_norms, sample_colors):
            # Simple color description
            r, g, b = col[:3]
            if name in ['curvature', 'laplacian']:
                if val < -0.001:
                    desc = "← Concave (hole)"
                elif val > 0.001:
                    desc = "← Convex (peak)"
                else:
                    desc = "← Flat"
            else:
                desc = ""
            scale_text += f"  {val:+.4f} → RGB({r:.2f}, {g:.2f}, {b:.2f}) {desc}\n"
        
        rr.log(f"colorbar/{name}/text_scale", rr.TextDocument(scale_text), static=True)

    def _log_rerun_properties_timeseries(self, rr):
        """Log property visualizations - each property to its own path with dedicated colormap."""
        # Build list of available properties
        properties = []
        
        if self.height_values is not None:
            properties.append(('height', self.height_values, 'RdBu_r', 'Height Profile'))
        if self.gradient_magnitude is not None:
            properties.append(('gradient', self.gradient_magnitude, 'hot', 'Gradient Magnitude'))
        if self.mean_curvature is not None:
            properties.append(('curvature', self.mean_curvature, 'coolwarm', 'Mean Curvature'))
        if self.laplacian is not None:
            properties.append(('laplacian', self.laplacian, 'coolwarm', 'Laplacian'))
        if hasattr(self, 'hole_energy') and self.hole_energy is not None:
            properties.append(('energy', self.hole_energy, 'viridis', 'Hole Energy'))
        
        if not properties:
            print("No properties computed to visualize.")
            return
        
        # Log each property to its own dedicated path (static, all visible at once)
        for name, values, cmap_name, title in properties:
            # Get colormap
            cmap = cm.get_cmap(cmap_name)
            
            # Normalize values
            if name in ['curvature', 'laplacian']:
                vmax = np.percentile(np.abs(values), 95)
                norm_values = (values + vmax) / (2 * vmax)
            else:
                vmin, vmax = np.percentile(values, [5, 95])
                norm_values = (values - vmin) / (vmax - vmin + 1e-10)
            
            norm_values = np.clip(norm_values, 0, 1)
            colors = cmap(norm_values)[:, :3]
            
            # Log to property-specific path
            rr.log(f"colored_points/{name}", rr.Points3D(
                positions=self.points,
                colors=colors,
                radii=0.001
            ), static=True)
            
            # Log info for this property
            rr.log(f"colored_points/{name}/info", rr.TextDocument(
                f"Property: {title}\nColormap: {cmap_name}\nRange: [{values.min():.4f}, {values.max():.4f}]"
            ), static=True)
            
            # Log colorbar/scale for this property
            self._log_colorbar(rr, name, title, values, cmap_name)
            
                
       
            # Voxel visualization for this property
            if self.voxel_centers is not None and name in self.voxel_properties:
                voxel_vals = self.voxel_properties[name]
                
                if name in ['curvature', 'laplacian']:
                    vmax_v = np.percentile(np.abs(voxel_vals), 95)
                    norm_voxel = (voxel_vals + vmax_v) / (2 * vmax_v)
                else:
                    vmin_v, vmax_v = np.percentile(voxel_vals, [5, 95])
                    norm_voxel = (voxel_vals - vmin_v) / (vmax_v - vmin_v + 1e-10)
                
                norm_voxel = np.clip(norm_voxel, 0, 1)
                voxel_colors = cmap(norm_voxel)[:, :3]
                
                rr.log(f"colored_points/{name}/voxels", rr.Points3D(
                    positions=self.voxel_centers,
                    colors=voxel_colors,
                    radii=0.005
                ), static=True)
                
                # Log all points colored by their voxel's property value
                # Each point in a voxel gets the same color as its voxel
                if self._voxel_point_mapping is not None:
                    # Create per-point colors based on voxel assignment
                    point_colors = np.zeros((len(self.points), 3))
                    
                    # Points with valid voxel assignment get the voxel's color
                    valid_mask = self._voxel_point_mapping >= 0
                    point_colors[valid_mask] = voxel_colors[self._voxel_point_mapping[valid_mask]]
                    
                    # Points without valid voxel get gray color
                    point_colors[~valid_mask] = [0.5, 0.5, 0.5]
                    
                    rr.log(f"colored_points_voxels/{name}", rr.Points3D(
                        positions=self.points,
                        colors=point_colors,
                        radii=0.001
                    ), static=True)
                    
                    rr.log(f"colored_points_voxels/{name}/info", rr.TextDocument(
                        f"Voxel-based coloring: {title}\n"
                        f"Colormap: {cmap_name}\n"
                        f"Total points: {len(self.points)}\n"
                        f"Points in valid voxels: {valid_mask.sum()}\n"
                        f"Number of voxels: {len(self.voxel_centers)}"
                    ), static=True)
        
        # Hole detection summary (if we have enough properties)
        if not self._can_compute_hole_score():
            return
        
        # Compute hole score using available properties
        props = self.voxel_properties
        
        def normalize(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-10)
        
        # Build hole score from available properties
        score_components = []
        weights = []
        
        height = props['height']
        height_norm = 1 - normalize(height)
        score_components.append(height_norm)
        weights.append(0.4)
        
        if 'curvature' in props:
            curv_norm = 1 - normalize(props['curvature'])
            score_components.append(curv_norm)
            weights.append(0.25)
        
        if 'gradient' in props:
            grad_norm = normalize(props['gradient'])
            score_components.append(grad_norm)
            weights.append(0.15)
        
        if 'laplacian' in props:
            lapl_norm = normalize(np.maximum(0, props['laplacian']))
            score_components.append(lapl_norm)
            weights.append(0.2)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        hole_score = sum(w * c for w, c in zip(weights, score_components))
        threshold = np.percentile(hole_score, 70)
        
        colors = np.where(
            hole_score[:, None] > threshold,
            np.array([[1.0, 0.0, 0.0]]),  # Red for hole
            np.array([[0.5, 0.5, 0.5]])   # Gray otherwise
        ).reshape(-1, 3)
        
        rr.log("colored_points/hole_detection", rr.Points3D(
            positions=self.voxel_centers,
            colors=colors,
            radii=0.005
        ), static=True)
        
        rr.log("colored_points/hole_detection/info", rr.TextDocument(
            f"Hole Detection Summary\nThreshold: {threshold:.4f}\n"
            f"Detected voxels: {(hole_score > threshold).sum()}"
        ), static=True)
        
        # Voxel-based hole detection - all points colored by their voxel's hole score
        if self._voxel_point_mapping is not None:
            point_colors = np.zeros((len(self.points), 3))
            
            valid_mask = self._voxel_point_mapping >= 0
            point_colors[valid_mask] = colors[self._voxel_point_mapping[valid_mask]]
            point_colors[~valid_mask] = [0.5, 0.5, 0.5]
            
            rr.log("colored_points_voxels/hole_detection", rr.Points3D(
                positions=self.points,
                colors=point_colors,
                radii=0.001
            ), static=True)
            
            rr.log("colored_points_voxels/hole_detection/info", rr.TextDocument(
                f"Voxel-based Hole Detection\n"
                f"Threshold: {threshold:.4f}\n"
                f"Total points: {len(self.points)}\n"
                f"Points in detected holes: {np.sum(self._voxel_point_mapping[valid_mask] >= 0) if valid_mask.any() else 0}"
            ), static=True)


def test_hole_property_visualization():
    """Test the hole property visualization with sample data."""
    print("=" * 60)
    print("Hole Property Visualization Test")
    print("=" * 60)
    
    # Load test data
    ply_path = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_WoundDetection\Ferse1\dense\fused_Reduced_235k.ply")
    model_path = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_WoundDetection\Ferse1\dense")
    
    analyzer = PointCloudGeometryAnalyzer()
    
    # Load point cloud
    print("\n=== Loading Point Cloud ===")
    analyzer.load_from_ply(ply_path)
    
    # Load camera positions and orientations from COLMAP
    print("\n=== Loading Camera Positions and Orientations ===")
    try:
        from read_write_model import read_model, qvec2rotmat
        cameras, images, _ = read_model(model_path)
        
        camera_positions = []
        camera_orientations = []
        for img in images.values():
            R = qvec2rotmat(img.qvec)
            t = img.tvec
            # Camera center: C = -R^T * t
            C = -R.T @ t
            camera_positions.append(C)
            camera_orientations.append(R)  # Store rotation matrix
        
        analyzer.set_camera_positions(camera_positions, camera_orientations)
        print(f"  Loaded {len(camera_positions)} cameras with orientations")
    except Exception as e:
        print(f"Could not load cameras: {e}")
    
    # Compute properties
    print("\n=== Computing Properties ===")
    # Compute center of gravity using camera orientations (approach 1 & 2)
    #analyzer.compute_center_of_gravity(use_orientations=True)
    # Compute from point cloud normals (approach 3)
    analyzer.compute_center_of_gravity_from_normals(voxel_size=0.05, use_median=True)
    
    # Visualize the 3 reference plane normal approaches
    print("\n=== Visualizing Reference Plane Normal Approaches ===")
    #analyzer.visualize_reference_plane_normals_rerun()
    
    #analyzer.visualize_normals_center_of_gravity_rerun()
    #analyzer.visualize_center_of_gravity()
    analyzer.compute_height_profile(grid_resolution=0.1)
    analyzer.compute_gradient_magnitude()
    analyzer.compute_mean_curvature()
    analyzer.compute_laplacian()
    analyzer.compute_hole_energy()
    #analyzer.compute_voxel_properties(voxel_size=1.0)
    
    # Visualize
    print("\n=== Generating Visualization ===")
    save_path = model_path / "hole_property_analysis.png"
    #analyzer.visualize_all_properties(show_voxels=False)
    
    # Try Rerun visualization
    # print("\n=== Launching Rerun Visualization ===")
    analyzer.visualize_with_rerun("Laplacian_Test")
    
    print("\n=== Test Complete ===")
    return analyzer


if __name__ == "__main__":
    analyzer = test_hole_property_visualization()
