#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DenseDataFiltering.py - Module for filtering and processing dense point clouds
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import glob
import itertools
import open3d as o3d
import matplotlib.pyplot as plt
import pcl
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from skimage import color
import rerun as rr
import rerun.blueprint as rrb
from Pipe_Code.utils.read_write_model import Camera, read_model
import uuid
from typing import Tuple, Dict, Optional, List, Union, Any


class DenseDataFiltering:
    """
    A class for filtering and processing dense point clouds from COLMAP.
    
    This class provides methods to:
    - Load point clouds from PLY files
    - Filter points based on distance from center
    - Cluster points based on color and spatial information
    - Filter points based on planarity
    - Visualize point clouds using Rerun
    """
    
    def __init__(self, file_path: Optional[str] = None, session_id: Optional[str] = None):
        """
        Initialize the DenseDataFiltering class.
        
        Args:
            file_path: Path to the PLY file to load
            session_id: Optional session ID for Rerun visualization
        """
        self.file_path = file_path
        self.session_id = session_id or f"dense_filtering_{uuid.uuid4().hex[:8]}"
        self.pcd : Optional[o3d.geometry.PointCloud] = None  # Main point cloud object
        self.clusters = None  # Will store cluster labels as point cloud attribute
        self.cluster_centers = None

      
        # Initialize data structures if file path is provided
        if file_path:
            self.load_point_cloud(file_path)
    
    def init_rerun(self, recording_path: Optional[str] = None):
        """
        Initialize Rerun for visualization
        
        Args:
            recording_path: Optional path to save the recording
        """
        rr.init(self.session_id, recording_id=recording_path, spawn=True)
        rr.send_blueprint(rrb.Blueprint(
        rrb.Grid(
            rrb.Tabs(rrb.Spatial3DView(origin="/original"), rrb.Spatial3DView(origin="/original_95_distance")),
            rrb.Tabs(rrb.Spatial3DView(origin="/filtered_point_clouds/MLS"),rrb.Spatial3DView(origin="/filtered_point_clouds/Bilateral")),
        )
        ))
        

            
    def load_point_cloud(self, file_path: str, Trafo_matrix: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
        """
        Load a point cloud from a PLY file
        
        Args:
            file_path: Path to the PLY file
            Trafo_matrix: Optional 4x4 transformation matrix to apply to the point cloud
            
        Returns:
            Open3D point cloud object
        """
        self.file_path = file_path
        self.pcd = o3d.io.read_point_cloud(str(file_path))
        
        if self.pcd.is_empty():
            try:
                print(f"Point cloud is empty, trying to read dense model from {file_path}/dense")
                cameras, images, self.pcd = read_model(file_path/"dense", ext=".bin")
            except Exception as e:
                print(f"Error loading dense point cloud: {e}")
                return None  # Return None on error

        if Trafo_matrix is not None and isinstance(self.pcd, o3d.geometry.PointCloud):
            # For Open3D point cloud, apply the transformation matrix directly
            self.pcd.transform(Trafo_matrix)
        
        # Ensure we have normals
        if not self.pcd.has_normals():
            self.pcd.estimate_normals()
            
        # Initialize custom attributes
        # We'll store LAB color information and cluster labels directly in the point cloud
        # by adding them as custom attributes to be accessed later
        if self.pcd.has_colors():
            # Convert RGB to LAB
            colors = np.asarray(self.pcd.colors)
            rgb_255 = (colors * 255).astype(np.uint8)
            lab = color.rgb2lab(rgb_255.reshape(1, -1, 3)).reshape(-1, 3)
            
            # Add LAB colors as attributes to the point cloud
            # Since Open3D doesn't directly support custom attributes,
            # we'll store them as class attributes for now
            self.pcd_lab_l = lab[:, 0]
            self.pcd_lab_a = lab[:, 1]
            self.pcd_lab_b = lab[:, 2]
            self.pcd_lab_a_plus_b = lab[:, 1] + lab[:, 2]  # Combined a+b channels
        
        return self.pcd
    
    def distance_filtering(self, keep_ratio: float = 0.9) -> o3d.geometry.PointCloud:
        """
        Filter points based on distance from center of mass
        
        Args:
            keep_ratio: Ratio of points to keep (closest to center)
            
        Returns:
            Filtered point cloud
        """
        if self.pcd is None:
            raise ValueError("Point cloud not loaded. Call load_point_cloud first.")
        
        # Get points as numpy array
        points = np.asarray(self.pcd.points)
        
        # Calculate center of mass
        center_of_mass = np.mean(points, axis=0)
        
        # Calculate distances from center of mass
        distances = np.sqrt(np.sum((points - center_of_mass) ** 2, axis=1))
        
        # Determine threshold distance based on keep_ratio
        threshold = np.quantile(distances, keep_ratio)
        
        # Create mask for points to keep
        mask = distances <= threshold
        
        # Create new filtered point cloud
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])
        
        # Copy other attributes if they exist
        if self.pcd.has_normals():
            filtered_pcd.normals = o3d.utility.Vector3dVector(np.asarray(self.pcd.normals)[mask])
        
        if self.pcd.has_colors():
            filtered_pcd.colors = o3d.utility.Vector3dVector(np.asarray(self.pcd.colors)[mask])
            
        # Copy LAB color information if it exists
        if hasattr(self, 'pcd_lab_l'):
            self.pcd_lab_l = self.pcd_lab_l[mask]
            self.pcd_lab_a = self.pcd_lab_a[mask]
            self.pcd_lab_b = self.pcd_lab_b[mask]
            self.pcd_lab_a_plus_b = self.pcd_lab_a_plus_b[mask]
        
        # Update the point cloud
        self.pcd = filtered_pcd
        
        # Visualize with rerun if initialized
        
        self._visualize_point_cloud("distance_filtered", "Distance Filtering")
            
        return self.pcd
    
    def _apply_SOR(self, nb_neighbors: int = 20, std_ratio: float = 2.0, visualize: bool = False) -> o3d.geometry.PointCloud:
        """
        Apply Statistical Outlier Removal (SOR) to filter noise
        
        Args:
            nb_neighbors: Number of neighbors to consider for statistics
            std_ratio: Standard deviation ratio threshold
            visualize: Whether to visualize before/after results
            
        Returns:
            Filtered point cloud
        """
        if self.pcd is None:
            raise ValueError("Point cloud not loaded. Call load_point_cloud first.")
        
        # Apply Statistical Outlier Removal
        cl, ind = self.pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        
        # Visualize the point cloud if requested
        if visualize:
            # Create geometries for visualization
            inlier_cloud = self.pcd.select_by_index(ind)
            outlier_cloud = self.pcd.select_by_index(ind, invert=True)
            outlier_cloud.paint_uniform_color([1, 0, 0])  # Red for outliers
            
            # Visualization setup
            print(f"SOR removed {len(self.pcd.points) - len(inlier_cloud.points)} points out of {len(self.pcd.points)}")
            
            # Create visualization
            vis = o3d.visualization.Visualizer()
            vis.create_window("SOR Filtering Results")
            
            # Show options dialog
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
            opt.point_size = 2.0
            
            # Add both point clouds
            vis.add_geometry(inlier_cloud)
            vis.add_geometry(outlier_cloud)
            
            # Update camera
            vis.run()
            vis.destroy_window()
        
        # Update the point cloud
        self.pcd = cl
        
        # Visualize with rerun if initialized
        if hasattr(self, 'rr_initialized') and self.rr_initialized:
            self._visualize_point_cloud("SOR", "Statistical Outlier Removal")
            
        return self.pcd

    def cluster_points(self, method: str = 'kmeans', n_clusters: int = 2,
                      use_lab: bool = True, eps: float = 0.1,
                      min_samples: int = 50, visualize: bool = False) -> o3d.geometry.PointCloud:
        """
        Cluster point cloud based on color and spatial information
        
        Args:
            method: Clustering method ('kmeans' or 'dbscan')
            n_clusters: Number of clusters for kmeans
            use_lab: Whether to use LAB color space (True) or RGB (False)
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN min_samples parameter
            visualize: Whether to visualize the clusters using Open3D
            
        Returns:
            Point cloud with cluster information
        """
        if self.pcd is None:
            raise ValueError("Point cloud not loaded. Call load_point_cloud first.")
        
        # Get points as numpy array
        points = np.asarray(self.pcd.points)
        
        # Prepare color features
        if use_lab and hasattr(self, 'pcd_lab_a_plus_b'):
            # Using the a+b channels for maximum contrast
            color_features = self.pcd_lab_a_plus_b.reshape(-1, 1)
        elif self.pcd.has_colors():
            color_features = np.asarray(self.pcd.colors)
        else:
            # If no color information is available, use only spatial features
            color_features = np.ones((len(points), 1))
        
        # Normalize spatial and color features separately
        #points_normalized = (points - points.mean(axis=0)) / (points.std(axis=0) + 1e-10)
        color_normalized = (color_features - color_features.mean(axis=0)) / (color_features.std(axis=0) + 1e-10)
        
        # Combine features with equal weighting
        features = color_normalized #np.hstack((points_normalized, color_normalized))
        
        # Apply clustering
        if method.lower() == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(features)
            self.cluster_centers = clusterer.cluster_centers_
        elif method.lower() == 'dbscan':
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1  )
            labels = clusterer.fit_predict(features)
            # For DBSCAN, calculate cluster centers manually
            unique_labels = np.unique(labels)
            self.cluster_centers = []
            for label in unique_labels:
                if label != -1:  # Skip noise points
                    center = features[labels == label].mean(axis=0)
                    self.cluster_centers.append(center)
            if self.cluster_centers:
                self.cluster_centers = np.vstack(self.cluster_centers)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Store cluster labels
        self.clusters = labels
        
        # Visualize with rerun if initialized
        if hasattr(self, 'rr_initialized') and self.rr_initialized:
            self._visualize_clusters("clustering", "Clustering Results")
        
        # Visualize with Open3D if requested
        if visualize:
            self._visualize_clusters_o3d()
            
        return self.pcd
    
    def mls_filtering(self, radius: float = 0.1, polynomial_order: int = 2, min_neighbors: int = 10) -> o3d.geometry.PointCloud:
        """
        Apply Moving Least Squares filtering to the point cloud
        
        Args:
            radius: Radius to search for neighboring points
            polynomial_order: Order of polynomial for surface fitting (1=plane, 2=quadratic)
            min_neighbors: Minimum number of neighbors required for fitting
            
        Returns:
            Filtered point cloud
        """
        if self.pcd is None:
            raise ValueError("Point cloud not loaded. Call load_point_cloud first.")
        
        # Apply MLS filtering to the point cloud
        self.pcd = self._apply_Moving_Least_Squares(self.pcd, radius, polynomial_order, min_neighbors)
        
        # Visualize with rerun if initialized
        #self._visualize_point_cloud("MLS", "MLS Filtering")
        
        return self.pcd
        
    def bilateral_filtering(self, radius: float = 0.05, spatial_sigma: float = 0.05, 
                           normal_sigma: float = 0.1, num_iterations: int = 3) -> o3d.geometry.PointCloud:
        """
        Apply bilateral filtering to the point cloud based on the paper implementation
        
        Bilateral filtering smooths point positions while preserving edges by considering
        both spatial distance and distance along surface normal.
        
        Args:
            radius: Search radius for neighboring points
            spatial_sigma: Standard deviation for spatial distance weighting (σd)
            normal_sigma: Standard deviation for normal distance weighting (σn)
            num_iterations: Number of filtering iterations to apply
            
        Returns:
            Filtered point cloud
        """
        if self.pcd is None:
            raise ValueError("Point cloud not loaded. Call load_point_cloud first.")
        
        # Apply bilateral filtering directly on the class's point cloud
        filtered_pcd = self.pcd
        for _ in range(num_iterations):
            filtered_pcd = self._apply_bilateral_filter(filtered_pcd, radius, spatial_sigma, normal_sigma)
        
        # Update the class's point cloud
        self.pcd = filtered_pcd
        
        # Print information about filtering
        print("Bilateral filtering complete. Points:", len(np.asarray(self.pcd.points)))
        
        # Visualize with rerun if initialized
        #self._visualize_point_cloud("bilateral", "Bilateral Filtering")

        return self.pcd
        
    def _apply_bilateral_filter(self, pcd: o3d.geometry.PointCloud, radius: float, 
                              spatial_sigma: float, normal_sigma: float) -> o3d.geometry.PointCloud:
        """
        Helper method to apply one iteration of bilateral filtering based on the paper implementation
        
        This implementation follows the algorithm:
        Input: A point p, a radius r, two Gaussian weights σd and σn
        Output: A denoised point p′
        1. Nr(p) ← neighbors of p
        2. Compute the unit normal to the regression plane np from Nr(p)
        3. sumw = 0
        4. δp = 0
        5. for q ∈ Nr(p) do
        6.   dd ← ‖q − p‖
        7.   dn ← 〈q − p, np〉
        8.   w = exp(-dd²/(2σd²)) * exp(-dn²/(2σn²))
        9.   δp = δp + w*dn
        10.  sumw = sumw + w
        11. p′ ← p + (δp/sumw) * np
        
        Args:
            pcd: Input point cloud
            radius: Search radius for neighboring points
            spatial_sigma: Standard deviation for spatial distance (σd)
            normal_sigma: Standard deviation for normal direction (σn)
            
        Returns:
            Filtered point cloud
        """
        # Build KD-tree for nearest neighbor search
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        points = np.asarray(pcd.points)
        
        # Output point cloud with filtered positions
        filtered_points = np.zeros_like(points)
        
        # Process each point
        for i in range(len(points)):
            p = points[i]  # Current point
            
            # 1. Find neighbors within radius (Nr(p))
            [_, idx, _] = pcd_tree.search_radius_vector_3d(p, radius)
            
            if len(idx) < 3:  # Skip if not enough neighbors for plane fitting
                filtered_points[i] = p
                continue
            
            # Get neighbor points
            neighbors = points[idx]
            
            # 2. Compute the unit normal to the regression plane from neighbors
            # Use PCA to find the normal of the best-fitting plane
            centroid = np.mean(neighbors, axis=0)
            neighbors_centered = neighbors - centroid
            
            # Use SVD for robust normal estimation
            U, S, Vh = np.linalg.svd(neighbors_centered)
            np_normal = Vh[2]  # The third eigenvector is the normal to the plane
            np_normal = np_normal / np.linalg.norm(np_normal)  # Ensure unit normal
            
            # 3 & 4. Initialize sumw and δp
            sumw = 0
            delta_p = 0
            
            # 5. For each neighbor q
            for q in neighbors:
                # 6. Calculate spatial distance dd
                dd = np.linalg.norm(q - p)
                
                # 7. Calculate normal distance dn (projection along normal)
                dn = np.dot(q - p, np_normal)
                
                # 8. Calculate weight w using both distance components
                w = np.exp(-dd**2 / (2 * spatial_sigma**2)) * np.exp(-dn**2 / (2 * normal_sigma**2))
                
                # 9 & 10. Accumulate weighted distances and weights
                delta_p += w * dn
                sumw += w
            
            # 11. Compute the new position by moving along the normal
            if sumw > 0:
                filtered_points[i] = p + (delta_p / sumw) * np_normal
            else:
                filtered_points[i] = p  # Keep original point if weights sum to zero
        
        # Create output point cloud
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        
        # Copy other attributes from the original point cloud
        if pcd.has_colors():
            filtered_pcd.colors = pcd.colors
        
        if pcd.has_normals():
            filtered_pcd.normals = pcd.normals
            
        return filtered_pcd
  

    def _apply_Moving_Least_Squares(self, pcd: o3d.geometry.PointCloud, search_radius: float, 
                                   polynomial_order: int = 2, min_neighbors: int = 10) -> o3d.geometry.PointCloud:
        """
        Apply Moving Least Squares (MLS) surface reconstruction to smooth the point cloud
        
        This method fits local polynomial surfaces to points and projects them onto these surfaces.
        MLS is effective for noise reduction while preserving surface details.
        
        Args:
            pcd: Input point cloud
            search_radius: Radius to search for neighboring points
            polynomial_order: Order of the polynomial for surface fitting (1=plane, 2=quadratic)
            min_neighbors: Minimum number of neighbors required for fitting
            
        Returns:
            Smoothed point cloud
        """
        points = np.asarray(pcd.points)
        filtered_points = np.zeros_like(points)
        normals = np.zeros_like(points)
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        
        # Process each point
        for i in range(points.shape[0]):
            query_point = points[i]
            
            # Find neighbors within radius
            [k, idx, _] = kdtree.search_radius_vector_3d(query_point, search_radius)
            
            if k < min_neighbors:  # Not enough neighbors for reliable fitting
                filtered_points[i] = query_point
                if pcd.has_normals():
                    normals[i] = np.asarray(pcd.normals)[i]
                continue
                
            # Get neighboring points
            neighbors = points[idx]
            
            # Step 1: Find local reference frame (LRF) using PCA
            centroid = np.mean(neighbors, axis=0)
            neighbors_centered = neighbors - centroid
            
            # Use SVD for robust normal estimation
            U, _, Vh = np.linalg.svd(neighbors_centered)
            normal = Vh[2, :]  # The least significant right singular vector is the normal
            
            # Create local coordinate system
            z_axis = normal / np.linalg.norm(normal)
            
            # Find arbitrary vector not parallel to z_axis
            if np.abs(z_axis[0]) > 0.9:
                x_temp = np.array([0, 1, 0])
            else:
                x_temp = np.array([1, 0, 0])
                
            # Create orthogonal coordinate system
            y_axis = np.cross(z_axis, x_temp)
            y_axis = y_axis / np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)
            
            # Transformation matrix from global to local coordinates
            transform = np.vstack((x_axis, y_axis, z_axis)).T
            
            # Step 2: Project neighbors to local 2D coordinate system
            local_neighbors = neighbors_centered @ transform
            
            # We only need x and y for surface fitting (z is along normal)
            local_2d = local_neighbors[:, :2]
            heights = local_neighbors[:, 2]
            
            # Step 3: Fit polynomial surface
            # For polynomial_order=1, this is a plane; for polynomial_order=2, a quadratic surface
            if polynomial_order == 1:
                # Linear fit (plane): z = ax + by + c
                A = np.column_stack((local_2d, np.ones(local_2d.shape[0])))
                coeffs, _, _, _ = np.linalg.lstsq(A, heights, rcond=None)
                
                # Project query point to local coordinates
                query_local = (query_point - centroid) @ transform
                query_local_2d = query_local[:2]
                
                # Evaluate height at query point
                projected_height = coeffs[0] * query_local_2d[0] + coeffs[1] * query_local_2d[1] + coeffs[2]
                
                # Convert back to 3D point in local coordinates
                query_projected_local = np.array([query_local_2d[0], query_local_2d[1], projected_height])
                
            else:  # polynomial_order >= 2 (quadratic or higher)
                # Quadratic fit: z = ax^2 + by^2 + cxy + dx + ey + f
                x = local_2d[:, 0]
                y = local_2d[:, 1]
                A = np.column_stack((x**2, y**2, x*y, x, y, np.ones_like(x)))
                coeffs, _, _, _ = np.linalg.lstsq(A, heights, rcond=None)
                
                # Project query point to local coordinates
                query_local = (query_point - centroid) @ transform
                query_local_2d = query_local[:2]
                x_q, y_q = query_local_2d
                
                # Evaluate height at query point using the polynomial
                projected_height = (
                    coeffs[0] * x_q**2 + 
                    coeffs[1] * y_q**2 + 
                    coeffs[2] * x_q * y_q + 
                    coeffs[3] * x_q + 
                    coeffs[4] * y_q + 
                    coeffs[5]
                )
                
                # Convert back to 3D point in local coordinates
                query_projected_local = np.array([query_local_2d[0], query_local_2d[1], projected_height])
                
                # Calculate normal at this point by taking the gradient of the polynomial surface
                # For a quadratic surface: grad = [2ax + cy + d, 2by + cx + e]
                surface_gradient = np.array([
                    2*coeffs[0]*x_q + coeffs[2]*y_q + coeffs[3],
                    2*coeffs[1]*y_q + coeffs[2]*x_q + coeffs[4]
                ])
                
                # Convert gradient to 3D normal in local coordinates (-grad_x, -grad_y, 1)
                local_normal = np.array([-surface_gradient[0], -surface_gradient[1], 1.0])
                local_normal = local_normal / np.linalg.norm(local_normal)
                
                # Transform normal back to global coordinates
                normals[i] = transform @ local_normal
            
            # Transform back to global coordinates
            filtered_points[i] = centroid + transform @ query_projected_local
        
        # Create output point cloud
        output_pcd = o3d.geometry.PointCloud()
        output_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        
        # Copy colors if available
        if pcd.has_colors():
            output_pcd.colors = pcd.colors
            
        # Set calculated normals
        output_pcd.normals = o3d.utility.Vector3dVector(normals)
            
        return output_pcd


    def planarity_filtering(self, std_threshold: float = 2.0, 
                           k_neighbors: int = 20,
                           per_cluster: bool = True) -> o3d.geometry.PointCloud:
        """
        Filter points based on local planarity
        
        Args:
            std_threshold: Standard deviation threshold for outlier removal
            k_neighbors: Number of neighbors to consider for planarity calculation
            per_cluster: Whether to perform filtering per cluster
            
        Returns:
            Filtered point cloud
        """
        if self.pcd is None:
            raise ValueError("Point cloud not loaded. Call load_point_cloud first.")
        
        # Check if clustering has been performed
        if per_cluster and (self.clusters is None or len(np.unique(self.clusters)) <= 1):
            print("Warning: No valid clusters found. Run cluster_points first or set per_cluster=False.")
            per_cluster = False
        
        # Function to calculate point to plane distances
        def calc_point_to_plane_distances(points):
            if len(points) < k_neighbors:
                return np.zeros(len(points))  # Return zero distances for small point sets
                
            # PCA to find the principal components
            pca = PCA(n_components=3)
            pca.fit(points)
            
            # The normal of the best-fitting plane is the eigenvector with the smallest eigenvalue
            normal = pca.components_[2]  # Third component (least variance direction)
            
            # Calculate the centroid of the points
            centroid = points.mean(axis=0)
            
            # Calculate the distance from each point to the plane
            # d = |dot(point - centroid, normal)|
            dists = np.abs(np.dot(points - centroid, normal))
            
            return dists
        
        # Get points as numpy array
        points = np.asarray(self.pcd.points)
        mask = np.ones(len(points), dtype=bool)
        
        if per_cluster:
            # Process each cluster separately
            unique_clusters = np.unique(self.clusters)
            for cluster in unique_clusters:
                if cluster == -1:  # Skip noise points
                    continue
                
                cluster_mask = (self.clusters == cluster)
                cluster_points = points[cluster_mask]
                
                # Calculate distances to local plane
                distances = calc_point_to_plane_distances(cluster_points)
                
                # Identify outliers (points with distance > std_threshold * std)
                std_dist = np.std(distances)
                mean_dist = np.mean(distances)
                threshold = mean_dist + std_threshold * std_dist
                
                # Update mask for this cluster
                outlier_indices = np.where(cluster_mask)[0][distances > threshold]
                mask[outlier_indices] = False
        else:
            # Process all points together
            distances = calc_point_to_plane_distances(points)
            
            # Identify outliers
            std_dist = np.std(distances)
            mean_dist = np.mean(distances)
            threshold = mean_dist + std_threshold * std_dist
            
            # Update mask
            mask[distances > threshold] = False
        
        # Create filtered point cloud
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])
        
        # Copy other attributes if they exist
        if self.pcd.has_normals():
            filtered_pcd.normals = o3d.utility.Vector3dVector(np.asarray(self.pcd.normals)[mask])
        
        if self.pcd.has_colors():
            filtered_pcd.colors = o3d.utility.Vector3dVector(np.asarray(self.pcd.colors)[mask])
        
        # Update the clusters array if it exists
        if self.clusters is not None:
            self.clusters = self.clusters[mask]
            
        # Copy LAB color information if it exists
        if hasattr(self, 'pcd_lab_l'):
            self.pcd_lab_l = self.pcd_lab_l[mask]
            self.pcd_lab_a = self.pcd_lab_a[mask]
            self.pcd_lab_b = self.pcd_lab_b[mask]
            self.pcd_lab_a_plus_b = self.pcd_lab_a_plus_b[mask]
        
        # Update the point cloud
        self.pcd = filtered_pcd
        
        # Visualize with rerun if initialized
        if self.rr_initialized:
            self._visualize_point_cloud("planarity_filtered", "Planarity Filtering")
            
        return self.pcd
    
    def save_point_cloud(self, output_path: str) -> None:
        """
        Save the processed point cloud to a PLY file
        
        Args:
            output_path: Path to save the filtered point cloud
        """
        if self.pcd is None:
            raise ValueError("No point cloud data to save")
        
        # Save to file
        o3d.io.write_point_cloud(output_path, self.pcd)
        print(f"Point cloud saved to {output_path}")
    
    def _plot_normals(self, pcd: o3d.geometry.PointCloud, scale: float = 0.2, sample_ratio: float = 0.1) -> None:
        """
        Plot normals of the point cloud using Open3D
        
        Args:
            pcd: Open3D point cloud object
            scale: Scale factor for normal vectors
            sample_ratio: Ratio of points to display normals for (to avoid clutter)
        """
        if not pcd.has_normals():
            print("Point cloud has no normals, estimating them first...")
            pcd.estimate_normals()
        
        # Create a copy of the point cloud for visualization
        vis_pcd = o3d.geometry.PointCloud(pcd)
        
        # Sample points to reduce clutter
        n_points = len(np.asarray(pcd.points))
        sample_size = max(1, int(n_points * sample_ratio))
        indices = np.random.choice(n_points, sample_size, replace=False)
        
        # Extract sampled points and normals
        points = np.asarray(pcd.points)[indices]
        normals = np.asarray(pcd.normals)[indices]
        
        # Create lines for normal visualization
        lines = []
        colors = []
        for i in range(len(points)):
            p1 = points[i]
            p2 = p1 + normals[i] * scale
            lines.append([p1, p2])
            colors.append([1, 0, 0])  # Red arrows for normals
        
        # Create LineSet for normals
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.vstack([p for line in lines for p in line]))
        line_set.lines = o3d.utility.Vector2iVector(np.array([[i*2, i*2+1] for i in range(len(lines))]))
        line_set.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        # Create visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window("Point Cloud with Normals")
        
        # Add geometries
        vis.add_geometry(vis_pcd)
        vis.add_geometry(line_set)
        
        # Set rendering options
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.line_width = 2.0
        opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
        
        # Run the visualizer
        vis.run()
        vis.destroy_window()
        
    
    def visualize(self, title: str = "Point Cloud Visualization") -> None:
        """
        Visualize the current point cloud in Rerun
        
        Args:
            title: Title for the visualization
        """
        if not self.rr_initialized:
            self.init_rerun()
        
        self._visualize_point_cloud("final_result", title)
        
        
    
    def _visualize_point_cloud(self, name: str, title: str) -> None:
        """
        Internal method to visualize point cloud in Rerun
        """
        if self.pcd is None:
            return
        
        points = np.asarray(self.pcd.points)
        colors = None
        
        if self.pcd.has_colors():
            colors = np.asarray(self.pcd.colors)

        if name == "original":
            # Log original point cloud
            rr.log("/original", rr.Points3D(points, colors=colors), static=True)
            #rr.log("/original_95_distance", rr.Points3D(points, colors=colors), static=True)
        elif name == "distance_filtered":
            # Log distance filtered point cloud
            rr.log("/original_95_distance", rr.Points3D(points, colors=colors), static=True)
        else:
            rr.log(f"/filtered_point_clouds/{name}", rr.Points3D(points, colors=colors), static=True)

        
    
    def _visualize_clusters(self, name: str, title: str) -> None:
        """
        Internal method to visualize clusters in Rerun
        """
        if self.pcd is None or self.clusters is None:
            return
        
        points = np.asarray(self.pcd.points)
        
        # Generate colors based on cluster labels
        unique_clusters = np.unique(self.clusters)
        
        # Create a colormap (excluding black for noise points)
        cmap = plt.cm.get_cmap('tab10', len(unique_clusters))
        cluster_colors = np.zeros((len(points), 3))
        
        # Generate cluster counts
        cluster_counts = {}
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = (self.clusters == cluster_id)
            count = np.sum(mask)
            cluster_counts[int(cluster_id)] = int(count)
            
            if cluster_id == -1:  # Noise points in DBSCAN
                cluster_colors[mask] = [0.5, 0.5, 0.5]  # Gray for noise
            else:
                color = cmap(i)[:3]  # RGB components
                cluster_colors[mask] = color
        
        # Log points to rerun with cluster colors
        rr.log(
            f"point_clouds/{name}",
            rr.Points3D(
                positions=points,
                colors=cluster_colors,
                radii=0.01,
            ),
            timeless=True,
        )
        
        # Add text annotation
        cluster_info = "\n".join([f"Cluster {k}: {v} points" for k, v in cluster_counts.items()])
        
        rr.log(
            f"point_clouds/{name}/title",
            rr.TextDocument(f"# {title}\n\n{cluster_info}"),
            timeless=True,
        )
    
    def _visualize_clusters_o3d(self):
        """
        Visualize the clustered point cloud using Open3D
        
        This creates a new point cloud with colors based on cluster labels
        and displays it in an Open3D visualization window.
        """
        if self.pcd is None or self.clusters is None:
            print("No point cloud or cluster data available")
            return
            
        # Create a copy of the point cloud for visualization
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(np.asarray(self.pcd.points))
        
        # Generate colors based on cluster labels
        unique_clusters = np.unique(self.clusters)
        num_clusters = len(unique_clusters)
        
        # Create a colormap (excluding black for noise points)
        # Using a rainbow color scheme for good contrast
        colors = plt.cm.rainbow(np.linspace(0, 1, num_clusters))[:, :3]
        
        # Map cluster labels to colors
        point_colors = np.zeros((len(self.clusters), 3))
        
        for i, cluster_id in enumerate(unique_clusters):
            if cluster_id == -1:  # Noise points in DBSCAN
                mask = (self.clusters == cluster_id)
                point_colors[mask] = [0.5, 0.5, 0.5]  # Gray for noise
            else:
                cluster_idx = np.where(unique_clusters == cluster_id)[0][0]
                mask = (self.clusters == cluster_id)
                point_colors[mask] = colors[cluster_idx % len(colors)]
        
        # Set the colors for the visualization point cloud
        vis_pcd.colors = o3d.utility.Vector3dVector(point_colors)
        
        # Create a visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window("Clustered Point Cloud")
        
        # Add the point cloud to the visualizer
        vis.add_geometry(vis_pcd)
        
        # Set rendering options
        opt = vis.get_render_option()
        opt.point_size = 3.0
        opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
        
        # Set view control
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        
        # Add cluster information text to the visualizer
        cluster_counts = {}
        for cluster_id in unique_clusters:
            count = np.sum(self.clusters == cluster_id)
            cluster_counts[int(cluster_id)] = int(count)
        
        # Run the visualizer
        vis.run()
        vis.destroy_window()
        
        # Print cluster information
        print("Cluster information:")
        for cluster_id, count in cluster_counts.items():
            if cluster_id == -1:
                print(f"  Noise points: {count}")
            else:
                print(f"  Cluster {cluster_id}: {count} points")

def geiler_Prozess_Digger():
    
    output_base_path_MLS = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_DenseFiltering\MLS")
    output_base_path_MLS.mkdir(exist_ok=True)
    
    output_base_path_Bilateral = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_DenseFiltering\Bilateral")
    output_base_path_Bilateral.mkdir(exist_ok=True)
    # Find PLY files in the workspace
    ply_files=[r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_DenseFiltering\ply_data\Trial28_Benchmark_reduced.ply"

               ]
    from time import time
    # Define parameter values for grid search
    radius_values = [0.2, 0.3, 0.4]
    polynomial_orders = [ 2]
    min_neighbors_values = [ 5, 10, 20]
    
    # Use the first PLY file found
    for ply_file in ply_files:
        
        # print(f"Processing PLY file: {ply_file}")
    
        # ply_filename = Path(ply_file).stem
        # file_output_dir = output_base_path_MLS / ply_filename
        # file_output_dir.mkdir(exist_ok=True)
        
        
        # # Process all parameter combinations
        # for radius, poly_order, min_neighbors in itertools.product(radius_values, polynomial_orders, min_neighbors_values):
        #     time_start = time()
        #     print(f"  Parameters: radius={radius}, polynomial_order={poly_order}, min_neighbors={min_neighbors}")
        #     # Create filtering object
        #     filter_obj = DenseDataFiltering(session_id=f"mls_filtering_{ply_filename}")
        #     try:
        #         # Load point cloud (fresh load for each parameter set)
        #         filter_obj.load_point_cloud(ply_file)
                
        #         # Apply MLS filtering with current parameters
        #         filter_obj.mls_filtering(radius=radius, polynomial_order=poly_order, min_neighbors=min_neighbors)
                
        #         # Save the filtered point cloud with parameters in filename
        #         output_filename = f"mls_r{radius}_p{poly_order}_n{min_neighbors}.ply"
        #         output_path = str(file_output_dir / output_filename)
        #         filter_obj.save_point_cloud(output_path)
                
        #         print(f"    Saved to: {output_path}")
        #         time_end = time()
        #         print(f"    Processing time: {time_end - time_start:.2f} seconds")
        #     except Exception as e:
        #         print(f"    Error processing with parameters {radius}, {poly_order}, {min_neighbors}: {str(e)}")
                
        # print(f"Completed processing {ply_file}")

        # print("All files processed. Results saved in:", str(output_base_path_MLS))
    
     #bilater parameters
        radius_values_bilateral = [0.05, 0.1]
        spatial_sigma_values = [0.05, 0.1]
        normal_sigma_values = [0.1]
        num_iterations_values = [3]
        

        
        ply_filename = Path(ply_file).stem
        file_output_dir = output_base_path_Bilateral / ply_filename
        file_output_dir.mkdir(exist_ok=True)

        # Process all parameter combinations for bilateral filtering
        for radius, spatial_sigma, normal_sigma, num_iterations in itertools.product(radius_values_bilateral, spatial_sigma_values, normal_sigma_values, num_iterations_values):
            print(f"  Bilateral Parameters: radius={radius}, spatial_sigma={spatial_sigma}, normal_sigma={normal_sigma}, num_iterations={num_iterations}")
            # Create filtering object
            filter_obj = DenseDataFiltering(session_id=f"bilateral_filtering_{ply_filename}")
            try:
                # Load point cloud (fresh load for each parameter set)
                filter_obj.load_point_cloud(ply_file)
                filter_obj.bilateral_filtering(radius=radius, spatial_sigma=spatial_sigma, normal_sigma=normal_sigma, num_iterations=num_iterations)
                output_filename = f"bilateral_r{radius}_s{spatial_sigma}_n{normal_sigma}_i{num_iterations}.ply"
                output_path = str(file_output_dir / output_filename)
                filter_obj.save_point_cloud(output_path)
                print(f"    Saved to: {output_path}")
            except Exception as e:
                print(f"    Error processing with parameters {radius}, {spatial_sigma}, {normal_sigma}, {num_iterations}: {str(e)}")

        print(f"Completed processing {ply_file}")

        print("All files processed. Results saved in:", str(output_base_path_Bilateral))

    
def demo():
    """
    Demonstration of the DenseDataFiltering class
    """
    
    # Find PLY files in the workspace
    base_path = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_DenseFiltering")
    #ply_files = list(glob.glob(str(base_path / "**/*.ply"), recursive=False))
    # Define output path
    ply_file = r"Experiments/Experiments_DenseFiltering/ply_data/Trial28_Benchmark_reduced.ply"
    # Create filtering object with rerun visualization
    filter_obj = DenseDataFiltering(session_id="demo_filtering")
    #filter_obj.init_rerun()
    
        # # Load point cloud
    filter_obj.load_point_cloud(ply_file)
        # filter_obj.mls_filtering(radius=0.1, polynomial_order=2, min_neighbors=10)
    #filter_obj._visualize_point_cloud("original", "Original Point Cloud")
    # Apply distance filtering
    #filter_obj.distance_filtering(keep_ratio=0.8)
    
    # Apply Statistical Outlier Removal filtering
    #filter_obj._apply_SOR(nb_neighbors=2, std_ratio=3.0, visualize=True)
    
    # Apply clustering
    filter_obj.cluster_points(method='dbscan', use_lab=True, visualize=True)
    
    # Visualize normals
    #filter_obj._plot_normals(filter_obj.pcd)
    
    # Apply planarity filtering
    #filter_obj.planarity_filtering(std_threshold=2.0, per_cluster=True)
    
    # Save the filtered point cloud
    # output_path = str(Path(ply_file).parent / "filtered.ply")
    # filter_obj.save_point_cloud(output_path)
    
    # Visualize the final result
    #filter_obj.visualize("Final Filtered Point Cloud")

  # Create a directory for this file's outputs
   
    
   
if __name__ == "__main__":
    #demo()
    geiler_Prozess_Digger()