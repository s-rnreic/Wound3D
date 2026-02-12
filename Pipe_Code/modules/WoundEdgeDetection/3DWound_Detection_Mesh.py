"""
3D Wound Edge Detection - Mesh-Based Version

This module provides mesh-based 3D wound segmentation from COLMAP dense reconstructions.
Unlike the point cloud version, this exploits mesh topology (faces, edges, connectivity)
for more robust wound detection and boundary extraction.

Key Differences from Point Cloud Version:
- Input is a mesh (PLY file with vertices and faces)
- Uses face visibility and connectivity for wound detection
- Exploits mesh topology for region growing and boundary detection
- Applies mesh-based filtering using connected components
- Computes surface properties (curvature, smoothness) for better segmentation

Classes:
    MeshFaceRegion: Groups of connected faces forming local mesh regions
    Spectator: Handles camera information and face visibility through masks
    WoundModel3DMesh: Main class for mesh-based 3D wound segmentation

Author: Your Name
Date: February 2026
"""

import os
import sys
import numpy as np
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any

# Add paths for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_pipe_code_dir = os.path.abspath(os.path.join(_current_dir, '..', '..'))
_workspace_root = os.path.abspath(os.path.join(_current_dir, '..', '..', '..'))

if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)
if _pipe_code_dir not in sys.path:
    sys.path.insert(0, _pipe_code_dir)

from CODE_Evaluation_Viewer_Align.Transformation import *
from utils.read_write_model import read_model, qvec2rotmat
from CODE_Evaluation_Viewer_Align.read_write_dense import read_array
from modules.WoundEdgeDetection.WoundMeshCorrection import correct_wound_mesh_from_detection



class MeshFaceRegion:
    """
    A region of connected mesh faces for local surface analysis.
    
    This class represents a group of connected triangular faces and computes
    local properties like mean curvature, surface smoothness, and wound likelihood.
    """
    
    def __init__(self, region_id: int, seed_face_idx: int = None):
        """
        Initialize a MeshFaceRegion.
        
        Args:
            region_id: Unique identifier for this region
            seed_face_idx: Initial face index for region growing
        """
        self.region_id = region_id
        self.seed_face_idx = seed_face_idx
        
        # Face and vertex data
        self.face_indices = []  # List of face indices in this region
        self.vertex_indices = set()  # Set of vertex indices (unique)
        
        # Computed properties
        self.centroid = None
        self.mean_normal = None
        self.mean_curvature = None
        self.surface_area = 0.0
        self.mean_color = None
        self.color_variance = None
        
        # Boundary information
        self.boundary_edges = []  # List of edge tuples on region boundary
        self.is_boundary_region = False
        
        # Wound likelihood score
        self.wound_score = 0.0
        
        # Validity flags
        self.properties_computed = False
        self.is_valid = True
    
    def add_faces(self, face_indices: List[int], vertices: np.ndarray, 
                  faces: np.ndarray, vertex_colors: np.ndarray = None):
        """
        Add faces to this region and update vertex set.
        
        Args:
            face_indices: List of face indices to add
            vertices: Global vertex array (N, 3)
            faces: Global face array (M, 3)
            vertex_colors: Global vertex color array (N, 3)
        """
        self.face_indices = face_indices
        
        # Extract unique vertices from these faces
        self.vertex_indices.clear()
        for face_idx in face_indices:
            face = faces[face_idx]
            self.vertex_indices.update(face)
        
        self.is_valid = len(self.face_indices) > 0
    
    def compute_properties(self, vertices: np.ndarray, faces: np.ndarray, 
                          vertex_normals: np.ndarray = None,
                          vertex_colors: np.ndarray = None):
        """Compute all geometric and color properties of the region."""
        if not self.is_valid or len(self.face_indices) == 0:
            return
        
        # Compute centroid (average of face centroids weighted by area)
        face_centroids = []
        face_areas = []
        
        for face_idx in self.face_indices:
            face = faces[face_idx]
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            
            # Face centroid
            face_centroid = (v0 + v1 + v2) / 3.0
            face_centroids.append(face_centroid)
            
            # Face area
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
            face_areas.append(face_area)
        
        face_centroids = np.array(face_centroids)
        face_areas = np.array(face_areas)
        
        total_area = np.sum(face_areas)
        self.surface_area = total_area
        
        if total_area > 0:
            self.centroid = np.average(face_centroids, axis=0, weights=face_areas)
        else:
            self.centroid = np.mean(face_centroids, axis=0)
        
        # Compute mean normal (weighted by face area)
        if vertex_normals is not None:
            face_normals = []
            for face_idx in self.face_indices:
                face = faces[face_idx]
                # Average vertex normals
                face_normal = np.mean(vertex_normals[face], axis=0)
                face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-6)
                face_normals.append(face_normal)
            
            face_normals = np.array(face_normals)
            if len(face_areas) == len(face_normals):
                mean_normal_raw = np.average(face_normals, axis=0, weights=face_areas)
            else:
                mean_normal_raw = np.mean(face_normals, axis=0)
            
            norm = np.linalg.norm(mean_normal_raw)
            if norm > 1e-6:
                self.mean_normal = mean_normal_raw / norm
            else:
                self.mean_normal = np.array([0, 0, 1])
        
        # Compute mean color
        if vertex_colors is not None:
            vertex_list = list(self.vertex_indices)
            region_colors = vertex_colors[vertex_list]
            self.mean_color = np.mean(region_colors, axis=0)
            self.color_variance = np.var(region_colors, axis=0)
        
        self.properties_computed = True
    
    def compute_boundary(self, faces: np.ndarray, adjacency: Dict[Tuple[int, int], List[int]]):
        """
        Compute boundary edges of this region.
        
        Args:
            faces: Global face array (M, 3)
            adjacency: Edge adjacency dictionary: edge -> list of face indices
        """
        self.boundary_edges = []
        face_set = set(self.face_indices)
        
        for face_idx in self.face_indices:
            face = faces[face_idx]
            # Check all three edges of the triangle
            edges = [
                tuple(sorted([face[0], face[1]])),
                tuple(sorted([face[1], face[2]])),
                tuple(sorted([face[2], face[0]]))
            ]
            
            for edge in edges:
                adjacent_faces = adjacency.get(edge, [])
                # Edge is on boundary if not all adjacent faces are in this region
                adjacent_in_region = [f for f in adjacent_faces if f in face_set]
                if len(adjacent_in_region) < len(adjacent_faces):
                    self.boundary_edges.append(edge)
        
        self.is_boundary_region = len(self.boundary_edges) > 0


class Spectator:
    """
    A class representing a camera viewpoint for mesh-based wound detection.
    
    Handles face visibility, mask projection, and mesh-based filtering.
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
        
        # Camera parameters
        self.rotation_matrix = None
        self.translation_vector = None
        self.intrinsics = None
        self.image_width = None
        self.image_height = None
        
        # Image and mask data
        self.image_data = None
        self.mask_data = None
        self.mask_loaded = False
        
        # Depth map data
        self.depth_map = None
        self.depth_map_loaded = False
        
        # Face visibility tracking (mesh-specific)
        self.visible_face_indices = []  # Faces visible through mask
        self.visible_vertex_indices = set()  # Vertices from visible faces
        self.filtered_out_faces = {}  # Dict: filter_name -> list of filtered face indices
        
        # Processing flags
        self.camera_data_loaded = False
        self.faces_processed = False
    
    def load_camera_data(self, colmap_image, colmap_camera):
        """Load camera intrinsics and extrinsics from COLMAP data."""
        quat = colmap_image.qvec
        tvec = colmap_image.tvec
        self.rotation_matrix = qvec2rotmat(quat)
        self.translation_vector = tvec.reshape(3, 1)
        
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
        """Load wound mask for this camera view."""
        mask_dir = Path(mask_dir)
        
        possible_names = [
            f"{self.image_name}.png",
            self.image_name.replace('.jpg', '.png'),
            self.image_name.replace('.jpeg', '.png'),
        ]
        
        for mask_name in possible_names:
            mask_path = mask_dir / mask_name
            if mask_path.exists():
                import cv2
                self.mask_data = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                self.mask_loaded = True
                return True
        
        print(f"Warning: No mask found for {self.image_name}")
        return False
    
    def project_points_to_image(self, points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project 3D points to this camera's image plane."""
        if not self.camera_data_loaded:
            raise ValueError("Camera data must be loaded before projecting points")
        
        # Transform to camera coordinates
        points_cam = self.rotation_matrix @ points_3d.T + self.translation_vector
        x_cam, y_cam, z_cam = points_cam
        
        # Project to image
        fx, fy, cx, cy = self.intrinsics['fx'], self.intrinsics['fy'], self.intrinsics['cx'], self.intrinsics['cy']
        u = fx * x_cam / z_cam + cx
        v = fy * y_cam / z_cam + cy
        
        return u, v, z_cam
    
    def filter_faces_through_mask(self, vertices: np.ndarray, faces: np.ndarray,
                                  face_centers: np.ndarray = None,
                                  vertex_colors: np.ndarray = None,
                                  face_normals: np.ndarray = None) -> Dict[str, Any]:
        """
        Filter mesh faces through this camera's mask.
        
        Uses face-specific criteria:
        - Face visibility (backface culling)
        - Face center projection through mask
        - All vertices of face must be visible
        
        Args:
            vertices: Vertex array (N, 3)
            faces: Face array (M, 3)
            face_centers: Pre-computed face centers (M, 3)
            vertex_colors: Vertex color array (N, 3)
            face_normals: Face normal array (M, 3)
            
        Returns:
            Dict containing filtering results and statistics
        """
        if not self.camera_data_loaded or not self.mask_loaded:
            raise ValueError("Camera data and mask must be loaded")
        
        n_faces = len(faces)
        all_face_indices = np.arange(n_faces)
        valid_face_mask = np.ones(n_faces, dtype=bool)
        
        filtering_results = {
            'total_faces': n_faces,
            'visible_face_indices': [],
            'filtered_steps': {},
            'processing_successful': True
        }
        
        # Project ALL face centers once (optimization)
        u_all, v_all, depths_all = None, None, None
        if face_centers is not None:
            u_all, v_all, depths_all = self.project_points_to_image(face_centers)
        
        # Step 1: Backface culling (faces facing away from camera)
        if face_normals is not None and face_centers is not None:
            # Camera center in world coordinates
            camera_center = -self.rotation_matrix.T @ self.translation_vector.flatten()
            
            # View vectors from face centers to camera
            view_vectors = camera_center - face_centers
            view_vectors = view_vectors / (np.linalg.norm(view_vectors, axis=1, keepdims=True) + 1e-6)
            
            # Dot product with face normals
            dot_products = np.sum(view_vectors * face_normals, axis=1)
            
            # Backfacing if dot product <= 0
            backfacing = dot_products <= 0
            backfacing_indices = all_face_indices[backfacing & valid_face_mask]
            valid_face_mask &= ~backfacing
            filtering_results['filtered_steps']['backface_culling'] = backfacing_indices.tolist()
        
        # Step 2: Filter by projection constraints using pre-computed projections
        if np.any(valid_face_mask) and u_all is not None:
            valid_face_list = all_face_indices[valid_face_mask]
            
            # Use pre-computed projections (just index into arrays)
            u = u_all[valid_face_list]
            v = v_all[valid_face_list]
            depths = depths_all[valid_face_list]
            
            # Filter behind camera
            behind_camera = depths <= 0
            behind_indices = valid_face_list[behind_camera]
            valid_face_mask[behind_indices] = False
            filtering_results['filtered_steps']['behind_camera'] = behind_indices.tolist()
            
            # Update valid list
            valid_face_list = all_face_indices[valid_face_mask]
            if len(valid_face_list) > 0:
                # Use pre-computed projections
                u = u_all[valid_face_list]
                v = v_all[valid_face_list]
                depths = depths_all[valid_face_list]
                
                # Filter out of bounds
                out_of_bounds = ((u < 0) | (u >= self.image_width) | 
                                (v < 0) | (v >= self.image_height))
                out_bounds_indices = valid_face_list[out_of_bounds]
                valid_face_mask[out_bounds_indices] = False
                filtering_results['filtered_steps']['out_of_bounds'] = out_bounds_indices.tolist()
                
                # Update valid list again
                valid_face_list = all_face_indices[valid_face_mask]
                if len(valid_face_list) > 0:
                    # Use pre-computed projections
                    u = u_all[valid_face_list]
                    v = v_all[valid_face_list]
                    depths = depths_all[valid_face_list]
                    
                    # Check mask
                    u_int = np.clip(u.astype(int), 0, self.mask_data.shape[1] - 1)
                    v_int = np.clip(v.astype(int), 0, self.mask_data.shape[0] - 1)
                    mask_values = self.mask_data[v_int, u_int]
                    
                    outside_mask = mask_values == 0
                    outside_mask_indices = valid_face_list[outside_mask]
                    valid_face_mask[outside_mask_indices] = False
                    filtering_results['filtered_steps']['outside_mask'] = outside_mask_indices.tolist()
        
        # Step 3: Require all vertices of a face to be projectable and within mask
        valid_face_list = all_face_indices[valid_face_mask]
        additional_filtered = []
        
        for face_idx in valid_face_list:
            face = faces[face_idx]
            face_vertices = vertices[face]
            
            # Project all three vertices
            u_verts, v_verts, z_verts = self.project_points_to_image(face_vertices)
            
            # Check if any vertex fails constraints
            any_behind = np.any(z_verts <= 0)
            any_out_bounds = np.any((u_verts < 0) | (u_verts >= self.image_width) |
                                   (v_verts < 0) | (v_verts >= self.image_height))
            
            if any_behind or any_out_bounds:
                additional_filtered.append(face_idx)
                valid_face_mask[face_idx] = False
                continue
            
            # Check mask for all vertices
            u_verts_int = np.clip(u_verts.astype(int), 0, self.mask_data.shape[1] - 1)
            v_verts_int = np.clip(v_verts.astype(int), 0, self.mask_data.shape[0] - 1)
            mask_vals = self.mask_data[v_verts_int, u_verts_int]
            
            if np.any(mask_vals == 0):
                additional_filtered.append(face_idx)
                valid_face_mask[face_idx] = False
        
        filtering_results['filtered_steps']['incomplete_face_visibility'] = additional_filtered
        
        # Final visible faces
        filtering_results['visible_face_indices'] = all_face_indices[valid_face_mask].tolist()
        
        # Store results
        self.visible_face_indices = filtering_results['visible_face_indices']
        self.filtered_out_faces = filtering_results['filtered_steps']
        
        # Extract visible vertices
        self.visible_vertex_indices.clear()
        for face_idx in self.visible_face_indices:
            face = faces[face_idx]
            self.visible_vertex_indices.update(face)
        
        self.faces_processed = True
        
        return filtering_results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for this spectator."""
        summary = {
            'image_name': self.image_name,
            'image_id': self.image_id,
            'camera_data_loaded': self.camera_data_loaded,
            'mask_loaded': self.mask_loaded,
            'faces_processed': self.faces_processed,
            'visible_faces_count': len(self.visible_face_indices),
            'visible_vertices_count': len(self.visible_vertex_indices)
        }
        
        if self.faces_processed:
            total_filtered = sum(len(indices) for indices in self.filtered_out_faces.values())
            summary['total_filtered_faces'] = total_filtered
            summary['filtering_breakdown'] = {
                step: len(indices) for step, indices in self.filtered_out_faces.items()
            }
        
        return summary


class WoundModel3DMesh:
    """
    Main class for mesh-based 3D wound segmentation.
    
    Exploits mesh topology (faces, edges, connectivity) for robust wound detection.
    """
    
    def __init__(self):
        """Initialize empty WoundModel3DMesh."""
        # Mesh data
        self.vertices = None  # (N, 3) vertex coordinates
        self.faces = None  # (M, 3) face indices
        self.vertex_colors = None  # (N, 3) vertex colors
        self.vertex_normals = None  # (N, 3) vertex normals
        self.face_normals = None  # (M, 3) face normals
        self.face_centers = None  # (M, 3) face center coordinates
        self._raycast_scene = None
        self._raycast_mesh_id = None
        
        # Mesh topology
        self.edge_adjacency = {}  # edge -> list of face indices
        self.vertex_adjacency = {}  # vertex -> list of adjacent vertices
        self.face_adjacency = {}  # face -> list of adjacent faces
        
        # Camera and image management
        self.spectators = {}  # Dict: image_name -> Spectator
        self.cameras = {}
        self.images = {}
        
        # Face visibility tracking
        self.face_visibility_count = {}  # face_idx -> count
        self.wound_faces_global = set()  # Faces visible through masks
        self.wound_vertices_global = set()  # Vertices from wound faces

        # Wound edge data
        self.wound_edge_edges = []  # List of (v0, v1) tuples
        self.wound_edge_vertices = set()  # Set of vertex indices on edge
        self.wound_edge_faces = set()  # Wound faces adjacent to non-wound

        # Filtering diagnostics
        self.removed_faces_visibility = set()
        self.removed_faces_smoothness = set()
        self.removed_faces_connectivity = set()
        self.added_faces_enclosed = set()
        
        # Connected components
        self.wound_regions = []  # List of connected wound regions
        
        # Model state
        self.model_loaded = False
        self.masks_loaded = False
        self.faces_processed = False
        self.topology_computed = False
    
    def load_mesh(self, mesh_path: Path, compute_normals: bool = True):
        """
        Load mesh from PLY file.
        
        Args:
            mesh_path: Path to mesh PLY file
            compute_normals: Whether to compute vertex normals if not present
        """
        mesh_path = Path(mesh_path)
        print(f"Loading mesh from: {mesh_path}")
        
        try:
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            
            self.vertices = np.asarray(mesh.vertices)
            self.faces = np.asarray(mesh.triangles)
            
            # Load colors if available
            if mesh.has_vertex_colors():
                self.vertex_colors = np.asarray(mesh.vertex_colors) # Scale to 0-255
            
            # Load or compute normals
            if mesh.has_vertex_normals():
                self.vertex_normals = np.asarray(mesh.vertex_normals)
            elif compute_normals:
                print("Computing vertex normals...")
                mesh.compute_vertex_normals()
                self.vertex_normals = np.asarray(mesh.vertex_normals)
            
            # Compute face normals
            if compute_normals:
                print("Computing face normals...")
                mesh.compute_triangle_normals()
                self.face_normals = np.asarray(mesh.triangle_normals)
            
            print(f"Loaded mesh: {len(self.vertices)} vertices, {len(self.faces)} faces")
            
        except ImportError:
            print("Open3D not available, attempting manual PLY parsing...")
            self._load_mesh_manual(mesh_path)
        
        # Build mesh topology
        self._compute_mesh_topology()
        
        # Compute face centers (only once)
        self._compute_face_centers()
        
        self.model_loaded = True
    
    def _load_mesh_manual(self, mesh_path: Path):
        """Manual PLY mesh loading (fallback)."""
        # Basic PLY parser for mesh data
        # This is a simplified version - expand as needed
        raise NotImplementedError("Manual mesh loading not yet implemented. Please install Open3D.")
    
    def _compute_mesh_topology(self):
        """Compute mesh topology: edge adjacency, vertex adjacency, face adjacency."""
        print("Computing mesh topology...")
        
        # Edge adjacency: each edge maps to list of faces containing it
        self.edge_adjacency = defaultdict(list)
        
        # Vertex adjacency: each vertex maps to adjacent vertices
        self.vertex_adjacency = defaultdict(set)
        
        # Face adjacency: each face maps to adjacent faces (sharing an edge)
        self.face_adjacency = defaultdict(list)
        
        for face_idx, face in enumerate(self.faces):
            v0, v1, v2 = face
            
            # Add edges (sorted for consistency)
            edges = [
                tuple(sorted([v0, v1])),
                tuple(sorted([v1, v2])),
                tuple(sorted([v2, v0]))
            ]
            
            for edge in edges:
                self.edge_adjacency[edge].append(face_idx)
                # Also update vertex adjacency
                self.vertex_adjacency[edge[0]].add(edge[1])
                self.vertex_adjacency[edge[1]].add(edge[0])
        
        # Build face adjacency from edge adjacency
        for edge, face_list in self.edge_adjacency.items():
            if len(face_list) >= 2:
                # Faces sharing this edge are adjacent
                for i, face_i in enumerate(face_list):
                    for face_j in face_list[i+1:]:
                        self.face_adjacency[face_i].append(face_j)
                        self.face_adjacency[face_j].append(face_i)
        
        self.topology_computed = True
        print(f"Topology computed: {len(self.edge_adjacency)} edges")
    
    def _compute_face_centers(self):
        """Compute face centers once for the entire mesh."""
        print("Computing face centers...")
        
        if self.vertices is None or self.faces is None:
            return
        
        n_faces = len(self.faces)
        self.face_centers = np.zeros((n_faces, 3), dtype=np.float64)
        
        for face_idx in range(n_faces):
            face = self.faces[face_idx]
            self.face_centers[face_idx] = np.mean(self.vertices[face], axis=0)
        
        print(f"Computed {n_faces} face centers")
    
    def load_colmap_cameras(self, model_path: Path):
        """
        Load camera information from COLMAP sparse model.
        
        Args:
            model_path: Path to COLMAP sparse model directory
        """
        model_path = Path(model_path)
        print(f"Loading COLMAP cameras from: {model_path}")
        
        self.cameras, self.images, _ = read_model(model_path)
        
        # Create Spectator objects
        self._create_spectators()
        
        print(f"Loaded {len(self.spectators)} cameras")
    
    def _create_spectators(self):
        """Create Spectator objects for each COLMAP image."""
        self.spectators = {}
        
        for img_id, img in self.images.items():
            spectator = Spectator(img_id, img.name, img.camera_id)
            cam = self.cameras[img.camera_id]
            spectator.load_camera_data(img, cam)
            self.spectators[img.name] = spectator
        
        print(f"Created {len(self.spectators)} spectators")
    
    def load_masks(self, mask_dir: Path):
        """Load wound masks for all spectators."""
        mask_dir = Path(mask_dir)
        print(f"Loading masks from: {mask_dir}")
        
        masks_loaded = 0
        for spectator in self.spectators.values():
            if spectator.load_mask(mask_dir):
                masks_loaded += 1
        
        self.masks_loaded = masks_loaded > 0
        print(f"Loaded masks for {masks_loaded}/{len(self.spectators)} cameras")
    
    def process_wound_segmentation(self,
                                   filter_mode: str = "open3d",
                                   benchmark_cameras: int = 1,
                                   max_mask_rays: int = 200000,
                                   speedup_threshold: float = 1.1):
        """
        Process mesh-based wound segmentation.
        
        Filters faces through all camera masks and builds wound regions.
        """
        if not self.model_loaded:
            raise ValueError("Mesh must be loaded first")
        if not self.masks_loaded:
            raise ValueError("Masks must be loaded first")
        
        print(f"Processing mesh-based wound segmentation...")
        
        # Reset tracking
        self.face_visibility_count = {}
        self.wound_faces_global = set()
        self.wound_vertices_global = set()
        
        # Decide filtering mode (numpy or Open3D) once
        filter_mode = filter_mode.lower().strip()
        if filter_mode not in {"numpy", "open3d", "auto"}:
            raise ValueError("filter_mode must be 'numpy', 'open3d', or 'auto'")

        selected_mode = filter_mode
       

        print(f"Using filter mode: {selected_mode}")

        # Process each spectator
        for spectator_name, spectator in self.spectators.items():
            if not spectator.mask_loaded:
                continue
            
            print(f"Processing spectator: {spectator_name}")
            
            # Filter faces through mask
            if selected_mode == "open3d":
                self._run_open3d_filter(spectator, max_mask_rays=max_mask_rays)
            else:
                spectator.filter_faces_through_mask(
                    self.vertices, self.faces, self.face_centers,
                    self.vertex_colors, self.face_normals
                )
            
            # Update global tracking
            for face_idx in spectator.visible_face_indices:
                self.face_visibility_count[face_idx] = self.face_visibility_count.get(face_idx, 0) + 1
                self.wound_faces_global.add(face_idx)
            
            self.wound_vertices_global.update(spectator.visible_vertex_indices)
            
            print(f"  Visible faces: {len(spectator.visible_face_indices)}")
        
        print(f"Total wound faces (any visibility): {len(self.wound_faces_global)}")
        print(f"Total wound vertices: {len(self.wound_vertices_global)}")
        
        self.faces_processed = True

    def _auto_select_filter_mode(self,
                                 benchmark_cameras: int,
                                 max_mask_rays: int,
                                 speedup_threshold: float) -> str:
        """Pick the faster filter mode using a small benchmark."""
        try:
            import open3d as _
        except ImportError:
            print("Open3D not available, using numpy filtering")
            return "numpy"

        candidates = [s for s in self.spectators.values() if s.mask_loaded]
        if not candidates:
            return "numpy"

        benchmark_targets = candidates[:max(1, benchmark_cameras)]
        numpy_time = 0.0
        open3d_time = 0.0

        for spectator in benchmark_targets:
            state = self._snapshot_spectator_state(spectator)

            start = time.perf_counter()
            spectator.filter_faces_through_mask(
                self.vertices, self.faces, self.face_centers,
                self.vertex_colors, self.face_normals
            )
            numpy_time += time.perf_counter() - start
            self._restore_spectator_state(spectator, state)

            start = time.perf_counter()
            self._run_open3d_filter(spectator, max_mask_rays=max_mask_rays)
            open3d_time += time.perf_counter() - start
            self._restore_spectator_state(spectator, state)

        print(f"Benchmark time (numpy): {numpy_time:.4f}s")
        print(f"Benchmark time (Open3D): {open3d_time:.4f}s")

        if open3d_time > 0 and open3d_time <= (numpy_time / speedup_threshold):
            return "open3d"

        return "numpy"

    def _snapshot_spectator_state(self, spectator: Spectator) -> Dict[str, Any]:
        """Capture spectator filtering state for benchmarking."""
        return {
            "visible_face_indices": list(spectator.visible_face_indices),
            "visible_vertex_indices": set(spectator.visible_vertex_indices),
            "filtered_out_faces": dict(spectator.filtered_out_faces),
            "faces_processed": spectator.faces_processed
        }

    def _restore_spectator_state(self, spectator: Spectator, state: Dict[str, Any]) -> None:
        """Restore spectator filtering state after benchmarking."""
        spectator.visible_face_indices = state["visible_face_indices"]
        spectator.visible_vertex_indices = state["visible_vertex_indices"]
        spectator.filtered_out_faces = state["filtered_out_faces"]
        spectator.faces_processed = state["faces_processed"]

    def _run_open3d_filter(self, spectator: Spectator, max_mask_rays: int) -> None:
        """Filter faces using Open3D raycasting through the mask."""
        visible_faces = self._filter_faces_with_open3d_raycast(
            spectator=spectator,
            max_mask_rays=max_mask_rays
        )

        spectator.visible_face_indices = visible_faces
        spectator.filtered_out_faces = {"open3d_raycast": []}
        spectator.visible_vertex_indices.clear()
        for face_idx in visible_faces:
            face = self.faces[face_idx]
            spectator.visible_vertex_indices.update(face)
        spectator.faces_processed = True

    def _filter_faces_with_open3d_raycast(self,
                                          spectator: Spectator,
                                          max_mask_rays: int) -> List[int]:
        """Use Open3D raycasting to find faces visible through the mask."""
        try:
            import open3d as o3d
        except ImportError:
            print("Open3D not available, falling back to numpy filtering")
            spectator.filter_faces_through_mask(
                self.vertices, self.faces, self.face_centers,
                self.vertex_colors, self.face_normals
            )
            return spectator.visible_face_indices

        if spectator.mask_data is None or self.vertices is None or self.faces is None:
            return []

        if self._raycast_scene is None:
            mesh_legacy = o3d.geometry.TriangleMesh()
            mesh_legacy.vertices = o3d.utility.Vector3dVector(self.vertices)
            mesh_legacy.triangles = o3d.utility.Vector3iVector(self.faces)
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
            self._raycast_scene = o3d.t.geometry.RaycastingScene()
            self._raycast_mesh_id = self._raycast_scene.add_triangles(mesh_t)

        mask_coords = np.argwhere(spectator.mask_data > 0)
        if mask_coords.size == 0:
            return []

        if max_mask_rays and len(mask_coords) > max_mask_rays:
            rng = np.random.default_rng(0)
            sample_idx = rng.choice(len(mask_coords), size=max_mask_rays, replace=False)
            mask_coords = mask_coords[sample_idx]

        v_coords = mask_coords[:, 0].astype(np.float64)
        u_coords = mask_coords[:, 1].astype(np.float64)

        fx = spectator.intrinsics['fx']
        fy = spectator.intrinsics['fy']
        cx = spectator.intrinsics['cx']
        cy = spectator.intrinsics['cy']

        x_cam = (u_coords - cx) / fx
        y_cam = (v_coords - cy) / fy
        z_cam = np.ones_like(x_cam)
        dirs_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
        dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True) + 1e-9

        # World-space rays
        camera_center = -spectator.rotation_matrix.T @ spectator.translation_vector.flatten()
        dirs_world = (spectator.rotation_matrix.T @ dirs_cam.T).T
        dirs_world /= np.linalg.norm(dirs_world, axis=1, keepdims=True) + 1e-9

        origins = np.repeat(camera_center[None, :], len(dirs_world), axis=0)
        rays = np.hstack([origins, dirs_world]).astype(np.float32)

        rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        hits = self._raycast_scene.cast_rays(rays_t)

        # Plot hit distances in image space (reshape to mask size)
        # try:
        #     import matplotlib.pyplot as plt
        #     t_hit = hits['t_hit'].numpy()
        #     depth_img = np.full(spectator.mask_data.shape, np.nan, dtype=np.float32)
        #     depth_img[mask_coords[:, 0], mask_coords[:, 1]] = t_hit
        #     plt.imshow(depth_img)
        #     plt.colorbar()
        #     plt.title("Hit distance (depth)")
        #     plt.show()
        # except Exception as e:
        #     print(f"Depth plot skipped: {e}")

        prim_ids = hits['primitive_ids'].numpy()
        max_invalid = np.iinfo(prim_ids.dtype).max
        prim_ids = prim_ids[(prim_ids != max_invalid) & (prim_ids >= 0) & (prim_ids < len(self.faces))]

        if prim_ids.size == 0:
            return []

        return np.unique(prim_ids).tolist()
    
    def filter_by_visibility(self, min_visibility_count: int = 2) -> Set[int]:
        """
        Filter faces by minimum visibility count.
        
        Args:
            min_visibility_count: Minimum number of cameras that must see a face
            
        Returns:
            Set of face indices meeting visibility threshold
        """
        if not self.faces_processed:
            raise ValueError("Must process wound segmentation first")
        
        high_visibility_faces = {
            face_idx for face_idx, count in self.face_visibility_count.items()
            if count >= min_visibility_count
        }
        
        print(f"Visibility filtering: {len(high_visibility_faces)}/{len(self.wound_faces_global)} faces "
              f"seen by >= {min_visibility_count} cameras")
        
        return high_visibility_faces
    
    def extract_connected_components(self, face_set: Set[int], min_faces: int = 10) -> List[Set[int]]:
        """
        Extract connected components from a set of faces using mesh topology.
        
        Args:
            face_set: Set of face indices
            min_faces: Minimum number of faces for a valid component
            
        Returns:
            List of connected components (each is a set of face indices)
        """
        if not self.topology_computed:
            raise ValueError("Mesh topology must be computed first")
        
        unvisited = face_set.copy()
        components = []
        
        while unvisited:
            # Start new component with arbitrary face
            seed_face = next(iter(unvisited))
            component = set()
            queue = deque([seed_face])
            
            # BFS to find all connected faces
            while queue:
                current_face = queue.popleft()
                
                if current_face in unvisited:
                    unvisited.remove(current_face)
                    component.add(current_face)
                    
                    # Add adjacent faces that are in face_set
                    for adj_face in self.face_adjacency.get(current_face, []):
                        if adj_face in unvisited:
                            queue.append(adj_face)
            
            # Add component if it meets minimum size
            if len(component) >= min_faces:
                components.append(component)
        
        print(f"Extracted {len(components)} connected components from {len(face_set)} faces")
        return components
    
    def filter_by_surface_smoothness(self, face_set: Set[int], 
                                    max_curvature_variance: float = 0.5) -> Set[int]:
        """
        Filter faces by surface smoothness (low normal variation).
        
        This exploits mesh topology to identify smooth wound regions.
        
        Args:
            face_set: Set of face indices to filter
            max_curvature_variance: Maximum allowed normal angle variance
            
        Returns:
            Filtered set of face indices
        """
        if self.face_normals is None:
            print("Warning: No face normals available for smoothness filtering")
            return face_set
        
        smooth_faces = set()
        
        for face_idx in face_set:
            face_normal = self.face_normals[face_idx]
            
            # Get adjacent faces
            adjacent_faces = self.face_adjacency.get(face_idx, [])
            adjacent_in_set = [f for f in adjacent_faces if f in face_set]
            
            if not adjacent_in_set:
                # Isolated face, keep it
                smooth_faces.add(face_idx)
                continue
            
            # Compute normal angles with adjacent faces
            angles = []
            for adj_face in adjacent_in_set:
                adj_normal = self.face_normals[adj_face]
                dot_product = np.clip(np.dot(face_normal, adj_normal), -1.0, 1.0)
                angle = np.arccos(dot_product)
                angles.append(angle)
            
            # Check if variance is acceptable
            if len(angles) > 0:
                angle_variance = np.var(angles)
                if angle_variance <= max_curvature_variance:
                    smooth_faces.add(face_idx)
        
        print(f"Smoothness filtering: {len(smooth_faces)}/{len(face_set)} faces are smooth")
        return smooth_faces

    def expand_wound_by_enclosed_faces(self, wound_faces: Set[int]) -> Set[int]:
        """Expand wound to include enclosed non-wound faces.

        Any non-wound faces that are not reachable from mesh boundary without
        crossing a wound face are treated as inside the wound region.
        """
        import rerun as rr
        rr.init ("3D Wound Detection", spawn=True)
        
        
        if not self.topology_computed or not wound_faces:
            return wound_faces

        boundary_faces = self._get_boundary_faces()
        non_wound_faces = set(range(len(self.faces))) - set(wound_faces)

        # Extract boundary face vertices and normals for visualization
        if boundary_faces:
            boundary_face_list = list(boundary_faces)
            boundary_vertex_set = set()
            for face_idx in boundary_face_list:
                face = self.faces[face_idx]
                boundary_vertex_set.update(face)
            
            boundary_vertex_list = sorted(list(boundary_vertex_set))
            vertex_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(boundary_vertex_list)}
            
            # Extract vertex positions and normals
            boundary_positions = self.vertices[boundary_vertex_list]
            boundary_normals = self.vertex_normals[boundary_vertex_list] if self.vertex_normals is not None else None
            
            # Remap faces to new vertex indices
            boundary_triangles = []
            for face_idx in boundary_face_list:
                old_face = self.faces[face_idx]
                new_face = [vertex_remap[v] for v in old_face]
                boundary_triangles.append(new_face)
            
            # Log boundary faces mesh
            rr.log(
                "/wound/boundary_faces",
                rr.Mesh3D(
                    vertex_positions=boundary_positions,
                    vertex_normals=boundary_normals,
                    vertex_colors=[[0, 255, 255]] * len(boundary_positions),  # Cyan
                    triangle_indices=boundary_triangles,
                ),
            )
            
            # Log wound faces
            wound_face_list = list(wound_faces)
            rr.log(
                "/wound/wound_faces",
                rr.Mesh3D(
                    vertex_positions=self.vertices,
                    vertex_normals=self.vertex_normals,
                    vertex_colors=self.vertex_colors,  
                    triangle_indices=[self.faces[f] for f in wound_face_list],
                ),
            )

            # Log non-wound faces
            non_wound_face_list = list(non_wound_faces)
            non_wound_vertex_set = set()
            for face_idx in non_wound_face_list:
                face = self.faces[face_idx]
                non_wound_vertex_set.update(face)
            
            non_wound_vertex_list = sorted(list(non_wound_vertex_set))
            non_wound_vertex_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(non_wound_vertex_list)}
            
            non_wound_positions = self.vertices[non_wound_vertex_list]
            non_wound_normals = self.vertex_normals[non_wound_vertex_list] if self.vertex_normals is not None else None
            
            non_wound_triangles = []
            for face_idx in non_wound_face_list:
                old_face = self.faces[face_idx]
                new_face = [non_wound_vertex_remap[v] for v in old_face]
                non_wound_triangles.append(new_face)
            
            rr.log(
                "/wound/non_wound_faces",
                rr.Mesh3D(
                    vertex_positions=non_wound_positions,
                    vertex_normals=non_wound_normals,
                    vertex_colors=[[0, 255, 30]] * len(non_wound_positions),  # Green
                    triangle_indices=non_wound_triangles,
                ),
            )
        
        queue = deque([f for f in boundary_faces if f in non_wound_faces])
        outside = set()

        while queue:
            face_idx = queue.popleft()
            if face_idx in outside:
                continue
            outside.add(face_idx)

            for adj_face in self.face_adjacency.get(face_idx, []):
                if adj_face in non_wound_faces and adj_face not in outside:
                    queue.append(adj_face)

        enclosed_faces = non_wound_faces - outside
        if enclosed_faces:
            print(f"Expanded wound with {len(enclosed_faces)} enclosed faces")

        return set(wound_faces) | enclosed_faces

    def _get_boundary_faces(self) -> Set[int]:
        """Get faces on the mesh boundary (edges with only one adjacent face)."""
        boundary_faces = set()
        for edge, face_list in self.edge_adjacency.items():
            if len(face_list) == 1:
                boundary_faces.add(face_list[0])
        return boundary_faces

    def compute_wound_edge(self, wound_faces: Set[int]) -> None:
        """Compute wound edge edges and vertices from wound/non-wound boundary."""
        self.wound_edge_edges = []
        self.wound_edge_vertices = set()
        self.wound_edge_faces = set()

        wound_faces_set = set(wound_faces)
        all_faces_set = set(range(len(self.faces)))
        non_wound_faces = all_faces_set - wound_faces_set

        for edge, face_list in self.edge_adjacency.items():
            wound_count = sum(1 for f in face_list if f in wound_faces_set)
            non_wound_count = sum(1 for f in face_list if f in non_wound_faces)

            if wound_count > 0 and (non_wound_count > 0 or len(face_list) == 1):
                self.wound_edge_edges.append(edge)
                self.wound_edge_vertices.update(edge)
                for f in face_list:
                    if f in wound_faces_set:
                        self.wound_edge_faces.add(f)
    
    def filter_wound_faces(self, min_visibility: int = 2, 
                          use_smoothness: bool = True) -> Set[int]:
        """
        Filter wound faces by visibility and surface smoothness.
        
        Args:
            min_visibility: Minimum camera visibility count
            use_smoothness: Whether to apply smoothness filtering
            
        Returns:
            Set of filtered face indices
        """
        if not self.faces_processed:
            raise ValueError("Must process wound segmentation first")
        
        # Start with visibility filtering
        filtered_faces = self.filter_by_visibility(min_visibility)
        self.removed_faces_visibility = set(self.wound_faces_global) - set(filtered_faces)
        
        # Apply smoothness filtering
        if use_smoothness and len(filtered_faces) > 0:
            pre_smooth = set(filtered_faces)
            filtered_faces = self.filter_by_surface_smoothness(filtered_faces)
            self.removed_faces_smoothness = pre_smooth - set(filtered_faces)
        else:
            self.removed_faces_smoothness = set()
        
        return filtered_faces
    
    def morphological_closing(self, face_set: Set[int], iterations: int = 1, kernel_size: int = 1) -> Set[int]:
        """
        Perform morphological closing on wound faces (dilation followed by erosion).
        
        Similar to 2D closing, this operation fills small holes and gaps in the wound region
        by first expanding (dilating) the wound faces to include adjacent non-wound faces,
        then contracting (eroding) back to remove temporary additions.
        
        Args:
            face_set: Set of face indices to perform closing on
            iterations: Number of closing iterations
            kernel_size: Number of adjacency layers to expand in each dilation step (1=immediate neighbors, 2=2 layers, etc.)
            
        Returns:
            Set of face indices after closing operation
        """
        if not self.topology_computed or not face_set:
            return face_set
        
        def _erode_faces(face_ids: Set[int], drop_isolated: bool = False) -> Set[int]:
            eroded_faces = set()
            for face_idx in face_ids:
                adjacent = self.face_adjacency.get(face_idx, [])
                if drop_isolated and not adjacent:
                    continue
                wound_neighbors = sum(1 for adj in adjacent if adj in face_ids)
                if wound_neighbors >= len(adjacent) / 2:
                    if not drop_isolated or wound_neighbors > 0:
                        eroded_faces.add(face_idx)
            return eroded_faces

        current_faces = face_set.copy()

        # Pre-erosion to drop tiny isolated structures before closing.
        current_faces = _erode_faces(current_faces, drop_isolated=True)
        
        for iteration in range(iterations):
            # Dilation: expand by kernel_size layers of adjacent faces
            dilated = current_faces.copy()
            expansion_front = current_faces.copy()
            
            
            for layer in range(kernel_size):
                next_front = set()
                for face_idx in expansion_front:
                    adjacent = self.face_adjacency.get(face_idx, [])
                    for adj_face in adjacent:
                        if adj_face not in dilated:
                            next_front.add(adj_face)
                dilated.update(next_front)
                expansion_front = next_front
                if not expansion_front:  # No more faces to expand
                    break
            
            # Erosion: remove faces that don't have enough wound neighbors
            eroded = _erode_faces(dilated, drop_isolated=False)
            
            current_faces = eroded
        
        added_faces = current_faces - face_set
        if added_faces:
            print(f"Morphological closing added {len(added_faces)} faces")
        
        return current_faces
    
    def expand_and_extract_components(self, face_set: Set[int],
                                     use_connectivity: bool = True,
                                     min_component_faces: int = 10) -> Set[int]:
        """
        Expand wound to include enclosed faces and extract connected components.
        
        Args:
            face_set: Set of face indices to process
            use_connectivity: Whether to extract connected components
            min_component_faces: Minimum faces for valid component
            
        Returns:
            Set of final face indices
        """
        if not face_set:
            return face_set
        
        # Expand wound to include enclosed faces
        if use_connectivity:
            pre_expand = set(face_set)
            expanded_faces = self.expand_wound_by_enclosed_faces(face_set)
            self.added_faces_enclosed = set(expanded_faces) - pre_expand
        else:
            expanded_faces = face_set
            self.added_faces_enclosed = set()
        
        # Extract connected components
        if use_connectivity and len(expanded_faces) > 0:
            pre_connectivity = set(expanded_faces)
            components = self.extract_connected_components(expanded_faces, min_component_faces)
            
            if components:
                # Take largest component as primary wound
                largest_component = max(components, key=len)
                final_faces = largest_component
                print(f"Selected largest component with {len(largest_component)} faces")
            else:
                print("Warning: No valid connected components found")
                final_faces = set()

            self.removed_faces_connectivity = pre_connectivity - set(final_faces)
        else:
            final_faces = expanded_faces
            self.removed_faces_connectivity = set()
        
        return final_faces
    
    def get_wound_mesh(self, min_visibility: int = 2, 
                      use_connectivity: bool = True,
                      min_component_faces: int = 10,
                      use_smoothness: bool = True,
                      use_closing: bool = True,
                      closing_iterations: int = 1,
                      closing_kernel_size: int = 1,
                      use_fill_holes: bool = False,
                      hole_size: float = -1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get final wound mesh after all filtering steps.
        
        Args:
            min_visibility: Minimum camera visibility count
            use_connectivity: Whether to extract connected components
            min_component_faces: Minimum faces for valid component
            use_smoothness: Whether to apply smoothness filtering
            use_closing: Whether to apply morphological closing
            closing_iterations: Number of closing iterations
            closing_kernel_size: Number of adjacency layers to expand per iteration (1=neighbors, 2=2 layers, etc.)
            use_fill_holes: Whether to fill holes in the mesh using trimesh
            hole_size: Maximum hole size to fill (-1.0 for all holes)
            
        Returns:
            Tuple of (wound_vertices, wound_faces, wound_colors)
        """
        if not self.faces_processed:
            raise ValueError("Must process wound segmentation first")
        
        # Step 1: Filter by visibility and smoothness
        filtered_faces = self.filter_wound_faces(min_visibility, use_smoothness)
        
        if not filtered_faces:
            print("Warning: No wound faces remain after filtering")
            return np.array([]), np.array([]), np.array([])
        
        # Step 2: Apply morphological closing
        if use_closing:
            filtered_faces = self.morphological_closing(filtered_faces, closing_iterations, closing_kernel_size)
        
        if not filtered_faces:
            print("Warning: No wound faces remain after closing")
            return np.array([]), np.array([]), np.array([])
        
        # Step 3: Expand and extract components
        filtered_faces = self.expand_and_extract_components(
            filtered_faces, use_connectivity, min_component_faces
        )
        
        if not filtered_faces:
            print("Warning: No wound faces remain after component extraction")
            return np.array([]), np.array([]), np.array([])
        
        # Extract wound vertices and remap faces
        wound_face_list = list(filtered_faces)
        wound_vertex_set = set()
        
        for face_idx in wound_face_list:
            face = self.faces[face_idx]
            wound_vertex_set.update(face)
        
        wound_vertex_list = sorted(list(wound_vertex_set))
        vertex_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(wound_vertex_list)}
        
        # Extract vertices and colors
        wound_vertices = self.vertices[wound_vertex_list]
        wound_colors = self.vertex_colors[wound_vertex_list] if self.vertex_colors is not None else None
        
        # Remap faces
        wound_faces = []
        for face_idx in wound_face_list:
            old_face = self.faces[face_idx]
            new_face = [vertex_remap[v] for v in old_face]
            wound_faces.append(new_face)
        wound_faces = np.array(wound_faces)
        
        # Step 4: Fill holes in the mesh (optional)
        if use_fill_holes and len(wound_vertices) > 0:
            print("Filling holes in wound mesh...")
            wound_vertices, wound_faces, wound_colors = self._fill_mesh_holes(
                wound_vertices, wound_faces, wound_colors, hole_size
            )
        
        print(f"Final wound mesh: {len(wound_vertices)} vertices, {len(wound_faces)} faces")
        
        # Update global tracking
        self.wound_faces_global = filtered_faces
        self.wound_vertices_global = wound_vertex_set

        # Compute wound edge boundary
        self.compute_wound_edge(filtered_faces)
        
        return wound_vertices, wound_faces, wound_colors
    
    def _fill_mesh_holes(self, vertices: np.ndarray, faces: np.ndarray, 
                        colors: np.ndarray = None, hole_size: float = -1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fill holes in the mesh using trimesh's fill_holes method.
        
        Args:
            vertices: Vertex array (N, 3)
            faces: Face array (M, 3)
            colors: Optional color array (N, 3)
            hole_size: Maximum hole size to fill (-1.0 for all holes)
            
        Returns:
            Tuple of (vertices, faces, colors) after hole filling
        """
        try:
            import trimesh

            if hole_size not in (-1.0, None):
                print("Note: trimesh.fill_holes ignores hole_size and fills all holes")

            colors_uint8 = None
            if colors is not None:
                if np.max(colors) <= 1.0:
                    colors_uint8 = (colors * 255.0).astype(np.uint8)
                else:
                    colors_uint8 = colors.astype(np.uint8)

            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=colors_uint8,
                process=False
            )

            initial_faces = len(mesh.faces)
            print("Attempting to fill holes in wound mesh...")
            added = mesh.fill_holes()

            vertices_filled = np.asarray(mesh.vertices)
            faces_filled = np.asarray(mesh.faces)

            colors_filled = colors
            if colors_uint8 is not None and hasattr(mesh.visual, "vertex_colors"):
                vcols = np.asarray(mesh.visual.vertex_colors)
                if vcols.ndim == 2 and vcols.shape[1] >= 3:
                    colors_filled = vcols[:, :3]

            filled_faces = len(faces_filled)
            print(f"Hole filling complete: {initial_faces} -> {filled_faces} faces (added {added})")

            return vertices_filled, faces_filled, colors_filled

        except ImportError:
            print("Trimesh not available for hole filling, returning original mesh")
            return vertices, faces, colors
        except Exception as e:
            print(f"Error during hole filling: {e}, returning original mesh")
            return vertices, faces, colors
    
    def save_wound_mesh(self, output_path: Path, vertices: np.ndarray, 
                       faces: np.ndarray, colors: np.ndarray = None):
        """
        Save wound mesh to PLY file.
        
        Args:
            output_path: Output PLY file path
            vertices: Vertex array (N, 3)
            faces: Face array (M, 3)
            colors: Optional color array (N, 3)
        """
        output_path = Path(output_path)
        print(f"Saving wound mesh to: {output_path}")
        
        try:
            import open3d as o3d
            
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            
            if colors is not None:
                # Normalize colors to [0, 1]
                colors_normalized = colors / 255.0 if np.max(colors) > 1.0 else colors
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors_normalized)
            
            # Compute normals
            mesh.compute_vertex_normals()
            
            o3d.io.write_triangle_mesh(str(output_path), mesh)
            print(f"Saved mesh successfully")
            
        except ImportError:
            print("Open3D not available for mesh export")
    
    def visualize_with_rerun(self, session_name: str = "WoundModel3DMesh"):
        """
        Visualize mesh-based wound segmentation using Rerun.
        
        Args:
            session_name: Name for the Rerun session
        """
        if not self.model_loaded:
            raise ValueError("Mesh must be loaded first")
        
        try:
            import rerun as rr
            import rerun.blueprint as rrb
        except ImportError:
            print("Rerun not available for visualization")
            return
        
        # Initialize Rerun
        try:
            rr.init(session_name, spawn=True)
        except RuntimeError:
            rr.init(session_name)
            rr.serve()
        
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        
        # Create blueprint
        blueprint = rrb.Blueprint(
            rrb.Tabs(
                rrb.Spatial3DView(origin="/world", name="Full Mesh"),
                rrb.Spatial3DView(origin="/wound", name="Wound Region"),
                rrb.TextDocumentView(origin="/stats", name="Statistics"),
            )
        )
        rr.send_blueprint(blueprint)
        
        # Log full mesh
        vertex_positions = self.vertices.astype(np.float32)
        
        # Create mesh entity
        if self.vertex_colors is not None:
            colors = self.vertex_colors.astype(np.uint8)
        else:
            colors = np.ones((len(self.vertices), 3), dtype=np.uint8) * 128
        
        # Log mesh using individual triangles (Rerun doesn't have direct mesh support yet)
        # We'll display it as triangles
        rr.log("/world/full_mesh", rr.Points3D(
            positions=vertex_positions,
            colors=colors,
            radii=0.001
        ), static=True)
        
        # Log wound faces if processed
        if self.faces_processed and self.wound_faces_global:
            wound_face_list = list(self.wound_faces_global)
            wound_vertex_set = set()
            
            for face_idx in wound_face_list:
                face = self.faces[face_idx]
                wound_vertex_set.update(face)
            
            wound_vertex_list = list(wound_vertex_set)
            wound_positions = self.vertices[wound_vertex_list]
            
            if self.vertex_colors is not None:
                wound_colors = self.vertex_colors[wound_vertex_list]
            else:
                wound_colors = np.array([[255, 0, 0]] * len(wound_vertex_list))  # Red
            
            rr.log("/wound/vertices", rr.Points3D(
                positions=wound_positions,
                colors=wound_colors,
                radii=0.002
            ), static=True)

        # Log wound edge vertices and edges
        if self.wound_edge_vertices:
            edge_vertex_list = list(self.wound_edge_vertices)
            edge_positions = self.vertices[edge_vertex_list]
            edge_colors = np.array([[255, 255, 0]] * len(edge_vertex_list), dtype=np.uint8)
            rr.log("/wound/edge_vertices", rr.Points3D(
                positions=edge_positions,
                colors=edge_colors,
                radii=0.003
            ), static=True)

        if self.wound_edge_edges:
            edge_strips = []
            for v0, v1 in self.wound_edge_edges:
                edge_strips.append(np.array([self.vertices[v0], self.vertices[v1]], dtype=np.float32))
            rr.log("/wound/edge_edges", rr.LineStrips3D(
                strips=edge_strips,
                colors=[[255, 255, 0]] * len(edge_strips),
                radii=0.0008
            ), static=True)

        # Log removed faces by filtering steps
        self._log_removed_faces_rr(rr, "/wound/removed/visibility", self.removed_faces_visibility, [160, 160, 160])
        self._log_removed_faces_rr(rr, "/wound/removed/smoothness", self.removed_faces_smoothness, [0, 128, 255])
        self._log_removed_faces_rr(rr, "/wound/removed/connectivity", self.removed_faces_connectivity, [255, 128, 0])
        
        # Log statistics
        stats_text = f"""
Mesh Statistics:
- Total Vertices: {len(self.vertices):,}
- Total Faces: {len(self.faces):,}
- Wound Faces: {len(self.wound_faces_global):,}
- Wound Vertices: {len(self.wound_vertices_global):,}
- Spectators: {len(self.spectators)}
- Masks Loaded: {sum(1 for s in self.spectators.values() if s.mask_loaded)}
        """
        
        rr.log("/stats", rr.TextDocument(stats_text), static=True)
        
        print(f"Rerun visualization started: {session_name}")

    def _log_removed_faces_rr(self, rr, path: str, face_set: Set[int], color: List[int]) -> None:
        """Log vertices from a face set for visualization in Rerun."""
        if not face_set:
            return
        vertex_set = set()
        for face_idx in face_set:
            face = self.faces[face_idx]
            vertex_set.update(face)

        vertex_list = list(vertex_set)
        positions = self.vertices[vertex_list]
        colors = np.array([color] * len(vertex_list), dtype=np.uint8)
        rr.log(path, rr.Points3D(
            positions=positions,
            colors=colors,
            radii=0.0018
        ), static=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about mesh-based wound segmentation."""
        stats = {
            'model_loaded': self.model_loaded,
            'masks_loaded': self.masks_loaded,
            'faces_processed': self.faces_processed,
            'total_vertices': len(self.vertices) if self.vertices is not None else 0,
            'total_faces': len(self.faces) if self.faces is not None else 0,
            'total_spectators': len(self.spectators),
            'spectators_with_masks': sum(1 for s in self.spectators.values() if s.mask_loaded),
            'topology_computed': self.topology_computed
        }
        
        if self.faces_processed:
            stats['wound_faces_count'] = len(self.wound_faces_global)
            stats['wound_vertices_count'] = len(self.wound_vertices_global)
            
            if self.face_visibility_count:
                counts = list(self.face_visibility_count.values())
                stats['visibility_stats'] = {
                    'min': min(counts),
                    'max': max(counts),
                    'mean': np.mean(counts),
                    'median': np.median(counts)
                }
        
        return stats


def main():
    """
    Example usage of mesh-based wound detection.
    """
    
    # Example paths (adjust to your data)
    mesh_path = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\test\dense\poisson.ply")
    model_path = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\test\dense")
    mask_dir = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\test\mask_wound_only_undistorted\images")
    output_path = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\test\dense\wound_mesh.ply")
    output_path_corrected = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\test\dense\wound_mesh_corrected.ply")
    
    
    # mesh_path = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_WoundDetection\Fuss1\dense\poisson.ply")
    # model_path = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_WoundDetection\Fuss1\dense")
    # mask_dir = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_WoundDetection\Fuss1\dense\images_woundmask")
    try:
        # Initialize model
        wound_model = WoundModel3DMesh()
        
        # Load mesh
        print("=== Loading Mesh ===")
        wound_model.load_mesh(mesh_path, compute_normals=True)
        
        # Load COLMAP cameras
        print("\n=== Loading COLMAP Cameras ===")
        wound_model.load_colmap_cameras(model_path)
        
        # Load masks
        print("\n=== Loading Masks ===")
        wound_model.load_masks(mask_dir)
        
        # Process wound segmentation
        print("\n=== Processing Wound Segmentation ===")
        wound_model.process_wound_segmentation()
        
        # Get statistics
        print("\n=== Statistics ===")
        stats = wound_model.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Get final wound mesh with mesh-specific filtering
        print("\n=== Extracting Wound Mesh ===")
        wound_vertices, wound_faces, wound_colors = wound_model.get_wound_mesh(
            min_visibility=3,  # Require visibility from multiple cameras
            use_connectivity=True,  # Extract connected components
            min_component_faces=100,  # Minimum size for valid component
            use_smoothness=False,  # Filter by surface smoothness
            use_closing=False,  # Apply morphological closing
            closing_iterations=5,  # Number of closing iterations
            closing_kernel_size=2,  # Number of adjacency layers per iteration (1-3 typical)
            use_fill_holes=True,  # We'll use trimesh instead in correction step
            hole_size=-1.0
        )
        
        # Get initial statistics before correction
        print("\n=== Initial Wound Mesh Statistics ===")
        print(f"Vertices: {len(wound_vertices)}")
        print(f"Faces: {len(wound_faces)}")
        
        # Correct and close wound mesh
        print("\n=== Correcting and Closing Wound Mesh ===")
        corrected_vertices, corrected_faces, corrected_colors = correct_wound_mesh_from_detection(
            wound_vertices, wound_faces, wound_colors,
            full_vertices=wound_model.vertices,
            full_faces=wound_model.faces,
            simplify_full_mesh=True,  # Only simplify if mesh is very large (300k+ vertices)
            verbose=True
        )
        
        # Get corrected statistics
        print("\n=== Corrected Wound Mesh Statistics ===")
        print(f"Vertices: {len(corrected_vertices)}")
        print(f"Faces: {len(corrected_faces)}")
        
        # Use corrected mesh for output
        wound_vertices = corrected_vertices
        wound_faces = corrected_faces
        if wound_colors is not None and corrected_colors is not None:
            wound_colors = corrected_colors
        # if len(wound_vertices) > 0:
        #     print("\n=== Saving Meshes ===")
        #     wound_model.save_wound_mesh(output_path, wound_vertices, wound_faces, wound_colors)
        #     print(f"Saved initial wound mesh to: {output_path}")
        
        # if len(corrected_vertices) > 0:
        #     wound_model.save_wound_mesh(output_path_corrected, corrected_vertices, corrected_faces, corrected_colors)
        #     print(f"Saved corrected wound mesh to: {output_path_corrected}")
        
        # Visualize
        # print("\n=== Launching Visualization ===")
        wound_model.visualize_with_rerun("WoundModel3DMesh_Main")
        
        print("\n=== Processing Complete ===")
        
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
