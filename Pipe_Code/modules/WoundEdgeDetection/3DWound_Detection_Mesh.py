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
        self.extruded_mask_data = None  # Extruded/extended wound mask
        self.extruded_mask_loaded = False
        
        # Depth map data
        self.depth_map = None
        self.depth_map_loaded = False
        
        # Face visibility tracking (mesh-specific)
        self.visible_face_indices = []  # Faces visible through mask
        self.visible_vertex_indices = set()  # Vertices from visible faces
        self.periwound_edge_face_indices = []  # Faces in the periwound region (mask difference)
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
        """Load wound mask for this camera view.
        
        Masks are expected to be named as: image_name.jpg.png
        For example, if image is 'image_001.jpg', mask should be 'image_001.jpg.png'
        """
        mask_dir = Path(mask_dir)
        
        # Only look for masks with the pattern: image_name.extension.png
        mask_name = f"{self.image_name}.png"
        mask_path = mask_dir / mask_name
        
        if mask_path.exists():
            import cv2
            self.mask_data = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            self.mask_loaded = True
            return True
        
        print(f"Warning: No mask found for {self.image_name} at {mask_path}")
        return False
    
    def load_extruded_mask(self, extruded_mask_dir: Path) -> bool:
        """Load extruded/extended wound mask for this camera view.
        
        Extruded masks are expected to use the same naming convention as regular masks.
        The difference between extruded and regular masks defines the periwound region.
        """
        extruded_mask_dir = Path(extruded_mask_dir)
        mask_name = f"{self.image_name}.png"
        mask_path = extruded_mask_dir / mask_name
        
        if mask_path.exists():
            import cv2
            self.extruded_mask_data = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            self.extruded_mask_loaded = True
            return True
        
        print(f"Warning: No extruded mask found for {self.image_name} at {mask_path}")
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
        # Single primary mesh storage as Open3D TriangleMesh
        self.mesh = None            # Full mesh (o3d.geometry.TriangleMesh)

        # Derived numpy arrays (synced from self.mesh for fast access)
        self.vertices = None  # (N, 3) vertex coordinates
        self.faces = None  # (M, 3) face indices
        self.vertex_colors = None  # (N, 3) vertex colors [0..1]
        self.vertex_normals = None  # (N, 3) vertex normals
        self.face_normals = None  # (M, 3) face normals
        self._raycast_scene = None
        self._raycast_mesh_id = None

        # Mesh topology (using trimesh/O3D adjacency)
        self.edge_adjacency = {}  # edge -> list of face indices
        self.face_adjacency = {}  # face -> list of adjacent faces
        
        # Camera and image management
        self.spectators = {}  # Dict: image_name -> Spectator
        self.cameras = {}
        self.images = {}
        
        # Face visibility tracking (indices into self.faces)
        self.face_visibility_count = {}  # face_idx -> count of camera views
        self.wound_face_indices = set()  # Faces visible through wound masks
        self.tissue_face_indices = set()  # Faces from extended_wound_mask - wound_mask (initial periwound)
        
        # Periwound tracking (boundary region detected from mask difference)
        self.periwound_edge_faces = set()  # Faces detected in periwound region
        self.periwound_edge_visibility_count = {}  # face_idx -> count of cameras seeing periwound
        self.periwound_edge_mean_color = {}  # face_idx -> mean color [R, G, B]
        self.periwound_edge_color_std = {}  # face_idx -> color std dev
        self.periwound_edge_camera_data = {}  # face_idx -> list of per-camera measurements
        
        # Normal tissue tracking (healthy tissue unaffected by wound - from undistorted mask difference)
        self.normal_tissue_face_indices = set()  # Faces detected in normal tissue region
        self.normal_tissue_visibility_count = {}  # face_idx -> count of cameras seeing normal tissue
        
        # Confidence maps based on visibility
        self.wound_confidence_map = {}  # face_idx -> confidence [0-1] based on wound visibility
        self.periwound_edge_confidence_map = {}  # face_idx -> confidence [0-1] based on periwound visibility
        self.overlapping_faces = set()  # Faces seen in both wound and periwound
        self.overlapping_wound_counts = {}  # face_idx -> wound visibility count for overlapping faces
        self.overlapping_tissue_counts = {}  # face_idx -> periwound visibility count for overlapping faces

        # Semantic classification (feature-based confidence)
        self.semantic_feature_names = []  # List of feature names
        self.semantic_face_indices = []  # Faces used for semantic classification
        self.semantic_features = None  # (N, D) feature matrix for semantic classification
        self.semantic_features_raw = None  # (N, D) unstandardized feature matrix for visualization
        self.semantic_posteriors = {}  # face_idx -> np.array([p_tissue, p_periwound, p_wound])
        self.semantic_posteriors_before_mrf = {}  # face_idx -> posteriors before MRF constraints
        self.semantic_labels = {}  # face_idx -> label string ("tissue", "periwound", or "wound")
        self.semantic_confidence = {}  # face_idx -> confidence [0-1]
        self.semantic_class_names = ["tissue", "periwound", "wound"]  # 3-class classification
        self.semantic_wound_prob = {}  # face_idx -> P(wound)
        self.semantic_periwound_prob = {}  # face_idx -> P(periwound)
        self.semantic_tissue_prob = {}  # face_idx -> P(tissue)
        
        # Intermediate data for visualization (feature extraction & GMM stages)
        self.semantic_fitted_plane = None  # (normal, d) for height deviation reference
        self.semantic_class_samples = {}  # class_name -> list of face indices in each class
        self.semantic_gmm_models = {}  # Stored GMM models for visualization
        self.semantic_feature_colormaps = {}  # feature_name -> dict of {face_idx: color}
        self.semantic_log_likelihoods = None  # (N, C) raw log-likelihoods from GMM (before softmax)
        self.semantic_wound_loglik = {}  # face_idx -> log P(features | wound)
        self.semantic_tissue_loglik = {}  # face_idx -> log P(features | tissue)
        self.semantic_mrf_posteriors = {}  # face_idx -> np.array([p_tissue, p_periwound, p_wound]) after MRF (mean field only)
        self.semantic_mrf_method = None  # "graph_cut" or "mean_field"

        # Wound edge data (boundary region between wound and tissue - computed after classification)
        self.wound_edge_edges = []  # List of (v0, v1) tuples on wound-tissue boundary
        self.wound_edge_vertices = set()  # Set of vertex indices on boundary
        self.wound_edge_face_indices = set()  # Faces adjacent to both wound and tissue

        # Current filtering result (indices into self.faces)
        self.current_wound_faces = set()  # Current wound face indices after filtering
        
        # Filtering diagnostics (all store face indices)
        self.removed_faces_visibility = set()
        self.removed_faces_smoothness = set()
        self.removed_faces_connectivity = set()
        self.added_faces_enclosed = set()
        self.added_faces_closing = set()
        
        # Last wound mesh results (for visualization)
        self.last_wound_vertices = None
        self.last_wound_faces = None
        self.last_wound_colors = None
        
        # Model state
        self.model_loaded = False
        self.masks_loaded = False
        self.faces_processed = False
        self.topology_computed = False

        # Cached geometry helpers
        self._face_centroids_cache = None
        self._face_neighbors_cache = {}
        self._trimesh_cache = None
        self._feature_mean = None
        self._feature_std = None
    
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
            self.mesh = o3d.io.read_triangle_mesh(str(mesh_path))

            if compute_normals:
                if not self.mesh.has_vertex_normals():
                    print("Computing vertex normals...")
                    self.mesh.compute_vertex_normals()
                if not self.mesh.has_triangle_normals():
                    print("Computing face normals...")
                    self.mesh.compute_triangle_normals()

        except ImportError:
            print("Open3D not available")
            
        # Sync numpy arrays from mesh
        self._sync_arrays_from_mesh()
        
        # Build mesh topology
        self._compute_mesh_topology()
        
        
        self.model_loaded = True
    
    def _load_mesh_manual(self, mesh_path: Path):
        """Manual PLY mesh loading (fallback)."""
        # Basic PLY parser for mesh data
        # This is a simplified version - expand as needed
        raise NotImplementedError("Manual mesh loading not yet implemented. Please install Open3D.")

    def _sync_arrays_from_mesh(self):
        """Sync numpy arrays from the Open3D TriangleMesh object (self.mesh)."""
        self.vertices = np.asarray(self.mesh.vertices)
        self.faces = np.asarray(self.mesh.triangles)
        self.vertex_colors = np.asarray(self.mesh.vertex_colors) if self.mesh.has_vertex_colors() else None
        self.vertex_normals = np.asarray(self.mesh.vertex_normals) if self.mesh.has_vertex_normals() else None
        self.face_normals = np.asarray(self.mesh.triangle_normals) if self.mesh.has_triangle_normals() else None

        # Reset cached geometry helpers
        self._face_centroids_cache = None
        self._face_neighbors_cache = {}
        self._trimesh_cache = None
    
    def _compute_mesh_topology(self):
        """Compute mesh topology using Open3D and numpy.
        
        Builds edge adjacency and face adjacency from mesh data.
        """
        print("Computing mesh topology...")

        # Try trimesh for fast topology extraction
        try:
            import trimesh
            self._trimesh_cache = trimesh.Trimesh(
                vertices=self.vertices,
                faces=self.faces,
                process=False,
                maintain_order=True
            )

            # Build face adjacency
            self.face_adjacency = defaultdict(list)
            for f0, f1 in self._trimesh_cache.face_adjacency:
                self.face_adjacency[int(f0)].append(int(f1))
                self.face_adjacency[int(f1)].append(int(f0))

            # Build edge adjacency: edge -> list of face indices
            self.edge_adjacency = defaultdict(list)
            edges_unique = np.asarray(self._trimesh_cache.edges_unique, dtype=np.int64)
            edges_unique_inverse = np.asarray(self._trimesh_cache.edges_unique_inverse, dtype=np.int64)

            if edges_unique_inverse.size != 3 * len(self.faces):
                raise ValueError("Unexpected edges_unique_inverse shape; cannot map faces to edges")

            for face_idx, face in enumerate(self.faces):
                face_edges = (
                    (int(face[0]), int(face[1])),
                    (int(face[1]), int(face[2])),
                    (int(face[2]), int(face[0])),
                )
                base = face_idx * 3
                for local_offset, _ in enumerate(face_edges):
                    unique_edge_idx = int(edges_unique_inverse[base + local_offset])
                    if unique_edge_idx < 0 or unique_edge_idx >= len(edges_unique):
                        continue
                    edge_vertices = tuple(sorted(edges_unique[unique_edge_idx].tolist()))
                    self.edge_adjacency[edge_vertices].append(face_idx)

            print(
                f"Topology computed (trimesh): {len(self.edge_adjacency)} edges, "
                f"{len(self.face_adjacency)} faces with neighbors"
            )
            self.topology_computed = True
            return
        except Exception as e:
            print(f"Trimesh not available or failed ({e}), falling back to manual topology")

        # Build edge adjacency: edge -> list of face indices
        self.edge_adjacency = defaultdict(list)
        for face_idx, face in enumerate(self.faces):
            v0, v1, v2 = face
            edges = [
                tuple(sorted([v0, v1])),
                tuple(sorted([v1, v2])),
                tuple(sorted([v2, v0]))
            ]
            for edge in edges:
                self.edge_adjacency[edge].append(face_idx)
        
        # Build face adjacency from edge adjacency
        # Faces that share an edge are adjacent
        self.face_adjacency = defaultdict(list)
        for edge, face_list in self.edge_adjacency.items():
            if len(face_list) >= 2:
                # All faces sharing this edge are adjacent to each other
                for i, face_i in enumerate(face_list):
                    for face_j in face_list[i+1:]:
                        self.face_adjacency[face_i].append(face_j)
                        self.face_adjacency[face_j].append(face_i)
        
        print(f"Topology computed: {len(self.edge_adjacency)} edges, {len(self.face_adjacency)} faces with neighbors")
        self.topology_computed = True

    
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
    
    def load_extruded_masks(self, extruded_mask_dir: Path):
        """Load extruded wound masks for all spectators.
        
        Args:
            extruded_mask_dir: Directory containing extruded/extended wound masks
        """
        extruded_mask_dir = Path(extruded_mask_dir)
        print(f"Loading extruded masks from: {extruded_mask_dir}")
        
        extruded_loaded = 0
        for spectator in self.spectators.values():
            if spectator.load_extruded_mask(extruded_mask_dir):
                extruded_loaded += 1
        
        print(f"Loaded extruded masks for {extruded_loaded}/{len(self.spectators)} cameras")
        return extruded_loaded > 0
    
    def process_wound_segmentation(self,
                                   filter_mode: str = "open3d",
                                   benchmark_cameras: int = 1,
                                   max_mask_rays: int = 200000,
                                   speedup_threshold: float = 1.1):
        """
        Process mesh-based wound segmentation.
        
        Filters faces through all camera masks and builds wound face indices.
        All results are stored as indices into self.faces.
        """
        if not self.model_loaded:
            raise ValueError("Mesh must be loaded first")
        if not self.masks_loaded:
            raise ValueError("Masks must be loaded first")
        
        print(f"Processing mesh-based wound segmentation...")
        
        # Reset tracking
        self.face_visibility_count = {}
        self.wound_face_indices = set()

        # Process each spectator
        for spectator_name, spectator in self.spectators.items():
            if not spectator.mask_loaded:
                continue
            
            print(f"Processing spectator: {spectator_name}")
            
            # Filter faces through mask using Open3D raycasting
            self._run_open3d_filter(spectator, max_mask_rays=max_mask_rays)
        
            # Update global tracking (face indices)
            for face_idx in spectator.visible_face_indices:
                self.face_visibility_count[face_idx] = self.face_visibility_count.get(face_idx, 0) + 1
                self.wound_face_indices.add(face_idx)
            
            print(f"  Visible faces: {len(spectator.visible_face_indices)}")
        
        print(f"Total wound faces (any visibility): {len(self.wound_face_indices)}")
        
        self.faces_processed = True


    

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
            print("Open3D not available, cannot perform raycasting filter")
            return []
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

        prim_ids = hits['primitive_ids'].numpy()
        max_invalid = np.iinfo(prim_ids.dtype).max
        prim_ids = prim_ids[(prim_ids != max_invalid) & (prim_ids >= 0) & (prim_ids < len(self.faces))]

        if prim_ids.size == 0:
            return []

        return np.unique(prim_ids).tolist()
    
    def process_periwound_face_detection(self, max_mask_rays: int = 200000):
        """
        Detect periwound faces from the difference between extended masks and wound masks.
        
        Periwound faces are defined as pixels in the extended_wound_mask but NOT in the wound_mask.
        These represent the initial periwound region (boundary tissue around the wound).
        
        Args:
            max_mask_rays: Maximum number of rays to cast per camera (for performance)
        """
        if not self.model_loaded:
            raise ValueError("Mesh must be loaded first")
        
        if not any(s.extruded_mask_loaded for s in self.spectators.values()):
            print("Warning: No extruded masks loaded. Cannot detect periwound faces.")
            return
        
        print(f"Processing periwound face detection from mask differences...")
        
        # Reset tracking (tissue_face_indices for initial periwound detection)
        self.tissue_face_indices = set()
        periwound_visibility_count = {}
        
        # Process each spectator
        for spectator_name, spectator in self.spectators.items():
            if not spectator.extruded_mask_loaded:
                continue
            
            # Create difference mask: extruded_mask AND NOT wound_mask
            # Guard against missing wound mask: treat entire extruded mask as periwound
            difference_mask = np.zeros_like(spectator.extruded_mask_data, dtype=np.uint8)
            if spectator.mask_data is not None:
                difference_mask[(spectator.extruded_mask_data > 0) & (spectator.mask_data == 0)] = 255
            else:
                difference_mask[spectator.extruded_mask_data > 0] = 255
            
            # Test if there's anything in the difference
            if np.sum(difference_mask) == 0:
                print(f"  No periwound pixels found for {spectator_name}")
                continue
            
            # Project periwound mask to mesh faces
            periwound_faces = self._filter_faces_with_mask(
                spectator=spectator,
                mask=difference_mask,
                max_mask_rays=max_mask_rays
            )
            
            # Update global periwound tracking
            for face_idx in periwound_faces:
                self.tissue_face_indices.add(face_idx)
                periwound_visibility_count[face_idx] = periwound_visibility_count.get(face_idx, 0) + 1
            
            print(f"  Periwound faces detected: {len(periwound_faces)}")
        
        # Store visibility counts
        self.periwound_edge_visibility_count = periwound_visibility_count
        
        # Maintain backwards compatibility: periwound_edge_faces is same as tissue_face_indices in 2-class model
        self.periwound_edge_faces = self.tissue_face_indices.copy()
        
        print(f"Total periwound faces detected: {len(self.tissue_face_indices)}")
        if periwound_visibility_count:
            print(f"  Average visibility: {np.mean(list(periwound_visibility_count.values())):.2f} cameras")

    def process_tissue_face_detection(self, 
                                     undistorted_mask_dir: Path = None,
                                     max_mask_rays: int = 200000):
        """
        Detect normal tissue faces from the difference between undistorted masks and extruded wound masks.
        
        Normal tissue faces are defined as pixels in the undistorted_mask but NOT in the extruded_wound_mask.
        These represent the healthy tissue unaffected by the wound and its surroundings.
        
        Args:
            undistorted_mask_dir: Directory containing undistorted wound masks. 
                                 If None, loads from the spectator's mask_dir
            max_mask_rays: Maximum number of rays to cast per camera (for performance)
        """
        if not self.model_loaded:
            raise ValueError("Mesh must be loaded first")
        
        if not any(s.extruded_mask_loaded for s in self.spectators.values()):
            print("Warning: No extruded masks loaded. Cannot detect tissue faces.")
            return
        
        print(f"Processing normal tissue face detection from undistorted mask difference...")
        
        # Reset tracking
        self.normal_tissue_face_indices = set()
        normal_tissue_visibility_count = {}
        
        # Process each spectator
        for spectator_name, spectator in self.spectators.items():
            if not spectator.extruded_mask_loaded:
                continue
            
            # Try to load undistorted mask for this spectator
            undistorted_mask_data = None
            
            if undistorted_mask_dir is not None:
                # Load from specified undistorted mask directory
                undistorted_mask_path = Path(undistorted_mask_dir) / f"{spectator.image_name}.png"
                if undistorted_mask_path.exists():
                    try:
                        import cv2
                        undistorted_mask_data = cv2.imread(str(undistorted_mask_path), cv2.IMREAD_GRAYSCALE)
                        if undistorted_mask_data is None or undistorted_mask_data.size == 0:
                            print(f"  Warning: Could not load undistorted mask for {spectator_name}")
                            continue
                    except Exception as e:
                        print(f"  Warning: Error loading undistorted mask for {spectator_name}: {e}")
                        continue
                else:
                    print(f"  Warning: Undistorted mask not found for {spectator_name}")
                    continue
            elif spectator.mask_data is not None:
                # Fall back to regular mask if no undistorted directory provided
                undistorted_mask_data = spectator.mask_data.copy()
            else:
                print(f"  No mask data available for {spectator_name}")
                continue
            
            # Create tissue difference mask: undistorted_mask AND NOT extruded_mask
            # This identifies healthy tissue that's not in the extended wound area
            tissue_mask = np.zeros_like(undistorted_mask_data, dtype=np.uint8)
            tissue_mask[(undistorted_mask_data > 0) & (spectator.extruded_mask_data == 0)] = 255
            
            # Test if there's anything in the difference
            if np.sum(tissue_mask) == 0:
                print(f"  No normal tissue pixels found for {spectator_name}")
                continue
            
            # Project tissue mask to mesh faces
            tissue_faces = self._filter_faces_with_mask(
                spectator=spectator,
                mask=tissue_mask,
                max_mask_rays=max_mask_rays
            )
            
            # Update global tissue tracking
            for face_idx in tissue_faces:
                self.normal_tissue_face_indices.add(face_idx)
                normal_tissue_visibility_count[face_idx] = normal_tissue_visibility_count.get(face_idx, 0) + 1
            
            print(f"  Tissue faces detected: {len(tissue_faces)}")
        
        # Store visibility counts
        self.normal_tissue_visibility_count = normal_tissue_visibility_count
        
        print(f"Total normal tissue faces detected: {len(self.normal_tissue_face_indices)}")
        if normal_tissue_visibility_count:
            print(f"  Average visibility: {np.mean(list(normal_tissue_visibility_count.values())):.2f} cameras")

    def _get_total_views(self, face_idx: int) -> int:
        """Return total independent projections for a face.

        Uses sum of wound and periwound hits so that visibility ratios
        are coherent (r_wound + r_periwound <= 1.0).
        """
        wound_hits = self.face_visibility_count.get(face_idx, 0)
        edge_hits = self.periwound_edge_visibility_count.get(face_idx, 0)
        return wound_hits + edge_hits

    def _compute_visibility_ratios(self, face_idx: int) -> Tuple[float, float, int]:
        """Compute wound and periwound visibility ratios for a face."""
        total_views = self._get_total_views(face_idx)
        if total_views <= 0:
            return 0.0, 0.0, 0
        wound_hits = self.face_visibility_count.get(face_idx, 0)
        edge_hits = self.periwound_edge_visibility_count.get(face_idx, 0)
        r_wound = wound_hits / total_views
        r_edge = edge_hits / total_views
        return r_wound, r_edge, total_views

    def _ensure_face_normals(self) -> None:
        """Ensure face normals are available."""
        if self.face_normals is not None and len(self.face_normals) == len(self.faces):
            return
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
        self.face_normals = normals / norm

    def _get_face_centroids(self) -> np.ndarray:
        """Compute face centroids (cached)."""
        if self._face_centroids_cache is None:
            v0 = self.vertices[self.faces[:, 0]]
            v1 = self.vertices[self.faces[:, 1]]
            v2 = self.vertices[self.faces[:, 2]]
            self._face_centroids_cache = (v0 + v1 + v2) / 3.0
        return self._face_centroids_cache

    def _fit_plane(self,
                   points: np.ndarray,
                   face_normals: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """Fit a plane and orient it to align with the supplied face_normals if provided."""
        if points is None or len(points) < 3:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64), 0.0

        normal = None
        origin = None

        if normal is None or origin is None:
            centroid = np.mean(points, axis=0)
            centered = points - centroid
            cov = centered.T @ centered
            _, _, vh = np.linalg.svd(cov)
            normal = vh[-1]
            origin = centroid

        normal = normal / (np.linalg.norm(normal) + 1e-9)

        if face_normals is not None and len(face_normals) > 0:
            face_normals = np.asarray(face_normals, dtype=np.float64)
            mean_normal = np.mean(face_normals, axis=0)
            mean_norm = np.linalg.norm(mean_normal)
            if mean_norm > 1e-9:
                mean_normal /= mean_norm
                if np.dot(normal, mean_normal) < 0:
                    normal = -normal

        d = -np.dot(normal, origin)
        return normal, d

    def _get_face_neighbors_ring(self, face_idx: int, ring: int) -> Set[int]:
        """Get neighboring faces within a ring distance (cached)."""
        cache_key = (face_idx, ring)
        if cache_key in self._face_neighbors_cache:
            return self._face_neighbors_cache[cache_key]

        if ring <= 0:
            neighbors = {face_idx}
            self._face_neighbors_cache[cache_key] = neighbors
            return neighbors

        visited = {face_idx}
        frontier = {face_idx}
        for _ in range(ring):
            next_frontier = set()
            for f in frontier:
                for adj in self.face_adjacency.get(f, []):
                    if adj not in visited:
                        visited.add(adj)
                        next_frontier.add(adj)
            frontier = next_frontier
            if not frontier:
                break

        self._face_neighbors_cache[cache_key] = visited
        return visited

    def _compute_height_deviation_multiscale(self,
                                             face_indices: List[int],
                                             centroids: np.ndarray,
                                             plane_normal: np.ndarray,
                                             plane_d: float,
                                             ring_sizes: List[int]) -> Dict[int, List[float]]:
        """Compute multi-scale height deviation for faces."""
        signed_dist = centroids @ plane_normal + plane_d
        deviations = {}
        for face_idx in face_indices:
            devs = []
            for ring in ring_sizes:
                neighbors = self._get_face_neighbors_ring(face_idx, ring)
                if not neighbors:
                    neighbors = {face_idx}
                vals = [signed_dist[n] for n in neighbors]
                devs.append(float(np.mean(vals)))
            deviations[face_idx] = devs
        return deviations

    def _compute_mesh_curvature(self,
                                face_indices: List[int],
                                centroids: np.ndarray,
                                plane_normal: np.ndarray,
                                plane_d: float,
                                ratio_cache: Dict[int, Tuple[float, float, float, int]],
                                wound_r_thresh: float = 0.5) -> np.ndarray:
        """Compute scale-adaptive discrete mean curvature for each face.

        The neighbourhood ring size is derived from the estimated wound area so
        that the curvature captures the *global* concavity of an indentation
        rather than local surface noise.

        For every face the curvature value is the Laplacian-style difference
        between the face's signed height (distance to the fitted tissue plane)
        and the mean signed height of all neighbours within the adaptive ring.
        Negative values indicate a concave region (indentation / wound),
        positive values indicate a convex region (protrusion).

        Args:
            face_indices: List of face indices to compute curvature for.
            centroids: (M, 3) face centroids for the full mesh.
            plane_normal: Normal of the fitted tissue reference plane.
            plane_d: Offset d of the plane equation  n · x + d = 0.
            ratio_cache: {face_idx: (r_wound, r_periwound, r_tissue, total_views)}.
            wound_r_thresh: r_wound threshold used to estimate wound scale.

        Returns:
            curvature: (len(face_indices),) array of curvature values.
        """
        # --- Estimate wound scale as adaptive ring radius ---
        n_wound_faces = sum(
            1 for f in face_indices if ratio_cache.get(f, (0, 0, 0, 0))[0] > wound_r_thresh
        )
        # Heuristic: ring ~ sqrt(wound_faces) / 2, clamped to [2, 40]
        adaptive_ring = int(np.clip(np.sqrt(max(n_wound_faces, 1)) / 2.0, 2, 40))
        print(f"  Curvature adaptive ring = {adaptive_ring}  "
              f"(wound faces with r_wound > {wound_r_thresh}: {n_wound_faces})")

        # Signed height of every face relative to the tissue plane
        signed_height = centroids @ plane_normal + plane_d  # (M,)

        curvature = np.zeros(len(face_indices), dtype=np.float64)
        for i, face_idx in enumerate(face_indices):
            neighbors = self._get_face_neighbors_ring(face_idx, adaptive_ring)
            if not neighbors or len(neighbors) <= 1:
                curvature[i] = 0.0
                continue
            neighbor_heights = np.array([signed_height[n] for n in neighbors], dtype=np.float64)
            # Laplacian curvature: own height minus neighbourhood mean
            curvature[i] = signed_height[face_idx] - np.mean(neighbor_heights)

        return curvature

    def _compute_normal_deviation(self,
                                  face_normals: np.ndarray,
                                  plane_normal: np.ndarray) -> np.ndarray:
        """Compute normal deviation (angle in radians) from plane normal."""
        plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-9)
        dots = np.abs(face_normals @ plane_normal)
        dots = np.clip(dots, -1.0, 1.0)
        return np.arccos(dots)

    def _rgb_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB color [0-1] to LAB color space."""
        # Ensure RGB is in [0, 1]
        rgb_normalized = np.clip(rgb, 0, 1).astype(np.float64)
        
        # RGB to XYZ
        # Apply inverse sRGB companding
        mask = rgb_normalized > 0.04045
        linear_rgb = np.where(mask, np.power((rgb_normalized + 0.055) / 1.055, 2.4), rgb_normalized / 12.92)
        
        # XYZ transform matrix (D65 illuminant)
        transform_matrix = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        xyz = linear_rgb @ transform_matrix.T
        
        # Normalize by D65 white point
        xyz_normalized = xyz / np.array([0.95047, 1.00000, 1.08883])
        
        # XYZ to LAB
        delta = 6.0 / 29.0
        mask_xyz = xyz_normalized > (delta ** 3)
        f = np.where(mask_xyz, np.cbrt(xyz_normalized), xyz_normalized / (3 * delta ** 2) + 4.0 / 29.0)
        
        L = 116 * f[1] - 16
        a = 500 * (f[0] - f[1])
        b = 200 * (f[1] - f[2])
        
        return np.array([L, a, b], dtype=np.float32)
    
    def _rgb_to_ydrdb(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB color [0-1] to YDrDb color space.
        
        Returns: [Y, Dr, Db] array
        
        Transformation matrix:
        Y  =  0.299*R + 0.587*G + 0.114*B
        Db = -0.450*R - 0.883*G + 1.333*B
        Dr = -1.333*R + 1.116*G + 0.217*B
        """
        # Ensure RGB is in [0, 1]
        rgb_normalized = np.clip(rgb, 0, 1).astype(np.float64)
        
        # YDrDb transform matrix
        transform_matrix = np.array([
            [ 0.299,  0.587,  0.114],  # Y
            [-0.450, -0.883,  1.333],  # Db
            [-1.333,  1.116,  0.217]   # Dr
        ])
        
        ydrdb = rgb_normalized @ transform_matrix.T
        
        return np.array([ydrdb[0], ydrdb[1], ydrdb[2]], dtype=np.float32)

    def _build_feature_vectors(self,
                               periwound_candidate_wound_thresh: float = 0.1,
                               periwound_candidate_top_percent: float = 20.0,
                               ring_sizes: List[int] = None,
                               use_ratio_features: bool = True,
                               use_height_deviation: bool = True,
                               use_normal_deviation: bool = True,
                               use_curvature: bool = True,
                               use_centroid_features: bool = True,
                               color_space: str = "lab_b") -> Tuple[np.ndarray, List[int], Dict[int, Tuple[float, float, int]]]:
        """Build feature vectors for semantic classification.
        
        Args:
            periwound_candidate_wound_thresh: Maximum r_wound for periwound seed selection
            periwound_candidate_top_percent: Select periwound candidates from the top X% of
                the r_periwound distribution (e.g. 20.0 means the 80th-percentile threshold)
            min_total_views: Minimum views for a face to be included
            tissue_visibility_percentile: Percentile for selecting high-vis tissue seed faces
            ring_sizes: Ring sizes for multi-scale height deviation
            use_height_deviation: Include height deviation features
            use_normal_deviation: Include normal deviation features
            use_curvature: Include scale-adaptive discrete mean curvature feature
            color_space: Color space to use: "none", "rgb", "lab_b" (LAB b-channel only), or "ydrdb" (Dr and Db only)
        """
        if ring_sizes is None:
            ring_sizes = [1, 3]

        self._ensure_face_normals()
        centroids = self._get_face_centroids()

        face_indices = []
        ratio_cache = {}
        for face_idx in range(len(self.faces)):
            wound_visibility = self.face_visibility_count.get(face_idx, 0)
            periwound_visibility = self.periwound_edge_visibility_count.get(face_idx, 0)
            tissue_visibility = self.normal_tissue_visibility_count.get(face_idx, 0)
            total_views = len(self.cameras)  # Per-face denominator

            # Calculate ratios (coherent: r_wound + r_periwound <= 1.0)
            r_wound = wound_visibility / total_views if total_views > 0 else 0.0
            r_periwound = periwound_visibility / total_views if total_views > 0 else 0.0
            r_tissue = tissue_visibility / total_views if total_views > 0 else 0.0
            ratio_cache[face_idx] = (r_wound, r_periwound, r_tissue, total_views)

            face_indices.append(face_idx)

        # Derive periwound_candidate_periwound_thresh from the r_periwound distribution
        # so that the top `periwound_candidate_top_percent`% of faces pass the threshold
        r_periwound_values = np.array([ratio_cache[f][1] for f in face_indices], dtype=np.float64)
        periwound_candidate_periwound_thresh = min(float( np.percentile(r_periwound_values, 100.0 - periwound_candidate_top_percent)),0.05) if len(r_periwound_values) > 0 else 0.0
        print(
            f"  periwound_candidate_periwound_thresh: "
            f"{periwound_candidate_periwound_thresh:.4f}"
        )

        # Plot histogram of r_periwound values
        if len(r_periwound_values) > 0:
            try:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(8, 4))
                plt.hist(r_periwound_values, bins=50, color="steelblue", alpha=0.8, edgecolor="black")
                plt.axvline(
                    periwound_candidate_periwound_thresh,
                    color="crimson",
                    linestyle="--",
                    linewidth=2,
                    label=f"threshold={periwound_candidate_periwound_thresh:.3f}",
                )
                plt.title("Histogram of r_periwound values")
                plt.xlabel("r_periwound")
                plt.ylabel("Face count")
                plt.grid(alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.show()
            except ImportError:
                print("matplotlib not available; skipping r_periwound histogram plot.")

        # Select periwound seed faces for plane fit
        periwound_candidates = [
            f for f in face_indices
            if ratio_cache[f][1] > periwound_candidate_periwound_thresh and ratio_cache[f][0] < periwound_candidate_wound_thresh
        ]
        print(f"  periwound_candidates: {len(periwound_candidates)} / {len(face_indices)} faces")

        plane_points = centroids[periwound_candidates] if periwound_candidates else centroids[face_indices]
        plane_face_indices = periwound_candidates if periwound_candidates else face_indices
        plane_face_normals = self.face_normals[plane_face_indices] if len(plane_face_indices) > 0 else None
        plane_normal, plane_d = self._fit_plane(plane_points, face_normals=plane_face_normals)

        # Build feature list based on toggles
        if use_ratio_features:
            feature_names = ["r_wound", "r_periwound", "r_tissue"]
        else:
            feature_names = []
        features_list = []
        
        # Centroid position features
        if use_centroid_features:
            feature_names.extend(["centroid_x", "centroid_y", "centroid_z"])
        
        # Height deviation features
        if use_height_deviation:
            height_devs = self._compute_height_deviation_multiscale(
                face_indices, centroids, plane_normal, plane_d, ring_sizes
            )
            for ring in ring_sizes:
                feature_names.append(f"height_dev_r{ring}")
        
        # Normal deviation feature
        if use_normal_deviation:
            normal_dev = self._compute_normal_deviation(self.face_normals, plane_normal)
            feature_names.append("normal_dev")

        # Scale-adaptive curvature feature
        if use_curvature:
            curvature_vals = self._compute_mesh_curvature(
                face_indices, centroids, plane_normal, plane_d, ratio_cache
            )
            feature_names.append("curvature")

        # Color features based on selected color space
        if color_space != "none" and self.vertex_colors is not None:
            if color_space == "rgb":
                feature_names.extend(["R", "G", "B"])
            elif color_space == "lab_b":
                feature_names.append("lab_b")
            elif color_space == "ydrdb":
                feature_names.extend(["Dr", "Db"])
            else:
                print(f"Warning: Unknown color_space '{color_space}'. Using 'none'.")
                color_space = "none"

        # Build feature vectors
        features = []
        for i, face_idx in enumerate(face_indices):
            r_wound, r_periwound, r_tissue, total_views = ratio_cache[face_idx]
            
            if use_ratio_features:
                feat = [r_wound, r_periwound, r_tissue]
            else:
                feat = []

            
            if use_centroid_features:
                # Add centroid position features
                centroid = centroids[face_idx]
                feat.extend([float(centroid[0]), float(centroid[1]), float(centroid[2])])

            # Add height deviation features
            if use_height_deviation:
                feat.extend(height_devs[face_idx])
            
            # Add normal deviation feature
            if use_normal_deviation:
                feat.append(float(normal_dev[face_idx]))

            # Add curvature feature
            if use_curvature:
                feat.append(float(curvature_vals[i]))

            # Add color features based on selected color space
            if color_space != "none" and self.vertex_colors is not None:
                face = self.faces[face_idx]
                face_colors = self.vertex_colors[face]
                mean_color_rgb = np.mean(face_colors, axis=0)
                
                if color_space == "rgb":
                    feat.extend(mean_color_rgb.tolist())
                elif color_space == "lab_b":
                    lab_color = self._rgb_to_lab(mean_color_rgb)
                    feat.append(lab_color[2])  # b channel only
                elif color_space == "ydrdb":
                    ydrdb_color = self._rgb_to_ydrdb(mean_color_rgb)
                    feat.extend([ydrdb_color[2], ydrdb_color[1]])  # Dr, Db (not Y)
            
            features.append(feat)

        self.semantic_feature_names = feature_names
        self.semantic_face_indices = face_indices
        raw_features = np.array(features, dtype=np.float64)
        self.semantic_features_raw = raw_features.copy()

        # Standardize features so GMM treats all dimensions equally
        if raw_features.size > 0:
            self._feature_mean = np.mean(raw_features, axis=0)
            self._feature_std = np.std(raw_features, axis=0) + 1e-9
            self.semantic_features = (raw_features - self._feature_mean) / self._feature_std
        else:
            self._feature_mean = None
            self._feature_std = None
            self.semantic_features = raw_features
        
        # Store intermediate data for visualization
        self.semantic_fitted_plane = (plane_normal, plane_d)
        
        # Create feature colormaps for each computed feature using raw values
        self.semantic_feature_colormaps = {}
        if face_indices and raw_features.size > 0:
            for feat_idx, feat_name in enumerate(feature_names):
                colormap, _, _ = self._build_feature_colormap_assets(
                    face_indices, raw_features[:, feat_idx], feat_name, generate_artifacts=False
                )
                if colormap:
                    self.semantic_feature_colormaps[feat_name] = colormap
        
        return self.semantic_features, self.semantic_face_indices, ratio_cache

    def _fit_gaussian_models(self,
                             features: np.ndarray,
                             face_indices: List[int],
                             ratio_cache: Dict[int, Tuple[float, float, float, int]],
                             r_wound_thresh: float = 0.3,
                             r_periwound_thresh: float = 0.3,
                             r_tissue_thresh: float = 0.2,
                             regularization: float = 1e-3) -> Dict[str, Dict[str, np.ndarray]]:
        """Fit Gaussian models for 3 classes: tissue, periwound, and wound.
        
        Uses visibility ratios to assign class labels:
        - Wound class: faces with r_wound > r_wound_thresh
        - Periwound class: faces with r_periwound > r_periwound_thresh (area around wound)
        - Tissue class: faces with low visibility ratios (r_wound < r_wound_thresh AND r_periwound < r_periwound_thresh AND r_tissue > r_tissue_thresh)
        
        The periwound class represents the boundary tissue around the wound,
        detected from the difference between extruded and regular wound masks.
        """
        class_samples = {"tissue": [], "periwound": [], "wound": []}
        class_sample_indices = {"tissue": [], "periwound": [], "wound": []}
        idx_map = {face_idx: i for i, face_idx in enumerate(face_indices)}

        # Classify faces based on visibility ratios
        for face_idx in face_indices:
            feat_idx = idx_map[face_idx]
            r_wound, r_periwound, r_tissue, _ = ratio_cache[face_idx]
            
            is_wound = r_wound > r_wound_thresh
            is_periwound = r_periwound > r_periwound_thresh
            is_tissue = r_tissue > r_tissue_thresh
            
            # Priority: wound > periwound > tissue
            if is_wound:
                class_samples["wound"].append(features[feat_idx])
                class_sample_indices["wound"].append(face_idx)
            if is_periwound:
                class_samples["periwound"].append(features[feat_idx])
                class_sample_indices["periwound"].append(face_idx)
            if is_tissue:
                class_samples["tissue"].append(features[feat_idx])
                class_sample_indices["tissue"].append(face_idx)
            # Faces not meeting any threshold are excluded from training

        # Store class samples for visualization
        self.semantic_class_samples = class_sample_indices
        
        print(f"  Class samples: tissue={len(class_sample_indices['tissue'])}, "
              f"periwound={len(class_sample_indices['periwound'])}, "
              f"wound={len(class_sample_indices['wound'])}")
        
        models = {}

        # Prefer sklearn GaussianMixture for stability
        try:
            from sklearn.mixture import GaussianMixture
            for cls_name, samples in class_samples.items():
                if not samples:
                    print(f"Warning: No samples for class {cls_name}")
                    continue
                X = np.vstack(samples)
                # Use 2 components to capture multi-modal distributions;
                # fall back to 1 if insufficient samples.
                n_comp = min(1, max(1, len(X) // (X.shape[1] + 1)))
                gmm = GaussianMixture(
                    n_components=n_comp,
                    covariance_type="full",
                    reg_covar=regularization,
                    max_iter=300,
                    init_params="kmeans"
                )
                gmm.fit(X)
                models[cls_name] = {
                    "gmm": gmm,
                    "n_samples": len(X)
                }
            self.semantic_gmm_models = models
            return models
        except Exception as e:
            print(f"sklearn GaussianMixture unavailable, falling back to manual MVN: {e}")

        return models

    def _gaussian_logpdf(self, x: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray, log_det: float) -> float:
        """Compute log PDF of multivariate Gaussian."""
        diff = x - mean
        quad = diff @ inv_cov @ diff
        d = len(x)
        return -0.5 * (quad + log_det + d * np.log(2.0 * np.pi))

    def _compute_posteriors_from_models(self,
                                        features: np.ndarray,
                                        models: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """Compute posterior probabilities for each face.
        
        Includes class priors proportional to training sample counts.
        """
        n = features.shape[0]
        class_names = self.semantic_class_names
        log_likelihoods = np.full((n, len(class_names)), -1e12, dtype=np.float64)

        # Compute class priors from training sample counts
        total_training = sum(m.get("n_samples", 0) for m in models.values())
        log_priors = np.zeros(len(class_names), dtype=np.float64)
        for k, cls_name in enumerate(class_names):
            if cls_name in models and total_training > 0:
                n_cls = models[cls_name].get("n_samples", 0)
                log_priors[k] = np.log(max(n_cls / total_training, 1e-12))

        for k, cls_name in enumerate(class_names):
            if cls_name not in models:
                continue
            model = models[cls_name]
            if "gmm" in model:
                log_likelihoods[:, k] = model["gmm"].score_samples(features) + log_priors[k]
                continue

            mean = model["mean"]
            inv_cov = model["inv_cov"]
            log_det = model["log_det"]
            # Vectorized logpdf
            diff = features - mean
            quad = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
            d = features.shape[1]
            log_likelihoods[:, k] = -0.5 * (quad + log_det + d * np.log(2.0 * np.pi)) + log_priors[k]

        # Store raw log-likelihoods for visualization
        self.semantic_log_likelihoods = log_likelihoods.copy()

        # Softmax to posterior
        max_ll = np.max(log_likelihoods, axis=1, keepdims=True)
        exp_ll = np.exp(log_likelihoods - max_ll)
        post = exp_ll / (np.sum(exp_ll, axis=1, keepdims=True) + 1e-12)
        return post

    def _apply_mrf_constraints(self,
                               posteriors: np.ndarray,
                               face_indices: List[int],
                               lambda_smooth: float = 1.0,
                               max_iters: int = 5,
                               method: str = "graph_cut") -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply MRF smoothness constraints for N-class semantic classification.
        
        Args:
            posteriors: (N, K) array of class posteriors (K = number of classes)
            face_indices: List of face indices corresponding to each posterior row
            lambda_smooth: Smoothness weight for pairwise cost
            max_iters: Maximum iterations for iterative methods
            method: "graph_cut" for alpha-expansion, "mean_field" for mean field approximation
        
        Returns:
            labels: (N,) array of hard labels (0=tissue, 1=periwound, 2=wound for 3-class)
            mrf_posteriors: (N, K) array of refined posteriors (only for mean_field, None for graph_cut)
        """
        labels = np.argmax(posteriors, axis=1).astype(np.int32)
        face_to_idx = {face_idx: i for i, face_idx in enumerate(face_indices)}
        n_classes = posteriors.shape[1]  # Should be 2
        n_faces = len(face_indices)
        
        # ===================== MEAN FIELD APPROXIMATION =====================
        if method == "mean_field":
            print(f"Applying mean field approximation (lambda={lambda_smooth}, max_iters={max_iters})...")
            
            # Initialize beliefs with unary posteriors
            q = posteriors.copy()  # (N, 2) belief distributions
            
            # Build neighbor list for efficient iteration
            neighbor_lists = []
            for f in face_indices:
                neighbors = []
                for adj in self.face_adjacency.get(f, []):
                    if adj in face_to_idx:
                        neighbors.append(face_to_idx[adj])
                neighbor_lists.append(neighbors)
            
            # Mean field iterations
            for iteration in range(max_iters):
                q_old = q.copy()
                
                # Update each face's belief
                for i, face_idx in enumerate(face_indices):
                    # Unary potential: log P(class | features)
                    unary_log = np.log(posteriors[i] + 1e-12)
                    
                    # Pairwise message from neighbors
                    # For each class c, sum over neighbors: lambda * E_q[1(neighbor != c)]
                    pairwise_msg = np.zeros(n_classes)
                    for j in neighbor_lists[i]:
                        # Expected disagreement cost for each class assignment
                        # If we assign class c to face i, cost = lambda * sum_k P(neighbor=k) * 1(k != c)
                        for c in range(n_classes):
                            pairwise_msg[c] -= lambda_smooth * (1.0 - q_old[j, c])
                    
                    # Combine unary and pairwise
                    log_belief = unary_log + pairwise_msg
                    
                    # Softmax to get updated belief
                    log_belief -= np.max(log_belief)  # Numerical stability
                    q[i] = np.exp(log_belief)
                    q[i] /= np.sum(q[i]) + 1e-12
                
                # Check convergence
                max_change = np.max(np.abs(q - q_old))
                if max_change < 1e-4:
                    print(f"  Mean field converged at iteration {iteration + 1} (max_change={max_change:.6f})")
                    break
            else:
                print(f"  Mean field reached max_iters={max_iters} (max_change={max_change:.6f})")
            
            # Extract hard labels from beliefs
            labels = np.argmax(q, axis=1).astype(np.int32)
            
            # Store mean field posteriors for visualization
            self.semantic_mrf_posteriors = {}
            for i, face_idx in enumerate(face_indices):
                self.semantic_mrf_posteriors[face_idx] = q[i].copy()
            
            return labels, q
        
        # ===================== GRAPH CUT (ALPHA-EXPANSION) =====================
        # Attempt graph-cut if available
        try:
            import maxflow
            if hasattr(maxflow, "fastmin") and hasattr(maxflow.fastmin, "aexpansion"):
                unary = -np.log(posteriors + 1e-12)
                
                # Symmetric pairwise penalty for label disagreement
                # lambda_smooth is baked into the pairwise matrix; do NOT pass it again
                pairwise = lambda_smooth * np.ones((n_classes, n_classes), dtype=np.float64)
                np.fill_diagonal(pairwise, 0.0)  # No penalty for same labels

                edges = []
                for f in face_indices:
                    i = face_to_idx[f]
                    for adj in self.face_adjacency.get(f, []):
                        if adj not in face_to_idx:
                            continue
                        j = face_to_idx[adj]
                        if i < j:
                            edges.append([i, j])

                edges = np.array(edges, dtype=np.int32)
                if len(edges) > 0:
                    labels = maxflow.fastmin.aexpansion(unary, pairwise, edges)
                
                self.semantic_mrf_posteriors = {}  # No refined posteriors for graph cut
                return labels, None
        except Exception as e:
            print(f"MRF graph-cut unavailable, using greedy smoothness: {e}")

        # Greedy smoothness enforcement fallback (ICM-style)
        # Combines unary cost (negative log posterior) with pairwise smoothness
        for _ in range(max_iters):
            changed = False
            for f in face_indices:
                i = face_to_idx[f]
                # Compute total energy for each possible label
                best_label = labels[i]
                best_energy = float('inf')
                for candidate in range(n_classes):
                    # Unary cost: -log P(candidate | features)
                    unary_cost = -np.log(posteriors[i, candidate] + 1e-12)
                    # Pairwise cost: lambda_smooth * number of disagreeing neighbors
                    pairwise_cost = 0.0
                    for adj in self.face_adjacency.get(f, []):
                        if adj not in face_to_idx:
                            continue
                        j = face_to_idx[adj]
                        if labels[j] != candidate:
                            pairwise_cost += lambda_smooth
                    total_cost = unary_cost + pairwise_cost
                    if total_cost < best_energy:
                        best_energy = total_cost
                        best_label = candidate
                if best_label != labels[i]:
                    labels[i] = best_label
                    changed = True
            if not changed:
                break

        self.semantic_mrf_posteriors = {}  # No refined posteriors for ICM
        return labels, None

    def _compute_confidence_from_posteriors(self,
                                            posteriors: np.ndarray,
                                            method: str = "entropy") -> np.ndarray:
        """Compute confidence from posteriors."""
        if method == "max":
            return np.max(posteriors, axis=1)
        # Entropy-based confidence
        entropy = -np.sum(posteriors * np.log(posteriors + 1e-12), axis=1)
        entropy /= np.log(posteriors.shape[1])
        return 1.0 - entropy

    def run_semantic_classification(self,
                                    tissue_ratio_thresh: float = 0.05,
                                    wound_ratio_thresh: float = 0.6,
                                    ring_sizes: List[int] = None,
                                    confidence_method: str = "entropy",
                                    lambda_smooth: float = 1.0,
                                    mrf_method: str = "graph_cut",
                                    mrf_max_iters: int = 10,
                                    use_centroid_features: bool = True,
                                    use_height_deviation: bool = True,
                                    use_normal_deviation: bool = True,
                                    use_curvature: bool = True,
                                    color_space: str = "lab_b",
                                    undistorted_mask_dir: Path = None) -> None:
        """Run feature-based semantic classification for 3-class (tissue, periwound, wound) with feature toggles.
        
        Args:
            tissue_ratio_thresh: Threshold for tissue candidate identification
            wound_ratio_thresh: Threshold for wound class assignment in GMM training
            ring_sizes: Ring sizes for multi-scale features
            confidence_method: "entropy" or "max" for confidence computation
            lambda_smooth: Regularization weight for MRF smoothness
            mrf_method: "graph_cut" for alpha-expansion, "mean_field" for mean field approximation
            mrf_max_iters: Maximum iterations for MRF methods (mean field or ICM fallback)
            use_height_deviation: Include height deviation features
            use_normal_deviation: Include normal deviation features
            use_curvature: Include scale-adaptive discrete mean curvature feature
            color_space: Color space to use: "none", "rgb", "lab_b" (LAB b-channel only), or "ydrdb" (Dr and Db only)
            undistorted_mask_dir: Optional directory with undistorted masks for tissue detection
        """
        if not self.faces_processed:
            raise ValueError("Must process wound segmentation first")
        if not self.topology_computed:
            self._compute_mesh_topology()
        
        # Detect initial periwound faces from mask difference
        self.process_periwound_face_detection(max_mask_rays=200000)
        
        # Detect normal tissue faces (optional, requires undistorted masks)
        if undistorted_mask_dir is not None:
            self.process_tissue_face_detection(undistorted_mask_dir=undistorted_mask_dir, 
                                               max_mask_rays=200000)

        features, face_indices, ratio_cache = self._build_feature_vectors(
            ring_sizes=ring_sizes or [1, 3],
            use_ratio_features=False,
            use_centroid_features=use_centroid_features,
            use_height_deviation= use_height_deviation,
            use_normal_deviation=use_normal_deviation,
            use_curvature=use_curvature,
            color_space=color_space
        )

        if features is None or len(features) == 0:
            print("No features available for semantic classification")
            return

        models = self._fit_gaussian_models(
            features=features,
            face_indices=face_indices,
            ratio_cache=ratio_cache,
            r_wound_thresh=0.3,
            r_periwound_thresh=0.3,
        )

        if not models:
            print("No Gaussian models could be fitted (insufficient seed data)")
            return

        posteriors = self._compute_posteriors_from_models(features, models)
        
        # Store per-face raw log-likelihoods for visualization (3-class: tissue, periwound, wound)
        self.semantic_tissue_loglik = {}
        self.semantic_periwound_loglik = {}
        self.semantic_wound_loglik = {}
        if self.semantic_log_likelihoods is not None:
            for i, face_idx in enumerate(face_indices):
                self.semantic_tissue_loglik[face_idx] = float(self.semantic_log_likelihoods[i, 0])
                if self.semantic_log_likelihoods.shape[1] > 2:
                    self.semantic_periwound_loglik[face_idx] = float(self.semantic_log_likelihoods[i, 1])
                    self.semantic_wound_loglik[face_idx] = float(self.semantic_log_likelihoods[i, 2])
                else:
                    self.semantic_wound_loglik[face_idx] = float(self.semantic_log_likelihoods[i, 1])
        
        # Store posteriors before MRF for visualization
        self.semantic_posteriors_before_mrf = {}
        for i, face_idx in enumerate(face_indices):
            self.semantic_posteriors_before_mrf[face_idx] = posteriors[i].copy()
        
        # Apply MRF constraints
        self.semantic_mrf_method = mrf_method
        labels, mrf_posteriors = self._apply_mrf_constraints(
            posteriors, face_indices, 
            lambda_smooth=lambda_smooth, 
            max_iters=mrf_max_iters,
            method=mrf_method
        )
        
        # Use MRF posteriors for confidence if available (mean field), otherwise use GMM posteriors
        if mrf_posteriors is not None:
            confidences = self._compute_confidence_from_posteriors(mrf_posteriors, method=confidence_method)
            final_posteriors = mrf_posteriors
        else:
            confidences = self._compute_confidence_from_posteriors(posteriors, method=confidence_method)
            final_posteriors = posteriors

        # Store per-face results (3-class: tissue=0, periwound=1, wound=2)
        self.semantic_posteriors = {}
        self.semantic_labels = {}
        self.semantic_confidence = {}
        self.semantic_tissue_prob = {}
        self.semantic_periwound_prob = {}
        self.semantic_wound_prob = {}

        n_classes = final_posteriors.shape[1] if len(final_posteriors.shape) > 1 else 2
        for i, face_idx in enumerate(face_indices):
            probs = final_posteriors[i]
            label_name = self.semantic_class_names[int(labels[i])]
            self.semantic_posteriors[face_idx] = probs
            self.semantic_labels[face_idx] = label_name
            self.semantic_confidence[face_idx] = float(confidences[i])
            self.semantic_tissue_prob[face_idx] = float(probs[0])
            if n_classes > 2:
                self.semantic_periwound_prob[face_idx] = float(probs[1])
                self.semantic_wound_prob[face_idx] = float(probs[2])
            else:
                self.semantic_periwound_prob[face_idx] = 0.0
                self.semantic_wound_prob[face_idx] = float(probs[1])
        
        # Compute wound_edge as boundary between wound and tissue
        self._compute_wound_edge_boundary()

        print("Semantic classification complete")
        print(f"  Faces classified: {len(self.semantic_labels)}")
        print(f"  Confidence (mean): {np.mean(list(self.semantic_confidence.values())):.3f}")
    
    def _compute_wound_edge_boundary(self):
        """Compute wound_edge as the boundary region between classified wound and periwound/tissue faces."""
        self.wound_edge_face_indices = set()
        self.wound_edge_vertices = set()
        self.wound_edge_edges = []
        
        # Get classified faces
        wound_faces = {f for f, lbl in self.semantic_labels.items() if lbl == "wound"}
        periwound_faces = {f for f, lbl in self.semantic_labels.items() if lbl == "periwound"}
        tissue_faces = {f for f, lbl in self.semantic_labels.items() if lbl == "tissue"}
        non_wound_faces = periwound_faces | tissue_faces
        
        # Find faces adjacent to both wound and non-wound (boundary faces)
        for face_idx in wound_faces:
            for adj_face in self.face_adjacency.get(face_idx, []):
                if adj_face in non_wound_faces:
                    # This wound face is adjacent to periwound/tissue - it's on the boundary
                    self.wound_edge_face_indices.add(face_idx)
                    break
        
        # Also add non-wound faces on the boundary
        for face_idx in non_wound_faces:
            for adj_face in self.face_adjacency.get(face_idx, []):
                if adj_face in wound_faces:
                    # This periwound/tissue face is adjacent to wound - it's on the boundary
                    self.wound_edge_face_indices.add(face_idx)
                    break
        
        # Extract wound_edge vertices from boundary faces
        for face_idx in self.wound_edge_face_indices:
            self.wound_edge_vertices.update(self.faces[face_idx])
        
        # Extract wound_edge edges (edges between wound and non-wound faces)
        for edge, face_list in self.edge_adjacency.items():
            if len(face_list) == 2:
                f0, f1 = face_list
                f0_label = self.semantic_labels.get(f0)
                f1_label = self.semantic_labels.get(f1)
                # If one is wound and other is not wound
                if (f0_label == "wound" and f1_label in ("periwound", "tissue")) or \
                   (f0_label in ("periwound", "tissue") and f1_label == "wound"):
                    v0, v1 = edge
                    self.wound_edge_edges.append((v0, v1))
        
        print(f"Wound edge boundary faces: {len(self.wound_edge_face_indices)}")
        print(f"Wound edge vertices: {len(self.wound_edge_vertices)}")
        print(f"Wound edge edges: {len(self.wound_edge_edges)}")

    
    def plot_visibility_histograms(self, output_dir: Path = None, show_plot: bool = True) -> None:
        """
        Plot histograms of visibility counts for wound and periwound faces.
        
        Creates separate histograms showing:
        - Wound face visibility distribution
        - periwound face visibility distribution
        - Overlapping faces highlighted
        
        Args:
            output_dir: Directory to save plots (if None, only shows)
            show_plot: Whether to display the plot interactively
        """
        if not self.faces_processed:
            raise ValueError("Must process wound segmentation first")
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return
        
        print("\n=== Plotting Visibility Histograms ===")
        
        # Collect visibility counts
        wound_counts = np.array(list(self.face_visibility_count.values()))
        tissue_counts = np.array(list(self.periwound_edge_visibility_count.values()))
        
        # Create figure with two subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Visibility Count Distributions', fontsize=16, fontweight='bold')
        
        # Plot 1: Wound face visibility histogram
        ax1 = axes[0, 0]
        if len(wound_counts) > 0:
            ax1.hist(wound_counts, bins=min(50, int(np.max(wound_counts))), 
                    color='red', alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Visibility Count (# of cameras)', fontsize=12)
            ax1.set_ylabel('Number of Faces', fontsize=12)
            ax1.set_title(f'Wound Face Visibility (n={len(wound_counts)})', fontsize=14)
            ax1.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f"Mean: {np.mean(wound_counts):.2f}\nMedian: {np.median(wound_counts):.0f}\nMax: {np.max(wound_counts):.0f}"
            ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)
        else:
            ax1.text(0.5, 0.5, 'No wound faces', ha='center', va='center')
            ax1.set_title('Wound Face Visibility', fontsize=14)
        
        # Plot 2: periwound visibility histogram
        ax2 = axes[0, 1]
        if len(tissue_counts) > 0:
            ax2.hist(tissue_counts, bins=min(50, int(np.max(tissue_counts))),
                    color='blue', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Visibility Count (# of cameras)', fontsize=12)
            ax2.set_ylabel('Number of Faces', fontsize=12)
            ax2.set_title(f'periwound Visibility (n={len(tissue_counts)})', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f"Mean: {np.mean(tissue_counts):.2f}\nMedian: {np.median(tissue_counts):.0f}\nMax: {np.max(tissue_counts):.0f}"
            ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                    fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'No periwound faces', ha='center', va='center')
            ax2.set_title('periwound Visibility', fontsize=14)
        
        # Plot 3: Overlapping faces comparison
        ax3 = axes[1, 0]
        if len(self.overlapping_faces) > 0:
            overlap_wound = [self.overlapping_wound_counts[f] for f in self.overlapping_faces]
            overlap_tissue = [self.overlapping_tissue_counts[f] for f in self.overlapping_faces]
            
            ax3.scatter(overlap_wound, overlap_tissue, alpha=0.6, s=30, color='purple')
            ax3.set_xlabel('Wound Visibility Count', fontsize=12)
            ax3.set_ylabel('periwound Visibility Count', fontsize=12)
            ax3.set_title(f'Overlapping Faces (n={len(self.overlapping_faces)})', fontsize=14)
            ax3.grid(True, alpha=0.3)
            
            # Add diagonal line
            max_val = max(max(overlap_wound) if overlap_wound else 1, 
                         max(overlap_tissue) if overlap_tissue else 1)
            ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal visibility')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No overlapping faces', ha='center', va='center')
            ax3.set_title('Overlapping Faces', fontsize=14)
        
        # Plot 4: Combined histogram with overlap
        ax4 = axes[1, 1]
        
        # Separate overlapping and non-overlapping
        non_overlap_wound = [self.face_visibility_count[f] for f in self.face_visibility_count.keys() 
                            if f not in self.overlapping_faces]
        non_overlap_tissue = [self.periwound_edge_visibility_count[f] for f in self.periwound_edge_visibility_count.keys() 
                             if f not in self.overlapping_faces]
        
        max_count = max(
            max(wound_counts) if len(wound_counts) > 0 else 1,
            max(tissue_counts) if len(tissue_counts) > 0 else 1
        )
        bins = np.linspace(0, max_count, min(50, int(max_count) + 1))
        
        if len(non_overlap_wound) > 0:
            ax4.hist(non_overlap_wound, bins=bins, color='red', alpha=0.5, 
                    label=f'Wound only (n={len(non_overlap_wound)})', edgecolor='darkred')
        if len(non_overlap_tissue) > 0:
            ax4.hist(non_overlap_tissue, bins=bins, color='blue', alpha=0.5,
                    label=f'periwound only (n={len(non_overlap_tissue)})', edgecolor='darkblue')
        
        ax4.set_xlabel('Visibility Count (# of cameras)', fontsize=12)
        ax4.set_ylabel('Number of Faces', fontsize=12)
        ax4.set_title('Non-Overlapping Face Distribution', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if output directory provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "visibility_histograms.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved histogram to: {output_path}")
        
        # Show plot
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def _filter_faces_with_mask(self, spectator: Spectator, mask: np.ndarray, max_mask_rays: int) -> List[int]:
        """Project a custom mask to mesh faces using raycasting.
        
        Args:
            spectator: Camera/spectator with intrinsics and extrinsics
            mask: Binary mask to project
            max_mask_rays: Maximum rays to cast
            
        Returns:
            List of face indices
        """
        try:
            import open3d as o3d
        except ImportError:
            print("Open3D not available for periwound raycasting")
            return []
        
        if self.vertices is None or self.faces is None:
            return []
        
        # Use the existing raycast scene if available
        if self._raycast_scene is None:
            mesh_legacy = o3d.geometry.TriangleMesh()
            mesh_legacy.vertices = o3d.utility.Vector3dVector(self.vertices)
            mesh_legacy.triangles = o3d.utility.Vector3iVector(self.faces)
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
            self._raycast_scene = o3d.t.geometry.RaycastingScene()
            self._raycast_mesh_id = self._raycast_scene.add_triangles(mesh_t)
        
        mask_coords = np.argwhere(mask > 0)
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
        
        prim_ids = hits['primitive_ids'].numpy()
        max_invalid = np.iinfo(prim_ids.dtype).max
        prim_ids = prim_ids[(prim_ids != max_invalid) & (prim_ids >= 0) & (prim_ids < len(self.faces))]
        
        if prim_ids.size == 0:
            return []
        
        return np.unique(prim_ids).tolist()
    
    def _compute_periwound_edge_face_stats(self, face_idx: int):
        """Compute per-face color statistics for periwound faces across all cameras.
        
        Args:
            face_idx: Face index in self.faces
        """
        if face_idx not in self.periwound_edge_faces:
            return
        
        # Get vertex colors for this face
        face = self.faces[face_idx]
        if self.vertex_colors is None:
            # Default colors if not available
            self.periwound_edge_mean_color[face_idx] = np.array([0.5, 0.5, 0.5])
            self.periwound_edge_color_std[face_idx] = np.array([0.0, 0.0, 0.0])
            return
        
        # Collect vertex colors
        vertex_colors = self.vertex_colors[face]  # (3, 3) array
        mean_color = np.mean(vertex_colors, axis=0)
        std_color = np.std(vertex_colors, axis=0)
        
        self.periwound_edge_mean_color[face_idx] = mean_color
        self.periwound_edge_color_std[face_idx] = std_color
    
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
        
        print(f"Visibility filtering: {len(high_visibility_faces)}/{len(self.wound_face_indices)} faces "
              f"seen by >= {min_visibility_count} cameras")
        
        return high_visibility_faces
    
    def filter_periwound_edge_by_visibility(self, min_visibility_count: int = 2) -> Set[int]:
        """Filter periwound faces by minimum visibility count.
        
        Args:
            min_visibility_count: Minimum number of cameras that must see a periwound face
            
        Returns:
            Set of periwound face indices meeting visibility threshold
        """
        if not self.periwound_edge_faces:
            return set()
        
        high_visibility_periwound_edge = {
            face_idx for face_idx, count in self.periwound_edge_visibility_count.items()
            if count >= min_visibility_count
        }
        
        print(f"periwound visibility filtering: {len(high_visibility_periwound_edge)}/{len(self.periwound_edge_faces)} faces "
              f"seen by >= {min_visibility_count} cameras")
        
        return high_visibility_periwound_edge
    
    def extract_connected_components(self, face_set: Set[int], min_faces: int = 10) -> List[Set[int]]:
        """
        Extract connected components from a set of faces.

        Uses Open3D's ``cluster_connected_triangles()`` on a sub-mesh built from
        *face_set*, which is faster than the previous hand-rolled BFS and delegates
        all graph logic to the library.

        Args:
            face_set: Set of face indices (indices into self.faces)
            min_faces: Minimum number of faces for a valid component

        Returns:
            List of connected components (each is a set of **global** face indices)
        """
        import open3d as o3d

        if not face_set:
            return []

        # --- Build a compact sub-mesh from the requested face subset ---
        face_list = np.array(sorted(face_set), dtype=np.int64)
        sub_faces_global = self.faces[face_list]          # (K, 3)

        unique_verts, inv = np.unique(sub_faces_global, return_inverse=True)
        sub_faces_local = inv.reshape(-1, 3).astype(np.int32)
        sub_verts = self.vertices[unique_verts].astype(np.float64)

        sub_mesh = o3d.geometry.TriangleMesh()
        sub_mesh.vertices = o3d.utility.Vector3dVector(sub_verts)
        sub_mesh.triangles = o3d.utility.Vector3iVector(sub_faces_local)

        # --- Use O3D to label triangles by connected component ---
        labels, cluster_n_triangles, _ = sub_mesh.cluster_connected_triangles()
        labels = np.asarray(labels)  # shape (K,)

        components = []
        for label in np.unique(labels):
            local_idxs = np.where(labels == label)[0]
            if len(local_idxs) < min_faces:
                continue
            global_faces = set(face_list[local_idxs].tolist())
            components.append(global_faces)

        print(f"Extracted {len(components)} connected components from {len(face_set)} faces")
        return components


    
            
         

    def _get_boundary_faces(self) -> Set[int]:
        """Get faces on the mesh boundary (edges with only one adjacent face)."""
        boundary_faces = set()
        for edge, face_list in self.edge_adjacency.items():
            if len(face_list) == 1:
                boundary_faces.add(face_list[0])
        return boundary_faces
    
    def extract_submesh_from_faces(self, face_indices: Set[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract a standalone submesh from a set of face indices.
        
        Creates a self-contained mesh by:
        1. Collecting all unique vertices used by the faces
        2. Creating a compact vertex array (only used vertices)
        3. Remapping face indices to reference the new vertex array
        4. Extracting corresponding vertex colors
        
        Args:
            face_indices: Set of face indices (referencing self.faces)
            
        Returns:
            Tuple of (vertices, faces, colors) for the extracted submesh
            - vertices: (N, 3) array of vertex positions
            - faces: (M, 3) array of face indices (0-indexed into returned vertices)
            - colors: (N, 3) array of vertex colors (or None if not available)
        """
        if not face_indices:
            return np.array([]), np.array([]), None
        
        # Collect all unique vertex indices used by these faces
        face_list = list(face_indices)
        vertex_set = set()
        
        for face_idx in face_list:
            face = self.faces[face_idx]
            vertex_set.update(face)
        
        # Create sorted vertex list and remap dictionary
        vertex_list = sorted(list(vertex_set))
        vertex_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(vertex_list)}
        
        # Extract vertex positions
        submesh_vertices = self.vertices[vertex_list]
        
        # Extract vertex colors if available
        submesh_colors = self.vertex_colors[vertex_list] if self.vertex_colors is not None else None
        
        # Remap faces to new vertex indices
        submesh_faces = np.array([[vertex_remap[v] for v in self.faces[fi]] for fi in face_list])
        
        return submesh_vertices, submesh_faces, submesh_colors

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
        self.removed_faces_visibility = self.wound_face_indices - filtered_faces
        
        return filtered_faces
    
    def morphological_closing(self, face_set: Set[int], iterations: int = 1, kernel_size: int = 1,
                              use_probability_validation: bool = True,
                              wound_prob_threshold: float = 0.3) -> Set[int]:
        """
        Perform morphological closing on wound faces (dilation followed by erosion).
        
        Similar to 2D closing, this operation fills small holes and gaps in the wound region
        by first expanding (dilating) the wound faces to include adjacent non-wound faces,
        then contracting (eroding) back to remove temporary additions.
        
        Args:
            face_set: Set of face indices to perform closing on
            iterations: Number of closing iterations
            kernel_size: Number of adjacency layers to expand in each dilation step (1=immediate neighbors, 2=2 layers, etc.)
            use_probability_validation: If True, validate added faces using wound probability map
            wound_prob_threshold: Minimum wound probability for a face to be added during closing
            
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
        
        def _validate_face_probability(face_idx: int, threshold: float) -> bool:
            """Check if a face passes the wound probability threshold."""
            if not use_probability_validation:
                return True
            # Check semantic_wound_prob first
            if self.semantic_wound_prob and face_idx in self.semantic_wound_prob:
                return self.semantic_wound_prob[face_idx] >= threshold
            # Check semantic_mrf_posteriors (mean field) if available
            if hasattr(self, 'semantic_mrf_posteriors') and self.semantic_mrf_posteriors and face_idx in self.semantic_mrf_posteriors:
                return self.semantic_mrf_posteriors[face_idx][1] >= threshold  # Index 1 is wound prob
            # If no probability info available, allow the face
            return True

        current_faces = face_set.copy()

        # Pre-erosion to drop tiny isolated structures before closing.
        current_faces = _erode_faces(current_faces, drop_isolated=True)
        
        validated_additions = 0
        rejected_additions = 0
        
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
                            # Validate face probability before adding
                            if _validate_face_probability(adj_face, wound_prob_threshold):
                                next_front.add(adj_face)
                                validated_additions += 1
                            else:
                                rejected_additions += 1
                dilated.update(next_front)
                expansion_front = next_front
                if not expansion_front:  # No more faces to expand
                    break
            
            # Erosion: remove faces that don't have enough wound neighbors
            eroded = _erode_faces(dilated, drop_isolated=False)
            
            current_faces = eroded
        
        added_faces = current_faces - face_set
        if added_faces or rejected_additions > 0:
            print(f"Morphological closing: added {len(added_faces)} faces")
            if use_probability_validation:
                print(f"  Probability validation: {validated_additions} validated, {rejected_additions} rejected (threshold={wound_prob_threshold:.2f})")
        
        return current_faces
    
    def expand_and_extract_components(self, face_set: Set[int],
                                     use_connectivity: bool = True,
                                     min_component_faces: int = 10,
                                     use_probability_validation: bool = True,
                                     wound_prob_threshold: float = 0.3) -> Set[int]:
        """
        Expand wound to include enclosed faces and extract connected components.
        
        This solves the problem where faces inside the wound region are not marked
        as wound faces even though they are completely enclosed by wound faces.
        Uses flood-fill from mesh boundary to identify unreachable (enclosed) faces.
        
        Args:
            face_set: Set of face indices to process
            use_connectivity: Whether to extract connected components
            min_component_faces: Minimum faces for valid component
            use_probability_validation: If True, validate enclosed faces using wound probability
            wound_prob_threshold: Minimum wound probability for enclosed faces to be added
            
        Returns:
            Set of final face indices including enclosed faces
        """
        if not face_set:
            return face_set
        
        def _validate_face_probability(face_idx: int, threshold: float) -> bool:
            """Check if a face passes the wound probability threshold."""
            if not use_probability_validation:
                return True
            # Check semantic_wound_prob first
            if self.semantic_wound_prob and face_idx in self.semantic_wound_prob:
                return self.semantic_wound_prob[face_idx] >= threshold
            # Check semantic_mrf_posteriors (mean field) if available
            if hasattr(self, 'semantic_mrf_posteriors') and self.semantic_mrf_posteriors and face_idx in self.semantic_mrf_posteriors:
                return self.semantic_mrf_posteriors[face_idx][1] >= threshold  # Index 1 is wound prob
            # If no probability info available, allow the face (enclosed in wound region)
            return True
        
        # Step 1: Expand wound to include enclosed non-wound faces
        if use_connectivity:
            pre_expand = set(face_set)
            
            # Get mesh boundary faces (edges with only one adjacent face)
            boundary_faces = self._get_boundary_faces()
            
            # All non-wound faces that could potentially be enclosed
            all_faces_set = set(range(len(self.faces)))
            non_wound_faces = all_faces_set - face_set
            
            # Flood-fill from boundary through non-wound faces
            # Faces reachable from boundary are "outside", others are "enclosed"
            queue = deque([f for f in boundary_faces if f in non_wound_faces])
            outside = set()
            
            while queue:
                face_idx = queue.popleft()
                if face_idx in outside:
                    continue
                outside.add(face_idx)
                
                # Explore adjacent non-wound faces
                for adj_face in self.face_adjacency.get(face_idx, []):
                    if adj_face in non_wound_faces and adj_face not in outside:
                        queue.append(adj_face)
            
            # Enclosed faces are non-wound faces that couldn't be reached
            enclosed_faces_raw = non_wound_faces - outside
            
            # Validate enclosed faces using probability map
            if use_probability_validation and enclosed_faces_raw:
                validated_enclosed = set()
                rejected_enclosed = set()
                for face_idx in enclosed_faces_raw:
                    if _validate_face_probability(face_idx, wound_prob_threshold):
                        validated_enclosed.add(face_idx)
                    else:
                        rejected_enclosed.add(face_idx)
                enclosed_faces = validated_enclosed
                
                if rejected_enclosed:
                    print(f"Probability validation rejected {len(rejected_enclosed)} enclosed faces (threshold={wound_prob_threshold:.2f})")
            else:
                enclosed_faces = enclosed_faces_raw
            
            if enclosed_faces:
                print(f"Found {len(enclosed_faces)} enclosed faces, adding to wound region")
            
            # Combine original wound faces with enclosed faces
            expanded_faces = face_set | enclosed_faces
            self.added_faces_enclosed = enclosed_faces
        else:
            expanded_faces = face_set
            self.added_faces_enclosed = set()
        
        # Step 2: Extract connected components and select largest
        if use_connectivity and len(expanded_faces) > 0:
            pre_connectivity = set(expanded_faces)
            components = self.extract_connected_components(expanded_faces, min_component_faces)
            
            if components:
                # Take largest component as primary wound
                largest_component = max(components, key=len)
                final_faces = largest_component
                print(f"Selected largest component with {len(largest_component)} faces from {len(components)} components")
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
                      use_closing: bool = True,
                      closing_iterations: int = 1,
                      closing_kernel_size: int = 1,
                      use_fill_holes: bool = False,
                      hole_size: float = -1.0,
                      use_probability_validation: bool = True,
                      wound_prob_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get final wound mesh after all filtering steps.
        
        Args:
            min_visibility: Minimum camera visibility count
            use_connectivity: Whether to extract connected components
            min_component_faces: Minimum faces for valid component
            use_closing: Whether to apply morphological closing
            closing_iterations: Number of closing iterations
            closing_kernel_size: Number of adjacency layers to expand per iteration (1=neighbors, 2=2 layers, etc.)
            use_fill_holes: Whether to fill holes in the mesh using trimesh
            hole_size: Maximum hole size to fill (-1.0 for all holes)
            use_probability_validation: If True, validate added faces using wound probability map
            wound_prob_threshold: Minimum wound probability for faces added during closing/expansion
            
        Returns:
            Tuple of (wound_vertices, wound_faces, wound_colors)
        """
        if not self.faces_processed:
            raise ValueError("Must process wound segmentation first")
        
        # Step 1: Filter by visibility
        filtered_faces = self.filter_wound_faces(min_visibility)

        if not filtered_faces:
            print("Warning: No wound faces remain after filtering")
            return np.array([]), np.array([]), np.array([])
        
        # Step 2: Apply morphological closing with probability validation
        if use_closing:
            pre_close = set(filtered_faces)
            filtered_faces = self.morphological_closing(
                filtered_faces, 
                closing_iterations, 
                closing_kernel_size,
                use_probability_validation=use_probability_validation,
                wound_prob_threshold=wound_prob_threshold
            )
            self.added_faces_closing = set(filtered_faces) - pre_close
        else:
            self.added_faces_closing = set()
        
        if not filtered_faces:
            print("Warning: No wound faces remain after closing")
            return np.array([]), np.array([]), np.array([])
        
        # Step 3: Expand and extract components with probability validation
        filtered_faces = self.expand_and_extract_components(
            filtered_faces, 
            use_connectivity, 
            min_component_faces,
            use_probability_validation=use_probability_validation,
            wound_prob_threshold=wound_prob_threshold
        )
        
        if not filtered_faces:
            print("Warning: No wound faces remain after component extraction")
            return np.array([]), np.array([]), np.array([])
        
        # Extract wound submesh from face indices
        wound_vertices, wound_faces, wound_colors = self.extract_submesh_from_faces(filtered_faces)

        # Step 4: Fill holes in the mesh (optional)
        if use_fill_holes and len(wound_vertices) > 0:
            print("Filling holes in wound mesh...")
            wound_vertices, wound_faces, wound_colors = self._fill_mesh_holes(
                wound_vertices, wound_faces, wound_colors, hole_size
            )

        print(f"Final wound mesh: {len(wound_vertices)} vertices, {len(wound_faces)} faces")

        # Update global tracking with face indices in self.faces
        self.current_wound_faces = filtered_faces

        # Store last wound mesh for visualization
        self.last_wound_vertices = wound_vertices
        self.last_wound_faces = wound_faces
        self.last_wound_colors = wound_colors

        # Compute wound edge boundary
        self.compute_wound_edge(filtered_faces)

        return wound_vertices, wound_faces, wound_colors
    
    def _compute_boundary_n_edges(self, vertices: np.ndarray, faces: np.ndarray) -> int:
        """
        Compute n_edges parameter for pymeshfix based on longest boundary loop.
        Returns 10% of the longest boundary loop's edge count.
        
        Args:
            vertices: Vertex array (N, 3)
            faces: Face array (M, 3)
            
        Returns:
            n_edges value (at least 10)
        """
        # Build edge adjacency for this submesh
        edge_count = defaultdict(int)
        for face in faces:
            v0, v1, v2 = face
            edges = [(min(v0, v1), max(v0, v1)),
                     (min(v1, v2), max(v1, v2)),
                     (min(v2, v0), max(v2, v0))]
            for edge in edges:
                edge_count[edge] += 1
        
        # Boundary edges appear only once
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        
        if not boundary_edges:
            print("No boundary edges found, using default n_edges=50")
            return 50
        
        # Build boundary loops
        # Create adjacency for boundary vertices
        boundary_adjacency = defaultdict(list)
        for v0, v1 in boundary_edges:
            boundary_adjacency[v0].append(v1)
            boundary_adjacency[v1].append(v0)
        
        # Extract connected boundary loops
        visited_edges = set()
        boundary_loops = []
        
        for start_edge in boundary_edges:
            if start_edge in visited_edges:
                continue
            
            # Trace boundary loop
            loop_edges = []
            current_edge = start_edge
            v0, v1 = current_edge
            visited_edges.add(current_edge)
            loop_edges.append(current_edge)
            
            # Follow the boundary
            current_vertex = v1
            while True:
                neighbors = [v for v in boundary_adjacency[current_vertex] 
                            if (min(current_vertex, v), max(current_vertex, v)) not in visited_edges]
                if not neighbors:
                    break
                
                next_vertex = neighbors[0]
                edge = (min(current_vertex, next_vertex), max(current_vertex, next_vertex))
                visited_edges.add(edge)
                loop_edges.append(edge)
                current_vertex = next_vertex
                
                if current_vertex == v0:  # Closed loop
                    break
            
            if loop_edges:
                boundary_loops.append(loop_edges)
        
        # Find longest boundary loop
        if not boundary_loops:
            print("No boundary loops found, using default n_edges=50")
            return 50
        
        longest_loop = max(boundary_loops, key=len)
        longest_loop_length = len(longest_loop)
        
        # Use 10% of longest boundary
        n_edges = max(10, int(longest_loop_length * 0.1))
        
        print(f"Boundary analysis: {len(boundary_edges)} boundary edges, "
              f"{len(boundary_loops)} loops, longest loop: {longest_loop_length} edges")
        print(f"Using n_edges = {n_edges} (10% of longest boundary)")
        
        return n_edges
    
    def _fill_mesh_holes(self, vertices: np.ndarray, faces: np.ndarray, 
                        colors: np.ndarray = None, hole_size: float = -1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fill holes in the mesh using pymeshfix.
        
        Args:
            vertices: Vertex array (N, 3)
            faces: Face array (M, 3)
            colors: Optional color array (N, 3)
            hole_size: Unused (kept for API compatibility)
            
        Returns:
            Tuple of (vertices, faces, colors) after hole filling
        """
        try:
            import pymeshfix
            
            # Compute boundary edges and find longest boundary loop
            n_edges = self._compute_boundary_n_edges(vertices, faces)
            
            print(f"Filling holes with pymeshfix (n_edges={n_edges})...")
            mfix = pymeshfix.MeshFix(vertices.copy(), faces.copy())
            mfix.fill_holes(n_edges=n_edges)
            
            vertices_filled = mfix.points
            faces_filled = mfix.faces
            
            # Transfer colors to new vertices using nearest neighbor
            colors_filled = colors
            if colors is not None and len(vertices_filled) != len(vertices):
                try:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(vertices)
                    _, nn_idx = tree.query(vertices_filled, k=1)
                    colors_filled = colors[nn_idx]
                except Exception as e:
                    print(f"Warning: Could not transfer colors: {e}")
                    colors_filled = None
            
            print(f"Hole filling complete: {len(faces)} -> {len(faces_filled)} faces")
            
            return vertices_filled, faces_filled, colors_filled
            
        except ImportError:
            print("pymeshfix not available, returning original mesh")
            print("Install with: pip install pymeshfix")
            return vertices, faces, colors
        except Exception as e:
            print(f"Error during hole filling: {e}, returning original mesh")
            import traceback
            traceback.print_exc()
            return vertices, faces, colors
    
    def save_wound_mesh(self, output_path: Path, vertices: np.ndarray = None,
                       faces: np.ndarray = None, colors: np.ndarray = None):
        """
        Save mesh to PLY file.

        Args:
            output_path: Output PLY file path
            vertices: Vertex array (N, 3)
            faces: Face array (M, 3)
            colors: Optional color array (N, 3)
        """
        import open3d as o3d
        output_path = Path(output_path)
        print(f"Saving mesh to: {output_path}")

        try:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
            mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
            if colors is not None:
                colors_01 = colors / 255.0 if np.max(colors) > 1.0 else colors.copy()
                mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors_01, 0.0, 1.0))
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(str(output_path), mesh)
            print("Saved mesh successfully")

        except ImportError:
            print("Open3D not available for mesh export")
    
    def visualize_with_rerun(self, session_name: str = "WoundModel3DMesh", min_periwound_edge_visibility: int = 2):
        """
        Visualize mesh-based wound segmentation using Rerun.
        
        Args:
            session_name: Name for the Rerun session
            min_periwound_edge_visibility: Minimum visibility count for periwound faces to display
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
        rr.init(session_name, spawn=True)
        
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        
        # Create blueprint with tabs for each processing stage
        blueprint = rrb.Blueprint(
            rrb.Tabs(
                rrb.Spatial3DView(origin="/world", name="Full Mesh & periwound"),
                rrb.Spatial3DView(origin="/wound", name="Wound Region"),
                rrb.Spatial3DView(origin="/periwound_edge", name="periwound Details"),
                rrb.Spatial3DView(origin="/features", name="Feature Vector Calculation"),
                rrb.Spatial3DView(origin="/gmm", name="GMM Classification"),
                rrb.Spatial3DView(origin="/mrf", name="MRF Constraints"),
                rrb.Spatial3DView(origin="/confidence", name="Final Confidence Maps"),
                rrb.TextDocumentView(origin="/stats", name="Statistics"),
            )
        )
        rr.send_blueprint(blueprint)
        
        # Log full mesh
        vertex_positions = self.vertices.astype(np.float32)
        
        # Create mesh entity
        if self.vertex_colors is not None:
            colors = (self.vertex_colors * 255).astype(np.uint8) if np.max(self.vertex_colors) <= 1.0 else self.vertex_colors.astype(np.uint8)
        else:
            colors = np.ones((len(self.vertices), 3), dtype=np.uint8) * 128
        
        # Log full mesh as triangle mesh
        rr.log("/world/full_mesh", rr.Mesh3D(
            vertex_positions=vertex_positions,
            triangle_indices=self.faces,
            vertex_colors=colors
        ), static=True)

        # Log final segmented wound mesh on the full mesh
        if self.last_wound_vertices is not None and self.last_wound_faces is not None:
            if self.last_wound_colors is not None:
                wound_colors_full = (self.last_wound_colors * 255).astype(np.uint8) if np.max(self.last_wound_colors) <= 1.0 else self.last_wound_colors.astype(np.uint8)
            else:
                wound_colors_full = np.ones((len(self.last_wound_vertices), 3), dtype=np.uint8) * 0
                wound_colors_full[:, 1] = 255  # Green
            rr.log("/world/segmented_wound", rr.Mesh3D(
                vertex_positions=self.last_wound_vertices.astype(np.float32),
                triangle_indices=self.last_wound_faces,
                vertex_colors=wound_colors_full
            ), static=True)
        
        # Log wound faces by step in Wound Region
        if self.faces_processed and self.wound_face_indices:
            # Initial wound mesh
            wound_verts, wound_faces_idx, wound_colors_init = self.extract_submesh_from_faces(self.wound_face_indices)
            if len(wound_verts) > 0:
                if wound_colors_init is None:
                    wound_colors_init = np.ones((len(wound_verts), 3), dtype=np.uint8) * 200
                else:
                    wound_colors_init = (wound_colors_init * 255).astype(np.uint8) if np.max(wound_colors_init) <= 1.0 else wound_colors_init.astype(np.uint8)
                rr.log("/wound/initial", rr.Mesh3D(
                    vertex_positions=wound_verts.astype(np.float32),
                    triangle_indices=wound_faces_idx,
                    vertex_colors=wound_colors_init
                ), static=True)

            # Removed faces by filtering step
            self._log_faces_mesh_rr(rr, "/wound/removed/visibility", self.removed_faces_visibility, [255, 0, 0])
            self._log_faces_mesh_rr(rr, "/wound/removed/smoothness", self.removed_faces_smoothness, [0, 128, 255])
            self._log_faces_mesh_rr(rr, "/wound/removed/connectivity", self.removed_faces_connectivity, [255, 128, 0])

            # Added/filled faces
            self._log_faces_mesh_rr(rr, "/wound/added/enclosed", self.added_faces_enclosed, [0, 255, 0])
            self._log_faces_mesh_rr(rr, "/wound/added/closing", self.added_faces_closing, [0, 200, 0])

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
            if edge_strips:
                rr.log("/wound/edge_edges", rr.LineStrips3D(
                    strips=edge_strips,
                    colors=[[255, 255, 0]] * len(edge_strips),
                    radii=0.002
                ), static=True)

        # Log periwound detection results with visibility filtering
        if self.periwound_edge_faces:
            # Filter periwound by visibility
            periwound_edge_filtered = self.filter_periwound_edge_by_visibility(min_periwound_edge_visibility)
            
            if periwound_edge_filtered:
                print(f"Logging {len(periwound_edge_filtered)} periwound faces (filtered by visibility >= {min_periwound_edge_visibility})...")
                periwound_edge_verts, periwound_edge_faces_idx, periwound_edge_colors = self.extract_submesh_from_faces(periwound_edge_filtered)
            else:
                print(f"Warning: No periwound faces remain after visibility filtering (min={min_periwound_edge_visibility})")
                periwound_edge_verts = np.array([])
            
            if len(periwound_edge_verts) > 0:
                # Color by original mesh colors
                if periwound_edge_colors is None:
                    periwound_edge_colors = np.ones((len(periwound_edge_verts), 3), dtype=np.uint8) * 0
                    periwound_edge_colors[:, 0] = 200  # Red
                    periwound_edge_colors[:, 1] = 200  # Green
                else:
                    periwound_edge_colors = (periwound_edge_colors * 255).astype(np.uint8) if np.max(periwound_edge_colors) <= 1.0 else periwound_edge_colors.astype(np.uint8)
                
                # Log to world origin (full mesh view)
                rr.log("/world/periwound_edge", rr.Mesh3D(
                    vertex_positions=periwound_edge_verts.astype(np.float32),
                    triangle_indices=periwound_edge_faces_idx,
                    vertex_colors=periwound_edge_colors
                ), static=True)
                
                # Log to periwound_edge origin (dedicated view)
                rr.log("/periwound_edge/mesh_original_colors", rr.Mesh3D(
                    vertex_positions=periwound_edge_verts.astype(np.float32),
                    triangle_indices=periwound_edge_faces_idx,
                    vertex_colors=periwound_edge_colors
                ), static=True)
            
            # Log periwound by visibility (cyan to magenta gradient) - filtered version
            if periwound_edge_filtered:
                visibility_face_colors_filtered = self._create_visibility_colormap(
                    {f: self.periwound_edge_visibility_count[f] for f in periwound_edge_filtered if f in self.periwound_edge_visibility_count}
                )
                self._log_faces_mesh_rr_with_colors(rr, "/world/periwound_edge_by_visibility", 
                                                    periwound_edge_filtered, visibility_face_colors_filtered)
                self._log_faces_mesh_rr_with_colors(rr, "/periwound_edge/by_visibility", 
                                                    periwound_edge_filtered, visibility_face_colors_filtered)
            
            # Log periwound statistics as text
            if self.periwound_edge_visibility_count:
                vis_counts = list(self.periwound_edge_visibility_count.values())
                periwound_edge_stats = f"""
periwound Detection Results:
Total periwound Faces: {len(self.periwound_edge_faces)}
Filtered periwound Faces (visibility >= {min_periwound_edge_visibility}): {len(periwound_edge_filtered)}

Visibility Statistics (all detected):
- Min views: {min(vis_counts)}
- Max views: {max(vis_counts)}
- Mean views: {np.mean(vis_counts):.2f}
- Median views: {np.median(vis_counts):.2f}

Visibility Statistics (filtered):
                """
                if periwound_edge_filtered:
                    filtered_vis_counts = [self.periwound_edge_visibility_count[f] for f in periwound_edge_filtered]
                    periwound_edge_stats += f"""- Min views: {min(filtered_vis_counts)}
- Max views: {max(filtered_vis_counts)}
- Mean views: {np.mean(filtered_vis_counts):.2f}
- Median views: {np.median(filtered_vis_counts):.2f}
"""
                
                periwound_edge_stats += f"""\nColor Statistics (across detected periwounds):
                """
                if self.periwound_edge_mean_color:
                    mean_colors = np.array([self.periwound_edge_mean_color[f] for f in self.periwound_edge_faces if f in self.periwound_edge_mean_color])
                    if len(mean_colors) > 0:
                        periwound_edge_stats += f"\nMean Color (RGB):\n- R: {np.mean(mean_colors[:, 0]):.3f}\n- G: {np.mean(mean_colors[:, 1]):.3f}\n- B: {np.mean(mean_colors[:, 2]):.3f}\n"
                    
                    std_colors = np.array([self.periwound_edge_color_std[f] for f in self.periwound_edge_faces if f in self.periwound_edge_color_std])
                    if len(std_colors) > 0:
                        periwound_edge_stats += f"\nColor Std (RGB):\n- R: {np.mean(std_colors[:, 0]):.3f}\n- G: {np.mean(std_colors[:, 1]):.3f}\n- B: {np.mean(std_colors[:, 2]):.3f}"
                
                rr.log("/periwound_edge/stats", rr.TextDocument(periwound_edge_stats), static=True)
        
        # === Feature Vector Calculation Visualization ===
        if self.semantic_fitted_plane is not None and self.semantic_feature_colormaps:
            print("Logging feature vector calculations to Rerun...")
            
            # Visualize fitted tissue reference plane
            plane_normal, plane_d = self.semantic_fitted_plane
            
            # Create a planar mesh visualization
            # Generate a grid on the plane
            centroids = self._get_face_centroids()
            plane_centroids = centroids[self.semantic_face_indices]
            
            # Project centroids onto plane for visualization
            if len(plane_centroids) > 0:
                # Find plane bounds
                center = np.mean(plane_centroids, axis=0)
                plane_normal_norm = plane_normal / (np.linalg.norm(plane_normal) + 1e-9)
                
                # Create orthogonal vectors in the plane
                if np.abs(plane_normal_norm[0]) < 0.9:
                    u_axis = np.array([plane_normal_norm[1], -plane_normal_norm[0], 0])
                else:
                    u_axis = np.array([0, plane_normal_norm[2], -plane_normal_norm[1]])
                u_axis = u_axis / (np.linalg.norm(u_axis) + 1e-9)
                v_axis = np.cross(plane_normal_norm, u_axis)
                
                # Create grid
                extent = np.max(np.linalg.norm(plane_centroids - center, axis=1)) * 1.2
                grid_points = []
                for i in np.linspace(-1, 1, 10):
                    for j in np.linspace(-1, 1, 10):
                        pt = center + i * extent * u_axis + j * extent * v_axis
                        grid_points.append(pt)
                
                grid_points = np.array(grid_points, dtype=np.float32)
                rr.log("/features/tissue_reference_plane", rr.Points3D(
                    positions=grid_points,
                    colors=[[200, 200, 200]] * len(grid_points),
                    radii=0.5
                ), static=True)
            
            # Log feature colormaps
            for feat_name, colormap in self.semantic_feature_colormaps.items():
                if colormap:
                    self._log_faces_mesh_rr_with_colors(rr, f"/features/{feat_name}", 
                                                        set(colormap.keys()), 
                                                        colormap)
        
        # === GMM Classification Visualization ===
        if self.semantic_class_samples:
            print("Logging class samples and GMM models to Rerun...")
            
            # Define colors for each class (3-class: tissue, periwound, wound)
            class_colors = {
                "tissue": [0, 150, 255],    # Blue
                "periwound": [255, 165, 0], # Orange
                "wound": [255, 0, 0]        # Red
            }
            
            # Log class samples for all three classes
            for class_name, sample_indices in self.semantic_class_samples.items():
                if sample_indices:
                    color = class_colors.get(class_name, [128, 128, 128])
                    self._log_faces_mesh_rr(rr, f"/gmm/samples/{class_name}",
                                          set(sample_indices), color)
                    self._log_faces_mesh_rr(rr, f"/world/gmm_samples/{class_name}",
                                          set(sample_indices), color)
            
            # Log all samples together with class colours for world view
            sample_stats = "GMM Training Samples:\n\n"
            for class_name, sample_indices in self.semantic_class_samples.items():
                sample_stats += f"{class_name.upper()}: {len(sample_indices)} faces\n"
            rr.log("/gmm/samples/stats", rr.TextDocument(sample_stats), static=True)
            
            # Log GMM model statistics
            gmm_stats = "GMM Model Statistics:\n\n"
            if self.semantic_gmm_models:
                for class_name in self.semantic_class_names:
                    if class_name in self.semantic_gmm_models:
                        model = self.semantic_gmm_models[class_name]
                        n_samples = model.get("n_samples", 0)
                        gmm_stats += f"{class_name.upper()}:\n"
                        gmm_stats += f"  Sample count: {n_samples}\n"
                        if "gmm" in model:
                            gmm = model["gmm"]
                            gmm_stats += f"  Mean: {gmm.means_[0]}\n"
                            gmm_stats += f"  Covariance diagonal: {np.diag(gmm.covariances_[0])}\n"
                        else:
                            gmm_stats += f"  Mean: {model['mean']}\n"
                            gmm_stats += f"  Covariance diagonal: {np.diag(model['cov'])}\n"
                        gmm_stats += "\n"
            
            rr.log("/gmm/model_stats", rr.TextDocument(gmm_stats), static=True)
        
            # === GMM Softmaxed Posterior Probability Visualization ===
            # Shows the normalized posterior probabilities P(class | features) after softmax.
            # High values (green) = high confidence the face belongs to that class.
            # Low values (red)   = low confidence the face belongs to that class.
            if self.semantic_wound_prob and self.semantic_tissue_prob:
                print("Logging GMM softmaxed posteriors to Rerun...")
                
                # Visualize per-class posterior probabilities (3-class: tissue, periwound, wound)
                class_prob_maps = [("tissue", self.semantic_tissue_prob),
                                   ("wound", self.semantic_wound_prob)]
                if self.semantic_periwound_prob:
                    class_prob_maps.insert(1, ("periwound", self.semantic_periwound_prob))
                
                for cls_name, prob_map in class_prob_maps:
                    prob_colormap = self._create_confidence_colormap(prob_map)
                    self._log_faces_mesh_rr_with_colors(
                        rr, f"/gmm/{cls_name}_posterior_probability",
                        set(prob_colormap.keys()), prob_colormap
                    )
                
                # Also log the posterior probability difference (wound - tissue):
                # Positive (green) = more wound-like, Negative (red) = more tissue-like
                posterior_diff_map = {}
                for face_idx in self.semantic_wound_prob:
                    diff = self.semantic_wound_prob[face_idx] - self.semantic_tissue_prob.get(face_idx, 0.0)
                    posterior_diff_map[face_idx] = diff
                
                diff_values = np.array(list(posterior_diff_map.values()))
                abs_max = max(abs(np.percentile(diff_values, 2)), abs(np.percentile(diff_values, 98)), 1e-9)
                diff_colormap = {}
                for face_idx, diff_val in posterior_diff_map.items():
                    # Map [-1.0, +1.0] to [0, 1]: 0=tissue-like(red), 1=wound-like(green)
                    norm_val = np.clip((diff_val + 1.0) / 2.0, 0.0, 1.0)
                    diff_colormap[face_idx] = self._confidence_to_color(norm_val)
                
                self._log_faces_mesh_rr_with_colors(
                    rr, "/gmm/posterior_difference_wound_minus_tissue",
                    set(diff_colormap.keys()), diff_colormap
                )
                
                # Stats text
                wound_probs = np.array(list(self.semantic_wound_prob.values()))
                tissue_probs = np.array(list(self.semantic_tissue_prob.values()))
                posterior_stats = f"""GMM Softmaxed Posterior Probabilities:

Wound posterior P(wound | features):
  min: {np.min(wound_probs):.4f}  max: {np.max(wound_probs):.4f}
  mean: {np.mean(wound_probs):.4f}  std: {np.std(wound_probs):.4f}

Tissue posterior P(tissue | features):
  min: {np.min(tissue_probs):.4f}  max: {np.max(tissue_probs):.4f}
  mean: {np.mean(tissue_probs):.4f}  std: {np.std(tissue_probs):.4f}
"""
                if self.semantic_periwound_prob:
                    periwound_probs = np.array(list(self.semantic_periwound_prob.values()))
                    posterior_stats += f"""
Periwound posterior P(periwound | features):
  min: {np.min(periwound_probs):.4f}  max: {np.max(periwound_probs):.4f}
  mean: {np.mean(periwound_probs):.4f}  std: {np.std(periwound_probs):.4f}
"""
                posterior_stats += f"""
Difference (wound - tissue):
  min: {np.min(diff_values):.4f}  max: {np.max(diff_values):.4f}
  mean: {np.mean(diff_values):.4f}

Interpretation (Viridis colormap):
  Yellow (posterior > 0.5) = high confidence for that class
  Dark purple (posterior < 0.5) = low confidence for that class
  Posterior difference: yellow = wound-like (P(wound) > P(tissue)), purple = tissue-like
"""
                rr.log("/gmm/posterior_stats", rr.TextDocument(posterior_stats), static=True)

        # === MRF Constraints Visualization ===
        if self.semantic_posteriors_before_mrf:
            print("Logging posteriors (before and after MRF) to Rerun...")
            
            # Create posterior colormaps for 3-class model before MRF (tissue, periwound, wound)
            tissue_prob_before = {}
            periwound_prob_before = {}
            wound_prob_before = {}
            
            for face_idx, posteriors_arr in self.semantic_posteriors_before_mrf.items():
                tissue_prob_before[face_idx] = float(posteriors_arr[0])
                if len(posteriors_arr) > 2:
                    periwound_prob_before[face_idx] = float(posteriors_arr[1])
                    wound_prob_before[face_idx] = float(posteriors_arr[2])
                else:
                    wound_prob_before[face_idx] = float(posteriors_arr[1])
            
            # Log before MRF
            if tissue_prob_before:
                tissue_colors_before = self._create_confidence_colormap(tissue_prob_before)
                self._log_faces_mesh_rr_with_colors(rr, "/mrf/tissue_prob_before_mrf", 
                                                    set(tissue_prob_before.keys()), 
                                                    tissue_colors_before)
            
            if periwound_prob_before:
                periwound_colors_before = self._create_confidence_colormap(periwound_prob_before)
                self._log_faces_mesh_rr_with_colors(rr, "/mrf/periwound_prob_before_mrf", 
                                                    set(periwound_prob_before.keys()), 
                                                    periwound_colors_before)
            
            if wound_prob_before:
                wound_colors_before = self._create_confidence_colormap(wound_prob_before)
                self._log_faces_mesh_rr_with_colors(rr, "/mrf/wound_prob_before_mrf", 
                                                    set(wound_prob_before.keys()), 
                                                    wound_colors_before)
            
            # Log after MRF (from semantic_tissue_prob, semantic_periwound_prob, semantic_wound_prob for 3-class model)
            if self.semantic_tissue_prob:
                tissue_colors_after = self._create_confidence_colormap(self.semantic_tissue_prob)
                self._log_faces_mesh_rr_with_colors(rr, "/mrf/tissue_prob_after_mrf", 
                                                    set(self.semantic_tissue_prob.keys()), 
                                                    tissue_colors_after)
            
            if self.semantic_periwound_prob:
                periwound_colors_after = self._create_confidence_colormap(self.semantic_periwound_prob)
                self._log_faces_mesh_rr_with_colors(rr, "/mrf/periwound_prob_after_mrf", 
                                                    set(self.semantic_periwound_prob.keys()), 
                                                    periwound_colors_after)
            
            if self.semantic_wound_prob:
                wound_colors_after = self._create_confidence_colormap(self.semantic_wound_prob)
                self._log_faces_mesh_rr_with_colors(rr, "/mrf/wound_prob_after_mrf", 
                                                    set(self.semantic_wound_prob.keys()), 
                                                    wound_colors_after)
            
            # Log MRF label changes
            mrf_stats = "MRF Constraint Analysis:\n\n"
            mrf_stats += f"Method: {getattr(self, 'semantic_mrf_method', 'unknown')}\n\n"
            label_changes = 0
            for face_idx in self.semantic_face_indices:
                if face_idx in self.semantic_posteriors_before_mrf and face_idx in self.semantic_posteriors:
                    before_label = np.argmax(self.semantic_posteriors_before_mrf[face_idx])
                    probs = self.semantic_posteriors[face_idx]
                    after_label = np.argmax(probs) if len(probs) > 0 else 0
                    if before_label != after_label:
                        label_changes += 1
            
            mrf_stats += f"Faces relabeled by MRF: {label_changes}\n"
            mrf_stats += f"Total classified faces: {len(self.semantic_face_indices)}\n"
            mrf_stats += f"Relabeling percentage: {100*label_changes/max(1, len(self.semantic_face_indices)):.2f}%"
            
            rr.log("/mrf/constraint_stats", rr.TextDocument(mrf_stats), static=True)
            
            # === Mean Field Posteriors Visualization (only if mean_field method used) ===
            if getattr(self, 'semantic_mrf_method', None) == "mean_field" and getattr(self, 'semantic_mrf_posteriors', {}):
                print("Logging mean field refined posteriors to Rerun...")
                
                # Extract mean field posteriors (3-class: tissue, periwound, wound)
                mf_tissue_prob = {}
                mf_periwound_prob = {}
                mf_wound_prob = {}
                for face_idx, probs in self.semantic_mrf_posteriors.items():
                    mf_tissue_prob[face_idx] = float(probs[0])
                    if len(probs) > 2:
                        mf_periwound_prob[face_idx] = float(probs[1])
                        mf_wound_prob[face_idx] = float(probs[2])
                    else:
                        mf_wound_prob[face_idx] = float(probs[1])
                
                # Log mean field tissue probability
                if mf_tissue_prob:
                    mf_tissue_colors = self._create_confidence_colormap(mf_tissue_prob)
                    self._log_faces_mesh_rr_with_colors(rr, "/mrf/mean_field_tissue_prob", 
                                                        set(mf_tissue_prob.keys()), 
                                                        mf_tissue_colors)
                
                # Log mean field periwound probability
                if mf_periwound_prob:
                    mf_periwound_colors = self._create_confidence_colormap(mf_periwound_prob)
                    self._log_faces_mesh_rr_with_colors(rr, "/mrf/mean_field_periwound_prob", 
                                                        set(mf_periwound_prob.keys()), 
                                                        mf_periwound_colors)
                
                # Log mean field wound probability
                if mf_wound_prob:
                    mf_wound_colors = self._create_confidence_colormap(mf_wound_prob)
                    self._log_faces_mesh_rr_with_colors(rr, "/mrf/mean_field_wound_prob", 
                                                        set(mf_wound_prob.keys()), 
                                                        mf_wound_colors)
                
                # Log mean field probability difference (wound - tissue)
                mf_diff_map = {}
                for face_idx in mf_wound_prob:
                    diff = mf_wound_prob[face_idx] - mf_tissue_prob.get(face_idx, 0.0)
                    mf_diff_map[face_idx] = diff
                
                if mf_diff_map:
                    diff_values = np.array(list(mf_diff_map.values()))
                    mf_diff_colormap = {}
                    for face_idx, diff_val in mf_diff_map.items():
                        # Map [-1.0, +1.0] to [0, 1]: 0=tissue-like, 1=wound-like
                        norm_val = np.clip((diff_val + 1.0) / 2.0, 0.0, 1.0)
                        mf_diff_colormap[face_idx] = self._confidence_to_color(norm_val)
                    
                    self._log_faces_mesh_rr_with_colors(
                        rr, "/mrf/mean_field_prob_diff_wound_minus_tissue",
                        set(mf_diff_colormap.keys()), mf_diff_colormap
                    )
                    
                    # Mean field statistics
                    mf_wound_vals = np.array(list(mf_wound_prob.values()))
                    mf_tissue_vals = np.array(list(mf_tissue_prob.values()))
                    mf_stats = f"""Mean Field Approximation Results:

Tissue posterior P(tissue):
  min: {np.min(mf_tissue_vals):.4f}  max: {np.max(mf_tissue_vals):.4f}
  mean: {np.mean(mf_tissue_vals):.4f}  std: {np.std(mf_tissue_vals):.4f}
"""
                    if mf_periwound_prob:
                        mf_periwound_vals = np.array(list(mf_periwound_prob.values()))
                        mf_stats += f"""
Periwound posterior P(periwound):
  min: {np.min(mf_periwound_vals):.4f}  max: {np.max(mf_periwound_vals):.4f}
  mean: {np.mean(mf_periwound_vals):.4f}  std: {np.std(mf_periwound_vals):.4f}
"""
                    mf_stats += f"""
Wound posterior P(wound):
  min: {np.min(mf_wound_vals):.4f}  max: {np.max(mf_wound_vals):.4f}
  mean: {np.mean(mf_wound_vals):.4f}  std: {np.std(mf_wound_vals):.4f}

Probability difference (wound - tissue):
  min: {np.min(diff_values):.4f}  max: {np.max(diff_values):.4f}
  mean: {np.mean(diff_values):.4f}

Interpretation:
  Yellow = high probability for that class
  Dark purple = low probability for that class
  Difference: yellow = wound-like, purple = tissue-like
  
NOTE: Mean field returns soft probabilities (not hard labels)
      The visualization shows the refined belief distribution.
"""
                    rr.log("/mrf/mean_field_stats", rr.TextDocument(mf_stats), static=True)

        # Log confidence maps
        if self.wound_confidence_map or self.periwound_edge_confidence_map:

            print("Logging confidence maps to Rerun...")
            
            # Wound confidence map
            if self.wound_confidence_map:
                wound_conf_colors = self._create_confidence_colormap(self.wound_confidence_map)
                self._log_faces_mesh_rr_with_colors(rr, "/confidence/wound_confidence", 
                                                    set(self.wound_confidence_map.keys()), 
                                                    wound_conf_colors)
                self._log_faces_mesh_rr_with_colors(rr, "/world/wound_confidence", 
                                                    set(self.wound_confidence_map.keys()), 
                                                    wound_conf_colors)
            
            # periwound confidence map
            if self.periwound_edge_confidence_map:
                tissue_conf_colors = self._create_confidence_colormap(self.periwound_edge_confidence_map)
                self._log_faces_mesh_rr_with_colors(rr, "/confidence/periwound_edge_confidence", 
                                                    set(self.periwound_edge_confidence_map.keys()), 
                                                    tissue_conf_colors)
                self._log_faces_mesh_rr_with_colors(rr, "/world/periwound_edge_confidence", 
                                                    set(self.periwound_edge_confidence_map.keys()), 
                                                    tissue_conf_colors)
            
            # Overlapping faces visualization
            if self.overlapping_faces:
                # Color by wound confidence for overlapping faces
                overlap_wound_colors = {f: self._confidence_to_color(self.wound_confidence_map[f]) 
                                       for f in self.overlapping_faces if f in self.wound_confidence_map}
                self._log_faces_mesh_rr_with_colors(rr, "/confidence/overlapping_wound_view", 
                                                    self.overlapping_faces, 
                                                    overlap_wound_colors)
                
                # Color by periwound confidence for overlapping faces
                overlap_tissue_colors = {f: self._confidence_to_color(self.periwound_edge_confidence_map[f]) 
                                        for f in self.overlapping_faces if f in self.periwound_edge_confidence_map}
                self._log_faces_mesh_rr_with_colors(rr, "/confidence/overlapping_tissue_view", 
                                                    self.overlapping_faces, 
                                                    overlap_tissue_colors)
            
            # Confidence statistics
            conf_stats = self._get_confidence_stats_text()
            rr.log("/confidence/stats", rr.TextDocument(conf_stats), static=True)

        # Log semantic classification results
        if self.semantic_confidence:
            print("Logging semantic confidence maps to Rerun...")
            semantic_conf_colors = self._create_confidence_colormap(self.semantic_confidence)
            self._log_faces_mesh_rr_with_colors(rr, "/confidence/semantic_confidence",
                                                set(self.semantic_confidence.keys()),
                                                semantic_conf_colors)

            if self.semantic_wound_prob:
                wound_prob_colors = self._create_confidence_colormap(self.semantic_wound_prob)
                self._log_faces_mesh_rr_with_colors(rr, "/confidence/semantic_wound_prob",
                                                    set(self.semantic_wound_prob.keys()),
                                                    wound_prob_colors)

            if self.semantic_periwound_prob:
                periwound_prob_colors = self._create_confidence_colormap(self.semantic_periwound_prob)
                self._log_faces_mesh_rr_with_colors(rr, "/confidence/semantic_periwound_prob",
                                                    set(self.semantic_periwound_prob.keys()),
                                                    periwound_prob_colors)

            if self.semantic_tissue_prob:
                tissue_prob_colors = self._create_confidence_colormap(self.semantic_tissue_prob)
                self._log_faces_mesh_rr_with_colors(rr, "/confidence/semantic_tissue_prob",
                                                    set(self.semantic_tissue_prob.keys()),
                                                    tissue_prob_colors)

            # Label-based view (3-class: tissue, periwound, wound)
            tissue_faces = {f for f, lbl in self.semantic_labels.items() if lbl == "tissue"}
            periwound_faces = {f for f, lbl in self.semantic_labels.items() if lbl == "periwound"}
            wound_faces = {f for f, lbl in self.semantic_labels.items() if lbl == "wound"}
            self._log_faces_mesh_rr(rr, "/confidence/semantic_labels/tissue", tissue_faces, [0, 150, 255])
            self._log_faces_mesh_rr(rr, "/confidence/semantic_labels/periwound", periwound_faces, [255, 200, 0])
            self._log_faces_mesh_rr(rr, "/confidence/semantic_labels/wound", wound_faces, [255, 0, 0])
            
            # Boundary region (computed post-classification from adjacency)
            if hasattr(self, 'wound_edge_face_indices') and self.wound_edge_face_indices:
                self._log_faces_mesh_rr(rr, "/confidence/wound_edge_boundary", 
                                        self.wound_edge_face_indices, [255, 200, 0])

        # Log statistics
        stats_text = f"""
Mesh Statistics:
- Total Vertices: {len(self.vertices):,}
- Total Faces: {len(self.faces):,}
- Wound Faces: {len(self.wound_face_indices):,}
- periwound Faces: {len(self.periwound_edge_faces):,}
- Spectators: {len(self.spectators)}
- Masks Loaded: {sum(1 for s in self.spectators.values() if s.mask_loaded)}
- Extruded Masks Loaded: {sum(1 for s in self.spectators.values() if s.extruded_mask_loaded)}
        """
        
        if self.periwound_edge_visibility_count:
            avg_vis = np.mean(list(self.periwound_edge_visibility_count.values()))
            stats_text += f"\nperiwound Visibility:\n- Average: {avg_vis:.2f}\n"
        
        rr.log("/stats", rr.TextDocument(stats_text), static=True)
        
        print(f"Rerun visualization started: {session_name}")


    def _log_faces_mesh_rr(self, rr, path: str, face_set: Set[int], color: List[int]) -> None:
        """Log a mesh from a set of face indices for visualization in Rerun."""
        if not face_set:
            return
        
        # Extract submesh from face indices
        submesh_verts, submesh_faces, _ = self.extract_submesh_from_faces(face_set)
        
        if len(submesh_verts) == 0:
            return
        
        # Create uniform color for all vertices
        colors = np.array([color] * len(submesh_verts), dtype=np.uint8)
        
        rr.log(path, rr.Mesh3D(
            vertex_positions=submesh_verts.astype(np.float32),
            triangle_indices=submesh_faces,
            vertex_colors=colors
        ), static=True)
    
    def _create_visibility_colormap(self, visibility_count: Dict[int, int]) -> Dict[int, np.ndarray]:
        """Create a colormap for visualization based on visibility counts.
        
        Args:
            visibility_count: Dict mapping face index to visibility count
            
        Returns:
            Dict mapping face index to RGB color [0-255]
        """
        if not visibility_count:
            return {}
        
        counts = np.array(list(visibility_count.values()))
        min_count = np.min(counts)
        max_count = np.max(counts)
        
        # Normalize to [0, 1]
        if max_count == min_count:
            normalized = np.ones_like(counts, dtype=float) * 0.5
        else:
            normalized = (counts - min_count) / (max_count - min_count)
        
        colormap = {}
        for (face_idx, count), norm_val in zip(visibility_count.items(), normalized):
            # Cyan to Magenta gradient
            r = int(norm_val * 255)  # 0-255
            g = int((1 - norm_val) * 255)  # 255-0
            b = 255  # Always full blue
            colormap[face_idx] = np.array([r, g, b], dtype=np.uint8)
        
        return colormap
    
    def _create_confidence_colormap(self, confidence_map: Dict[int, float]) -> Dict[int, np.ndarray]:
        """Create a colormap for visualization based on confidence values.
        
        Uses a color gradient from red (low confidence) to green (high confidence).
        
        Args:
            confidence_map: Dict mapping face index to confidence [0-1]
            
        Returns:
            Dict mapping face index to RGB color [0-255]
        """
        if not confidence_map:
            return {}
        
        colormap = {}
        for face_idx, confidence in confidence_map.items():
            colormap[face_idx] = self._confidence_to_color(confidence)
        
        return colormap
    
    def _confidence_to_color(self, confidence: float) -> np.ndarray:
        """Convert a confidence value [0-1] to RGB color [0-255].
        
        Uses a viridis-like colormap for smooth perceptual gradient.
        
        Color scheme (viridis):
        - 0.0 (low): Dark purple [68, 1, 84]
        - 0.5 (medium): Teal [32, 145, 140]
        - 1.0 (high): Yellow [253, 231, 37]
        
        Args:
            confidence: Confidence value [0-1]
            
        Returns:
            RGB color as numpy array [0-255]
        """
        import matplotlib
        confidence = np.clip(confidence, 0.0, 1.0)
        
        cmap = matplotlib.colormaps['viridis']
        rgba = cmap(confidence)  # Returns RGBA values [0-1]
        rgb = np.array(rgba[:3]) * 255  # Extract RGB, ensure numpy array, scale to [0-255]
        return np.clip(rgb, 0, 255).astype(np.uint8)
    
    def _get_confidence_stats_text(self) -> str:
        """Generate statistics text for confidence maps.
        
        Returns:
            Formatted statistics string
        """
        stats = """
Confidence Map Statistics:
========================="""
        
        if self.wound_confidence_map:
            wound_values = list(self.wound_confidence_map.values())
            stats += f"""

Wound Confidence Map:
- Total faces: {len(self.wound_confidence_map)}
- Min confidence: {np.min(wound_values):.3f}
- Max confidence: {np.max(wound_values):.3f}
- Mean confidence: {np.mean(wound_values):.3f}
- Median confidence: {np.median(wound_values):.3f}
- Std dev: {np.std(wound_values):.3f}
"""
        
        if self.periwound_edge_confidence_map:
            tissue_values = list(self.periwound_edge_confidence_map.values())
            stats += f"""

periwound Confidence Map:
- Total faces: {len(self.periwound_edge_confidence_map)}
- Min confidence: {np.min(tissue_values):.3f}
- Max confidence: {np.max(tissue_values):.3f}
- Mean confidence: {np.mean(tissue_values):.3f}
- Median confidence: {np.median(tissue_values):.3f}
- Std dev: {np.std(tissue_values):.3f}
"""
        
        if self.overlapping_faces:
            stats += f"""

Overlapping Faces (seen in both wound and periwound):
- Total overlapping: {len(self.overlapping_faces)}
"""
            
            if self.overlapping_wound_counts:
                wound_counts = list(self.overlapping_wound_counts.values())
                tissue_counts = list(self.overlapping_tissue_counts.values())
                stats += f"""
- Wound visibility: min={np.min(wound_counts)}, max={np.max(wound_counts)}, mean={np.mean(wound_counts):.2f}
- Tissue visibility: min={np.min(tissue_counts)}, max={np.max(tissue_counts)}, mean={np.mean(tissue_counts):.2f}
"""
        
        stats += """

Color Legend (Viridis):
- Dark Purple: Low confidence (0.0)
- Teal: Medium confidence (0.5)
- Yellow: High confidence (1.0)
"""
        
        return stats

    def _build_feature_colormap_assets(self,
                                        face_indices: List[int],
                                        values: np.ndarray,
                                        feature_name: str,
                                        generate_artifacts: bool = True
                                        ) -> Tuple[Dict[int, np.ndarray], Optional[np.ndarray], Optional[str]]:
        """Create per-face colors, an optional gradient image, and stats text for a feature."""
        if not face_indices:
            return {}, None, None

        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return {}, None, None

        finite_mask = np.isfinite(values)
        if not np.all(finite_mask):
            if np.any(finite_mask):
                fill_value = np.nanmedian(values[finite_mask])
                values = np.where(finite_mask, values, fill_value)
            else:
                values = np.zeros_like(values)

        style = self._get_feature_visual_style(feature_name)

        if style.get("value_range") is not None:
            vmin, vmax = style["value_range"]
        else:
            p_low, p_high = style.get("percentiles", (2.0, 98.0))
            vmin, vmax = np.percentile(values, [p_low, p_high])
            if np.isclose(vmin, vmax):
                spread = max(abs(vmin), 1.0)
                vmin -= 0.5 * spread
                vmax += 0.5 * spread

        diverging = style.get("diverging", False)
        if diverging:
            abs_max = style.get("abs_max")
            if abs_max is None or abs_max <= 0:
                abs_max = max(abs(vmin), abs(vmax), 1e-9)
            vmin, vmax = -abs_max, abs_max
            norm_vals = (values / abs_max + 1.0) * 0.5
        else:
            norm_vals = (values - vmin) / (vmax - vmin + 1e-9)

        norm_vals = np.clip(norm_vals, 0.0, 1.0)
        palette = style["palette"].astype(np.float32)
        colors = self._interpolate_palette(norm_vals, palette)
        face_color_map = {int(face_idx): colors[i] for i, face_idx in enumerate(face_indices)}

        gradient_image = None
        stats_text = None
        if generate_artifacts:
            label_specs = []
            if diverging:
                def normalize_div(val: float) -> float:
                    if abs_max <= 1e-9:
                        return 0.5
                    return np.clip((val / abs_max + 1.0) * 0.5, 0.0, 1.0)

                label_specs.append((normalize_div(-abs_max), f"{-abs_max:.3f}"))
                label_specs.append((normalize_div(0.0), "0"))
                label_specs.append((normalize_div(abs_max), f"{abs_max:.3f}"))
            else:
                def normalize_lin(val: float) -> float:
                    return np.clip((val - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0)

                label_specs.append((normalize_lin(vmin), f"{vmin:.3f}"))
                mid_val = (vmin + vmax) * 0.5
                label_specs.append((normalize_lin(mid_val), f"{mid_val:.3f}"))
                label_specs.append((normalize_lin(vmax), f"{vmax:.3f}"))

            gradient_image = self._create_gradient_image(palette, label_specs)
            stats_text = self._format_feature_stats(feature_name, values, (vmin, vmax), style)

        return face_color_map, gradient_image, stats_text

    def _get_feature_visual_style(self, feature_name: str) -> Dict[str, Any]:
        """Return palette and normalization hints for a feature name."""
        default_palette = np.array([
            [68, 1, 84],
            [58, 82, 139],
            [32, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
        ], dtype=np.float32)

        style = {
            "palette": default_palette,
            "percentiles": (2.0, 98.0),
            "value_range": None,
            "diverging": False,
            "abs_max": None,
            "description": feature_name,
            "unit": None,
        }

        name = feature_name.lower()

        if name.startswith("height_dev"):
            style.update({
                "palette": np.array([[0, 32, 255], [255, 255, 255], [255, 40, 0]], dtype=np.float32),
                "diverging": False,
                "description": "Height deviation",
                "unit": "m",
            })
        elif name == "r_wound":
            style.update({
                "palette": np.array([[40, 60, 120], [255, 200, 0], [255, 40, 0]], dtype=np.float32),
                "value_range": (0.0, 1.0),
                "description": "Wound visibility ratio",
            })
        elif name == "r_tissue":
            style.update({
                "palette": np.array([[30, 20, 60], [60, 200, 140], [0, 255, 200]], dtype=np.float32),
                "value_range": (0.0, 1.0),
                "description": "Tissue visibility ratio",
            })
        elif name == "normal_dev":
            style.update({
                "palette": np.array([[0, 40, 130], [0, 200, 80], [255, 255, 0]], dtype=np.float32),
                "value_range": (0.0, float(np.pi / 2)),
                "description": "Normal deviation",
                "unit": "rad",
            })
        elif name == "lab_b":
            style.update({
                "palette": np.array([[0, 70, 200], [245, 245, 245], [255, 200, 0]], dtype=np.float32),
                "diverging": True,
                "abs_max": 50.0,
                "description": "LAB b channel",
            })
        elif name in ("dr", "db"):
            style.update({
                "palette": np.array([[0, 120, 255], [245, 245, 245], [255, 60, 60]], dtype=np.float32),
                "diverging": True,
                "abs_max": 1.5,
                "description": f"YDrDb {feature_name.upper()} component",
            })
        elif name == "r":
            style.update({
                "palette": np.array([[0, 0, 0], [255, 80, 80]], dtype=np.float32),
                "value_range": (0.0, 1.0),
                "description": "Mean red channel",
            })
        elif name == "g":
            style.update({
                "palette": np.array([[0, 0, 0], [90, 255, 120]], dtype=np.float32),
                "value_range": (0.0, 1.0),
                "description": "Mean green channel",
            })
        elif name == "b":
            style.update({
                "palette": np.array([[0, 0, 0], [80, 160, 255]], dtype=np.float32),
                "value_range": (0.0, 1.0),
                "description": "Mean blue channel",
            })
        elif name == "curvature":
            style.update({
                "palette": np.array([[0, 80, 255], [240, 240, 240], [255, 50, 0]], dtype=np.float32),
                "diverging": True,
                "description": "Scale-adaptive curvature",
                "unit": "m (Laplacian)",
            })

        return style

    def _interpolate_palette(self, norm_values: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """Linearly interpolate between palette colors for normalized values."""
        norm_values = np.asarray(norm_values, dtype=np.float64)
        if palette is None or len(palette) == 0:
            return np.tile(np.array([[200, 200, 200]], dtype=np.uint8), (len(norm_values), 1))

        if len(palette) == 1:
            return np.tile(palette[0].astype(np.uint8), (len(norm_values), 1))

        scaled = np.clip(norm_values, 0.0, 1.0) * (len(palette) - 1)
        lower_idx = np.floor(scaled).astype(int)
        upper_idx = np.clip(lower_idx + 1, 0, len(palette) - 1)
        frac = (scaled - lower_idx)[:, None]

        lower_colors = palette[lower_idx]
        upper_colors = palette[upper_idx]
        colors = lower_colors + (upper_colors - lower_colors) * frac
        return np.clip(colors, 0, 255).astype(np.uint8)

    def _create_gradient_image(self,
                                palette: np.ndarray,
                                value_labels: Optional[List[Tuple[float, str]]] = None,
                                width: int = 256,
                                height: int = 48) -> Optional[np.ndarray]:
        """Create a horizontal gradient image visualizing the palette with optional tick labels."""
        if palette is None or len(palette) == 0:
            return None

        ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)
        colors = self._interpolate_palette(ramp, palette)
        gradient = np.tile(colors[None, :, :], (height, 1, 1)).astype(np.uint8)

        if value_labels:
            for norm_pos, label in value_labels:
                norm = np.clip(norm_pos, 0.0, 1.0)
                x = int(norm * (width - 1))
                gradient[: height // 3, max(0, x - 1): min(width, x + 2)] = 255
                self._draw_text_on_image(gradient, label, x, height - 12)

        return gradient

    def _draw_text_on_image(self,
                             image: np.ndarray,
                             text: str,
                             center_x: int,
                             baseline_y: int,
                             color: Tuple[int, int, int] = (255, 255, 255)) -> None:
        """Render simple pixel text onto an RGB image (no external dependencies)."""
        if image.size == 0 or not text:
            return

        font = {
            "0": ["111", "101", "101", "101", "111"],
            "1": ["010", "110", "010", "010", "111"],
            "2": ["111", "001", "111", "100", "111"],
            "3": ["111", "001", "111", "001", "111"],
            "4": ["101", "101", "111", "001", "001"],
            "5": ["111", "100", "111", "001", "111"],
            "6": ["111", "100", "111", "101", "111"],
            "7": ["111", "001", "010", "010", "010"],
            "8": ["111", "101", "111", "101", "111"],
            "9": ["111", "101", "111", "001", "111"],
            "-": ["000", "010", "010", "000", "000"],
            ".": ["000", "000", "000", "000", "010"],
            " ": ["000", "000", "000", "000", "000"],
        }
        default_char = ["111", "101", "101", "101", "111"]

        char_width = 4  # 3 px glyph + 1 px spacing
        total_width = len(text) * char_width
        start_x = max(0, min(image.shape[1] - total_width, center_x - total_width // 2))
        y0 = max(0, min(image.shape[0] - 5, baseline_y))

        for idx, char in enumerate(text):
            glyph = font.get(char, default_char)
            for row, pattern in enumerate(glyph):
                for col, bit in enumerate(pattern):
                    if bit == "1":
                        x = start_x + idx * char_width + col
                        y = y0 + row
                        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                            image[y, x] = color

    def _format_feature_stats(self,
                                feature_name: str,
                                values: np.ndarray,
                                color_range: Tuple[float, float],
                                style: Dict[str, Any]) -> str:
        """Pretty-print statistics for a feature."""
        values = np.asarray(values, dtype=np.float64)
        stats = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
        }

        descriptor = style.get("description") or feature_name
        unit = style.get("unit")
        header = f"{descriptor} [{feature_name}]"
        if unit:
            header += f" ({unit})"

        lines = [
            header,
            f"Color range: {color_range[0]:.4f} → {color_range[1]:.4f}",
            f"Min / Max : {stats['min']:.4f} / {stats['max']:.4f}",
            f"Mean ± Std : {stats['mean']:.4f} ± {stats['std']:.4f}",
            f"Median (P25/P75): {stats['median']:.4f} ({stats['p25']:.4f} / {stats['p75']:.4f})",
        ]

        if style.get("diverging"):
            lines.append("Scale centered at zero (diverging colormap).")

        return "\n".join(lines)

    def _log_faces_mesh_rr_with_colors(self, rr, path: str, face_set: Set[int], 
                                       face_color_map: Dict[int, np.ndarray]) -> None:
        """Log a mesh with per-face colors for visualization in Rerun.
        
        Args:
            rr: Rerun module
            path: Path for logging in Rerun
            face_set: Set of face indices
            face_color_map: Dict mapping face index to RGB color
        """
        if not face_set:
            return
        
        # Extract submesh
        submesh_verts, submesh_faces, _ = self.extract_submesh_from_faces(face_set)
        
        if len(submesh_verts) == 0:
            return
        
        # Create vertex colors from face colors
        vertex_colors = np.ones((len(submesh_verts), 3), dtype=np.uint8) * 128
        
        # Map face colors to vertex colors (average of face colors for each vertex)
        vertex_face_count = np.zeros(len(submesh_verts), dtype=np.int32)
        vertex_color_sum = np.zeros((len(submesh_verts), 3), dtype=np.float32)
        
        # Build mapping from global face indices to local submesh face indices
        face_list = sorted(list(face_set))
        vertex_remap = {}
        unique_verts = set()
        for face_idx in face_list:
            face = self.faces[face_idx]
            unique_verts.update(face)
        
        vertex_list = sorted(list(unique_verts))
        for new_idx, old_idx in enumerate(vertex_list):
            vertex_remap[old_idx] = new_idx
        
        # Accumulate colors
        for submesh_face_idx, global_face_idx in enumerate(face_list):
            if global_face_idx not in face_color_map:
                continue
            
            color = face_color_map[global_face_idx].astype(np.float32)
            global_face = self.faces[global_face_idx]
            
            for vertex_id in global_face:
                local_vertex_id = vertex_remap.get(vertex_id)
                if local_vertex_id is not None:
                    vertex_color_sum[local_vertex_id] += color
                    vertex_face_count[local_vertex_id] += 1
        
        # Average colors
        for i in range(len(submesh_verts)):
            if vertex_face_count[i] > 0:
                vertex_colors[i] = np.clip(vertex_color_sum[i] / vertex_face_count[i], 0, 255).astype(np.uint8)
        
        rr.log(path, rr.Mesh3D(
            vertex_positions=submesh_verts.astype(np.float32),
            triangle_indices=submesh_faces,
            vertex_colors=vertex_colors
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
            'spectators_with_extruded_masks': sum(1 for s in self.spectators.values() if s.extruded_mask_loaded),
            'topology_computed': self.topology_computed
        }
        
        if self.faces_processed:
            stats['wound_faces_count'] = len(self.wound_face_indices)
            
            if self.face_visibility_count:
                counts = list(self.face_visibility_count.values())
                stats['visibility_stats'] = {
                    'min': min(counts),
                    'max': max(counts),
                    'mean': np.mean(counts),
                    'median': np.median(counts)
                }
        
        if self.periwound_edge_faces:
            stats['periwound_edge_faces_count'] = len(self.periwound_edge_faces)
            
            if self.periwound_edge_visibility_count:
                counts = list(self.periwound_edge_visibility_count.values())
                stats['periwound_edge_visibility_stats'] = {
                    'min': min(counts),
                    'max': max(counts),
                    'mean': np.mean(counts),
                    'median': np.median(counts)
                }
            
            if self.periwound_edge_mean_color and self.periwound_edge_color_std:
                # Collect color statistics
                mean_colors = np.array([self.periwound_edge_mean_color[f] for f in self.periwound_edge_faces if f in self.periwound_edge_mean_color])
                std_colors = np.array([self.periwound_edge_color_std[f] for f in self.periwound_edge_faces if f in self.periwound_edge_color_std])
                
                stats['periwound_edge_color_stats'] = {
                    'mean_color': {
                        'R': float(np.mean(mean_colors[:, 0])) if len(mean_colors) > 0 else 0,
                        'G': float(np.mean(mean_colors[:, 1])) if len(mean_colors) > 0 else 0,
                        'B': float(np.mean(mean_colors[:, 2])) if len(mean_colors) > 0 else 0
                    },
                    'color_std': {
                        'R': float(np.mean(std_colors[:, 0])) if len(std_colors) > 0 else 0,
                        'G': float(np.mean(std_colors[:, 1])) if len(std_colors) > 0 else 0,
                        'B': float(np.mean(std_colors[:, 2])) if len(std_colors) > 0 else 0
                    }
                }
        
        # Add confidence map statistics
        if self.wound_confidence_map or self.periwound_edge_confidence_map:
            stats['confidence_maps'] = {
                'wound_faces_with_confidence': len(self.wound_confidence_map),
                'periwound_edge_faces_with_confidence': len(self.periwound_edge_confidence_map),
                'overlapping_faces': len(self.overlapping_faces)
            }
            
            if self.wound_confidence_map:
                wound_conf_values = list(self.wound_confidence_map.values())
                stats['confidence_maps']['wound_confidence_stats'] = {
                    'min': float(np.min(wound_conf_values)),
                    'max': float(np.max(wound_conf_values)),
                    'mean': float(np.mean(wound_conf_values)),
                    'median': float(np.median(wound_conf_values))
                }
            
            if self.periwound_edge_confidence_map:
                tissue_conf_values = list(self.periwound_edge_confidence_map.values())
                stats['confidence_maps']['periwound_edge_confidence_stats'] = {
                    'min': float(np.min(tissue_conf_values)),
                    'max': float(np.max(tissue_conf_values)),
                    'mean': float(np.mean(tissue_conf_values)),
                    'median': float(np.median(tissue_conf_values))
                }
        
        return stats


def main():
    """
    Example usage of mesh-based wound detection with periwound detection.
    """
    
    # Example paths (adjust to your data)
    mesh_path = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse\dense\poisson_smoothed.ply")
    model_path = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse\dense")
    mask_dir = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse\dense\masks_wound_only_undistorted")
    extruded_mask_dir = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse\dense\masks_extruded_wound_only_undistorted")
    tissue_mask_dir = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse\dense\masks_undistorted")
    output_path = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse\dense\wound_mesh.ply")
    output_path_corrected = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse\dense\wound_mesh_corrected.ply")
    
    try:
        # Initialize model
        wound_model = WoundModel3DMesh()
        
        # Load mesh
        print("=== Loading Mesh ===")
        wound_model.load_mesh(mesh_path, compute_normals=True)
        
        # Load COLMAP cameras
        print("\n=== Loading COLMAP Cameras ===")
        wound_model.load_colmap_cameras(model_path)
        
        # Load wound masks
        print("\n=== Loading Wound Masks ===")
        wound_model.load_masks(mask_dir)
        
        # Load extruded masks for periwound detection
        print("\n=== Loading Extruded Masks ===")
        wound_model.load_extruded_masks(extruded_mask_dir)
        
        # Process wound segmentation
        print("\n=== Processing Wound Segmentation ===")
        wound_model.process_wound_segmentation()
        
        # Run semantic classification (feature-based, 3-class: tissue, periwound, wound)
        # Note: periwound and tissue are automatically detected within run_semantic_classification()
        print("\n=== Semantic Classification (Gaussian + MRF) ===")
        wound_model.run_semantic_classification(
            ring_sizes=[1, 3],
            confidence_method="entropy",
            lambda_smooth=2.0,
            use_centroid_features=True,
            use_height_deviation=True,
            use_normal_deviation=False,
            use_curvature=True,
            color_space="lab_b",
            mrf_method="mean_field",
            undistorted_mask_dir=tissue_mask_dir
        )

        
        # Plot visibility histograms
        print("\n=== Plotting Visibility Histograms ===")
        output_dir = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse\dense")
        wound_model.plot_visibility_histograms(
            output_dir=output_dir,
            show_plot=False  # Set to True to display interactively
        )
        
        # Get statistics
        print("\n=== Statistics ===")
        stats = wound_model.get_statistics()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        
        # Get final wound mesh with mesh-specific filtering
        # print("\n=== Extracting Wound Mesh ===")
        # wound_vertices, wound_faces, wound_colors = wound_model.get_wound_mesh(
        #     min_visibility=3,  # Require visibility from multiple cameras
        #     use_connectivity=True,  # Extract connected components
        #     min_component_faces=100,  # Minimum size for valid component
        #     use_closing=True,  # Apply morphological closing
        #     closing_iterations=5,  # Number of closing iterations
        #     closing_kernel_size=2,  # Number of adjacency layers per iteration (1-3 typical)
        #     use_fill_holes=False,  # We'll use trimesh instead in correction step
        #     hole_size=-1.0
        # )
        
        # Save the initial wound mesh
        # print("\n=== Saving Meshes ===")
        # wound_model.save_wound_mesh(output_path, wound_vertices, wound_faces, wound_colors)
        # print(f"Saved initial wound mesh to: {output_path}")
        
        # Launch visualization with all results
        print("\n=== Launching Visualization ===")
        wound_model.visualize_with_rerun("WoundModel3DMesh_TissueEdge", min_periwound_edge_visibility=10)
        
        print("\n=== Processing Complete ===")
        
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
