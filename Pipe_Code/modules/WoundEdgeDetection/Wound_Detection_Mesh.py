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
import importlib
import numpy as np
import time
import traceback
from dataclasses import dataclass
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
from Pipe_Code.modules.Mesh_postprocessing.MeshSimplification import MeshSimplifier


@dataclass
class Faces:
    """Per-face container for geometry and optional dynamic segmentation values."""

    face_index: int
    vertex_indices: Tuple[int, int, int]
    centroid: np.ndarray
    face_normal: Optional[np.ndarray]
    color_vector: Optional[np.ndarray]
    neighbor_face_indices: Set[int]

    # Populated once raycasting/semantic classification is executed.
    visibility_wound: Optional[int] = None
    visibility_periwound: Optional[int] = None
    visibility_tissue: Optional[int] = None
    r_wound: Optional[float] = None
    r_periwound: Optional[float] = None
    r_tissue: Optional[float] = None
    wound_prob: Optional[float] = None
    periwound_prob: Optional[float] = None
    posterior_log_ratio: Optional[float] = None
    posterior_log_ratio_strength: Optional[float] = None
    label: Optional[str] = None
    confidence: Optional[float] = None


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
        self.face_objects: List[Faces] = []  # indexed by face index
        self._raycast_scene = None
        self._raycast_mesh_id = None

        # Mesh topology (using trimesh/O3D adjacency)
        self.edge_adjacency = {}  # edge -> list of face indices
        self.face_adjacency = {}  # face -> list of adjacent faces
        
        # Camera and image management
        self.spectators = {}  # Dict: image_name -> Spectator
        self.cameras = {}
        self.images = {}
        
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
        self.semantic_posteriors = {}  # face_idx -> np.array([p_periwound, p_wound])
        self.semantic_posteriors_before_mrf = {}  # face_idx -> posteriors before MRF constraints
        self.semantic_class_names = []  # n-class classification
        self.semantic_log_ratio_values = {}  # face_idx -> log(p_wound / p_periwound)
        self.semantic_log_ratio_strength_values = {}  # face_idx -> abs(log(p_wound / p_periwound))
        self.semantic_log_ratio_labels = {}  # face_idx -> wound/periwound/uncertain
        self.semantic_log_ratio_thresholds = {"wound": 0.35, "periwound": 0.35}
        

        # Current filtering result (indices into self.faces)
        self.current_wound_faces = set()  # Current wound face indices after filtering
        self.current_periwound_faces = set()  # Current periwound face indices after filtering
        self.periwound_thickness = 2  # Number of adjacency layers used for the periwound shell
        

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
        self._face_distance_cache = {}
        
        # --- DEBUG_VIZ (safe to delete) ---
        # Stores debug-only semantic selection/visualization state.
        self.debug_gmm_sample_indices = {"periwound": [], "wound": []}
        self.debug_gmm_face_indices = []
        self.debug_confidence_vertex_colors = None
        self.debug_confidence_face_values = {}
        self.debug_wound_posterior_map = {}
        self.debug_tissue_posterior_map = {}
    
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

        # Build per-face data containers once geometry and adjacency are available.
        self._initialize_face_objects()

        
        self.model_loaded = True
    

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


    def _initialize_face_objects(self) -> None:
        """Create one Faces object per face and reset all dynamic per-face values."""
        if self.faces is None:
            self.face_objects = []
            return

        # Use geometry-derived centroids for all faces; avoid generator-to-ndarray pitfalls.
        centroids = self._get_face_centroids()

        self.face_objects = []
        for face_idx, face in enumerate(self.faces):
            color_vector = None
            if self.vertex_colors is not None and len(self.vertex_colors) > 0:
                color_vector = np.mean(self.vertex_colors[face], axis=0)

            normal = None
            if self.face_normals is not None and face_idx < len(self.face_normals):
                normal = self.face_normals[face_idx]

            self.face_objects.append(
                Faces(
                    face_index=int(face_idx),
                    vertex_indices=(int(face[0]), int(face[1]), int(face[2])),
                    centroid=centroids[face_idx],
                    face_normal=normal,
                    color_vector=color_vector,
                    neighbor_face_indices=set(self.face_adjacency.get(face_idx, [])),
                )
            )

        # Reset all dynamic per-face state whenever face objects are recreated.
        for face_obj in self.face_objects:
            face_obj.visibility_wound = None
            face_obj.visibility_periwound = None
            face_obj.visibility_tissue = None
            face_obj.r_wound = None
            face_obj.r_periwound = None
            face_obj.r_tissue = None
            face_obj.wound_prob = None
            face_obj.periwound_prob = None
            face_obj.label = None
            face_obj.confidence = None

    def _register_class_visibility(self, class_name: str, face_indices: List[int]) -> None:
        """Register raycast hits for a semantic class and update per-face fields."""
        for face_idx in face_indices:
            if face_idx < 0 or face_idx >= len(self.faces):
                continue

            if face_idx < len(self.face_objects):
                face_obj = self.face_objects[face_idx]
                if class_name == "wound":
                    face_obj.visibility_wound = int(face_obj.visibility_wound or 0) + 1
                elif class_name == "periwound":
                    face_obj.visibility_periwound = int(face_obj.visibility_periwound or 0) + 1
                elif class_name == "tissue":
                    face_obj.visibility_tissue = int(face_obj.visibility_tissue or 0) + 1


    
    def _compute_mesh_topology(self):
        """Compute mesh topology using Open3D and numpy.
        
        Builds edge adjacency and face adjacency from mesh data.
        """
        t0 = time.perf_counter()
        print("Computing mesh topology...")

        # Try trimesh for fast topology extraction
        try:
            import trimesh

            vertex_colors = None
            if self.vertex_colors is not None and len(self.vertex_colors) == len(self.vertices):
                vc = np.asarray(self.vertex_colors, dtype=np.float64)
                vc = np.clip(vc, 0.0, 1.0)
                vertex_colors = (vc * 255.0).astype(np.uint8)

            mesh = trimesh.Trimesh(
                vertices=self.vertices,
                faces=self.faces,
                vertex_colors=vertex_colors,
                process=False,
                maintain_order=True
            )
            self._trimesh_cache = mesh
            
            # Requested trimesh references:
            # FA: face adjacency index pairs
            # FA_edges: shared edge vertices for each FA pair
            FA = np.asarray(mesh.face_adjacency, dtype=np.int64)
            FA_edges = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)

            # Build face adjacency
            self.face_adjacency = defaultdict(list)
            for f0, f1 in FA:
                self.face_adjacency[int(f0)].append(int(f1))
                self.face_adjacency[int(f1)].append(int(f0))

            # Build edge adjacency from shared edges.
            edge_faces = defaultdict(set)
            for i, edge in enumerate(FA_edges):
                if i >= len(FA):
                    break
                f0, f1 = int(FA[i][0]), int(FA[i][1])
                edge_key = tuple(sorted((int(edge[0]), int(edge[1]))))
                edge_faces[edge_key].add(f0)
                edge_faces[edge_key].add(f1)

            # --- OPTIMIZED: Add boundary edges with O(1) per-edge lookup ---
            edges_unique = np.asarray(mesh.edges_unique, dtype=np.int64)
            counts = np.bincount(
                np.asarray(mesh.edges_unique_inverse, dtype=np.int64),
                minlength=len(edges_unique),
            )
            boundary_edge_ids = np.flatnonzero(counts == 1)
            
            if boundary_edge_ids.size > 0:
                # Pre-build a single edge-to-face dictionary from ALL faces
                # (one pass instead of O(boundary_edges × num_faces))
                boundary_edges_to_find = set()
                for edge_id in boundary_edge_ids:
                    boundary_edge = tuple(sorted((
                        int(edges_unique[edge_id][0]),
                        int(edges_unique[edge_id][1]),
                    )))
                    # Skip if already known from FA_edges (shouldn't happen, but be safe)
                    if boundary_edge not in edge_faces:
                        boundary_edges_to_find.add(boundary_edge)
                
                if boundary_edges_to_find:
                    # One pass over all faces to build edge -> face mappings for missing edges
                    face_arr = np.asarray(self.faces, dtype=np.int64)
                    for face_idx in range(len(face_arr)):
                        v0, v1, v2 = int(face_arr[face_idx, 0]), int(face_arr[face_idx, 1]), int(face_arr[face_idx, 2])
                        e0 = (v0, v1) if v0 < v1 else (v1, v0)
                        e1 = (v1, v2) if v1 < v2 else (v2, v1)
                        e2 = (v2, v0) if v2 < v0 else (v0, v2)
                        for e_key in (e0, e1, e2):
                            if e_key in boundary_edges_to_find:
                                edge_faces[e_key].add(face_idx)
                                boundary_edges_to_find.discard(e_key)
                                if not boundary_edges_to_find:
                                    break
                        if not boundary_edges_to_find:
                            break

            self.edge_adjacency = defaultdict(list)
            for edge_key, faces_for_edge in edge_faces.items():
                self.edge_adjacency[edge_key] = sorted(list(faces_for_edge))

            elapsed = time.perf_counter() - t0
            print(
                f"Topology computed (trimesh) in {elapsed:.2f}s: "
                f"{len(self.edge_adjacency)} edges, "
                f"{len(self.face_adjacency)} faces with neighbors"
            )
            self.topology_computed = True
            return
        except Exception as e:
            print(f"Trimesh not available or failed ({e}), falling back to manual topology")

    
       

    
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

        # Reset wound visibility before processing all spectators.
        for face_obj in self.face_objects:
            face_obj.visibility_wound = None
        

        # Process each spectator
        for spectator_name, spectator in self.spectators.items():
            if not spectator.mask_loaded:
                continue
            
            print(f"Processing spectator: {spectator_name}")
            
            # Filter faces through mask using Open3D raycasting
            self._run_open3d_filter(spectator, max_mask_rays=max_mask_rays)
    
            
            print(f"  Visible faces: {len(spectator.visible_face_indices)}")

        for face_obj in self.face_objects:
            face_obj.r_wound = face_obj.visibility_wound/len(self.spectators) if face_obj.visibility_wound is not None else 0.0
        
        
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
        self._register_class_visibility("wound", visible_faces)
        for face_idx in visible_faces:
            face = self.faces[face_idx]
            spectator.visible_vertex_indices.update(face)
        spectator.faces_processed = True

    def _filter_faces_with_open3d_raycast(self,
                                          spectator: Spectator,
                                          max_mask_rays: int,
                                          mask_data: Optional[np.ndarray] = None) -> List[int]:
        """Use Open3D raycasting to find faces visible through a mask.

        Args:
            spectator: Camera/spectator with intrinsics and extrinsics.
            max_mask_rays: Maximum rays to cast from active mask pixels.
            mask_data: Optional custom mask. If None, uses spectator.mask_data.
        """
        try:
            import open3d as o3d
        except ImportError:
            print("Open3D not available, cannot perform raycasting filter")
            return []

        active_mask = spectator.mask_data if mask_data is None else mask_data
        if active_mask is None or self.vertices is None or self.faces is None:
            return []

        if self._raycast_scene is None:
            mesh_legacy = o3d.geometry.TriangleMesh()
            mesh_legacy.vertices = o3d.utility.Vector3dVector(self.vertices)
            mesh_legacy.triangles = o3d.utility.Vector3iVector(self.faces)
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
            self._raycast_scene = o3d.t.geometry.RaycastingScene()
            self._raycast_mesh_id = self._raycast_scene.add_triangles(mesh_t)

        mask_coords = np.argwhere(active_mask > 0)
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

        for face_obj in self.face_objects:
            face_obj.visibility_periwound = None
        
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
            periwound_faces = self._filter_faces_with_open3d_raycast(
                spectator=spectator,
                max_mask_rays=max_mask_rays,
                mask_data=difference_mask,
            )

            # Accumulate periwound visibility across all spectators.
            self._register_class_visibility("periwound", periwound_faces)
        
            print(f"  Periwound faces detected: {len(periwound_faces)}")

        for face_obj in self.face_objects:
            face_obj.r_periwound = (
                face_obj.visibility_periwound / len(self.spectators)
                if face_obj.visibility_periwound is not None
                else 0.0
            )
    

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
        for face_obj in self.face_objects:
            face_obj.visibility_tissue = None
        
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
            tissue_faces = self._filter_faces_with_open3d_raycast(
                spectator=spectator,
                mask_data=tissue_mask,
                max_mask_rays=max_mask_rays
            )
            
            # Update global tissue tracking
            self._register_class_visibility("tissue", tissue_faces)
            
            print(f"  Tissue faces detected: {len(tissue_faces)}")
        
        # Store visibility and ratio directly from face objects.
        self.normal_tissue_visibility_count = {}
        self.normal_tissue_face_indices = set()
        for face_obj in self.face_objects:
            face_obj.r_tissue = (
                face_obj.visibility_tissue / len(self.spectators)
                if face_obj.visibility_tissue is not None
                else 0.0
            )
            if face_obj.visibility_tissue is not None and face_obj.visibility_tissue > 0:
                self.normal_tissue_visibility_count[face_obj.face_index] = int(face_obj.visibility_tissue)
                self.normal_tissue_face_indices.add(face_obj.face_index)
        
        print(f"Total normal tissue faces detected: {len(self.normal_tissue_face_indices)}")
        if self.normal_tissue_visibility_count:
            print(f"  Average visibility: {np.mean(list(self.normal_tissue_visibility_count.values())):.2f} cameras")


    def _get_face_centroids(self) -> np.ndarray:
        """Return per-face centroids as an (M, 3) array, computed lazily and cached."""
        if self._face_centroids_cache is not None:
            return self._face_centroids_cache

        if self.faces is None or self.vertices is None or len(self.faces) == 0:
            self._face_centroids_cache = np.zeros((0, 3), dtype=np.float64)
            return self._face_centroids_cache

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
                                             plane_normal: np.ndarray,
                                             plane_d: float,
                                             ring_sizes: List[int]) -> Dict[int, List[float]]:
        """Compute multi-scale height deviation for faces."""
        signed_dist = {
            face_obj.face_index: float(np.dot(face_obj.centroid, plane_normal) + plane_d)
            for face_obj in self.face_objects
        }
        deviations = {}
        for face_idx in face_indices:
            devs = []
            for ring in ring_sizes:
                neighbors = self._get_face_neighbors_ring(face_idx, ring)
                if not neighbors:
                    neighbors = {face_idx}
                vals = [signed_dist.get(n, 0.0) for n in neighbors]
                devs.append(float(np.mean(vals)))
            deviations[face_idx] = devs
        return deviations

    def _compute_mesh_curvature(self,
                                face_indices: List[int],
                                plane_normal: np.ndarray,
                                plane_d: float,
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
            plane_normal: Normal of the fitted tissue reference plane.
            plane_d: Offset d of the plane equation  n · x + d = 0.
            wound_r_thresh: r_wound threshold used to estimate wound scale.

        Returns:
            curvature: (len(face_indices),) array of curvature values.
        """
        # --- Estimate wound scale as adaptive ring radius ---
        n_wound_faces = sum(
            1
            for f in face_indices
            if 0 <= f < len(self.face_objects)
            and float(self.face_objects[f].r_wound or 0.0) > wound_r_thresh
        )
        # Heuristic: ring ~ sqrt(wound_faces) / 2, clamped to [2, 40]
        adaptive_ring = int(np.clip(np.sqrt(max(n_wound_faces, 1)) / 2.0, 2, 40))
        print(f"  Curvature adaptive ring = {adaptive_ring}  "
              f"(wound faces with r_wound > {wound_r_thresh}: {n_wound_faces})")

        # Signed height of every face relative to the tissue plane.
        signed_height = {
            face_obj.face_index: float(np.dot(face_obj.centroid, plane_normal) + plane_d)
            for face_obj in self.face_objects
        }

        curvature = np.zeros(len(face_indices), dtype=np.float64)
        for i, face_idx in enumerate(face_indices):
            neighbors = self._get_face_neighbors_ring(face_idx, adaptive_ring)
            if not neighbors or len(neighbors) <= 1:
                curvature[i] = 0.0
                continue
            neighbor_heights = np.array([signed_height.get(n, 0.0) for n in neighbors], dtype=np.float64)
            # Laplacian curvature: own height minus neighbourhood mean
            curvature[i] = signed_height.get(face_idx, 0.0) - np.mean(neighbor_heights)

        return curvature

    def _compute_normal_deviation(self,
                                  face_indices: List[int],
                                  plane_normal: np.ndarray) -> Dict[int, float]:
        """Compute normal deviation (angle in radians) from plane normal per face."""
        plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-9)
        deviations = {}
        for face_idx in face_indices:
            normal = None
            if 0 <= face_idx < len(self.face_objects):
                normal = self.face_objects[face_idx].face_normal
            if normal is None:
                deviations[face_idx] = 0.0
                continue
            n = np.asarray(normal, dtype=np.float64)
            n = n / (np.linalg.norm(n) + 1e-9)
            dot = float(np.clip(abs(np.dot(n, plane_normal)), -1.0, 1.0))
            deviations[face_idx] = float(np.arccos(dot))
        return deviations

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
                               periwound_candidate_top_percent: float = 60.0,
                               ring_sizes: List[int] = None,
                               use_ratio_features: bool = True,
                               use_height_deviation: bool = True,
                               use_normal_deviation: bool = True,
                               use_curvature: bool = True,
                               use_centroid_features: bool = True,
                               color_space: str = "lab_b") -> Tuple[np.ndarray, List[int]]:
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

        face_indices = list(range(len(self.face_objects))) #local defined face indices for feature building

        # Extract ratio arrays once to avoid repeated Python attribute lookups in loops.
        r_periwound_values = np.fromiter(
            (float(face_obj.r_periwound or 0.0) for face_obj in self.face_objects),
            dtype=np.float64,
            count=len(self.face_objects),
        )
        r_wound_values = np.fromiter(
            (float(face_obj.r_wound or 0.0) for face_obj in self.face_objects),
            dtype=np.float64,
            count=len(self.face_objects),
        )
    
        # Derive periwound_candidate_periwound_thresh from the r_periwound distribution
        # so that the top `periwound_candidate_top_percent`% of faces pass the threshold
        r_periwound_nonzero = r_periwound_values[r_periwound_values > 0.0]
        periwound_candidate_periwound_thresh = max(float(np.percentile(r_periwound_nonzero, 100.0 - periwound_candidate_top_percent)), 0.05) if len(r_periwound_nonzero) > 0 else 0.0
        
        plt.hist(r_periwound_values, bins=50, color='blue', alpha=0.7)
        plt.axvline(periwound_candidate_periwound_thresh, color='red', linestyle='dashed', label=f'Threshold: {periwound_candidate_periwound_thresh:.4f}')
        plt.title('Distribution of r_periwound values')
        plt.xlabel('r_periwound')
        plt.ylabel('Frequency')
        plt.legend()
        print(
            f"  periwound_candidate_periwound_thresh: "
            f"{periwound_candidate_periwound_thresh:.4f}"
        )


        # Select periwound seed faces for plane fit
        periwound_mask = (
            (r_periwound_values > periwound_candidate_periwound_thresh)
            & (r_wound_values < periwound_candidate_wound_thresh)
        )
        periwound_candidates = np.flatnonzero(periwound_mask).tolist()
        self.semantic_periwound_candidate_faces = set(periwound_candidates)
        print(f"  periwound_candidates: {len(periwound_candidates)} / {len(face_indices)} faces")

        plane_seed = periwound_candidates if periwound_candidates else face_indices
        plane_points = np.asarray(
            [self.face_objects[f].centroid for f in plane_seed],
            dtype=np.float64,
        )
        plane_face_indices = periwound_candidates if periwound_candidates else face_indices
        plane_face_normals = self.face_normals[plane_face_indices] if len(plane_face_indices) > 0 else None
        plane_normal, plane_d = self._fit_plane(plane_points, face_normals=plane_face_normals)

        # Build feature list based on toggles
        if use_ratio_features:
            feature_names = ["r_wound", "r_periwound"]
        else:
            feature_names = []
        # Centroid position features
        if use_centroid_features:
            feature_names.extend(["centroid_x", "centroid_y", "centroid_z"])
        
        # Height deviation features
        if use_height_deviation:
            height_devs = self._compute_height_deviation_multiscale(
                face_indices, plane_normal, plane_d, ring_sizes
            )
            for ring in ring_sizes:
                feature_names.append(f"height_dev_r{ring}")
        
        # Normal deviation feature
        if use_normal_deviation:
            normal_dev = self._compute_normal_deviation(face_indices, plane_normal)
            feature_names.append("normal_dev")

        # Scale-adaptive curvature feature
        if use_curvature:
            curvature_vals = self._compute_mesh_curvature(
                face_indices, plane_normal, plane_d
            )
            feature_names.append("curvature")

        # Color features based on selected color space
        if color_space != "none":
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
            r_wound = float(self.face_objects[face_idx].r_wound or 0.0)
            r_periwound = float(self.face_objects[face_idx].r_periwound or 0.0)
            
            if use_ratio_features:
                feat = [r_wound, r_periwound]
            else:
                feat = []

            
            if use_centroid_features:
                # Add centroid position features
                centroid = self.face_objects[face_idx].centroid
                feat.extend([float(centroid[0]), float(centroid[1]), float(centroid[2])])

            # Add height deviation features
            if use_height_deviation:
                feat.extend(height_devs[face_idx])
            
            # Add normal deviation feature
            if use_normal_deviation:
                feat.append(float(normal_dev.get(face_idx, 0.0)))

            # Add curvature feature
            if use_curvature:
                feat.append(float(curvature_vals[i]))

            # Add color features based on selected color space
            if color_space != "none":
                mean_color_rgb = self.face_objects[face_idx].color_vector
                if mean_color_rgb is None:
                    mean_color_rgb = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                
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
        
        
        return self.semantic_features, self.semantic_face_indices

    def _fit_gaussian_models(self,
                             features: np.ndarray,
                             face_indices: List[int],
                             r_wound_thresh: float = 0.3,
                             r_periwound_thresh: float = 0.3,
                             regularization: float = 1e-3) -> Dict[str, Dict[str, np.ndarray]]:
        """Fit Gaussian models for 2 classes: periwound and wound.
        
        Uses visibility ratios to assign class labels:
        - Wound class: faces with r_wound > r_wound_thresh
        - Periwound class: faces with r_periwound > r_periwound_thresh (area around wound)
        
        The periwound class represents the boundary tissue around the wound,
        detected from the difference between extruded and regular wound masks.
        """
        class_samples = {"periwound": [], "wound": []}
        class_sample_indices = {"periwound": [], "wound": []}

        # Classify faces based on visibility ratios
        for feat_idx, face_idx in enumerate(face_indices):
            r_wound = float(self.face_objects[face_idx].r_wound or 0.0)
            r_periwound = float(self.face_objects[face_idx].r_periwound or 0.0)

            is_wound = r_wound > r_wound_thresh
            is_periwound = r_periwound > r_periwound_thresh
            
            # Priority: wound > periwound
            if is_wound:
                class_samples["wound"].append(features[feat_idx])
                class_sample_indices["wound"].append(face_idx)
            if is_periwound:
                class_samples["periwound"].append(features[feat_idx])
                class_sample_indices["periwound"].append(face_idx)
            # Faces not meeting any threshold are excluded from training
        
        print(f"  Class samples: periwound={len(class_sample_indices['periwound'])}, "
              f"wound={len(class_sample_indices['wound'])}")

        # --- DEBUG_VIZ (safe to delete) ---
        # Persist selected training sample face IDs for visualization.
        self.debug_gmm_sample_indices = {
            "periwound": list(class_sample_indices["periwound"]),
            "wound": list(class_sample_indices["wound"]),
        }
        self.debug_gmm_face_indices = list(face_indices)
        
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
                n_comp = 1
                print(f"Fitting GMM for class '{cls_name}' with {len(X)} samples and {n_comp} components...")
                gmm = GaussianMixture(
                    n_components=n_comp,
                    covariance_type="full",
                    reg_covar=regularization,
                    max_iter=100,
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
                                        models: Dict[str, Dict[str, np.ndarray]],
                                        class_prior_mode: str = "uniform") -> np.ndarray:
        """Compute posterior probabilities for each face.
        
        Args:
            features: Feature matrix (N, D).
            models: Fitted class models.
            class_prior_mode:
                - "sample_proportional": prior[k] ∝ n_k (current behavior)
                - "inverse_count": prior[k] ∝ 1 / n_k (class balancing)
                - "uniform": prior[k] is equal for all available classes
        """
        n = features.shape[0]
        class_names = self.semantic_class_names

        if len(class_names) == 0:
            raise ValueError(
                "semantic_class_names is empty. Set class names before posterior computation "
                "(e.g. ['periwound', 'wound'])."
            )

        if len(models) == 0:
            raise ValueError("No class models available for posterior computation.")

        log_likelihoods = np.full((n, len(class_names)), -1e12, dtype=np.float64)

        log_priors = np.zeros(len(class_names), dtype=np.float64)

        # Compute configurable class priors from training sample counts.
        class_weights = np.zeros(len(class_names), dtype=np.float64)
        for k, cls_name in enumerate(class_names):
            if cls_name not in models:
                continue

            n_cls = float(models[cls_name].get("n_samples", 0))
            if n_cls <= 0:
                continue

            if class_prior_mode == "sample_proportional":
                class_weights[k] = n_cls
            elif class_prior_mode == "inverse_count":
                class_weights[k] = 1.0 / n_cls
            elif class_prior_mode == "uniform":
                class_weights[k] = 1.0
            else:
                raise ValueError(
                    f"Unknown class_prior_mode '{class_prior_mode}'. "
                    "Expected 'sample_proportional', 'inverse_count', or 'uniform'."
                )

        weight_sum = float(np.sum(class_weights))
        if weight_sum > 0:
            priors = class_weights / weight_sum
            for k in range(len(class_names)):
                if class_weights[k] > 0:
                    log_priors[k] = np.log(max(priors[k], 1e-12))

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
        # Purpose: map class posterior distributions to a single reliability score per face.
        if method == "max":
            return np.max(posteriors, axis=1)
        # Entropy-based confidence
        entropy = -np.sum(posteriors * np.log(posteriors + 1e-12), axis=1)
        entropy /= np.log(posteriors.shape[1])
        return 1.0 - entropy

    def _compute_wound_log_ratio_from_posteriors(self, posteriors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return signed and absolute log-ratio scores for wound vs periwound."""
        # Purpose: create a signed wound-vs-periwound evidence score and its magnitude.
        if posteriors is None or len(posteriors) == 0:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

        if posteriors.shape[1] < 2:
            raise ValueError("Expected at least two posterior columns for periwound and wound")

        class_index = {name: idx for idx, name in enumerate(self.semantic_class_names)}
        periwound_idx = class_index.get("periwound", 0)
        wound_idx = class_index.get("wound", 1 if posteriors.shape[1] > 1 else 0)

        periwound = np.clip(posteriors[:, periwound_idx], 1e-12, 1.0)
        wound = np.clip(posteriors[:, wound_idx], 1e-12, 1.0)
        log_ratio = np.log(wound) - np.log(periwound)
        return log_ratio.astype(np.float64), np.abs(log_ratio).astype(np.float64)

    def run_semantic_classification(self,
                                    r_wound_thresh: float = 0.3,
                                    r_periwound_thresh: float = 0.7,
                                    ring_sizes: List[int] = None,
                                    confidence_method: str = "entropy",
                                    lambda_smooth: float = 1.0,
                                    mrf_method: str = "graph_cut",
                                    mrf_max_iters: int = 10,
                                    class_prior_mode: str = "sample_proportional",
                                    use_ratio_features: bool = True,
                                    use_centroid_features: bool = True,
                                    use_height_deviation: bool = True,
                                    use_normal_deviation: bool = True,
                                    use_curvature: bool = True,
                                    use_mrf: bool = False,
                                    color_space: str = "lab_b",
                                    ) -> None:
        """Run feature-based semantic classification for 2-class (periwound, wound) with feature toggles.

        Args:
            tissue_ratio_thresh: Legacy parameter kept for API compatibility.
            wound_ratio_thresh: Legacy parameter kept for API compatibility.
            ring_sizes: Ring sizes for multi-scale features.
            confidence_method: "entropy" or "max" for confidence computation.
            lambda_smooth: Regularization weight for MRF smoothness.
            mrf_method: "graph_cut" for alpha-expansion, "mean_field" for mean field approximation.
            mrf_max_iters: Maximum iterations for MRF methods.
            use_mrf: If False, skip MRF and use raw Gaussian posteriors directly.
            class_prior_mode: "sample_proportional", "inverse_count", or "uniform" for class priors.
            use_height_deviation: Include height deviation features.
            use_normal_deviation: Include normal deviation features.
            use_curvature: Include scale-adaptive discrete mean curvature feature.
            color_space: Color space to use: "none", "rgb", "lab_b", or "ydrdb".
            undistorted_mask_dir: Unused in 2-class mode (kept for API compatibility).
        """
        # Purpose: execute the full semantic pipeline from features -> probabilistic labels -> per-face state.
        # Goal 1: infer per-face periwound/wound probabilities from geometric + photometric features.
        # Goal 2: derive log-ratio evidence and uncertainty bands for downstream decision making.
        # Goal 3: log interpretable rerun meshes so confidence and uncertainty can be inspected visually.
        


        features, face_indices = self._build_feature_vectors(
            ring_sizes=ring_sizes or [1, 3],
            use_ratio_features=use_ratio_features,
            use_centroid_features=use_centroid_features,
            use_height_deviation= use_height_deviation,
            use_normal_deviation=use_normal_deviation,
            use_curvature=use_curvature,
            color_space=color_space
        )

        # Default to 2-class setup for this pipeline if not explicitly configured.
        if not self.semantic_class_names:
            self.semantic_class_names = ["periwound", "wound"]

        if features is None or len(features) == 0:
            print("No features available for semantic classification")
            return

        # if features is None or len(features) == 0:
        #     print("No features available for semantic classification")
        #     return

        models = self._fit_gaussian_models(
            features=features,
            face_indices=face_indices,
            r_wound_thresh=r_wound_thresh  ,
            r_periwound_thresh=r_periwound_thresh,
        )

        if not models:
            print("No Gaussian models could be fitted (insufficient seed data)")
            return


        posteriors = self._compute_posteriors_from_models(
            features,
            models,
            class_prior_mode=class_prior_mode,
        )
        
  
        
        # Optionally apply MRF constraints.
        if use_mrf:
            self.semantic_mrf_method = mrf_method
            labels, mrf_posteriors = self._apply_mrf_constraints(
                posteriors, face_indices, 
                lambda_smooth=lambda_smooth, 
                max_iters=mrf_max_iters,
                method=mrf_method
            )

            if mrf_posteriors is not None:
                confidences = self._compute_confidence_from_posteriors(mrf_posteriors, method=confidence_method)
                final_posteriors = mrf_posteriors
            else:
                confidences = self._compute_confidence_from_posteriors(posteriors, method=confidence_method)
                final_posteriors = posteriors
        else:
            self.semantic_mrf_method = "none"
            self.semantic_mrf_posteriors = {}
            labels = np.argmax(posteriors, axis=1).astype(np.int32)
            confidences = self._compute_confidence_from_posteriors(posteriors, method=confidence_method)
            final_posteriors = posteriors

        self.semantic_posteriors_before_mrf = {
            face_idx: posteriors[i].copy()
            for i, face_idx in enumerate(face_indices)
        }

        log_ratio_values, log_ratio_strength_values = self._compute_wound_log_ratio_from_posteriors(final_posteriors)
  
        
        # Reset semantic state on all face objects before writing new labels/probabilities.
        for face_obj in self.face_objects:
            face_obj.label = None
            face_obj.confidence = None
            face_obj.periwound_prob = None
            face_obj.wound_prob = None
            face_obj.posterior_log_ratio = None
            face_obj.posterior_log_ratio_strength = None

        # Store per-face results (2-class: periwound=0, wound=1)
        self.semantic_posteriors = {}
        self.semantic_log_ratio_values = {}
        self.semantic_log_ratio_strength_values = {}


        for i, face_idx in enumerate(face_indices):
            probs = final_posteriors[i]
            label_name = self.semantic_class_names[int(labels[i])]
            log_ratio_value = float(log_ratio_values[i])
            log_ratio_strength = float(log_ratio_strength_values[i])
            self.semantic_posteriors[face_idx] = probs
            self.semantic_log_ratio_values[face_idx] = log_ratio_value
            self.semantic_log_ratio_strength_values[face_idx] = log_ratio_strength

            if face_idx < len(self.face_objects):
                face_obj = self.face_objects[face_idx]
                face_obj.periwound_prob = float(probs[0])
                face_obj.wound_prob = float(probs[1])
                face_obj.posterior_log_ratio = log_ratio_value
                face_obj.posterior_log_ratio_strength = log_ratio_strength
                face_obj.label = label_name
                face_obj.confidence = float(confidences[i])
        
        # Compute wound_edge as boundary between wound and tissue

    # Rerun visualization was moved to rerun_visualization.py to keep this file focused.


    


    

    def _compute_periwound_edge_face_stats(self, face_idx: int):
        """Compute per-face color statistics for periwound faces across all cameras.
        
        Args:
            face_idx: Face index in self.faces
        """
        # Purpose: summarize local color appearance for periwound boundary faces.
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
        # Purpose: split a face subset into topology-connected regions and discard tiny fragments.
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
        # Purpose: identify outer-boundary faces used by flooding and accessibility checks.
        boundary_faces = set()
        for edge, face_list in self.edge_adjacency.items():
            if len(face_list) == 1:
                boundary_faces.add(face_list[0])
        return boundary_faces

    def _count_boundary_loops_in_face_set(self, face_set: Set[int]) -> int:
        """Count boundary loops for a face subset using trimesh + networkx.

        The primary path follows:
        1) build submesh
        2) extract unique boundary edges (used by exactly one triangle)
        3) build graph from boundary edges
        4) count cycles via networkx.cycle_basis

        Falls back to edge-component counting if trimesh/networkx are unavailable.
        """
        # Purpose: estimate region topology complexity (holes/loops) before morphology operations.
        if not face_set:
            return 0

        # Primary implementation: trimesh unique edges + networkx cycle basis.
        try:
            import trimesh
            import networkx as nx

            face_list = np.array(sorted(face_set), dtype=np.int64)
            if face_list.size == 0:
                return 0

            submesh = trimesh.Trimesh(
                vertices=self.vertices,
                faces=self.faces[face_list],
                process=False,
                maintain_order=True,
            )

            edges_unique = np.asarray(submesh.edges_unique, dtype=np.int64)
            if len(edges_unique) == 0:
                return 0

            edges_unique_inverse = np.asarray(submesh.edges_unique_inverse, dtype=np.int64)
            if edges_unique_inverse.size == 0:
                return 0

            edge_use_count = np.bincount(edges_unique_inverse, minlength=len(edges_unique))
            boundary_edges = edges_unique[edge_use_count == 1]
            if len(boundary_edges) == 0:
                return 0

            graph = nx.Graph()
            graph.add_edges_from(boundary_edges.tolist())
            cycles = nx.cycle_basis(graph)
            return int(len(cycles))
        except Exception:
            # Fallback path below keeps behavior deterministic without extra deps.
            pass

        edge_count = defaultdict(int)
        for face_idx in face_set:
            if face_idx < 0 or face_idx >= len(self.faces):
                continue
            v0, v1, v2 = self.faces[face_idx]
            edges = [
                (min(v0, v1), max(v0, v1)),
                (min(v1, v2), max(v1, v2)),
                (min(v2, v0), max(v2, v0)),
            ]
            for edge in edges:
                edge_count[edge] += 1

        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        if not boundary_edges:
            return 0

        vertex_to_edges = defaultdict(list)
        for edge in boundary_edges:
            a, b = edge
            vertex_to_edges[a].append(edge)
            vertex_to_edges[b].append(edge)

        unvisited = set(boundary_edges)
        n_loops = 0

        while unvisited:
            n_loops += 1
            stack = [next(iter(unvisited))]
            while stack:
                edge = stack.pop()
                if edge not in unvisited:
                    continue
                unvisited.remove(edge)
                a, b = edge
                for next_edge in vertex_to_edges[a]:
                    if next_edge in unvisited:
                        stack.append(next_edge)
                for next_edge in vertex_to_edges[b]:
                    if next_edge in unvisited:
                        stack.append(next_edge)

        return n_loops

    def _get_boundary_loop_lengths_in_face_set(self, face_set: Set[int]) -> Tuple[float, List[int]]:
        """Return total boundary perimeter and boundary vertex indices for a face subset.

        The measure is the total boundary perimeter, computed as the sum of the
        Euclidean lengths of all boundary edges across all boundary loops.
        """
        if not face_set:
            return 0.0, []

        try:
            import trimesh
            import networkx as nx

            face_list = np.array(sorted(face_set), dtype=np.int64)
            if face_list.size == 0:
                return 0.0, []

            submesh = trimesh.Trimesh(
                vertices=self.vertices,
                faces=self.faces[face_list],
                process=False,
                maintain_order=True,
            )

            edges_unique = np.asarray(submesh.edges_unique, dtype=np.int64)
            if len(edges_unique) == 0:
                return 0.0, []

            edges_unique_inverse = np.asarray(submesh.edges_unique_inverse, dtype=np.int64)
            if edges_unique_inverse.size == 0:
                return 0.0, []

            edge_use_count = np.bincount(edges_unique_inverse, minlength=len(edges_unique))
            boundary_edges = edges_unique[edge_use_count == 1]
            if len(boundary_edges) == 0:
                return 0.0, []

            graph = nx.Graph()
            graph.add_edges_from(boundary_edges.tolist())

            loop_lengths = []
            total_perimeter = 0.0
            for component_nodes in nx.connected_components(graph):
                component_graph = graph.subgraph(component_nodes)
                if component_graph.number_of_edges() > 0:
                    loop_lengths.append(int(component_graph.number_of_edges()))

            for a, b in boundary_edges:
                va = np.asarray(self.vertices[int(a)], dtype=np.float64)
                vb = np.asarray(self.vertices[int(b)], dtype=np.float64)
                total_perimeter += float(np.linalg.norm(va - vb))

            boundary_vertices = np.unique(boundary_edges.reshape(-1)).astype(np.int64).tolist()
            return total_perimeter, boundary_vertices
        except Exception:
            pass

        edge_count = defaultdict(int)
        for face_idx in face_set:
            if face_idx < 0 or face_idx >= len(self.faces):
                continue
            v0, v1, v2 = self.faces[face_idx]
            edges = [
                (min(v0, v1), max(v0, v1)),
                (min(v1, v2), max(v1, v2)),
                (min(v2, v0), max(v2, v0)),
            ]
            for edge in edges:
                edge_count[edge] += 1

        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        if not boundary_edges:
            return 0.0, []

        vertex_to_edges = defaultdict(list)
        boundary_vertices = set()
        total_perimeter = 0.0
        for edge in boundary_edges:
            a, b = edge
            vertex_to_edges[a].append(edge)
            vertex_to_edges[b].append(edge)
            boundary_vertices.add(int(a))
            boundary_vertices.add(int(b))
            va = np.asarray(self.vertices[int(a)], dtype=np.float64)
            vb = np.asarray(self.vertices[int(b)], dtype=np.float64)
            total_perimeter += float(np.linalg.norm(va - vb))

        unvisited = set(boundary_edges)

        while unvisited:
            stack = [next(iter(unvisited))]
            while stack:
                edge = stack.pop()
                if edge not in unvisited:
                    continue
                unvisited.remove(edge)
                a, b = edge
                for next_edge in vertex_to_edges[a]:
                    if next_edge in unvisited:
                        stack.append(next_edge)
                for next_edge in vertex_to_edges[b]:
                    if next_edge in unvisited:
                        stack.append(next_edge)

        return total_perimeter, sorted(boundary_vertices)

    def _expand_faces_by_boundary_reduction(self,
                                            face_set: Set[int],
                                            max_iterations: int = 25) -> Set[int]:
        """Grow a face set by adding adjacent faces only when the outer boundary shrinks.

        The longest boundary loop is used as the objective when multiple loops are
        present. Growth stops when no adjacent face reduces that length or when the
        iteration cap is reached.
        """
        filtered_faces = set(face_set)
        print(f"Starting boundary-guided growth with {len(filtered_faces)} faces")
        if not filtered_faces:
            return set()

        current_boundary_perimeter, _boundary_vertices = self._get_boundary_loop_lengths_in_face_set(filtered_faces)
        if current_boundary_perimeter <= 0:
            return filtered_faces

        print(
            f"Boundary-guided growth start: boundary_perimeter={current_boundary_perimeter:.6f}, "
            f"max_iterations={max_iterations}"
        )
        
        for iteration in range(max_iterations):
            candidate_faces = set()
            for face_idx in filtered_faces:
                for adj_face_idx in self.face_adjacency.get(face_idx, []):
                    if adj_face_idx not in filtered_faces:
                        candidate_faces.add(adj_face_idx)
            if not candidate_faces:
                print(f"Boundary-guided growth stopped at iteration {iteration + 1}: no adjacent faces left")
                break

            # Test every candidate against the current boundary perimeter.
            # Add ALL candidates whose inclusion does not increase the perimeter.
            accepted_faces = set()
            for candidate_face in sorted(candidate_faces):
                trial_faces = filtered_faces | {candidate_face}
                trial_boundary_perimeter, _trial_boundary_vertices = self._get_boundary_loop_lengths_in_face_set(trial_faces)

                if trial_boundary_perimeter <= current_boundary_perimeter:
                    accepted_faces.add(candidate_face)

            if not accepted_faces:
                print(
                    f"Boundary-guided growth stopped at iteration {iteration + 1}: "
                    f"no adjacent face reduced the boundary perimeter "
                    f"returning {len(filtered_faces)} faces with boundary_perimeter={current_boundary_perimeter:.6f}"
                )
                return filtered_faces

            filtered_faces.update(accepted_faces)
            current_boundary_perimeter, _boundary_vertices = self._get_boundary_loop_lengths_in_face_set(filtered_faces)
            print(
                f"Boundary-guided growth iteration {iteration + 1}: added {len(accepted_faces)} faces, "
                f"boundary_perimeter={current_boundary_perimeter:.6f}"
            )

            if current_boundary_perimeter <= 0:
                break

        return filtered_faces
    
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
        # Purpose: convert selected global face IDs into a compact standalone mesh.
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
        # Purpose: capture the geometric wound contour as boundary edges for later measurement/visualization.
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
    

    def morphological_closing_opening(self,
                                     face_set: Set[int],
                                     operation: str = "closing",
                                     geodesic_radius: int = 1,
                                     iterations: int = 1) -> Set[int]:
        """Apply geodesic graph morphology to a face subset.

        Closing fills small gaps in the wound region, while opening removes
        thin spurs and isolated protrusions. The morphology is defined on the
        face-adjacency graph, with `geodesic_radius` controlling the number of
        adjacency hops used for the dilation/erosion neighborhood.
        """
        if not face_set:
            return set()

        operation = str(operation).lower().strip()
        geodesic_radius = max(1, int(geodesic_radius))
        iterations = max(1, int(iterations))

        if operation not in {"closing", "opening", "auto"}:
            raise ValueError(
                f"Unsupported morphology operation '{operation}'. "
                "Expected 'closing', 'opening', or 'auto'."
            )

        if operation == "auto":
            return set(face_set)

        def _geodesic_dilate(seed_faces: Set[int], radius: int) -> Set[int]:
            dilated = set(seed_faces)
            frontier = set(seed_faces)

            for _ in range(radius):
                next_frontier: Set[int] = set()
                for face_idx in frontier:
                    for adj_face_idx in self.face_adjacency.get(face_idx, []):
                        if adj_face_idx not in dilated:
                            next_frontier.add(adj_face_idx)
                if not next_frontier:
                    break
                dilated.update(next_frontier)
                frontier = next_frontier

            return dilated

        def _geodesic_erode(region_faces: Set[int], radius: int) -> Set[int]:
            if not region_faces:
                return set()

            eroded_faces: Set[int] = set()
            for face_idx in region_faces:
                neighborhood = self._get_face_neighbors_ring(face_idx, radius)
                if neighborhood.issubset(region_faces):
                    eroded_faces.add(face_idx)
            return eroded_faces

        current_faces = set(face_set)
        for _ in range(iterations):
            if operation == "closing":
                current_faces = _geodesic_erode(
                    _geodesic_dilate(current_faces, geodesic_radius),
                    geodesic_radius,
                )
            else:
                current_faces = _geodesic_dilate(
                    _geodesic_erode(current_faces, geodesic_radius),
                    geodesic_radius,
                )

            if not current_faces:
                break

        return current_faces

    def morphological_closing(self,
                              face_set: Set[int],
                              closing_iterations: int = 1,
                              closing_kernel_size: int = 1) -> Set[int]:
        """Compatibility wrapper for geodesic closing on face sets."""
        return self.morphological_closing_opening(
            face_set=face_set,
            operation="closing",
            geodesic_radius=closing_kernel_size,
            iterations=closing_iterations,
        )

    def morphological_opening(self,
                              face_set: Set[int],
                              opening_iterations: int = 1,
                              opening_kernel_size: int = 1) -> Set[int]:
        """Compatibility wrapper for geodesic opening on face sets."""
        return self.morphological_closing_opening(
            face_set=face_set,
            operation="opening",
            geodesic_radius=opening_kernel_size,
            iterations=opening_iterations,
        )
    
    def expand_and_extract_components(self, face_set: Set[int],
                                     min_component_faces: int = 100,
                                     ) -> Set[int]:
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
        
        
        # Step 1: Expand wound to include enclosed non-wound faces


        # Get mesh boundary faces (edges with only one adjacent face)
        boundary_faces = self._get_boundary_faces()
        
        # All non-wound faces that could potentially be enclosed
        all_faces_set = set(range(len(self.faces)))
        non_wound_faces = all_faces_set - face_set
        
        
        self.log_wound(non_wound_faces, stage_name="non_wound_faces_before_enclosed_expansion")
        # Restrict flood-fill to only faces classified as semantic wound candidates
        # or uncertain faces (from three-way classification in _select_wound_faces).
        # This prevents propagation through periwound or tissue regions.
        semantic_allowed = set()
        semantic_wound = getattr(self, "semantic_wound_candidates", None)
        semantic_uncertain = getattr(self, "semantic_uncertain_faces", None)

        semantic_allowed = set(semantic_wound) | set(semantic_uncertain)

        self.log_wound(non_wound_faces, stage_name="floodable_faces_before_enclosed_expansion")
        # Flood-fill from boundary through floodable faces
        # Faces reachable from boundary are "outside", others are "enclosed"
        queue = deque([f for f in boundary_faces if f in non_wound_faces])
        outside = set()
        
        while queue:
            face_idx = queue.popleft()
            if face_idx in outside:
                continue
            outside.add(face_idx)
            
            # Explore adjacent floodable faces
            for adj_face in self.face_adjacency.get(face_idx, []):
                if adj_face in non_wound_faces and adj_face not in outside:
                    queue.append(adj_face)
        
        # Enclosed faces are floodable faces that couldn't be reached
        enclosed_faces = non_wound_faces - outside
        
        enclosed_faces = {f for f in enclosed_faces if f in semantic_allowed}
        
        if enclosed_faces:
            print(f"Found {len(enclosed_faces)} enclosed faces, adding to wound region")
        
        # Combine original wound faces with enclosed faces
        expanded_faces = face_set | enclosed_faces
        self.added_faces_enclosed = enclosed_faces

        self.log_wound(expanded_faces, stage_name="post_enclosed_face_expansion")
        # Step 2: Extract connected components and select largest
        if len(expanded_faces) > 0:
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

        
        return final_faces

    def _get_face_wound_probability(self, face_idx: int) -> float:
        """Return wound probability for a face with graceful fallbacks."""
        # Purpose: provide a robust wound probability accessor across multiple storage paths.
        if 0 <= face_idx < len(self.face_objects):
            wound_prob = self.face_objects[face_idx].wound_prob
            if wound_prob is not None:
                return float(wound_prob)
        if getattr(self, "semantic_mrf_posteriors", None) and face_idx in self.semantic_mrf_posteriors:
            return float(self.semantic_mrf_posteriors[face_idx][1])
        if self.semantic_posteriors and face_idx in self.semantic_posteriors:
            return float(self.semantic_posteriors[face_idx][1])
        return 0.0

    def _get_face_posterior_pair(self, face_idx: int) -> Optional[np.ndarray]:
        """Return [p_periwound, p_wound] for a face when available."""
        # Purpose: centralize posterior lookup for downstream ratio/probability logic.
        if getattr(self, "semantic_mrf_posteriors", None) and face_idx in self.semantic_mrf_posteriors:
            return np.asarray(self.semantic_mrf_posteriors[face_idx], dtype=np.float64)
        if self.semantic_posteriors and face_idx in self.semantic_posteriors:
            return np.asarray(self.semantic_posteriors[face_idx], dtype=np.float64)
        if 0 <= face_idx < len(self.face_objects):
            face_obj = self.face_objects[face_idx]
            if face_obj.periwound_prob is not None and face_obj.wound_prob is not None:
                return np.asarray([face_obj.periwound_prob, face_obj.wound_prob], dtype=np.float64)
        return None

    def _get_face_wound_log_ratio(self, face_idx: int) -> float:
        """Return signed log(p_wound / p_periwound) for a face."""
        # Purpose: expose a numerically stable wound-vs-periwound evidence score per face.
        posterior_pair = self._get_face_posterior_pair(face_idx)
        if posterior_pair is None or len(posterior_pair) < 2:
            return 0.0
        periwound = float(np.clip(posterior_pair[0], 1e-12, 1.0))
        wound = float(np.clip(posterior_pair[1], 1e-12, 1.0))
        return float(np.log(wound) - np.log(periwound))

    def _get_face_wound_log_ratio_strength(self, face_idx: int) -> float:
        """Return absolute log-ratio magnitude for a face."""
        # Purpose: expose evidence strength independent of class sign.
        return abs(self._get_face_wound_log_ratio(face_idx))

    def _build_face_runtime_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build fast per-face arrays from face objects for wound mesh extraction.

        Returns:
            visibility_counts: integer wound visibility count per face
            wound_r_values: wound r-value per face
            wound_prob_values: wound posterior/confidence proxy per face
        """
        # Purpose: build vectorized per-face arrays to speed selection and filtering.
        n_faces = len(self.face_objects)
        n_views = max(len(self.spectators), 1)

        wound_r_values = np.fromiter(
            (float(face_obj.r_wound or 0.0) for face_obj in self.face_objects),
            dtype=np.float64,
            count=n_faces,
        )

        visibility_counts = np.fromiter(
            (
                int(face_obj.visibility_wound)
                if face_obj.visibility_wound is not None
                else int(np.floor(float(face_obj.r_wound or 0.0) * n_views + 1e-9))
                for face_obj in self.face_objects
            ),
            dtype=np.int32,
            count=n_faces,
        )

        wound_prob_values = np.fromiter(
            (
                float(face_obj.wound_prob)
                if face_obj.wound_prob is not None
                else float(self.semantic_posteriors.get(face_obj.face_index, [0.0, 0.0])[1])
                if self.semantic_posteriors
                else 0.0
                for face_obj in self.face_objects
            ),
            dtype=np.float64,
            count=n_faces,
        )

        return visibility_counts, wound_r_values, wound_prob_values

    def _extract_components_in_face_set(self, face_set: Set[int]) -> List[Set[int]]:
        """Extract connected components from a face subset using face adjacency."""
        # Purpose: lightweight BFS component extraction without external dependencies.
        if not face_set:
            return []

        unvisited = set(face_set)
        components = []

        while unvisited:
            start = next(iter(unvisited))
            queue = deque([start])
            component = set()

            while queue:
                face_idx = queue.popleft()
                if face_idx not in unvisited:
                    continue
                unvisited.remove(face_idx)
                component.add(face_idx)

                for adj in self.face_adjacency.get(face_idx, []):
                    if adj in unvisited:
                        queue.append(adj)

            if component:
                components.append(component)

        return components

    def _compute_probability_wound_segments(self, wound_prob_threshold: float) -> List[Set[int]]:
        """Compute connected wound segments from the wound probability map."""
        # Purpose: derive contiguous high-probability wound candidates for segment-level filtering.
        if not getattr(self, "semantic_mrf_posteriors", None) and not self.semantic_posteriors:
            return []

        probability_wound_faces = {
            face_idx for face_idx in range(len(self.faces))
            if self._get_face_wound_probability(face_idx) >= wound_prob_threshold
        }

        return self._extract_components_in_face_set(probability_wound_faces)

    def _is_component_accessible_from_boundary(self, component_faces: Set[int]) -> bool:
        """Check whether any face in a component touches the outer mesh boundary."""
        # Purpose: detect enclosed/internal components that may need special handling.
        if not component_faces:
            return False
        boundary_faces = self._get_boundary_faces()
        return any(face_idx in boundary_faces for face_idx in component_faces)

    def _region_grow_within_segment(self, seed_faces: Set[int], segment_faces: Set[int]) -> Set[int]:
        """Region-grow from seed faces while staying inside a target segment."""
        # Purpose: expand trusted seeds within a constrained probability segment.
        if not seed_faces or not segment_faces:
            return set()

        seeds = set(seed_faces) & set(segment_faces)
        if not seeds:
            return set()

        grown = set()
        queue = deque(seeds)

        while queue:
            face_idx = queue.popleft()
            if face_idx in grown or face_idx not in segment_faces:
                continue
            grown.add(face_idx)

            for adj in self.face_adjacency.get(face_idx, []):
                if adj in segment_faces and adj not in grown:
                    queue.append(adj)

        return grown

    def _select_probability_segment(self,
                                    probability_segments: List[Set[int]],
                                    initial_projected_faces: Set[int]) -> Set[int]:
        """Select probability segment with maximal overlap to initial wound projection."""
        # Purpose: choose the most plausible wound segment by overlap and size.
        if not probability_segments:
            return set()

        best_segment = max(
            probability_segments,
            key=lambda seg: (len(seg & initial_projected_faces), len(seg))
        )
        overlap = len(best_segment & initial_projected_faces)
        print(
            f"Selected probability wound segment with {len(best_segment)} faces "
            f"(overlap with initial projection: {overlap})"
        )
        return best_segment

    def _region_grow_combined_scores(self,
                                     seed_faces: Set[int],
                                     combined_scores: np.ndarray,
                                     face_distance_hops: Optional[np.ndarray] = None,
                                     max_distance_hops: Optional[int] = None,
                                     max_iterations: int = 100) -> Set[int]:
        """
        Perform region growing using combined weighted scores.
        
        Expands from seed faces to adjacent faces that fit the current selected-score
        distribution and exceed the growth threshold.
        Uses a queue-based BFS approach to grow the region iteratively.
        
        Args:
            seed_faces: Set of initial seed face indices (must have high combined scores)
            combined_scores: Array of combined scores for each face (length = num_faces)
            grow_threshold: Minimum combined score for a face to be included during growth
            max_iterations: Maximum number of iterations to prevent infinite loops
            
        Returns:
            Set of face indices after region growing
        """
        # Purpose: expand seed wound region based on multi-parameter combined scores with controlled growth.
        if not seed_faces:
            return set()
        
        std_multiplier = 1.6
        stats_refresh_interval = 50
        min_std = 1e-6

        def _compute_score_stats(face_ids: Set[int]) -> Tuple[float, float]:
            values = [float(combined_scores[face_idx]) for face_idx in face_ids if 0 <= face_idx < len(combined_scores)]
            if not values:
                return 0.0, 0.0
            return float(np.mean(values)), float(np.std(values))

        grown_faces = set(seed_faces)
        queue = deque(seed_faces)
        iterations = 0
        added_count = 0
        blocked_count = 0
        blocked_by_distance = 0

        selected_mean, selected_std = _compute_score_stats(grown_faces)
        print(
            f"Region grow seed stats: mean={selected_mean:.4f}, std={selected_std:.4f}, "
            f"seed_count={len(grown_faces)}"
        )
        
        while queue and iterations < max_iterations:
            iterations += 1
            face_idx = queue.popleft()
            
            # Get adjacent faces
            adjacent_faces = self.face_adjacency.get(face_idx, [])
            
            for adj_face_idx in adjacent_faces:
                # Skip if already in grown region
                if adj_face_idx in grown_faces:
                    continue
                
                # If a face was previously rejected, do not keep growing through it.
                if adj_face_idx < 0 or adj_face_idx >= len(combined_scores):
                    continue

                # Hard geometric gate: stop growth when a face is farther than the allowed hop distance
                # from wound-supporting source faces.
                if (
                    face_distance_hops is not None
                    and max_distance_hops is not None
                    and 0 <= adj_face_idx < len(face_distance_hops)
                ):
                    hop_distance = face_distance_hops[adj_face_idx]
                    if not np.isfinite(hop_distance) or hop_distance > float(max_distance_hops):
                        blocked_by_distance += 1
                        continue
                
                score = float(combined_scores[adj_face_idx])
                score_distance = abs(score - selected_mean)
                allowed_distance = std_multiplier * max(selected_std, min_std)

                # Candidate must be close to the current selected distribution and above the floor.
                if score_distance <= allowed_distance:
                    grown_faces.add(adj_face_idx)
                    queue.append(adj_face_idx)
                    added_count += 1
                else:
                    blocked_count += 1

            if iterations % stats_refresh_interval == 0 and grown_faces:
                selected_mean, selected_std = _compute_score_stats(grown_faces)
                print(
                    f"Region grow stats refresh at iteration {iterations}: "
                    f"mean={selected_mean:.4f}, std={selected_std:.4f}, "
                    f"selected_count={len(grown_faces)}"
                )
        
        print(
            f"Region grow completed in {iterations} iterations, added {added_count} faces, "
            f"blocked {blocked_count} faces, distance_blocked={blocked_by_distance}"
        )
        return grown_faces

    def _compute_face_hop_distance_map(self,
                                       source_faces: Set[int],
                                       allowed_faces: Optional[Set[int]] = None) -> np.ndarray:
        """Compute multi-source shortest hop distance to source faces on the face-adjacency graph.

        Distances are in topological hops (0 at source faces, 1 for immediate neighbors, ...).
        If allowed_faces is provided, BFS is restricted to that subset.
        """
        n_faces = len(self.faces) if self.faces is not None else 0
        distances = np.full(n_faces, np.inf, dtype=np.float64)
        if n_faces == 0 or not source_faces:
            return distances

        allowed = None if allowed_faces is None else set(allowed_faces)
        frontier = deque()

        for face_idx in source_faces:
            if face_idx < 0 or face_idx >= n_faces:
                continue
            if allowed is not None and face_idx not in allowed:
                continue
            distances[face_idx] = 0.0
            frontier.append(face_idx)

        while frontier:
            current_face = frontier.popleft()
            current_dist = distances[current_face]
            next_dist = current_dist + 1.0

            for adj_face in self.face_adjacency.get(current_face, []):
                if adj_face < 0 or adj_face >= n_faces:
                    continue
                if allowed is not None and adj_face not in allowed:
                    continue
                if next_dist < distances[adj_face]:
                    distances[adj_face] = next_dist
                    frontier.append(adj_face)

        return distances

    def _select_wound_faces(self, seed_face_threshold: float = 0.2) -> Set[int]:
        """Select wound faces from visibility/probability and segment checks."""
        # Purpose: produce a high-precision wound candidate set with staged evidence filters.
        # if the filtering steps are too aggressive, this function should perform further iterations with lower thresholds rather than returning an empty set.
   
        # Three-way constrained region grow
        # Step 1: Label all faces based on log_ratio_strength vs Tau and log_ratio sign
        #         - strength >= Tau and log_ratio > 0  -> "wound_candidate"
        #         - strength >= Tau and log_ratio <= 0 -> "periwound_candidate"
        #         - strength < Tau                      -> "uncertain"
        # Step 2: Perform the same region grow as in Option 2 but only on the
        #         "wound_candidate" and "uncertain" faces, using the same
        #         combined score and thresholds.
        # Step 3: Select the resulting region-grown set as the final wound faces
        #         but also keep the "uncertain" faces for later adjustments.


        print("\n=== Option 3: Constrained Region Grow (wound_candidate + uncertain) ===")

        # Option 3 hard distance gate parameters.
        option3_distance_source_r_wound_threshold = 0.15
        
        # ---- Calculate max_distance_hops from mesh density and wound extent ----
        # Strategy: estimate how many face-adjacency hops correspond to the wound's
        # characteristic size. Dense meshes need more hops to cover the same physical
        # distance; sparse meshes need fewer.
        try:
            # 1. Estimate mesh density: average face edge length
            if self.faces is not None and self.vertices is not None and len(self.faces) > 0:
                v0 = self.vertices[self.faces[:, 0]]
                v1 = self.vertices[self.faces[:, 1]]
                v2 = self.vertices[self.faces[:, 2]]
                e0 = np.linalg.norm(v1 - v0, axis=1)
                e1 = np.linalg.norm(v2 - v1, axis=1)
                e2 = np.linalg.norm(v0 - v2, axis=1)
                all_edges = np.concatenate([e0, e1, e2])
                # Use median edge length as robust density measure (ignore outliers)
                median_edge_length = float(np.median(all_edges))
                # Also compute mean for comparison
                mean_edge_length = float(np.mean(all_edges))
            else:
                median_edge_length = 1.0
                mean_edge_length = 1.0

            # 2. Find faces with r_wound above threshold and compute their bounding box
            high_rwound_faces = [
                fi for fi in range(len(self.face_objects))
                if float(self.face_objects[fi].r_wound or 0.0) > 0.0
            ]
            if high_rwound_faces and len(high_rwound_faces) > 1:
                centroids = np.asarray([
                    self.face_objects[fi].centroid for fi in high_rwound_faces
                ], dtype=np.float64)
                bbox_min = centroids.min(axis=0)
                bbox_max = centroids.max(axis=0)
                bbox_extent = float(np.linalg.norm(bbox_max - bbox_min))
            else:
                bbox_extent = 0.0

            # 3. Compute max hops: how many face-widths fit across the wound extent?
            #    Use a safety factor so the gate is generous (2x the wound diameter)
            if median_edge_length > 1e-9 and bbox_extent > 1e-9:
                hops_from_extent = int(np.ceil(4.0 * bbox_extent / median_edge_length))
            else:
                hops_from_extent = 3000  # fallback

            # Clamp to reasonable range: at least 5, at most 3000
            option3_max_distance_hops = int(np.clip(hops_from_extent, 5, 3000))

            print(
                f"  Adaptive max_distance_hops: {option3_max_distance_hops} "
                f"(median_edge={median_edge_length:.4f}, mean_edge={mean_edge_length:.4f}, "
                f"wound_bbox_extent={bbox_extent:.4f}, "
                f"high_rwound_faces={len(high_rwound_faces)})"
            )
        except Exception as e:
            option3_max_distance_hops = 3000
            print(f"  Could not compute adaptive max_distance_hops, using default 3000: {e}")

        # Recreate Tau and labelings if not available (safe idempotent)
        Tau = 5.0

        # Build arrays of log-ratio and strength indexed by global face id
        log_ratio_arr = np.fromiter(
            (self.semantic_log_ratio_values.get(fi, 0.0) for fi in range(len(self.faces))),
            dtype=np.float64,
            count=len(self.faces),
        )
        strength_arr = np.fromiter(
            (self.semantic_log_ratio_strength_values.get(fi, 0.0) for fi in range(len(self.faces))),
            dtype=np.float64,
            count=len(self.faces),
        )

        # STEP 1: Label faces
        wound_candidates = set()
        periwound_candidates = set()
        uncertain_faces = set()

        for face_idx in range(len(self.faces)):
            strength = float(strength_arr[face_idx])
            log_ratio = float(log_ratio_arr[face_idx])

            if strength >= Tau:
                if log_ratio > 0.0:
                    wound_candidates.add(face_idx)
                else:
                    periwound_candidates.add(face_idx)
            else:
                uncertain_faces.add(face_idx)

        print(f"Option3: classified faces - wound_candidates={len(wound_candidates)}, "
                f"periwound_candidates={len(periwound_candidates)}, uncertain={len(uncertain_faces)}")

        # Log three-way classification as colormap to rerun
        try:
            import rerun as rr
            from Pipe_Code.modules.WoundEdgeDetection.utils.rerun_visualization import (
                _safe_normalize,
                _viridis_like_colormap,
            )

            if self.vertices is not None and self.faces is not None and len(self.faces) > 0:
                # Create color array: red for wound, blue for periwound, gray for uncertain
                label_colors = np.zeros((len(self.faces), 3), dtype=np.uint8)
                
                # Red for wound candidates
                for fi in wound_candidates:
                    if 0 <= fi < len(self.faces):
                        label_colors[fi] = np.array([255, 0, 0], dtype=np.uint8)
                
                # Blue for periwound candidates
                for fi in periwound_candidates:
                    if 0 <= fi < len(self.faces):
                        label_colors[fi] = np.array([0, 0, 255], dtype=np.uint8)
                
                # Gray for uncertain faces
                for fi in uncertain_faces:
                    if 0 <= fi < len(self.faces):
                        label_colors[fi] = np.array([128, 128, 128], dtype=np.uint8)
                
                # Flatten for mesh visualization (repeat color for each vertex of triangle)
                flat_vertex_positions = self.vertices[self.faces].reshape(-1, 3)
                flat_triangle_indices = np.arange(len(self.faces) * 3, dtype=np.int32).reshape(-1, 3)
                flat_vertex_colors = np.repeat(label_colors, 3, axis=0)

                rr.init("WoundModel3D_Assessment_final", spawn=True)
                rr.log(
                    "/analysis/option3_three_way_labels",
                    rr.Mesh3D(
                        vertex_positions=np.ascontiguousarray(flat_vertex_positions, dtype=np.float64),
                        triangle_indices=np.ascontiguousarray(flat_triangle_indices, dtype=np.int32),
                        vertex_colors=np.ascontiguousarray(flat_vertex_colors, dtype=np.uint8),
                    ),
                    static=True,
                )
                print("✓ Logged three-way labels colormap to /analysis/option3_three_way_labels")
                print("  Red: wound_candidates, Blue: periwound_candidates, Gray: uncertain")
        except Exception as e:
            print(f"Could not log three-way labels colormap: {e}")

        # Ensure combined_scores exists (recompute same way as Option 2 for safety)
        weight_r_wound = 0.2
        weight_r_periwound = 0.2
        weight_log_ratio = 0.6
        combined_scores = np.zeros(len(self.faces), dtype=np.float64)
        for fi in range(len(self.faces)):
            r_wound = float(self.face_objects[fi].r_wound or 0.0)
            r_wound = np.clip(r_wound, 0.0, 1.0)
            r_periwound = float(self.face_objects[fi].r_periwound or 0.0)
            r_periwound = np.clip(r_periwound, 0.0, 1.0)
            lr = float(log_ratio_arr[fi])
            lr_norm = 1.0 / (1.0 + np.exp(-lr))
            combined_scores[fi] = (
                weight_r_wound * r_wound +
                weight_r_periwound * (1.0 - r_periwound) +
                weight_log_ratio * lr_norm
            )

        
        try:
            import rerun as rr
            from Pipe_Code.modules.WoundEdgeDetection.utils.rerun_visualization import (
                _safe_normalize,
                _viridis_like_colormap,
            )

            if self.vertices is not None and self.faces is not None and len(self.faces) > 0:
                combined_score_colors = _viridis_like_colormap(_safe_normalize(combined_scores))
                flat_vertex_positions = self.vertices[self.faces].reshape(-1, 3)
                flat_triangle_indices = np.arange(len(self.faces) * 3, dtype=np.int32).reshape(-1, 3)
                flat_vertex_colors = np.repeat(combined_score_colors, 3, axis=0)

                rr.log(
                    "/analysis/option3_combined_score",
                    rr.Mesh3D(
                        vertex_positions=np.ascontiguousarray(flat_vertex_positions, dtype=np.float64),
                        triangle_indices=np.ascontiguousarray(flat_triangle_indices, dtype=np.int32),
                        vertex_colors=np.ascontiguousarray(flat_vertex_colors, dtype=np.uint8),
                    ),
                    static=True,
                )
        except Exception as e:
            print(f"Could not log masked combined score colormap: {e}")
        
        # STEP 2: Restrict region grow to wound_candidate + uncertain faces
        allowed_faces = set(wound_candidates) | set(uncertain_faces)

        # Build source faces for distance map from r_wound support within the allowed set.
        distance_source_faces = {
            fi for fi in allowed_faces
            if 0 <= fi < len(self.face_objects)
            and float(self.face_objects[fi].r_wound or 0.0) >= option3_distance_source_r_wound_threshold
        }

        if not distance_source_faces:
            # Fallback keeps map valid even when r_wound support is sparse.
            distance_source_faces = set(seed for seed in wound_candidates if seed in allowed_faces)

        # Compute hop distance on the full mesh so the gate reflects global topology.
        face_distance_hops = self._compute_face_hop_distance_map(
            source_faces=distance_source_faces,
        )

        # Log the hop-distance field as a full-mesh colormap for rerun.
        try:
            import rerun as rr
            from Pipe_Code.modules.WoundEdgeDetection.utils.rerun_visualization import (
                _safe_normalize,
                _viridis_like_colormap,
            )

            if self.vertices is not None and self.faces is not None and len(self.faces) > 0:
                distance_for_viz = np.asarray(face_distance_hops, dtype=np.float64).copy()
                finite_mask = np.isfinite(distance_for_viz)
                if np.any(finite_mask):
                    max_finite_distance = float(np.max(distance_for_viz[finite_mask]))
                    distance_for_viz[~finite_mask] = max_finite_distance + 1.0
                else:
                    distance_for_viz[:] = 0.0

                distance_colors = _viridis_like_colormap(_safe_normalize(distance_for_viz))
                flat_vertex_positions = self.vertices[self.faces].reshape(-1, 3)
                flat_triangle_indices = np.arange(len(self.faces) * 3, dtype=np.int32).reshape(-1, 3)
                flat_vertex_colors = np.repeat(distance_colors, 3, axis=0)

                rr.log(
                    "/analysis/option3_distance_hops",
                    rr.Mesh3D(
                        vertex_positions=np.ascontiguousarray(flat_vertex_positions, dtype=np.float64),
                        triangle_indices=np.ascontiguousarray(flat_triangle_indices, dtype=np.int32),
                        vertex_colors=np.ascontiguousarray(flat_vertex_colors, dtype=np.uint8),
                    ),
                    static=True,
                )
                print("✓ Logged option3 hop-distance colormap to /analysis/option3_distance_hops")
        except Exception as e:
            print(f"Could not log option3 hop-distance colormap: {e}")
        
        finite_distances = face_distance_hops[np.isfinite(face_distance_hops)]
        if finite_distances.size > 0:
            print(
                f"Option3 distance gate: source_faces={len(distance_source_faces)}, "
                f"max_hops={option3_max_distance_hops}, "
                f"reachable_faces={len(finite_distances)}, "
                f"max_reachable_hops={int(np.max(finite_distances))}"
            )
        else:
            print(
                f"Option3 distance gate: source_faces={len(distance_source_faces)}, "
                f"max_hops={option3_max_distance_hops}, reachable_faces=0"
            )
        
        
        # Seed faces: use high r_wound like in Option 2 but constrained to allowed faces
        seed_faces = set(
            fi for fi in range(len(self.faces))
            if float(self.face_objects[fi].r_wound or 0.0) > seed_face_threshold and fi in allowed_faces
        )

        if not seed_faces:
            # fall back to wound_candidates as seeds
            seed_faces = set(wound_candidates) & allowed_faces

        print(f"Option3: seed faces for constrained grow: {len(seed_faces)}")

        grown_faces = set()
        if seed_faces:
            grown_faces = self._region_grow_combined_scores(
                seed_faces=seed_faces,
                combined_scores=combined_scores,
                face_distance_hops=face_distance_hops,
                max_distance_hops=option3_max_distance_hops,
                max_iterations=20000,
            )
            print(f"Option3: constrained region grow expanded to {len(grown_faces)} faces")
            try:
                self.log_wound(grown_faces, stage_name="option3_region_grown")
            except Exception:
                pass
        else:
            print("Option3: no seeds available for constrained grow; grown set empty")



        # Store for inspection/visualization
        self.semantic_wound_candidates = wound_candidates
        self.semantic_periwound_candidates = periwound_candidates
        self.semantic_uncertain_faces = uncertain_faces
        self.semantic_grown = grown_faces

        filtered_faces = set(grown_faces)
        print(f"Option3: final selected faces={len(filtered_faces)} (grown={len(grown_faces)}, uncertain_kept={len(uncertain_faces)})")
        try:
            self.log_wound(filtered_faces, stage_name="option3_selected_wound")
        except Exception:
            pass

        return set(filtered_faces)

    def log_wound(self, face_indices: Set[int], stage_name: str = "wound_stage"):
        """Log wound faces to rerun viewer at any stage of the pipeline.
        
        Purpose: visualize wound segmentation at any intermediate or final stage.
        
        Args:
            face_indices: Set of face indices to visualize
            stage_name: Name of the stage (e.g., 'selected', 'filtered', 'final', 'closed')
                       Will be logged to /analysis/wound_{stage_name}
        """
        try:
            import rerun as rr
            rr.init("WoundModel3D_Assessment_final", spawn=True)
            from Pipe_Code.modules.WoundEdgeDetection.utils.rerun_visualization import (
                _safe_normalize,
                _viridis_like_colormap,
            )
            
            if not face_indices or self.vertices is None or self.faces is None:
                print(f"log_wound({stage_name}): no faces to log")
                return
            
            print(f"Logging {len(face_indices)} wound faces (stage: {stage_name}) to rerun...")
            
            # Create per-face flattened mesh for consistent per-face coloring
            flat_vertex_positions = self.vertices[self.faces].reshape(-1, 3)
            flat_triangle_indices = np.arange(len(self.faces) * 3, dtype=np.int32).reshape(-1, 3)
            
            # Color non-wound faces gray, wound faces by log-ratio value
            flat_vertex_colors = np.full((len(self.faces) * 3, 3), 200, dtype=np.uint8)
            
            # Get log-ratio values for all faces
            wound_log_ratio_values = np.array([
                self._get_face_wound_log_ratio(face_idx) 
                for face_idx in range(len(self.faces))
            ], dtype=np.float64)
            
            # Normalize and colormap
            normalized_log_ratio = _safe_normalize(wound_log_ratio_values)
            face_colors = _viridis_like_colormap(normalized_log_ratio)
            
            # Set colors: gray for non-wound, viridis for wound
            for face_idx in range(len(self.faces)):
                if face_idx in face_indices:
                    # Wound face: use viridis colormap
                    vertex_indices = np.arange(face_idx * 3, face_idx * 3 + 3)
                    flat_vertex_colors[vertex_indices] = face_colors[face_idx]
                # else: already gray, leave as is
            
            # Log to rerun
            rr.log(
                f"/analysis/wound_{stage_name}",
                rr.Mesh3D(
                    vertex_positions=np.ascontiguousarray(flat_vertex_positions, dtype=np.float64),
                    triangle_indices=np.ascontiguousarray(flat_triangle_indices, dtype=np.int32),
                    vertex_colors=np.ascontiguousarray(flat_vertex_colors, dtype=np.uint8),
                ),
                static=True,
            )
            print(f"✓ Logged to /analysis/wound_{stage_name}")
            
        except Exception as e:
            print(f"Could not log wound mesh: {e}")

    def _refine_wound_faces(self,
                            face_set: Set[int],
                            min_component_faces: int,
                            closing_iterations: int,
                            closing_kernel_size: int,
                            boundary_growth_max_iterations: int = 25) -> Set[int]:
        """Refine selected wound faces (closing, smoothing, connectivity cleanup)."""
                    # Purpose: convert coarse wound candidates into a geometrically coherent final region.
        filtered_faces = set(face_set)
        if not filtered_faces:
            return set()


        boundary_loop_count = self._count_boundary_loops_in_face_set(filtered_faces)
        print(f"Boundary loop count before closing: {boundary_loop_count}")


        filtered_faces = self.morphological_closing_opening(
            filtered_faces,
            operation="closing",
            geodesic_radius=closing_kernel_size,
            iterations=closing_iterations,
        )
        
        filtered_faces = self.morphological_closing_opening(
            filtered_faces,
            operation="opening",
            geodesic_radius=closing_kernel_size,
            iterations=closing_iterations,
        )

        self.log_wound(filtered_faces, stage_name="post_closing")

        if not filtered_faces:
            print("Warning: No wound faces remain after closing")
            return set()

        filtered_faces = self.expand_and_extract_components(
            filtered_faces,
            min_component_faces,
        )

        self.log_wound(filtered_faces, stage_name="post_closing_and_connectivity")
        
        filtered_faces = self._expand_faces_by_boundary_reduction(
            filtered_faces,
            max_iterations=boundary_growth_max_iterations,
        )
        
        self.log_wound(filtered_faces, stage_name="post_boundary_growth")
        
        if not filtered_faces:
            print("Warning: No wound faces remain after component extraction")
            return set()

        return set(filtered_faces)

    
    
    def get_wound_mesh(self,
                      min_component_faces: int = 10,
                      closing_iterations: int = 1,
                      closing_kernel_size: int = 1,
                      boundary_growth_max_iterations: int = 25,
                      periwound_thickness: int = 30,
                      periwound_use_semantic_candidates: bool = True,
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            segmentation_tags: Optional tags that select the segmentation method.
                Use {'log_ratio_band'} or {'ratio_band'} for the new three-way log-ratio mode.
            wound_log_ratio_threshold: Lower bound for the wound class in log-ratio mode.
            periwound_log_ratio_threshold: Lower bound for the periwound class in log-ratio mode.
            use_probability_validation: If True, validate added faces using wound probability map
            wound_prob_threshold: Minimum wound probability for faces added during closing/expansion
            use_posterior_ratio_validation: If True, also validate faces using log(p_wound / p_periwound)
            posterior_ratio_threshold: Minimum absolute log-ratio required for wound selection
            prob_segment_region_grow_ratio: If a selected probability segment is not boundary-accessible
                and initial coverage ratio is below this value, grow/select the full segment
            periwound_thickness: Number of adjacency layers to include around the final wound region.
            periwound_use_semantic_candidates: If True, use semantic periwound candidates as soft support.
            
        Returns:
            Tuple of (wound_vertices, wound_faces, wound_colors)
        """
        # Purpose: orchestrate wound-face selection/refinement and return an extractable wound submesh.
        # Goal 1: choose a segmentation strategy (standard evidence filters or log-ratio band mode).
        # Goal 2: apply confidence-aware thresholds that preserve likely wound tissue.
        # Goal 3: prepare face sets for refinement/submesh extraction in downstream steps.
        if not self.faces_processed:
            raise ValueError("Must process wound segmentation first")

        selected_faces = self._select_wound_faces()

        
        # Step 2: Refinement stage (closing, boundary smoothing, connectivity cleanup).
        filtered_faces = self._refine_wound_faces(
            face_set=selected_faces,
            min_component_faces=min_component_faces,
            closing_iterations=closing_iterations,
            closing_kernel_size=closing_kernel_size,
            boundary_growth_max_iterations=boundary_growth_max_iterations,
        )
        self.log_wound(filtered_faces, stage_name="refined")
        
        self.current_wound_faces = set(filtered_faces)
        self.periwound_thickness = int(periwound_thickness)
        periwound_faces = self._segment_periwound_faces(
            wound_faces=filtered_faces,
            thickness=periwound_thickness,
            use_semantic_candidates=periwound_use_semantic_candidates,
        )
        self.current_periwound_faces = set(periwound_faces)
        self.periwound_edge_faces = set(periwound_faces)
        self.log_wound(periwound_faces, stage_name="periwound_region")
        self.compute_wound_edge(filtered_faces)
        self.log_wound(self.wound_edge_faces, stage_name="wound_edge")

        # for f_name in ["gaussian", "linear", "multiquadric", "inverse_multiquadric", "thin_plate"]:
        #     for weight_mode in ["linear_floor", "linear", "sigmoid"]:
        #         for smoothing in [0.0, 0.1, 0.5]:
        #             for voxel_size in [1.0, 2.0, 3.0]:
                        #self.fit_periwound_surface(voxel_size=voxel_size, grid_resolution=100, smoothing=smoothing, outer_weighting_exponent=1.0, function_name=f_name, weighting_mode=weight_mode)
                        #self.visualize_periwound_surface(session_path=f"/analysis/periwound_surface_{f_name}_voxel{voxel_size:.1f}_smooth{smoothing:.2f}_weight{weight_mode}")
        self.fit_periwound_surface(voxel_size=1.0, grid_resolution=100, smoothing=0.0, outer_weighting_exponent=0.0, function_name="linear", weighting_mode="linear")
        self.visualize_periwound_surface(session_path=f"/analysis/periwound_surface_linear_voxel2.0")
        print(
            f"Periwound segmentation complete: {len(periwound_faces)} faces "
            f"(thickness={self.periwound_thickness}, semantic_support={periwound_use_semantic_candidates})"
        )

        if not filtered_faces:
            return np.array([]), np.array([]), np.array([])
        
        #Extract wound submesh from face indices
        wound_vertices, wound_faces, wound_colors = self.extract_submesh_from_faces(filtered_faces)

        # Step 4: Fill holes in the mesh (optional)
        if len(wound_vertices) > 0:
            print("Filling holes in wound mesh...")
            wound_vertices, wound_faces, wound_colors = self._fill_mesh_holes(
                wound_vertices, wound_faces, wound_colors
            )

        print(f"Final wound mesh: {len(wound_vertices)} vertices, {len(wound_faces)} faces")
        
        # Update global tracking with face indices in self.faces
        self.current_wound_faces = filtered_faces

        # Compute wound edge boundary
        self.compute_wound_edge(filtered_faces)

        return wound_vertices, wound_faces, wound_colors
        
    def _segment_periwound_faces(self,
                                 wound_faces: Set[int],
                                 thickness: int = 2,
                                 use_semantic_candidates: bool = True) -> Set[int]:
        """Build a periwound shell by growing outward from the wound boundary."""

        thickness = max(0, int(thickness))

        wound_faces = set(wound_faces)
        periwound_faces: Set[int] = set()
        visited_faces = set(wound_faces)
        frontier = set(wound_faces)

        for _ in range(thickness):
            next_frontier = set()
            for face_idx in frontier:
                for adj_face in self.face_adjacency.get(face_idx, []):
                    if adj_face in visited_faces:
                        continue
                    visited_faces.add(adj_face)
                    next_frontier.add(adj_face)
            periwound_faces.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break

        periwound_faces -= wound_faces

        if use_semantic_candidates:
            semantic_candidates = set(getattr(self, "semantic_periwound_candidates", set()) or set())
            if semantic_candidates:
                shell_support = wound_faces | periwound_faces
                supported_semantic_faces = {
                    face_idx
                    for face_idx in semantic_candidates
                    if any(adj in shell_support for adj in self.face_adjacency.get(face_idx, []))
                }
                periwound_faces.update(supported_semantic_faces - wound_faces)

        boundary_faces = {
            face_idx
            for face_idx in wound_faces
            if any(adj not in wound_faces for adj in self.face_adjacency.get(face_idx, []))
        }
        boundary_normals = np.asarray([
            self.face_objects[idx].face_normal
            for idx in boundary_faces
            if 0 <= idx < len(self.face_objects) and self.face_objects[idx].face_normal is not None
        ], dtype=np.float64)
        periwound_normals = np.asarray([
            self.face_objects[idx].face_normal
            for idx in periwound_faces
            if 0 <= idx < len(self.face_objects) and self.face_objects[idx].face_normal is not None
        ], dtype=np.float64)

        self.periwound_boundary_normal_reference = None
        if boundary_normals.size > 0:
            self.periwound_boundary_normal_reference = np.median(boundary_normals, axis=0)
            self.periwound_boundary_normal_reference /= np.linalg.norm(self.periwound_boundary_normal_reference) + 1e-9

        self.periwound_face_normal_reference = None
        if periwound_normals.size > 0:
            self.periwound_face_normal_reference = np.median(periwound_normals, axis=0)
            self.periwound_face_normal_reference /= np.linalg.norm(self.periwound_face_normal_reference) + 1e-9

        self.periwound_boundary_alignment = None
        if self.periwound_boundary_normal_reference is not None and self.periwound_face_normal_reference is not None:
            self.periwound_boundary_alignment = float(
                abs(np.dot(self.periwound_boundary_normal_reference, self.periwound_face_normal_reference))
            )

        return periwound_faces

        
    
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
        # Purpose: estimate a stable hole-filling edge parameter from boundary complexity.
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

    def fit_periwound_surface(self,
                              method: str = "rbf",
                              voxel_size: Optional[float] = None,
                              grid_resolution: int = 128,
                              smoothing: float = 0.0,
                              spline_degree: int = 3,
                              return_points: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Fit a smooth surface through the current periwound faces and return a mesh or sample points.

        Strategy:
        - Project periwound face centroids into a fitted local tissue plane (u,v coords).
        - Fit either an RBF (recommended) or a B-spline-like smooth bivariate surface z(u,v).
        - Return a dense grid-mesh sampled from the fitted function (vertices, faces) or points.

        Args:
            method: 'rbf' or 'spline' (falls back to a simple poly fit if SciPy not available)
            voxel_size: Optional voxel size for aggregating normals/points before fitting.
            grid_resolution: Grid resolution for sampled output (per axis).
            smoothing: Regularization/smoothing parameter for spline/rbf where applicable.
            spline_degree: Degree for bivariate spline when used.
            return_points: If True, return sampled points instead of a triangulated mesh.

        Returns:
            (vertices, faces) mesh tuple or (points, None) if return_points=True. Returns None on failure.
        """
        # Validate periwound data
        periwound_faces = getattr(self, "current_periwound_faces", None)
        if not periwound_faces:
            print("fit_periwound_surface: no periwound faces available (run get_wound_mesh first)")
            return None

        # Collect centroids and normals
        pts = np.asarray([self.face_objects[f].centroid for f in sorted(periwound_faces)], dtype=np.float64)
        normals = np.asarray([self.face_objects[f].face_normal for f in sorted(periwound_faces) if self.face_objects[f].face_normal is not None], dtype=np.float64)
        if pts.size == 0:
            print("fit_periwound_surface: no centroid points available")
            return None

        # Fit local plane and build local coordinate system
        plane_normal, plane_d = self._fit_plane(pts, face_normals=normals if normals.size > 0 else None)
        origin = -plane_d * plane_normal
        # Create orthonormal basis (u, v, n)
        n = plane_normal / (np.linalg.norm(plane_normal) + 1e-9)
        # pick arbitrary vector not parallel to n
        a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(a, n)) > 0.9:
            a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = np.cross(n, a)
        u /= (np.linalg.norm(u) + 1e-9)
        v = np.cross(n, u)
        v /= (np.linalg.norm(v) + 1e-9)

        # Local coordinates (u_coord, v_coord) and heights (z)
        rel = pts - origin[None, :]
        uv = np.vstack([rel @ u, rel @ v]).T
        z = (rel @ n).ravel()

        # Optional voxelization/aggregation of points and normals
        if voxel_size is not None and voxel_size > 0.0:
            # voxelize in uv plane
            vox_idx = np.floor(uv / float(voxel_size)).astype(np.int64)
            keys = [tuple(k) for k in vox_idx.tolist()]
            agg = {}
            for i, k in enumerate(keys):
                if k not in agg:
                    agg[k] = {'uv': [], 'z': [], 'n': []}
                agg[k]['uv'].append(uv[i])
                agg[k]['z'].append(z[i])
                if i < len(normals):
                    agg[k]['n'].append(normals[i])

            uv_list = []
            z_list = []
            for k, vdata in agg.items():
                uv_list.append(np.mean(vdata['uv'], axis=0))
                z_list.append(np.mean(vdata['z'], axis=0))
            uv = np.asarray(uv_list, dtype=np.float64)
            z = np.asarray(z_list, dtype=np.float64)

        # Choose fitting backend: try SciPy RBF / SmoothBivariateSpline
        fitted = None
        try:
            if method == 'rbf':
                from scipy.interpolate import Rbf
                rbf = Rbf(uv[:, 0], uv[:, 1], z, function='thin_plate', smooth=float(smoothing))
                def evaluator(xi, yi):
                    return rbf(xi, yi)
                fitted = evaluator
            elif method == 'spline':
                from scipy.interpolate import SmoothBivariateSpline
                s = float(smoothing) if smoothing is not None else 0.0
                spline = SmoothBivariateSpline(uv[:, 0], uv[:, 1], z, kx=spline_degree, ky=spline_degree, s=s)
                def evaluator(xi, yi):
                    return spline.ev(xi, yi)
                fitted = evaluator
        except Exception:
            # SciPy not available or failed -> fallback to quadratic polynomial least-squares
            A = np.vstack([np.ones(len(uv)), uv[:, 0], uv[:, 1], uv[:, 0] ** 2, uv[:, 0] * uv[:, 1], uv[:, 1] ** 2]).T
            coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
            def evaluator(xi, yi):
                xi = np.asarray(xi)
                yi = np.asarray(yi)
                xi_flat = xi.ravel()
                yi_flat = yi.ravel()
                X = np.vstack([np.ones_like(xi_flat), xi_flat, yi_flat, xi_flat ** 2, xi_flat * yi_flat, yi_flat ** 2])
                zz = (coeffs @ X).reshape(xi.shape)
                return zz
            fitted = evaluator

        # Build grid in uv space covering data (with small padding)
        pad_u = max(1e-3, (uv[:, 0].max() - uv[:, 0].min()) * 0.05)
        pad_v = max(1e-3, (uv[:, 1].max() - uv[:, 1].min()) * 0.05)
        u_min, u_max = uv[:, 0].min() - pad_u, uv[:, 0].max() + pad_u
        v_min, v_max = uv[:, 1].min() - pad_v, uv[:, 1].max() + pad_v

        gx = np.linspace(u_min, u_max, int(max(3, grid_resolution)))
        gy = np.linspace(v_min, v_max, int(max(3, grid_resolution)))
        GX, GY = np.meshgrid(gx, gy)
        GZ = fitted(GX, GY)

        # Convert grid back to 3D points
        flat_uvx = GX.ravel()
        flat_uvy = GY.ravel()
        flat_z = GZ.ravel()
        vert_xyz = origin[None, :] + np.outer(flat_uvx, u) + np.outer(flat_uvy, v) + np.outer(flat_z, n)

        if return_points:
            self.periwound_surface_points = vert_xyz.reshape(-1, 3)
            return self.periwound_surface_points, None

        # Build faces from grid quads
        nx = len(gx)
        ny = len(gy)
        verts = vert_xyz.reshape(ny * nx, 3)
        faces_idx = []
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                a = iy * nx + ix
                b = a + 1
                c = a + nx
                d = c + 1
                faces_idx.append([a, b, c])
                faces_idx.append([b, d, c])

        faces_idx = np.asarray(faces_idx, dtype=np.int32)
        self.periwound_surface_mesh = (verts, faces_idx)
        return verts, faces_idx

    def visualize_periwound_surface(self,
                                     session_path: str = "/analysis/periwound_surface",
                                     spawn: bool = True,
                                     color: Tuple[int, int, int] = (220, 120, 40)) -> None:
        """
        Visualize the last fitted periwound surface in rerun.

        Uses `self.periwound_surface_mesh` if available, otherwise falls back to
        `self.periwound_surface_points`. The helper calls `rr.init()` and logs
        the mesh or points to the supplied `session_path`.
        """
        try:
            import rerun as rr
        except Exception as e:
            print(f"visualize_periwound_surface: rerun unavailable: {e}")
            return

        try:
            rr.init("WoundModel3D_Assessment_final", spawn=spawn)

            if hasattr(self, "periwound_surface_mesh") and self.periwound_surface_mesh is not None:
                verts, faces = self.periwound_surface_mesh
                if verts is None or faces is None or len(verts) == 0 or len(faces) == 0:
                    print("visualize_periwound_surface: fitted mesh empty")
                    return

                rr.log(
                    session_path,
                    rr.Mesh3D(
                        vertex_positions=np.ascontiguousarray(verts, dtype=np.float64),
                        triangle_indices=np.ascontiguousarray(faces, dtype=np.int32),
                        vertex_colors=np.ascontiguousarray(np.tile(np.array(color, dtype=np.uint8)[None, :], (len(verts), 1)), dtype=np.uint8),
                    ),
                    static=True,
                )
                print(f"Logged periwound surface mesh to {session_path}")
                return

            if hasattr(self, "periwound_surface_points") and self.periwound_surface_points is not None:
                pts = self.periwound_surface_points
                if pts is None or len(pts) == 0:
                    print("visualize_periwound_surface: no periwound surface points to log")
                    return

                try:
                    rr.log(
                        session_path,
                        rr.Points3D(
                            positions=np.ascontiguousarray(pts, dtype=np.float64),
                            radii=np.full(len(pts), 0.5, dtype=np.float32),
                            colors=np.ascontiguousarray(np.tile(np.array(color, dtype=np.uint8)[None, :], (len(pts), 1)), dtype=np.uint8),
                        ),
                        static=True,
                    )
                    print(f"Logged periwound surface points to {session_path}")
                    return
                except Exception:
                    # Fallback: log as a small mesh (each point as a tiny triangle is expensive)
                    print("visualize_periwound_surface: Points3D logging failed, ensure rerun supports Points3D")

            print("visualize_periwound_surface: no fitted surface available to visualize")

        except Exception as exc:
            print(f"visualize_periwound_surface: failed to log: {exc}")
    
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

    # =========================================================================
    # NEW FUNCTION: Detect outer boundary of periwound faces and remove outside faces
    # =========================================================================
    def trim_mesh_to_periwound_boundary(self,
                                        periwound_faces: Optional[Set[int]] = None,
                                        wound_faces: Optional[Set[int]] = None,
                                        keep_enclosed_faces: bool = True,
                                        return_removed_faces: bool = False) -> Set[int]:
        """
        Detect the outer boundary of the periwound region and identify/remove all
        faces outside it.

        Strategy:
        ---------
        Starting from the mesh's global outer boundary faces, flood-fill inward
        through all faces that are *not* part of the periwound region (and optionally
        also not part of the wound region). The flood-fill stops when it reaches
        periwound faces — this "outer boundary" of the periwound region is where
        the periwound meets the exterior of the mesh.

        Faces visited by the flood-fill are "outside" the periwound and should be
        removed. Their complement — the faces *inside* the periwound outer boundary
        — is returned.

        This is typically called *after* `_segment_periwound_faces()` (or
        equivalently after `get_wound_mesh()`) when `self.current_periwound_faces`
        is populated.

        Args:
            periwound_faces: Set of periwound face indices. If None, uses
                `self.current_periwound_faces`.
            wound_faces: Optional set of wound face indices. If provided, the wound
                region is also treated as "blocking" the flood-fill (i.e. kept).
            keep_enclosed_faces: If True, any faces that are completely enclosed by
                the periwound (but not themselves periwound faces) are also kept.
                This prevents "holes" inside the kept region.
            return_removed_faces: If True, also return the set of removed face indices.

        Returns:
            If return_removed_faces is False (default):
                Set of face indices to KEEP (inside the periwound outer boundary).
            If return_removed_faces is True:
                Tuple of (kept_faces, removed_faces).
        """
        # --- Resolve input face sets ---
        if periwound_faces is None:
            periwound_faces = getattr(self, "current_periwound_faces", None)
            if periwound_faces is None:
                raise ValueError(
                    "periwound_faces not provided and self.current_periwound_faces is empty. "
                    "Run get_wound_mesh() or _segment_periwound_faces() first."
                )

        periwound_faces = set(periwound_faces)
        if not periwound_faces:
            print("trim_mesh_to_periwound_boundary: empty periwound_faces, returning empty kept set")
            return (set(), set()) if return_removed_faces else set()

        wound_faces = set(wound_faces) if wound_faces is not None else getattr(self, "current_wound_faces", set())
        wound_faces = set(wound_faces)

        # Combined "blocking" set: flood-fill cannot cross these faces
        blocking_faces = periwound_faces | wound_faces

        # --- Sanity checks ---
        n_faces = len(self.faces) if self.faces is not None else 0
        if n_faces == 0:
            print("trim_mesh_to_periwound_boundary: mesh has no faces")
            return (set(), set()) if return_removed_faces else set()

        all_faces = set(range(n_faces))
        non_blocking_faces = all_faces - blocking_faces

        if not non_blocking_faces:
            # All faces are either periwound or wound - nothing to remove
            print("trim_mesh_to_periwound_boundary: all faces are periwound or wound, nothing to remove")
            kept_faces = all_faces
            removed_faces = set()
            self._outside_periwound_faces = removed_faces
            self._trimmed_kept_faces = kept_faces
            if return_removed_faces:
                return kept_faces, removed_faces
            return kept_faces

        # --- Flood-fill from mesh boundary through non-blocking faces ---
        boundary_faces = self._get_boundary_faces()

        # Seed the queue with boundary faces that are NOT blocking (i.e. not periwound/wound)
        seed_faces = [f for f in boundary_faces if f in non_blocking_faces]

        outside_faces = set()  # Faces reachable from the boundary through non-blocking faces
        queue = deque(seed_faces)

        while queue:
            face_idx = queue.popleft()
            if face_idx in outside_faces:
                continue
            outside_faces.add(face_idx)

            for adj_face in self.face_adjacency.get(face_idx, []):
                # Only propagate through non-blocking faces that haven't been visited
                if adj_face in non_blocking_faces and adj_face not in outside_faces:
                    queue.append(adj_face)

        # --- Determine kept vs removed ---
        # 'outside_faces' are those reachable from the mesh boundary through non-blocking faces
        # Periwound and wound faces are always kept (they block the flood-fill)
        kept_faces = blocking_faces.copy()  # periwound + wound faces are kept

        # Identify faces enclosed by the periwound that should also be kept
        if keep_enclosed_faces:
            # Enclosed faces: non-blocking faces that are NOT reachable from the boundary
            enclosed_faces = non_blocking_faces - outside_faces
            if enclosed_faces:
                print(f"  Found {len(enclosed_faces)} enclosed faces inside periwound boundary, keeping them")
                kept_faces.update(enclosed_faces)
                self._enclosed_faces_inside_periwound = enclosed_faces

        # Everything outside the periwound (reachable from boundary) is removed
        removed_faces = outside_faces - kept_faces

        print(
            f"trim_mesh_to_periwound_boundary:\n"
            f"  Total faces: {n_faces}\n"
            f"  Periwound faces: {len(periwound_faces)}\n"
            f"  Wound faces: {len(wound_faces)}\n"
            f"  Blocking faces (kept): {len(kept_faces)}\n"
            f"  Outside faces (removed): {len(removed_faces)}"
        )

        # Cache for external access
        self._outside_periwound_faces = removed_faces
        self._trimmed_kept_faces = kept_faces

        if return_removed_faces:
            return kept_faces, removed_faces
        return kept_faces

    def extract_periwound_trimmed_mesh(self,
                                       periwound_threshold: float = 0.1,
                                       wound_threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Extract a submesh that is trimmed to the periwound outer boundary.

        Works directly after `process_periwound_face_detection()` by deriving
        periwound faces from the `r_periwound` visibility ratios (no need to
        run the full semantic classification first).

        The function:
        1. Selects periwound faces based on `r_periwound > periwound_threshold`
           (and optionally wound faces based on `r_wound > wound_threshold`).
        2. Calls `trim_mesh_to_periwound_boundary()` to flood-fill from the mesh
           outer boundary and remove everything outside the periwound ring.
        3. Extracts the compact submesh via `extract_submesh_from_faces()`.

        Args:
            periwound_threshold: Minimum r_periwound value for a face to be
                considered part of the periwound region (default: 0.1).
            wound_threshold: Minimum r_wound value for a face to be treated as
                blocking the flood-fill (default: 0.0 — disabled). Set to e.g.
                0.3 to also treat high-confidence wound faces as blocking.

        Returns:
            Tuple of (vertices, faces, colors) for the trimmed mesh.
        """
        # --- Derive periwound faces from visibility ratios if not already set ---
        periwound_faces = getattr(self, "current_periwound_faces", None)
        if periwound_faces is None or len(periwound_faces) == 0:
            # No semantic periwound mask yet — use r_periwound threshold directly
            periwound_faces = set()
            for face_obj in self.face_objects:
                if face_obj.r_periwound is not None and float(face_obj.r_periwound) > periwound_threshold:
                    periwound_faces.add(face_obj.face_index)
            if not periwound_faces:
                print("extract_periwound_trimmed_mesh: no periwound faces found from r_periwound data")
                return np.array([]), np.array([]), None
            print(f"extract_periwound_trimmed_mesh: derived {len(periwound_faces)} periwound faces from r_periwound > {periwound_threshold}")

        # --- Optionally derive wound faces from r_wound ---
        wound_faces = getattr(self, "current_wound_faces", set())
        if wound_threshold > 0.0 and len(wound_faces) == 0:
            wound_faces = set()
            for face_obj in self.face_objects:
                if face_obj.r_wound is not None and float(face_obj.r_wound) > wound_threshold:
                    wound_faces.add(face_obj.face_index)
            if wound_faces:
                print(f"extract_periwound_trimmed_mesh: derived {len(wound_faces)} wound faces from r_wound > {wound_threshold}")

        kept_faces = self.trim_mesh_to_periwound_boundary(
            periwound_faces=periwound_faces,
            wound_faces=wound_faces if wound_faces else None,
            return_removed_faces=False,
        )
        if not kept_faces:
            print("extract_periwound_trimmed_mesh: no faces kept, returning empty mesh")
            return np.array([]), np.array([]), None

        print(f"Extracting trimmed mesh with {len(kept_faces)} faces...")
        vertices, faces, colors = self.extract_submesh_from_faces(kept_faces)
        print(f"Trimmed mesh: {len(vertices)} vertices, {len(faces)} faces")
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


def MeshReduction(mesh_path: str, model_path: Path, mask_dir: Path, extruded_mask_dir: Path, decimation_threshold: float = 0.7)-> Path:
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
        
        # Detect initial periwound faces from mask difference
        wound_model.process_periwound_face_detection()
        
        # Save the periwound-trimmed mesh (removes everything outside the periwound outer boundary)
        print("\n=== Saving Periwound-Trimmed Mesh ===")
        trimmed_vertices, trimmed_faces, trimmed_colors = wound_model.extract_periwound_trimmed_mesh()
        if len(trimmed_vertices) > 0:
            trimmed_output_path = Path(model_path) / "wound_and_periwound_trimmed.ply"
            wound_model.save_wound_mesh(trimmed_output_path, trimmed_vertices, trimmed_faces, trimmed_colors)
            print(f"Saved periwound-trimmed mesh to: {trimmed_output_path}")
        
            simplifier = MeshSimplifier(str(trimmed_output_path))
            simplifier.run(smooth_iterations=3, decimation_threshold=decimation_threshold, min_component_face_count=25)
            simplifier.save(trimmed_output_path)
        else:
            print("No vertices in trimmed mesh, skipping saving and simplification")
            return mesh_path  # Return original mesh path if trimming failed
        
        return trimmed_output_path
        
def main(workspace_path: str, mesh_path: Path=None, visualization: bool = True):
    """
    Example usage of mesh-based wound detection with periwound detection.
    """
    
    # Example paths (adjust to your data)
    if mesh_path is None:
        mesh_path = Path(workspace_path) / "dense" / "poisson.ply"
    model_path = Path(workspace_path) / "dense"
    mask_dir = Path(workspace_path) / "dense" / "masks_wound_only_undistorted"
    extruded_mask_dir = Path(workspace_path) / "dense" / "masks_extruded_wound_only_undistorted"

    try:
        
        mesh_path = MeshReduction(mesh_path, model_path, mask_dir, extruded_mask_dir, decimation_threshold=0.7) #Creates a periwound-trimmed mesh based on mask projection. Basically removes everything outside the periwound outer boundary. It then simplifies the mesh to reduce face count while preserving detail.
        wound_model = WoundModel3DMesh()
        
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
        
        # Detect initial periwound faces from mask difference
        wound_model.process_periwound_face_detection()
        #Run semantic classification (feature-based, 2-class: periwound, wound)
        print("\n=== Semantic Classification (Gaussian, MRF optional) ===")
        wound_model.run_semantic_classification(
            r_wound_thresh=0.3,
            r_periwound_thresh=0.5,
            ring_sizes=[1, 3],
            confidence_method="entropy",
            lambda_smooth=1.0,
            use_ratio_features=False,
            use_centroid_features=True,
            use_height_deviation=True,
            use_normal_deviation=False,
            use_curvature=True,
            use_mrf=True,  # Set to True to enable MRF smoothing
            mrf_max_iters=10,
            color_space="rgb",
            class_prior_mode="sample_proportional",
            mrf_method="mean_field"
        )

       
      

        
        # # Get final wound mesh with mesh-specific filtering
        print("\n=== Extracting Wound Mesh ===")
        wound_vertices, wound_faces, wound_colors = wound_model.get_wound_mesh(
            min_component_faces=100,  # Minimum size for valid component
            closing_iterations=5,  # Number of closing iterations
            closing_kernel_size=2,  # Number of adjacency layers per iteration (1-3 typical)
        )

        if visualization:
            from Pipe_Code.modules.WoundEdgeDetection.utils.rerun_visualization import (
                visualize_wound_assessment_rerun,
            )

        visualize_wound_assessment_rerun(
            wound_model,
            session_name="WoundModel3D_Assessment",
            spawn=True,
            image_root=model_path.parent,
        )

        print("\n=== Measuring Wound Dimensions ===")
        wound_measurement_result = None
        try:
            measurement_module = importlib.import_module(
                "Pipe_Code.modules.WoundEdgeDetection.Wound_Measurement_Mesh"
            )
            wound_measurement_result = measurement_module.measure_wound_dimensions_from_model(wound_model, visualization=True)
            print(measurement_module.format_measurement_summary(wound_measurement_result))
        except Exception as measurement_error:
            print(f"Warning: wound measurement failed: {measurement_error}")
            traceback.print_exc()

     

        # Save the initial wound mesh
        print("\n=== Saving Meshes ===")
        output_path = Path(model_path) / "wound_mesh.ply"
        wound_model.save_wound_mesh(output_path, wound_vertices, wound_faces, wound_colors)
        print(f"Saved initial wound mesh to: {output_path}")
        
        
        print("\n=== Processing Complete ===")
        
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main("C:\\Users\\Reich\\LocalCoding\\COLMAP_Pipline\\Experiments\\Bein3", r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Bein3\dense\poisson.ply", visualization=True)