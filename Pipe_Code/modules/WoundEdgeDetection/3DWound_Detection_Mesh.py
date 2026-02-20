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
        self.wound_face_indices = set()  # Faces visible through masks

        # Wound edge data (all indices reference self.faces and self.vertices)
        self.wound_edge_edges = []  # List of (v0, v1) tuples
        self.wound_edge_vertices = set()  # Set of vertex indices on edge
        self.wound_edge_face_indices = set()  # Wound faces adjacent to non-wound

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
    
    def _compute_mesh_topology(self):
        """Compute mesh topology using Open3D and numpy.
        
        Builds edge adjacency and face adjacency from mesh data.
        """
        print("Computing mesh topology...")

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
        
        This solves the problem where faces inside the wound region are not marked
        as wound faces even though they are completely enclosed by wound faces.
        Uses flood-fill from mesh boundary to identify unreachable (enclosed) faces.
        
        Args:
            face_set: Set of face indices to process
            use_connectivity: Whether to extract connected components
            min_component_faces: Minimum faces for valid component
            
        Returns:
            Set of final face indices including enclosed faces
        """
        if not face_set:
            return face_set
        
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
            enclosed_faces = non_wound_faces - outside
            
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
        
        # Step 1: Filter by visibility
        filtered_faces = self.filter_wound_faces(min_visibility)

        if not filtered_faces:
            print("Warning: No wound faces remain after filtering")
            return np.array([]), np.array([]), np.array([])
        
        # Step 2: Apply morphological closing
        if use_closing:
            pre_close = set(filtered_faces)
            filtered_faces = self.morphological_closing(filtered_faces, closing_iterations, closing_kernel_size)
            self.added_faces_closing = set(filtered_faces) - pre_close
        else:
            self.added_faces_closing = set()
        
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

        # Log statistics
        stats_text = f"""
Mesh Statistics:
- Total Vertices: {len(self.vertices):,}
- Total Faces: {len(self.faces):,}
- Wound Faces: {len(self.wound_face_indices):,}
- Spectators: {len(self.spectators)}
- Masks Loaded: {sum(1 for s in self.spectators.values() if s.mask_loaded)}
        """
        
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
            stats['wound_faces_count'] = len(self.wound_face_indices)
            
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
    mesh_path = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse\dense\poisson.ply")
    model_path = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse\dense")
    mask_dir = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse\dense\masks_wound_only_undistorted")
    output_path = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse\dense\wound_mesh.ply")
    output_path_corrected = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse\dense\wound_mesh_corrected.ply")
    
    
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
            use_closing=True,  # Apply morphological closing
            closing_iterations=5,  # Number of closing iterations
            closing_kernel_size=2,  # Number of adjacency layers per iteration (1-3 typical)
            use_fill_holes=False,  # We'll use trimesh instead in correction step
            hole_size=-1.0
        )
        
        # Get initial statistics before correction
        print("\n=== Initial Wound Mesh Statistics ===")
        print(f"Vertices: {len(wound_vertices)}")
        print(f"Faces: {len(wound_faces)}")
        
        
        #     print("\n=== Saving Meshes ===")
        wound_model.save_wound_mesh(output_path, wound_vertices, wound_faces, wound_colors)
        print(f"Saved initial wound mesh to: {output_path}")
    
    
        # print("\n=== Launching Visualization ===")
        wound_model.visualize_with_rerun("WoundModel3DMesh_Main")
        
        print("\n=== Processing Complete ===")
        
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
