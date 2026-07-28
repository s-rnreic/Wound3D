"""
Mesh Simplification Module
This module provides functionality to simplify 3D meshes using quadratic edge collapse
or marching cubes algorithms, calculate face normals, and visualize results in Rerun.
"""

import numpy as np
import trimesh
import pyfqmr
import pymeshlab
import rerun as rr
from typing import Tuple, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MeshSimplifier:
    """
    A class for simplifying 3D meshes using PyMeshLab.

    Workflow:
        1. Load a mesh from a file path
        2. Apply Laplacian smoothing (``apply_coord_laplacian_smoothing``)
        3. Apply clustering decimation (``meshing_decimation_clustering``)
        4. Optionally save / retrieve the result
    """

    def __init__(self, filepath: str):
        """
        Initialize the MeshSimplifier by loading a mesh from *filepath*.

        Args:
            filepath: Absolute or relative path to a mesh file
                      (PLY, OBJ, STL, OFF, …  – any format supported by PyMeshLab).
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Mesh file not found: {filepath}")

        self.filepath = filepath
        self.ms = pymeshlab.MeshSet()
        self.ms.load_new_mesh(filepath)

        mesh = self.ms.current_mesh()
        logger.info(
            f"Loaded mesh from '{filepath}' – "
            f"{mesh.vertex_number()} vertices, {mesh.face_number()} faces"
        )

    # ------------------------------------------------------------------ #
    # Core processing steps
    # ------------------------------------------------------------------ #

    def apply_laplacian_smoothing(self, iterations: int = 3,
                                  stepsmoothnum: int = 1,
                                  cotangentweight: bool = True,
                                  selected: bool = False) -> "MeshSimplifier":
        """
        Apply Laplacian coordinate smoothing to the current mesh.

        Args:
            iterations: Number of smoothing passes to run.
            stepsmoothnum: Number of smoothing steps per pass
                           (maps to ``stepsmoothnum`` in PyMeshLab).
            cotangentweight: If True, use cotangent weighting scheme;
                             otherwise uniform weights.
            selected: If True, smooth only selected vertices.

        Returns:
            self (for method chaining).
        """
        for i in range(iterations):
            self.ms.apply_coord_laplacian_smoothing(
                stepsmoothnum=stepsmoothnum,
                cotangentweight=cotangentweight,
                selected=selected,
            )

        mesh = self.ms.current_mesh()
        logger.info(
            f"Laplacian smoothing applied ({iterations} iteration(s)) – "
            f"{mesh.vertex_number()} vertices, {mesh.face_number()} faces"
        )
        return self

    def apply_decimation_clustering(self, threshold: pymeshlab.PercentageValue = pymeshlab.PercentageValue(1.0)) -> "MeshSimplifier":
        """
        Apply clustering decimation to reduce the mesh complexity.

        The algorithm merges vertices that fall within a spatial grid whose
        cell size is controlled by *threshold* (percentage of the bounding-box
        diagonal).

        Args:
            threshold: Grid-cell size expressed as a ``pymeshlab.PercentageValue``
                       of the mesh bounding-box diagonal (default 1 %).

        Returns:
            self (for method chaining).
        """
        self.ms.meshing_decimation_clustering(
            threshold=threshold,
        )

        mesh = self.ms.current_mesh()
        logger.info(
            f"Clustering decimation applied (threshold={threshold}) – "
            f"{mesh.vertex_number()} vertices, {mesh.face_number()} faces"
        )
        return self
    
    def apply_quadric_edge_collapse_decimation(self, target_face_count: int) -> "MeshSimplifier":
        """
        Apply quadric edge collapse decimation to reduce the mesh complexity.

        Args:
            target_face_count: The desired number of faces after decimation.

        Returns:
            self (for method chaining).
        """
        self.ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_face_count,
            boundaryweight=1.0,
            preservenormal=True,
            preservetopology=True,
        )

        mesh = self.ms.current_mesh()
        logger.info(
            f"Quadric edge collapse decimation applied (target_face_count={target_face_count}) – "
            f"{mesh.vertex_number()} vertices, {mesh.face_number()} faces"
        )
        return self

    def apply_taubin_smoothing(self, iterations: int = 3) -> "MeshSimplifier":
        """
        Apply Taubin smoothing to the current mesh.

        Args:
            iterations: Number of smoothing passes to run.

        Returns:
            self (for method chaining).
        """
        for i in range(iterations):
            self.ms.apply_coord_taubin_smoothing()

        mesh = self.ms.current_mesh()
        logger.info(
            f"Taubin smoothing applied ({iterations} iteration(s)) – "
            f"{mesh.vertex_number()} vertices, {mesh.face_number()} faces"
        )
        return self

    # ------------------------------------------------------------------ #
    # Mesh cleaning
    # ------------------------------------------------------------------ #

    def preclean_mesh(self, min_component_face_count: int = 500,
                   remove_unreferenced: bool = True) -> "MeshSimplifier":
        """
        Run a comprehensive mesh-cleaning pass using PyMeshLab filters.

        The following operations are applied **in order**:

        1. Remove duplicate faces
        2. Remove duplicate vertices
        3. Remove degenerate / null faces (zero-area triangles)
        4. Remove folded faces (collapsed edges / zero-length edges)
        5. Repair non-manifold edges
        6. Repair non-manifold vertices
        7. Remove small disconnected components (by face count)
        8. Recompute face & vertex normals (fixes flipped normals)
        9. Remove unreferenced vertices (optional)

        Args:
            min_component_face_count: Connected components with fewer faces
                than this value are removed (default 25).
            remove_unreferenced: If True, remove vertices that are not
                referenced by any face after cleaning.

        Returns:
            self (for method chaining).
        """
        mesh = self.ms.current_mesh()
        v_before, f_before = mesh.vertex_number(), mesh.face_number()
        logger.info(f"Cleaning mesh ({v_before} vertices, {f_before} faces) …")

        # 1. Duplicate faces
        self.ms.meshing_remove_duplicate_faces()
        # 2. Duplicate vertices
        self.ms.meshing_remove_duplicate_vertices()
        # 3. Degenerate / zero-area faces
        self.ms.meshing_remove_null_faces()
        # 4. Folded / collapsed edges (zero-length edges)
        self.ms.meshing_remove_folded_faces()
        # 5. Non-manifold edges
        self.ms.meshing_repair_non_manifold_edges()
        # 6. Non-manifold vertices
        self.ms.meshing_repair_non_manifold_vertices()
        # 7. Small disconnected components
        self.ms.meshing_remove_connected_component_by_face_number(
            mincomponentsize=min_component_face_count,
            removeunref=remove_unreferenced,
        )
        
        if remove_unreferenced:
            self.ms.meshing_remove_unreferenced_vertices()

        mesh = self.ms.current_mesh()
        v_after, f_after = mesh.vertex_number(), mesh.face_number()
        logger.info(
            f"Cleaning done – removed {v_before - v_after} vertices, "
            f"{f_before - f_after} faces "
            f"({v_after} vertices, {f_after} faces remaining)"
        )

        print(self.get_mesh_info())

       
        
        return self
    
    def postclean_mesh(self, min_component_face_count: int = 500,
                       remove_unreferenced: bool = True) -> "MeshSimplifier":
            """
            Run a comprehensive mesh-cleaning pass using PyMeshLab filters.
    
            The following operations are applied **in order**:
    
            1. Remove duplicate faces
            2. Remove duplicate vertices
            3. Remove degenerate / null faces (zero-area triangles)
            4. Remove folded faces (collapsed edges / zero-length edges)
            5. Repair non-manifold edges
            6. Repair non-manifold vertices
            7. Remove small disconnected components (by face count)
            8. Recompute face & vertex normals (fixes flipped normals)
            9. Remove unreferenced vertices (optional)
    
            Args:
                min_component_face_count: Connected components with fewer faces
                    than this value are removed (default 500).
                remove_unreferenced: If True, remove vertices that are not
                    referenced by any face after cleaning.
    
            Returns:
                self (for method chaining).
            """
            mesh = self.ms.current_mesh()
            v_before, f_before = mesh.vertex_number(), mesh.face_number()
            logger.info(f"Cleaning mesh ({v_before} vertices, {f_before} faces) …")
    
            # 3. Degenerate / zero-area faces
            self.ms.meshing_remove_null_faces()
            # 4. Folded / collapsed edges (zero-length edges)
            self.ms.meshing_remove_folded_faces()
            # 5. Non-manifold edges
            self.ms.meshing_repair_non_manifold_edges()
            # 6. Non-manifold vertices
            self.ms.meshing_repair_non_manifold_vertices()
            # 7. Small disconnected components
            self.ms.meshing_remove_connected_component_by_face_number(
                mincomponentsize=min_component_face_count,
                removeunref=remove_unreferenced,
            )
            # 8. Recompute normals (fixes flipped normals)
            self.ms.compute_normal_per_face()
            self.ms.compute_normal_per_vertex()
            # 9. Unreferenced vertices
            if remove_unreferenced:
                self.ms.meshing_remove_unreferenced_vertices()
    
            mesh = self.ms.current_mesh()
            v_after, f_after = mesh.vertex_number(), mesh.face_number()
            logger.info(
                f"Cleaning done – removed {v_before - v_after} vertices, "
                f"{f_before - f_after} faces "
                f"({v_after} vertices, {f_after} faces remaining)"
            )
    
            # Final topology check via trimesh: 2-manifold with boundary allowed.
            # Condition: every unique undirected edge must have face-incidence 1
            # (boundary edge) or 2 (interior edge), and no edge may have >2 faces.

            mesh_info = self.get_mesh_info()
            print(mesh_info)

            if not mesh_info["is_watertight_except_outer_boundary"]:
                diagnostics = self.get_boundary_diagnostics()
                logger.warning(
                    "Mesh is not watertight except for the outer boundary. "
                    f"Boundary loops: {diagnostics['boundary_loop_count']}, "
                    f"boundary edges: {diagnostics['boundary_edge_count']}."
                )
                for loop in diagnostics["boundary_loops"]:
                    logger.warning(
                        "Boundary loop %s: %s edges, faces=%s, edges=%s",
                        loop["loop_index"],
                        loop["edge_count"],
                        loop["face_indices"],
                        loop["edge_indices"],
                    )

                logger.info("Applying targeted small-hole repair pass (PyMeshLab).")
                repaired = self.repair_small_holes(maxhole_size=30, remesh_iterations=5)

                if repaired:
                    # Re-orient triangle winding after repair so the mesh displays with
                    # consistent face normals from the outside.
                    mesh = self.ms.current_mesh()
                    vertices = mesh.vertex_matrix()
                    faces = mesh.face_matrix()

                    try:
                        repaired_mesh = trimesh.Trimesh(
                            vertices=vertices,
                            faces=faces,
                            process=False,
                        )

                        try:
                            import open3d

                            open3d_mesh = open3d.geometry.TriangleMesh(
                                vertices=open3d.utility.Vector3dVector(repaired_mesh.vertices),
                                triangles=open3d.utility.Vector3iVector(repaired_mesh.faces),
                            )
                            open3d_mesh.orient_triangles()
                            open3d_mesh.compute_triangle_normals()
                            repaired_mesh = trimesh.Trimesh(
                                vertices=np.asarray(open3d_mesh.vertices),
                                faces=np.asarray(open3d_mesh.triangles),
                                process=False,
                            )
                        except Exception:
                            pass

                        trimesh.repair.fix_normals(repaired_mesh)
                        if repaired_mesh.is_watertight and repaired_mesh.volume < 0:
                            repaired_mesh.invert()

                        self.ms.clear()
                        self.ms.add_mesh(
                            pymeshlab.Mesh(
                                vertex_matrix=repaired_mesh.vertices,
                                face_matrix=repaired_mesh.faces,
                            )
                        )
                        self.ms.compute_normal_per_face()
                        self.ms.compute_normal_per_vertex()
                        if remove_unreferenced:
                            self.ms.meshing_remove_unreferenced_vertices()
                    except Exception as exc:
                        logger.warning(f"Normal re-orientation skipped due to failure: {exc}")

                self.ms.meshing_remove_null_faces()
               
                
                final_info = self.get_mesh_info()
                if not final_info["is_watertight_except_outer_boundary"]:
                    final_diagnostics = self.get_boundary_diagnostics()
                    logger.warning(
                        "Mesh is still not watertight except for the outer boundary after repair."
                    )
                    for loop in final_diagnostics["boundary_loops"]:
                        logger.warning(
                            "Remaining boundary loop %s: %s edges, faces=%s, edges=%s",
                            loop["loop_index"],
                            loop["edge_count"],
                            loop["face_indices"],
                            loop["edge_indices"],
                        )
            else:
                logger.info("Postclean mesh is watertight except for the outer boundary.")
    
    
            
            
            return self

    # ------------------------------------------------------------------ #
    # Convenience pipeline
    # ------------------------------------------------------------------ #

    def run(self, smooth_iterations: int = 3,
            min_component_face_count: int = 500) -> "MeshSimplifier":
        """
        Run the full simplification pipeline:
        clean → taubin → decimate → taubin → clean.

        Args:
            smooth_iterations: Number of Laplacian smoothing passes.
            decimation_threshold: Clustering threshold as a percentage of the
                                  bounding-box diagonal.
            min_component_face_count: Connected components with fewer faces
                than this are removed during cleaning.

        Returns:
            self (for method chaining).
        """
        # Pre-simplification cleaning
        self.preclean_mesh(min_component_face_count=min_component_face_count)

        # Core simplification
        self.apply_taubin_smoothing(iterations=smooth_iterations)
        
        # 1. Duplicate faces
        self.ms.meshing_remove_duplicate_faces()
        # 2. Duplicate vertices
        self.ms.meshing_remove_duplicate_vertices()
        # 3. Degenerate / zero-area faces
        self.ms.meshing_remove_null_faces()
        # 4. Folded / collapsed edges (zero-length edges)
        self.ms.meshing_remove_folded_faces()
        # 5. Non-manifold edges
        self.ms.meshing_repair_non_manifold_edges()
        # 6. Non-manifold vertices
        self.ms.meshing_repair_non_manifold_vertices()

        
        self.apply_quadric_edge_collapse_decimation(target_face_count=10000)  # Example target face count
        
        self.ms.meshing_remove_duplicate_faces()
                # 2. Duplicate vertices
        self.ms.meshing_remove_duplicate_vertices()
        # 3. Degenerate / zero-area faces
        self.ms.meshing_remove_null_faces()
        # 4. Folded / collapsed edges (zero-length edges)
        self.ms.meshing_remove_folded_faces()
        # 5. Non-manifold edges
        self.ms.meshing_repair_non_manifold_edges()
        # 6. Non-manifold vertices
        self.ms.meshing_repair_non_manifold_vertices()
        
        self.apply_taubin_smoothing(iterations=smooth_iterations)

        print(self.get_mesh_info())
        # # Post-simplification cleaning
        self.postclean_mesh(min_component_face_count=min_component_face_count) #--> currently this produces holes in the mesh, so we skip it for now. We can consider adding a more targeted cleaning pass if needed.

        return self

    # ------------------------------------------------------------------ #
    # Output helpers
    # ------------------------------------------------------------------ #

    def save(self, output_path: str) -> str:
        """
        Save the current (simplified) mesh to *output_path*.

        Args:
            output_path: Destination file path (str or Path).

        Returns:
            The absolute path of the saved file.
        """
        self.ms.save_current_mesh(str(output_path))
        logger.info(f"Saved simplified mesh to '{output_path}'")
        return os.path.abspath(str(output_path))

    def to_trimesh(self) -> trimesh.Trimesh:
        """
        Convert the current PyMeshLab mesh to a ``trimesh.Trimesh`` object.

        Returns:
            A Trimesh instance with vertices, faces and (if available)
            vertex colors copied from the PyMeshLab mesh.
        """
        mesh = self.ms.current_mesh()
        vertices = mesh.vertex_matrix()
        faces = mesh.face_matrix()

        vertex_colors = None
        if mesh.has_vertex_color():
            # PyMeshLab returns colors as float RGBA in [0, 1]
            vertex_colors = (mesh.vertex_color_matrix() * 255).astype(np.uint8)

        trimesh_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
            process=False,
        )

        try:
            import open3d

            open3d_mesh = open3d.geometry.TriangleMesh(
                vertices=open3d.utility.Vector3dVector(trimesh_mesh.vertices),
                triangles=open3d.utility.Vector3iVector(trimesh_mesh.faces),
            )
            open3d_mesh.orient_triangles()
            open3d_mesh.compute_triangle_normals()

            trimesh_mesh = trimesh.Trimesh(
                vertices=np.asarray(open3d_mesh.vertices),
                faces=np.asarray(open3d_mesh.triangles),
                vertex_colors=vertex_colors,
                process=False,
            )
        except Exception:
            pass

        trimesh.repair.fix_normals(trimesh_mesh)
        if trimesh_mesh.is_watertight and trimesh_mesh.volume < 0:
            trimesh_mesh.invert()

        return trimesh_mesh

    def get_mesh_info(self) -> dict:
        """
        Return basic information about the current mesh state.

        Returns:
            Dict with keys ``vertices``, ``faces``, ``has_vertex_color``,
            ``is_watertight``, ``boundary_edge_count``, ``boundary_loop_count``,
            and ``is_watertight_except_outer_boundary``.
        """
        mesh = self.ms.current_mesh()
        vertices = mesh.vertex_matrix()
        faces = mesh.face_matrix()

        trimesh_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
        )

        # Count boundary edges and connected boundary loops using trimesh.
        edge_face_incidence = np.bincount(trimesh_mesh.edges_unique_inverse)
        boundary_edge_mask = edge_face_incidence == 1
        boundary_edges = trimesh_mesh.edges_unique[boundary_edge_mask]
        boundary_edge_count = int(boundary_edges.shape[0])

        boundary_loop_count = 0
        if boundary_edge_count > 0:
            try:
                boundary_components = trimesh.graph.connected_components(
                    boundary_edges,
                    min_len=1,
                )
                boundary_loop_count = len(boundary_components)
            except Exception:
                # Fall back to a conservative count if trimesh graph utilities
                # are unavailable in the installed version.
                boundary_loop_count = 1

        is_watertight = bool(trimesh_mesh.is_watertight)
        is_edge_manifold_with_boundary = bool(
            trimesh_mesh.is_watertight or boundary_edge_count > 0
        )

        try:
            import open3d

            open3d_mesh = open3d.geometry.TriangleMesh(
                vertices=open3d.utility.Vector3dVector(vertices),
                triangles=open3d.utility.Vector3iVector(faces),
            )
            is_edge_manifold_with_boundary = bool(
                open3d_mesh.is_edge_manifold(allow_boundary_edges=True)
            )
            is_watertight = bool(open3d_mesh.is_watertight())
        except Exception:
            pass

        is_watertight_except_outer_boundary = bool(
            is_watertight
            or (
                is_edge_manifold_with_boundary
                and boundary_loop_count == 1
            )
        )

        return {
            "vertices": mesh.vertex_number(),
            "faces": mesh.face_number(),
            "has_vertex_color": mesh.has_vertex_color(),
            "is_watertight": is_watertight,
            "is_edge_manifold_with_boundary": is_edge_manifold_with_boundary,
            "boundary_edge_count": boundary_edge_count,
            "boundary_loop_count": boundary_loop_count,
            "is_watertight_except_outer_boundary": is_watertight_except_outer_boundary,
        }

    def _current_trimesh(self) -> trimesh.Trimesh:
        """Build a trimesh view of the current PyMeshLab mesh."""
        mesh = self.ms.current_mesh()
        return trimesh.Trimesh(
            vertices=mesh.vertex_matrix(),
            faces=mesh.face_matrix(),
            process=False,
        )

    def get_boundary_diagnostics(self) -> dict:
        """
        Return boundary-edge and boundary-loop diagnostics for the current mesh.

        Returns:
            Dict with boundary edges, boundary loops, and the faces incident
            to each boundary loop.
        """
        trimesh_mesh = self._current_trimesh()
        edge_face_incidence = np.bincount(trimesh_mesh.edges_unique_inverse)
        boundary_edge_ids = np.flatnonzero(edge_face_incidence == 1)
        boundary_edges = trimesh_mesh.edges_unique[boundary_edge_ids]

        edge_to_faces = [set() for _ in range(len(trimesh_mesh.edges_unique))]
        faces_unique_edges = getattr(trimesh_mesh, "faces_unique_edges", None)
        if faces_unique_edges is not None:
            for face_idx, edge_ids in enumerate(faces_unique_edges):
                for edge_id in edge_ids:
                    edge_to_faces[int(edge_id)].add(int(face_idx))
        else:
            face_edges = trimesh_mesh.edges_unique_inverse.reshape((-1, 3))
            for face_idx, edge_ids in enumerate(face_edges):
                for edge_id in edge_ids:
                    edge_to_faces[int(edge_id)].add(int(face_idx))

        loops = []
        loop_face_indices = set()
        if len(boundary_edges) > 0:
            boundary_components = trimesh.graph.connected_components(boundary_edges, min_len=1)
            for loop_index, component_vertices in enumerate(boundary_components):
                component_vertices = set(np.asarray(component_vertices, dtype=int).tolist())
                component_edge_ids = [
                    int(edge_id)
                    for edge_id in boundary_edge_ids
                    if set(map(int, trimesh_mesh.edges_unique[int(edge_id)])).issubset(component_vertices)
                ]
                component_faces = sorted(
                    {
                        face_id
                        for edge_id in component_edge_ids
                        for face_id in edge_to_faces[edge_id]
                    }
                )
                loop_face_indices.update(component_faces)
                loops.append(
                    {
                        "loop_index": loop_index,
                        "vertex_indices": sorted(component_vertices),
                        "edge_indices": component_edge_ids,
                        "edge_count": len(component_edge_ids),
                        "face_indices": component_faces,
                    }
                )

        loops.sort(key=lambda item: item["edge_count"], reverse=True)
        return {
            "boundary_edge_count": int(len(boundary_edge_ids)),
            "boundary_edges": boundary_edges,
            "boundary_loop_count": int(len(loops)),
            "boundary_loops": loops,
            "boundary_face_indices": sorted(loop_face_indices),
        }

    def repair_small_holes(self, maxhole_size: int = 30,
                           remesh_iterations: int = 5) -> bool:
        """
        Try to close small boundary loops and locally regularize the area.

        Returns:
            True if the repair pass completed without raising, otherwise False.
        """
        try:
            self.ms.meshing_close_holes(
                maxholesize=maxhole_size,
                selected=False,
                newfaceselected=False,
                selfintersection=False,
            )
            if remesh_iterations > 0:
                self.ms.meshing_isotropic_explicit_remeshing(iterations=remesh_iterations)
            self.ms.meshing_close_holes(
                maxholesize=maxhole_size,
                selected=False,
                newfaceselected=False,
                selfintersection=False,
            )
            self.ms.compute_normal_per_face()
            self.ms.compute_normal_per_vertex()
            self.ms.meshing_remove_unreferenced_vertices()
            return True
        except Exception as exc:
            logger.warning(f"Small-hole repair skipped due to filter failure: {exc}")
            return False

    def _extract_face_submesh(self, face_indices: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract compact vertex/face arrays and original vertex ids for a face subset."""
        trimesh_mesh = self._current_trimesh()
        face_indices = np.asarray(face_indices, dtype=np.int64)
        if face_indices.size == 0:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
            )

        sub_faces = np.asarray(trimesh_mesh.faces[face_indices], dtype=np.int64)
        unique_vertex_ids, inverse = np.unique(sub_faces.reshape(-1), return_inverse=True)
        sub_vertices = np.asarray(trimesh_mesh.vertices[unique_vertex_ids], dtype=np.float64)
        sub_faces = inverse.reshape((-1, 3)).astype(np.int64)
        return sub_vertices, sub_faces, unique_vertex_ids.astype(np.int64)



class VoxelMesh:
    """
    A class for voxelizing 3D meshes and computing voxel-level properties.
    
    This class converts a mesh into a voxel grid and computes various statistics
    for each voxel including:
    - Average/median normal vectors
    - Normal variance (measure of surface smoothness)
    - Average color and color variance
    - Curvature proxy (based on normal variation)
    """
    
    def __init__(self, mesh_vertices: np.ndarray, mesh_faces: np.ndarray,
                 mesh_vertex_colors: Optional[np.ndarray] = None,
                 voxel_size: Optional[float] = None,
                 num_voxels: Optional[int] = None):
        """
        Initialize the VoxelMesh.
        
        Args:
            mesh_vertices: Array of shape (N, 3) containing vertex coordinates
            mesh_faces: Array of shape (M, 3) containing face indices
            mesh_vertex_colors: Array of shape (N, 3) or (N, 4) containing RGB or RGBA colors
            voxel_size: Size of each voxel (in mesh units). If None, calculated from num_voxels
            num_voxels: Approximate number of voxels along longest axis. Default is 50
        """
        self.vertices = mesh_vertices
        self.faces = mesh_faces
        self.vertex_colors = mesh_vertex_colors
        
        # Create trimesh object
        self.mesh = trimesh.Trimesh(
            vertices=self.vertices,
            faces=self.faces,
            process=False
        )
        
        # Calculate vertex normals
        self.vertex_normals = self.mesh.vertex_normals
        
        # Determine voxel size
        if voxel_size is None:
            if num_voxels is None:
                num_voxels = 50
            mesh_extent = self.mesh.extents.max()
            self.voxel_size = mesh_extent / num_voxels
        else:
            self.voxel_size = voxel_size
        
        # Initialize voxel properties
        self.voxel_grid = None
        self.voxel_centers = None
        self.voxel_properties = {}
        
        logger.info(f"Initialized VoxelMesh with {len(mesh_vertices)} vertices, "
                   f"{len(mesh_faces)} faces, voxel_size={self.voxel_size:.4f}")
    
    def voxelize(self):
        """
        Voxelize the mesh and compute voxel indices for each vertex.
        """
        logger.info("Voxelizing mesh...")
        
        # Get mesh bounds
        min_bound = self.vertices.min(axis=0)
        max_bound = self.vertices.max(axis=0)
        
        # Calculate grid dimensions
        grid_dims = np.ceil((max_bound - min_bound) / self.voxel_size).astype(int) + 1
        
        # Assign vertices to voxels
        voxel_indices = np.floor((self.vertices - min_bound) / self.voxel_size).astype(int)
        
        # Create unique voxel identifiers
        self.voxel_grid = {}
        for i, voxel_idx in enumerate(voxel_indices):
            voxel_key = tuple(voxel_idx)
            if voxel_key not in self.voxel_grid:
                self.voxel_grid[voxel_key] = {
                    'vertex_indices': [],
                    'normals': [],
                    'colors': []
                }
            self.voxel_grid[voxel_key]['vertex_indices'].append(i)
            self.voxel_grid[voxel_key]['normals'].append(self.vertex_normals[i])
            if self.vertex_colors is not None:
                self.voxel_grid[voxel_key]['colors'].append(self.vertex_colors[i])
        
        # Calculate voxel centers
        self.voxel_centers = {}
        for voxel_key in self.voxel_grid:
            voxel_coords = np.array(voxel_key)
            center = min_bound + (voxel_coords + 0.5) * self.voxel_size
            self.voxel_centers[voxel_key] = center
        
        logger.info(f"Created {len(self.voxel_grid)} voxels")
    
    def compute_voxel_properties(self):
        """
        Compute statistical properties for each voxel.
        
        Properties computed:
        - average_normal: Mean normal vector
        - median_normal: Median normal vector
        - normal_variance: Variance of normals (smoothness measure)
        - average_color: Mean color
        - color_variance: Variance of color
        - curvature_proxy: Measure of local curvature based on normal variation
        """
        if self.voxel_grid is None:
            self.voxelize()
        
        logger.info("Computing voxel properties...")
        
        for voxel_key, voxel_data in self.voxel_grid.items():
            normals = np.array(voxel_data['normals'])
            
            # Compute normal statistics
            avg_normal = np.mean(normals, axis=0)
            avg_normal_normalized = avg_normal / (np.linalg.norm(avg_normal) + 1e-8)
            
            median_normal = np.median(normals, axis=0)
            median_normal_normalized = median_normal / (np.linalg.norm(median_normal) + 1e-8)
            
            # Normal variance (measure of how much normals vary in this voxel)
            normal_variance = np.var(normals, axis=0).mean()
            
            # Curvature proxy: measure angular deviation from mean normal
            # Higher values indicate more curvature
            dot_products = np.dot(normals, avg_normal_normalized)
            dot_products = np.clip(dot_products, -1.0, 1.0)
            angles = np.arccos(dot_products)
            curvature_proxy = np.std(angles)
            
            # Compute color statistics if available
            avg_color = None
            color_variance = None
            if voxel_data['colors']:
                colors = np.array(voxel_data['colors'])
                avg_color = np.mean(colors, axis=0)
                color_variance = np.var(colors, axis=0).mean()
            
            # Store properties
            self.voxel_properties[voxel_key] = {
                'average_normal': avg_normal_normalized,
                'median_normal': median_normal_normalized,
                'normal_variance': normal_variance,
                'average_color': avg_color,
                'color_variance': color_variance,
                'curvature_proxy': curvature_proxy,
                'num_vertices': len(voxel_data['vertex_indices'])
            }
        
        logger.info(f"Computed properties for {len(self.voxel_properties)} voxels")
    
    def get_voxel_property_arrays(self):
        """
        Get voxel properties as numpy arrays for easy analysis.
        
        Returns:
            Dictionary containing arrays of voxel properties
        """
        if not self.voxel_properties:
            self.compute_voxel_properties()
        
        num_voxels = len(self.voxel_properties)
        centers = np.zeros((num_voxels, 3))
        avg_normals = np.zeros((num_voxels, 3))
        median_normals = np.zeros((num_voxels, 3))
        normal_variances = np.zeros(num_voxels)
        curvature_proxies = np.zeros(num_voxels)
        vertex_counts = np.zeros(num_voxels, dtype=int)
        
        avg_colors = []
        color_variances = []
        
        voxel_keys = []
        for i, (voxel_key, props) in enumerate(self.voxel_properties.items()):
            voxel_keys.append(voxel_key)
            centers[i] = self.voxel_centers[voxel_key]
            avg_normals[i] = props['average_normal']
            median_normals[i] = props['median_normal']
            normal_variances[i] = props['normal_variance']
            curvature_proxies[i] = props['curvature_proxy']
            vertex_counts[i] = props['num_vertices']
            
            if props['average_color'] is not None:
                avg_colors.append(props['average_color'])
                color_variances.append(props['color_variance'])
        
        result = {
            'centers': centers,
            'average_normals': avg_normals,
            'median_normals': median_normals,
            'normal_variances': normal_variances,
            'curvature_proxies': curvature_proxies,
            'vertex_counts': vertex_counts,
            'voxel_keys': voxel_keys
        }
        
        if avg_colors:
            result['average_colors'] = np.array(avg_colors)
            result['color_variances'] = np.array(color_variances)
        
        return result
    
    def visualize_in_rerun(self, session_name: str = "voxel_mesh",
                          show_normals: bool = True,
                          show_curvature: bool = True,
                          normal_scale: float = 0.02):
        """
        Visualize voxel properties in Rerun.
        
        Args:
            session_name: Name of the Rerun session
            show_normals: Whether to show average normal vectors
            show_curvature: Whether to color voxels by curvature
            normal_scale: Scale factor for normal vector visualization
        """
        if not self.voxel_properties:
            self.compute_voxel_properties()
        
        rr.init(session_name, spawn=True)
        
        logger.info("Visualizing voxel mesh in Rerun...")
        
        # Get property arrays
        props = self.get_voxel_property_arrays()
        centers = props['centers']
        avg_normals = props['average_normals']
        curvature = props['curvature_proxies']
        
        # Visualize voxel centers as points
        if show_curvature:
            # Normalize curvature for color mapping (0 = blue/low, 1 = red/high)
            curvature_normalized = (curvature - curvature.min()) / (curvature.max() - curvature.min() + 1e-8)
            colors = self._curvature_to_color(curvature_normalized)
            
            rr.log(
                "voxels/centers",
                rr.Points3D(
                    positions=centers,
                    colors=colors,
                    radii=self.voxel_size * 0.3
                )
            )
        elif 'average_colors' in props:
            rr.log(
                "voxels/centers",
                rr.Points3D(
                    positions=centers,
                    colors=props['average_colors'],
                    radii=self.voxel_size * 0.3
                )
            )
        else:
            rr.log(
                "voxels/centers",
                rr.Points3D(
                    positions=centers,
                    radii=self.voxel_size * 0.3
                )
            )
        
        # Visualize normals
        if show_normals:
            normal_length = self.mesh.extents.max() * normal_scale
            end_points = centers + avg_normals * normal_length
            
            # Log all normal vectors as line segments in a single static call
            line_strips = [np.array([start, end]) for start, end in zip(centers, end_points)]
            rr.log(
                "voxels/normals",
                rr.LineStrips3D(strips=line_strips)
            )
        
        logger.info("Voxel visualization logged to Rerun")
    
    def _curvature_to_color(self, curvature_normalized: np.ndarray) -> np.ndarray:
        """
        Convert normalized curvature values to RGB colors (blue to red colormap).
        
        Args:
            curvature_normalized: Normalized curvature values (0-1)
            
        Returns:
            RGB colors array
        """
        colors = np.zeros((len(curvature_normalized), 3), dtype=np.uint8)
        
        # Blue (low curvature) to Red (high curvature) via green
        colors[:, 2] = (255 * (1 - curvature_normalized)).astype(np.uint8)  # Blue
        colors[:, 0] = (255 * curvature_normalized).astype(np.uint8)         # Red
        colors[:, 1] = (255 * (1 - np.abs(curvature_normalized - 0.5) * 2)).astype(np.uint8)  # Green
        
        return colors
    
    def get_high_curvature_voxels(self, threshold_percentile: float = 90) -> np.ndarray:
        """
        Get voxel centers with high curvature (potential edge/feature locations).
        
        Args:
            threshold_percentile: Percentile threshold for high curvature (0-100)
            
        Returns:
            Array of voxel centers with high curvature
        """
        if not self.voxel_properties:
            self.compute_voxel_properties()
        
        props = self.get_voxel_property_arrays()
        curvature = props['curvature_proxies']
        centers = props['centers']
        
        threshold = np.percentile(curvature, threshold_percentile)
        high_curvature_mask = curvature >= threshold
        
        logger.info(f"Found {high_curvature_mask.sum()} high-curvature voxels "
                   f"(threshold: {threshold:.4f})")
        
        return centers[high_curvature_mask]
    
    def get_statistics(self) -> dict:
        """
        Get overall statistics about the voxel mesh.
        
        Returns:
            Dictionary containing voxel mesh statistics
        """
        if not self.voxel_properties:
            self.compute_voxel_properties()
        
        props = self.get_voxel_property_arrays()
        
        stats = {
            'num_voxels': len(self.voxel_properties),
            'voxel_size': self.voxel_size,
            'avg_vertices_per_voxel': np.mean(props['vertex_counts']),
            'normal_variance_mean': np.mean(props['normal_variances']),
            'normal_variance_std': np.std(props['normal_variances']),
            'curvature_proxy_mean': np.mean(props['curvature_proxies']),
            'curvature_proxy_std': np.std(props['curvature_proxies']),
            'curvature_proxy_max': np.max(props['curvature_proxies'])
        }
        
        if 'average_colors' in props:
            stats['color_variance_mean'] = np.mean(props['color_variances'])
            stats['color_variance_std'] = np.std(props['color_variances'])
        
        return stats


# Example usage
if __name__ == "__main__":
    MESH_PATH = r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse\dense\poisson.ply"

    # ---- MeshSimplifier workflow ----
    simplifier = MeshSimplifier(MESH_PATH)
    print("Before simplification:", simplifier.get_mesh_info())

    # Full pipeline: 3× Laplacian smoothing, then clustering decimation at 1 %
    simplifier.get_mesh_info()
    simplifier.run(smooth_iterations=3)
    print("After simplification:", simplifier.get_mesh_info())

    #simplifier.save(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Bein4\dense\test.ply")
    # Convert to trimesh for downstream use (e.g. VoxelMesh)
    mesh = simplifier.to_trimesh()
    mesh.show()
    # # Extract vertex colors if available, otherwise pass None for default colors
    # vertex_colors = mesh.visual.vertex_colors if hasattr(mesh.visual, 'vertex_colors') else None

    # voxel_mesh = VoxelMesh(
    #     mesh.vertices,
    #     mesh.faces,
    #     mesh_vertex_colors=vertex_colors,
    #     num_voxels=50  # Create ~50 voxels along longest axis
    # )
    
    # # Compute voxel properties
    # voxel_mesh.compute_voxel_properties()
    
    # # Get statistics
    # voxel_stats = voxel_mesh.get_statistics()
    # print("\nVoxel Mesh Statistics:")
    # for key, value in voxel_stats.items():
    #     print(f"  {key}: {value}")
    
    # # Get high curvature voxels (potential edges/features)
    # high_curv_voxels = voxel_mesh.get_high_curvature_voxels(threshold_percentile=90)
    # print(f"\nHigh curvature voxels: {len(high_curv_voxels)}")
    
    # # Visualize voxel mesh
    # voxel_mesh.visualize_in_rerun(
    #     session_name="voxel_analysis",
    #     show_normals=True,
    #     show_curvature=True,
    #     normal_scale=0.02
    # )


