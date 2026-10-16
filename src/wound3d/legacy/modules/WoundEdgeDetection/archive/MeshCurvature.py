"""
Mesh Curvature Module

Computes discrete curvature on triangular meshes with configurable neighbourhood
sizes.  Supports separate curvature statistics for wound and non-wound regions
and provides comparative analysis (mean difference, std deviation) between them.

Curvature methods implemented:
    - Mean curvature  (discrete Laplace-Beltrami approximation)
    - Gaussian curvature (angle-deficit method)
    - Shape index  (derived from principal curvatures)

Classes:
    MeshCurvature: Core curvature computation and region analysis.
"""

import numpy as np
import trimesh
from typing import Optional, Dict, Set, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MeshCurvature:
    """
    Discrete curvature computation on a triangular mesh.

    The class computes per-vertex curvature values using a configurable
    neighbourhood ring size.  It can hold separate curvature arrays for
    wound and non-wound regions, and compute comparative statistics
    (difference of means, per-region std, etc.).

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)
        Mesh vertex positions.
    faces : np.ndarray, shape (M, 3)
        Triangle indices.
    vertex_colors : np.ndarray, optional, shape (N, 3|4)
        Per-vertex RGB(A) colours.
    neighbourhood_rings : int
        Number of adjacency rings to include in the local neighbourhood.
        Higher values = larger neighbourhood = smoother curvature estimate.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        vertex_colors: Optional[np.ndarray] = None,
        neighbourhood_rings: int = 2,
    ):
        self.vertices = vertices
        self.faces = faces
        self.vertex_colors = vertex_colors
        self.neighbourhood_rings = neighbourhood_rings

        # Build trimesh for helper queries
        self.mesh = trimesh.Trimesh(
            vertices=self.vertices,
            faces=self.faces,
            process=False,
        )

        # Adjacency: vertex -> set of neighbouring vertex indices
        self._vertex_adjacency: Dict[int, Set[int]] = {}
        self._build_vertex_adjacency()

        # Per-vertex curvature arrays (computed lazily)
        self.mean_curvature: Optional[np.ndarray] = None
        self.gaussian_curvature: Optional[np.ndarray] = None

        # Region masks & curvature splits
        self.wound_vertex_mask: Optional[np.ndarray] = None  # bool array
        self.wound_curvature: Optional[np.ndarray] = None
        self.non_wound_curvature: Optional[np.ndarray] = None

        # Comparative statistics
        self.comparison_stats: Optional[Dict] = None

        logger.info(
            f"MeshCurvature initialised – {len(vertices)} verts, "
            f"{len(faces)} faces, neighbourhood_rings={neighbourhood_rings}"
        )

    # ------------------------------------------------------------------
    # Adjacency helpers
    # ------------------------------------------------------------------
    def _build_vertex_adjacency(self):
        """Build a 1-ring vertex adjacency from the face array."""
        adj: Dict[int, Set[int]] = {i: set() for i in range(len(self.vertices))}
        for f in self.faces:
            v0, v1, v2 = int(f[0]), int(f[1]), int(f[2])
            adj[v0].update([v1, v2])
            adj[v1].update([v0, v2])
            adj[v2].update([v0, v1])
        self._vertex_adjacency = adj

    def get_neighbourhood(self, vertex_idx: int, rings: Optional[int] = None) -> Set[int]:
        """
        Expand neighbourhood around *vertex_idx* by the given number of
        adjacency rings (defaults to ``self.neighbourhood_rings``).

        Returns
        -------
        Set[int]
            All vertex indices within the specified ring distance (excluding
            the seed vertex).
        """
        if rings is None:
            rings = self.neighbourhood_rings
        visited: Set[int] = {vertex_idx}
        frontier: Set[int] = {vertex_idx}
        for _ in range(rings):
            next_frontier: Set[int] = set()
            for v in frontier:
                next_frontier.update(self._vertex_adjacency.get(v, set()))
            next_frontier -= visited
            visited.update(next_frontier)
            frontier = next_frontier
        visited.discard(vertex_idx)
        return visited

    # ------------------------------------------------------------------
    # Curvature computation
    # ------------------------------------------------------------------
    def _get_curvature_radius(self) -> float:
        """Estimate a curvature neighborhood radius from mesh edge lengths."""
        if hasattr(self.mesh, "edges_unique_length") and len(self.mesh.edges_unique_length) > 0:
            avg_edge = float(np.mean(self.mesh.edges_unique_length))
        else:
            avg_edge = float(np.linalg.norm(self.mesh.extents) / max(len(self.vertices), 1))
        radius = max(avg_edge * max(self.neighbourhood_rings, 1), avg_edge * 1.5)
        return radius

    def compute_mean_curvature(self) -> np.ndarray:
        """
        Compute mean curvature using trimesh.curvature.discrete_mean_curvature_measure.

        The curvature is estimated within a ball of radius derived from the
        neighbourhood ring count and the average edge length.

        Returns
        -------
        np.ndarray, shape (N,)
            Per-vertex mean curvature measure.
        """
        logger.info("Computing mean curvature (trimesh.curvature)…")

        radius = self._get_curvature_radius()
        H = trimesh.curvature.discrete_mean_curvature_measure(
            self.mesh,
            self.vertices,
            radius=3,
        )

        self.mean_curvature = np.asarray(H, dtype=np.float64)
        logger.info("Mean curvature computed.")
        return self.mean_curvature

    def compute_gaussian_curvature(self) -> np.ndarray:
        """
        Compute Gaussian curvature using trimesh.curvature.discrete_gaussian_curvature_measure.

        Returns
        -------
        np.ndarray, shape (N,)
            Per-vertex Gaussian curvature measure.
        """
        logger.info("Computing Gaussian curvature (trimesh.curvature)…")

        radius = self._get_curvature_radius()
        K = trimesh.curvature.discrete_gaussian_curvature_measure(
            self.mesh,
            self.vertices,
            radius=0.1,
        )

        self.gaussian_curvature = np.asarray(K, dtype=np.float64)
        logger.info("Gaussian curvature computed.")
        return self.gaussian_curvature

    def compute_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience: compute both mean and Gaussian curvature."""
        H = self.compute_mean_curvature()
        #K = self.compute_gaussian_curvature()
        return H#, K

    # ------------------------------------------------------------------
    # Wound / non-wound region handling
    # ------------------------------------------------------------------
    def set_wound_mask_from_face_set(
        self,
        wound_face_indices: Set[int],
    ):
        """
        Mark vertices as wound / non-wound based on a set of wound face indices.

        A vertex is classified as *wound* if it belongs to at least one wound
        face.  This mirrors how ``WoundModel3DMesh`` tracks wound vertices.

        Parameters
        ----------
        wound_face_indices : Set[int]
            Indices into ``self.faces`` that belong to the wound region.
        """
        mask = np.zeros(len(self.vertices), dtype=bool)
        for fi in wound_face_indices:
            for vi in self.faces[fi]:
                mask[int(vi)] = True
        self.wound_vertex_mask = mask
        logger.info(
            f"Wound mask set: {mask.sum()} wound vertices, "
            f"{(~mask).sum()} non-wound vertices"
        )

    def set_wound_mask_from_vertex_set(
        self,
        wound_vertex_indices: Set[int],
    ):
        """
        Directly set the wound mask from a set of vertex indices.

        Parameters
        ----------
        wound_vertex_indices : Set[int]
            Vertex indices belonging to the wound.
        """
        mask = np.zeros(len(self.vertices), dtype=bool)
        for vi in wound_vertex_indices:
            mask[int(vi)] = True
        self.wound_vertex_mask = mask
        logger.info(
            f"Wound mask set: {mask.sum()} wound vertices, "
            f"{(~mask).sum()} non-wound vertices"
        )

    def split_curvature_by_region(self, curvature: Optional[np.ndarray] = None):
        """
        Split a curvature array into wound and non-wound parts.

        Parameters
        ----------
        curvature : np.ndarray, optional
            Per-vertex curvature array.  Defaults to ``self.mean_curvature``.

        Populates
        ---------
        self.wound_curvature, self.non_wound_curvature
        """
        if curvature is None:
            if self.mean_curvature is None:
                self.compute_mean_curvature()
            curvature = self.mean_curvature

        if self.wound_vertex_mask is None:
            raise ValueError(
                "Wound mask not set. Call set_wound_mask_from_face_set() "
                "or set_wound_mask_from_vertex_set() first."
            )

        self.wound_curvature = curvature[self.wound_vertex_mask]
        self.non_wound_curvature = curvature[~self.wound_vertex_mask]
        logger.info(
            f"Curvature split – wound: {len(self.wound_curvature)} verts, "
            f"non-wound: {len(self.non_wound_curvature)} verts"
        )

    # ------------------------------------------------------------------
    # Comparative statistics
    # ------------------------------------------------------------------
    def compute_comparison_statistics(
        self,
        curvature: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compute comparative statistics between wound and non-wound curvature.

        Computes for each region:
            mean, median, std, min, max

        And across regions:
            mean_difference  = mean(wound) − mean(non-wound)
            median_difference
            std_ratio        = std(wound) / std(non-wound)

        Parameters
        ----------
        curvature : np.ndarray, optional
            Per-vertex curvature.  Defaults to ``self.mean_curvature``.

        Returns
        -------
        dict
            Nested dictionary with keys ``wound``, ``non_wound``, ``comparison``.
        """
        self.split_curvature_by_region(curvature)

        def _region_stats(arr: np.ndarray) -> Dict:
            if len(arr) == 0:
                return {k: 0.0 for k in ("mean", "median", "std", "min", "max", "count")}
            return {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": int(len(arr)),
            }

        wound_stats = _region_stats(self.wound_curvature)
        non_wound_stats = _region_stats(self.non_wound_curvature)

        # Cross-region comparison
        mean_diff = wound_stats["mean"] - non_wound_stats["mean"]
        median_diff = wound_stats["median"] - non_wound_stats["median"]
        std_ratio = (
            wound_stats["std"] / non_wound_stats["std"]
            if non_wound_stats["std"] > 1e-12
            else float("inf")
        )

        self.comparison_stats = {
            "wound": wound_stats,
            "non_wound": non_wound_stats,
            "comparison": {
                "mean_difference": mean_diff,
                "median_difference": median_diff,
                "std_ratio": std_ratio,
            },
        }
        logger.info(
            f"Curvature comparison – mean diff: {mean_diff:.6f}, "
            f"std ratio: {std_ratio:.4f}"
        )
        return self.comparison_stats

    # ------------------------------------------------------------------
    # Neighbourhood size adaption
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_neighbourhood_rings(
        total_faces: int,
        wound_faces: int,
        base_rings: int = 2,
        min_rings: int = 1,
        max_rings: int = 10,
    ) -> int:
        """
        Heuristic: adapt the neighbourhood ring count to the wound size
        relative to the full mesh.

        Smaller wounds (few faces) benefit from a larger neighbourhood to
        gather enough context, while larger wounds can use a tighter ring.

        Parameters
        ----------
        total_faces : int
            Total number of mesh faces.
        wound_faces : int
            Number of wound faces.
        base_rings : int
            Default ring count.
        min_rings, max_rings : int
            Allowed range.

        Returns
        -------
        int
            Recommended neighbourhood ring count.
        """
        if wound_faces <= 0 or total_faces <= 0:
            return base_rings
        ratio = wound_faces / total_faces
        # Small wound → more rings; large wound → fewer rings
        if ratio < 0.01:
            rings = base_rings + 3
        elif ratio < 0.05:
            rings = base_rings + 2
        elif ratio < 0.15:
            rings = base_rings + 1
        elif ratio > 0.40:
            rings = max(base_rings - 1, min_rings)
        else:
            rings = base_rings
        return int(np.clip(rings, min_rings, max_rings))

    # ------------------------------------------------------------------
    # Rerun visualisation
    # ------------------------------------------------------------------
    def visualize_in_rerun(
        self,
        session_name: str = "mesh_curvature",
        curvature: Optional[np.ndarray] = None,
        show_wound_boundary: bool = True,
    ):
        """
        Visualise the mesh coloured by curvature in Rerun.

        Parameters
        ----------
        session_name : str
            Rerun recording name.
        curvature : np.ndarray, optional
            Per-vertex curvature to colour-code.  Defaults to mean curvature.
        show_wound_boundary : bool
            If True and a wound mask exists, highlight wound vs non-wound.
        """
        import rerun as rr

        if curvature is None:
            if self.mean_curvature is None:
                self.compute_mean_curvature()
            curvature = self.mean_curvature

        rr.init(session_name, spawn=True)

        # Normalise curvature for colour mapping
        c_min, c_max = np.percentile(curvature, [2, 98])
        if abs(c_max - c_min) < 1e-12:
            c_norm = np.zeros_like(curvature)
        else:
            c_norm = np.clip((curvature - c_min) / (c_max - c_min), 0, 1)

        colors = self._scalar_to_color(c_norm)

        rr.log(
            "curvature/mesh",
            rr.Mesh3D(
                vertex_positions=self.vertices,
                triangle_indices=self.faces,
                vertex_colors=colors,
            ),
            static=True,
        )

        # Overlay wound / non-wound as points
        if show_wound_boundary and self.wound_vertex_mask is not None:
            wound_pts = self.vertices[self.wound_vertex_mask]
            non_wound_pts = self.vertices[~self.wound_vertex_mask]

            if len(wound_pts) > 0:
                rr.log(
                    "curvature/wound_vertices",
                    rr.Points3D(
                        positions=wound_pts,
                        colors=np.array([[255, 0, 0]] * len(wound_pts), dtype=np.uint8),
                        radii=0.001,
                    ),
                    static=True,
                )

        # Log statistics as text
        if self.comparison_stats is not None:
            stats_lines = ["Curvature Comparison Statistics", "=" * 40]
            for region in ("wound", "non_wound"):
                stats_lines.append(f"\n{region.upper()}:")
                for k, v in self.comparison_stats[region].items():
                    stats_lines.append(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
            stats_lines.append("\nCOMPARISON:")
            for k, v in self.comparison_stats["comparison"].items():
                stats_lines.append(f"  {k}: {v:.6f}")

            rr.log("curvature/stats", rr.TextDocument("\n".join(stats_lines)), static=True)

        logger.info("Curvature visualization logged to Rerun.")

    @staticmethod
    def _scalar_to_color(values: np.ndarray) -> np.ndarray:
        """Map normalised 0-1 scalars to a blue → green → red colour-map."""
        colors = np.zeros((len(values), 3), dtype=np.uint8)
        colors[:, 0] = (255 * values).astype(np.uint8)  # Red
        colors[:, 2] = (255 * (1 - values)).astype(np.uint8)  # Blue
        colors[:, 1] = (255 * (1 - np.abs(values - 0.5) * 2)).astype(np.uint8)  # Green
        return colors

    # ------------------------------------------------------------------
    # Convenience summary
    # ------------------------------------------------------------------
    def get_statistics(self) -> Dict:
        """Return a summary dictionary of all computed curvature data."""
        stats: Dict = {
            "num_vertices": len(self.vertices),
            "num_faces": len(self.faces),
            "neighbourhood_rings": self.neighbourhood_rings,
            "mean_curvature_computed": self.mean_curvature is not None,
            "gaussian_curvature_computed": self.gaussian_curvature is not None,
            "wound_mask_set": self.wound_vertex_mask is not None,
        }
        if self.mean_curvature is not None:
            stats["mean_curvature_mean"] = float(np.mean(self.mean_curvature))
            stats["mean_curvature_std"] = float(np.std(self.mean_curvature))
        if self.gaussian_curvature is not None:
            stats["gaussian_curvature_mean"] = float(np.mean(self.gaussian_curvature))
            stats["gaussian_curvature_std"] = float(np.std(self.gaussian_curvature))
        if self.comparison_stats is not None:
            stats["comparison"] = self.comparison_stats
        return stats


"""
Curvature estimation methods (PCA and Principal) with Rerun visualization.
"""

import numpy as np
import trimesh
import rerun as rr
from scipy.spatial import cKDTree
from typing import Tuple, Optional


def estimate_pca_curvature(vertices: np.ndarray, k_neighbors: int = 30) -> np.ndarray:
	"""
	Estimate curvature using PCA on local neighborhoods.

	Curvature proxy = smallest eigenvalue / sum(eigenvalues).
	Values are in [0, 1], where:
		- near 0: locally planar
		- higher values: sharper curvature / edges
	"""
	if k_neighbors < 5:
		raise ValueError("k_neighbors must be >= 5")

	tree = cKDTree(vertices)
	curvature = np.zeros(len(vertices), dtype=np.float64)

	for i, v in enumerate(vertices):
		dists, idx = tree.query(v, k=k_neighbors)
		neigh = vertices[idx]
		neigh_centered = neigh - np.mean(neigh, axis=0, keepdims=True)
		cov = (neigh_centered.T @ neigh_centered) / max(len(neigh_centered), 1)
		eigvals = np.linalg.eigvalsh(cov)
		eigvals = np.maximum(eigvals, 0.0)
		denom = np.sum(eigvals)
		curvature[i] = (eigvals[0] / denom) if denom > 1e-12 else 0.0

	return curvature


def curvature_to_colors(values: np.ndarray, clip_percentiles: Tuple[float, float] = (2.0, 98.0)) -> np.ndarray:
	"""
	Map curvature values to RGB colors (blue -> green -> red).
	"""
	lo, hi = np.percentile(values, clip_percentiles)
	if abs(hi - lo) < 1e-12:
		norm = np.zeros_like(values)
	else:
		norm = np.clip((values - lo) / (hi - lo), 0.0, 1.0)

	colors = np.zeros((len(values), 3), dtype=np.uint8)
	colors[:, 0] = (255 * norm).astype(np.uint8)
	colors[:, 2] = (255 * (1.0 - norm)).astype(np.uint8)
	colors[:, 1] = (255 * (1.0 - np.abs(norm - 0.5) * 2.0)).astype(np.uint8)
	return colors


def visualize_curvature_rerun(vertices: np.ndarray, faces: np.ndarray, curvature: np.ndarray):
	"""
	Visualize mesh with PCA-curvature colors and legend in Rerun.
	"""
	colors = curvature_to_colors(curvature)
	rr.init("pca_curvature", spawn=True)

	rr.log(
		"mesh",
		rr.Mesh3D(
			vertex_positions=vertices,
			triangle_indices=faces,
			vertex_colors=colors,
		),
		static=True,
	)

	curv_min = float(np.min(curvature))
	curv_max = float(np.max(curvature))
	curv_mean = float(np.mean(curvature))

	legend = (
		"PCA Curvature Proxy (lambda_min / sum(lambda))\n"
		"----------------------------------------------\n"
		f"min:  {curv_min:.6f}\n"
		f"mean: {curv_mean:.6f}\n"
		f"max:  {curv_max:.6f}\n"
		"\nColor scale: blue=low (planar), red=high (edge/curved)"
	)
	rr.log("curvature_legend", rr.TextDocument(legend), static=True)


def estimate_principal_curvature(
	mesh: trimesh.Trimesh, 
	radius: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Estimate mean and Gaussian curvature using discrete differential geometry.
	
	Uses trimesh's discrete curvature measures:
	- Mean curvature: H = (k1 + k2) / 2 (average of principal curvatures)
	- Gaussian curvature: K = k1 * k2 (product of principal curvatures)
	
	Args:
		mesh: Input trimesh mesh
		radius: Sampling radius. If None, auto-computed as 5% of aabb diagonal
		
	Returns:
		Tuple of (mean_curvature, gaussian_curvature) arrays per vertex
	"""
	if radius is None:
		bbox = mesh.bounds
		diag = np.linalg.norm(bbox[1] - bbox[0])
		radius = 0.05 * diag
	
	# Compute discrete mean curvature measure
	mean_curv = trimesh.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices, radius)
	
	# Compute discrete Gaussian curvature measure
	gauss_curv = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, radius)
	
	return mean_curv, gauss_curv


def visualize_principal_curvature_rerun(
	vertices: np.ndarray, 
	faces: np.ndarray, 
	mean_curvature: np.ndarray, 
	gaussian_curvature: np.ndarray
):
	"""
	Visualize mesh with mean and Gaussian curvature in separate Rerun entities.
	"""
	rr.init("principal_curvature", spawn=True)
	
	# Mean curvature visualization
	mean_colors = curvature_to_colors(mean_curvature)
	rr.log(
		"mesh/mean_curvature",
		rr.Mesh3D(
			vertex_positions=vertices,
			triangle_indices=faces,
			vertex_colors=mean_colors,
		),
		static=True,
	)
	
	mean_min = float(np.min(mean_curvature))
	mean_max = float(np.max(mean_curvature))
	mean_mean = float(np.mean(mean_curvature))
	mean_legend = (
		"Mean Curvature H = (k1 + k2) / 2\n"
		"----------------------------------\n"
		f"min:  {mean_min:.8f}\n"
		f"mean: {mean_mean:.8f}\n"
		f"max:  {mean_max:.8f}\n"
		"\nColor: blue=negative (saddle), red=positive (convex/concave)"
	)
	rr.log("legends/mean_curvature", rr.TextDocument(mean_legend), static=True)
	
	# Gaussian curvature visualization
	gauss_colors = curvature_to_colors(gaussian_curvature)
	rr.log(
		"mesh/gaussian_curvature",
		rr.Mesh3D(
			vertex_positions=vertices,
			triangle_indices=faces,
			vertex_colors=gauss_colors,
		),
		static=True,
	)
	
	gauss_min = float(np.min(gaussian_curvature))
	gauss_max = float(np.max(gaussian_curvature))
	gauss_mean = float(np.mean(gaussian_curvature))
	gauss_legend = (
		"Gaussian Curvature K = k1 * k2\n"
		"-------------------------------\n"
		f"min:  {gauss_min:.8f}\n"
		f"mean: {gauss_mean:.8f}\n"
		f"max:  {gauss_max:.8f}\n"
		"\nColor: blue=negative (saddle), red=positive (elliptic)"
	)
	rr.log("legends/gaussian_curvature", rr.TextDocument(gauss_legend), static=True)


def compute_shape_index(
	mesh: trimesh.Trimesh,
	radius: Optional[float] = None,
	basin_threshold: float = -0.5,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Compute the Shape Index S from principal curvatures k1, k2.

	Definition (Koenderink & van Doorn, 1992):
		S = (2/pi) * arctan((k1 + k2) / (k2 - k1))  with k1 <= k2
		Equivalently: S = (2/pi) * arctan(H / sqrt(H^2 - K))

	Shape Index ranges [-1, 1]:
		 S = -1.0  spherical cup  (deepest concave basin)
		 S = -0.5  trough / rut
		 S =  0.0  saddle
		 S =  0.5  ridge
		 S =  1.0  spherical cap  (convex dome)

	Basin detection: vertices with S < basin_threshold are classified as basin.

	Args:
		mesh: Input trimesh mesh
		radius: Sampling radius for discrete curvature measures.
		        If None, auto-set to 5% of bounding-box diagonal.
		basin_threshold: Shape-index threshold for basin classification (default -0.5).

	Returns:
		shape_index: Per-vertex shape index in [-1, 1] (NaN where undefined)
		basin_mask:  Boolean array, True where S < basin_threshold
	"""
	if radius is None:
		diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
		radius = 0.05 * diag

	# Discrete curvature measures (integrated over ball of radius r)
	H_meas = trimesh.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices, radius)
	K_meas = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, radius)

	# Normalize by ball-area approximation (pi * r^2) to get local curvature density
	ball_area = np.pi * radius ** 2
	H = H_meas / ball_area   # approx mean curvature
	K = K_meas / ball_area   # approx Gaussian curvature

	# Discriminant: H^2 - K  (must be >= 0 for real principal curvatures)
	discrim = H ** 2 - K

	# Shape index is undefined where discriminant is near zero (umbilic points)
	valid = discrim > 1e-12
	sqrt_discrim = np.where(valid, np.sqrt(np.maximum(discrim, 0.0)), 0.0)

	shape_index = np.full(len(mesh.vertices), np.nan, dtype=np.float64)
	shape_index[valid] = (2.0 / np.pi) * np.arctan2(H[valid], sqrt_discrim[valid])

	# Fill umbilic points with 0 (flat saddle)
	shape_index = np.where(valid, shape_index, 0.0)
	# Clamp to valid range
	shape_index = np.clip(shape_index, -1.0, 1.0)

	basin_mask = shape_index < basin_threshold

	return shape_index, basin_mask


def shape_index_to_colors(shape_index: np.ndarray) -> np.ndarray:
	"""
	Map shape index [-1, 1] to a perceptually meaningful colormap.

	Color meaning:
		Deep blue  : S = -1  (basin / cup)
		Cyan       : S = -0.5 (trough)
		Green      : S =  0  (saddle)
		Yellow     : S =  0.5 (ridge)
		Red        : S =  1  (dome / cap)
	"""
	# Normalize from [-1, 1] to [0, 1]
	norm = np.clip((shape_index + 1.0) / 2.0, 0.0, 1.0)

	colors = np.zeros((len(norm), 3), dtype=np.uint8)

	# Segment the colormap into 4 bands:
	# [0, 0.25] => blue  -> cyan  (basin)
	# [0.25, 0.50] => cyan -> green (trough->saddle)
	# [0.50, 0.75] => green -> yellow (saddle->ridge)
	# [0.75, 1.00] => yellow -> red (ridge->dome)
	t = norm

	r = np.where(
		t < 0.25, 0,
		np.where(t < 0.50, 0,
		np.where(t < 0.75, ((t - 0.50) / 0.25 * 255),
		255))
	)
	g = np.where(
		t < 0.25, (t / 0.25 * 255),
		np.where(t < 0.50, 255,
		np.where(t < 0.75, 255,
		(1.0 - (t - 0.75) / 0.25) * 255))
	)
	b = np.where(
		t < 0.25, 255,
		np.where(t < 0.50, (1.0 - (t - 0.25) / 0.25) * 255,
		0, )
	)

	colors[:, 0] = np.clip(r, 0, 255).astype(np.uint8)
	colors[:, 1] = np.clip(g, 0, 255).astype(np.uint8)
	colors[:, 2] = np.clip(b, 0, 255).astype(np.uint8)
	return colors


def visualize_shape_index_rerun(
	vertices: np.ndarray,
	faces: np.ndarray,
	shape_index: np.ndarray,
	basin_mask: np.ndarray,
	basin_threshold: float = -0.5,
	radius: float = 0.0,
):
	"""
	Visualize shape index and detected basins in Rerun.

	Two mesh layers are logged:
	  shape_index/full     - full mesh colored by shape index [-1, 1]
	  shape_index/basins   - only basin vertices highlighted in bright magenta
	"""
	rr.init("shape_index", spawn=True)

	# --- Full mesh colored by shape index ---
	si_colors = shape_index_to_colors(shape_index)
	rr.log(
		"shape_index/full",
		rr.Mesh3D(
			vertex_positions=vertices,
			triangle_indices=faces,
			vertex_colors=si_colors,
		),
		static=True,
	)

	# --- Basin vertices as a point cloud overlay ---
	basin_verts = vertices[basin_mask]
	if len(basin_verts) > 0:
		rr.log(
			"shape_index/basins",
			rr.Points3D(
				positions=basin_verts,
				colors=np.array([[255, 0, 220]], dtype=np.uint8).repeat(len(basin_verts), axis=0),
				radii=np.full(len(basin_verts), 0.3),
			),
			static=True,
		)

	n_basins = int(np.sum(basin_mask))
	si_min = float(np.nanmin(shape_index))
	si_max = float(np.nanmax(shape_index))
	si_mean = float(np.nanmean(shape_index))

	legend = (
		f"Shape Index S = (2/pi) * arctan(H / sqrt(H^2 - K))\n"
		f"Sampling radius: {radius:.4f}\n"
		f"Basin threshold: S < {basin_threshold}\n"
		"-----------------------------------------------\n"
		f"min:       {si_min:.4f}\n"
		f"mean:      {si_mean:.4f}\n"
		f"max:       {si_max:.4f}\n"
		f"basins:    {n_basins} vertices ({100*n_basins/len(shape_index):.1f}%)\n"
		"\nColor map:\n"
		"  Deep blue  S=-1  basin / cup\n"
		"  Cyan       S=-0.5  trough\n"
		"  Green      S= 0  saddle\n"
		"  Yellow     S= 0.5  ridge\n"
		"  Red        S=+1  dome / cap\n"
		"  Magenta points = detected basins"
	)
	rr.log("shape_index/legend", rr.TextDocument(legend), static=True)


# ---------------------------------------------------------------------------

def compute_cotangent_laplacian(mesh: trimesh.Trimesh, n_rings: int = 1) -> np.ndarray:
	"""
	Compute cotangent Laplacian for each vertex using n-ring neighborhoods.
	
	The cotangent Laplacian is a discrete differential operator:
	L(v_i) = (1 / (2*A_i)) * Σ_j (cot(α_ij) + cot(β_ij)) * (v_j - v_i)
	
	Args:
		n_rings: Number of vertex rings to include (1=immediate neighbors, 2=neighbors of neighbors, etc.)
		         Higher values capture more global curvature behavior.
	
	Returns the magnitude of the Laplacian vector, which is related to mean curvature.
	"""
	vertices = mesh.vertices
	faces = mesh.faces
	n_verts = len(vertices)
	
	# Build vertex adjacency for n-ring neighborhoods
	vertex_neighbors = [set() for _ in range(n_verts)]
	for face in faces:
		for i in range(3):
			v1, v2 = face[i], face[(i + 1) % 3]
			vertex_neighbors[v1].add(v2)
			vertex_neighbors[v2].add(v1)
	
	# Expand to n-ring neighborhoods
	n_ring_neighbors = [set() for _ in range(n_verts)]
	for v_idx in range(n_verts):
		current_ring = {v_idx}
		visited = {v_idx}
		for _ in range(n_rings):
			next_ring = set()
			for v in current_ring:
				next_ring.update(vertex_neighbors[v])
			next_ring -= visited
			visited.update(next_ring)
			current_ring = next_ring
		n_ring_neighbors[v_idx] = visited - {v_idx}
	
	# Initialize
	laplacian = np.zeros((n_verts, 3), dtype=np.float64)
	vertex_areas = np.zeros(n_verts, dtype=np.float64)
	
	# Build edge-face adjacency
	edges_to_faces = {}
	for face_idx, face in enumerate(faces):
		for i in range(3):
			v1, v2 = face[i], face[(i + 1) % 3]
			edge = tuple(sorted([v1, v2]))
			if edge not in edges_to_faces:
				edges_to_faces[edge] = []
			edges_to_faces[edge].append((face_idx, face[(i + 2) % 3]))  # opposite vertex
	
	# Compute cotangent weights (only for edges within n-ring neighborhoods)
	for edge, face_data in edges_to_faces.items():
		v1, v2 = edge
		
		# Check if edge is within n-ring neighborhood for both vertices
		if v2 not in n_ring_neighbors[v1] and v1 not in n_ring_neighbors[v2]:
			continue
		
		edge_vec = vertices[v2] - vertices[v1]
		
		cot_sum = 0.0
		for face_idx, v_opp in face_data:
			# Compute cotangent of angle at opposite vertex
			vec1 = vertices[v1] - vertices[v_opp]
			vec2 = vertices[v2] - vertices[v_opp]
			
			cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-12)
			cos_angle = np.clip(cos_angle, -1.0, 1.0)
			sin_angle = np.sqrt(1.0 - cos_angle**2)
			
			cot = cos_angle / (sin_angle + 1e-12)
			cot_sum += cot
		
		# Apply cotangent weight to Laplacian for vertices within their n-ring neighborhoods
		if v2 in n_ring_neighbors[v1]:
			laplacian[v1] += cot_sum * (vertices[v2] - vertices[v1])
		if v1 in n_ring_neighbors[v2]:
			laplacian[v2] += cot_sum * (vertices[v1] - vertices[v2])
	
	# Compute vertex Voronoi areas (using mixed area)
	for face in faces:
		v0, v1, v2 = face
		
		# Get triangle edges
		e0 = vertices[v1] - vertices[v0]
		e1 = vertices[v2] - vertices[v1]
		e2 = vertices[v0] - vertices[v2]
		
		# Triangle area
		face_area = 0.5 * np.linalg.norm(np.cross(e0, -e2))
		
		# Distribute area to vertices (simple uniform distribution)
		for v in face:
			vertex_areas[v] += face_area / 3.0
	
	# Normalize by vertex area
	for i in range(n_verts):
		if vertex_areas[i] > 1e-12:
			laplacian[i] /= (2.0 * vertex_areas[i])
	
	# Return magnitude of Laplacian (related to mean curvature)
	laplacian_magnitude = np.linalg.norm(laplacian, axis=1)
	
	return laplacian_magnitude


def visualize_cotangent_laplacian_rerun(
	vertices: np.ndarray,
	faces: np.ndarray,
	laplacian: np.ndarray,
	n_rings: int = 1
):
	"""
	Visualize mesh with cotangent Laplacian magnitude colors in Rerun.
	"""
	colors = curvature_to_colors(laplacian)
	rr.init("cotangent_laplacian", spawn=True)
	
	rr.log(
		"mesh",
		rr.Mesh3D(
			vertex_positions=vertices,
			triangle_indices=faces,
			vertex_colors=colors,
		),
		static=True,
	)
	
	lap_min = float(np.min(laplacian))
	lap_max = float(np.max(laplacian))
	lap_mean = float(np.mean(laplacian))
	
	legend = (
		f"Cotangent Laplacian Magnitude ||L(v)|| ({n_rings}-ring)"
		"--------------------------------------\n"
		f"min:  {lap_min:.8f}\n"
		f"mean: {lap_mean:.8f}\n"
		f"max:  {lap_max:.8f}\n"
		"\nRelated to mean curvature via ||L(v)|| ≈ 2H\n"
		"Color: blue=low curvature (flat), red=high curvature"
	)
	rr.log("laplacian_legend", rr.TextDocument(legend), static=True)


if __name__ == "__main__":
	mesh_path = r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse_Phantom\dense\simplified_mesh_10k.ply"
	mesh = trimesh.load(mesh_path)
	vertices = np.asarray(mesh.vertices)
	faces = np.asarray(mesh.faces)
	
	# ===== SCALE PARAMETER: Adjust to see local vs global curvature =====
	# For local details: k_neighbors=30-100, radius=1-3, n_rings=1-2
	# For global behavior: k_neighbors=500-2000, radius=10-50, n_rings=5-10
	SCALE_K_NEIGHBORS = 500   # PCA: number of neighbors
	SCALE_RADIUS = 0.2      # Principal curvature: spatial radius
	SCALE_N_RINGS = 3          # Cotangent Laplacian: n-ring neighborhoods
	# ===================================================================

	# # Option 1: PCA-based curvature proxy
	# print(f"Computing PCA curvature (k={SCALE_K_NEIGHBORS})...")
	# pca_curvature = estimate_pca_curvature(vertices, k_neighbors=SCALE_K_NEIGHBORS)
	# visualize_curvature_rerun(vertices, faces, pca_curvature)
	
	# # Option 2: Principal curvature (mean and Gaussian)
	# print(f"\nComputing principal curvature (radius={SCALE_RADIUS})...")
	# mean_curv, gauss_curv = estimate_principal_curvature(mesh, radius=SCALE_RADIUS)
	# visualize_principal_curvature_rerun(vertices, faces, mean_curv, gauss_curv)
	
	# # Option 3: Cotangent Laplacian
	# print(f"\nComputing cotangent Laplacian ({SCALE_N_RINGS}-ring)...")
	# laplacian = compute_cotangent_laplacian(mesh, n_rings=SCALE_N_RINGS)
	# visualize_cotangent_laplacian_rerun(vertices, faces, laplacian, n_rings=SCALE_N_RINGS)

	# Option 4: Shape Index + Basin Detection
	SCALE_BASIN_THRESHOLD = -0.1   # S < threshold => basin  (-1=cup,-0.5=trough)
	print(f"\nComputing shape index (radius={SCALE_RADIUS}, basin_threshold={SCALE_BASIN_THRESHOLD})...")
	shape_idx, basin_mask = compute_shape_index(mesh, radius=SCALE_RADIUS, basin_threshold=SCALE_BASIN_THRESHOLD)
	visualize_shape_index_rerun(vertices, faces, shape_idx, basin_mask,
								basin_threshold=SCALE_BASIN_THRESHOLD, radius=SCALE_RADIUS)
	print(f"  -> {int(np.sum(basin_mask))} basin vertices detected")

	print("\nVisualization complete. Check Rerun windows.")

	import trimesh as _trimesh

	mesh = _trimesh.load(
		r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_WoundDetection\Ferse1\dense\fused_Reduced_235k_mesh_10k.ply"
	)
	vertex_colors = (
		mesh.visual.vertex_colors if hasattr(mesh.visual, "vertex_colors") else None
	)

	curv = MeshCurvature(
		mesh.vertices,
		mesh.faces,
		vertex_colors=vertex_colors,
		neighbourhood_rings=3,
	)

	H = curv.compute_all()

	print("Mean curvature  – mean:", np.mean(H), " std:", np.std(H))
	#print("Gaussian curv.  – mean:", np.mean(K), " std:", np.std(K))

	curv.visualize_in_rerun(session_name="curvature_standalone")
