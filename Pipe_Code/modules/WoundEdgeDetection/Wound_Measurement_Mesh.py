"""
3D wound measurement utilities for mesh-based segmentation.

Provides multiple geometric approaches to estimate wound dimensions:
1) Boundary-oriented 3D bounding box (length, width, height)
2) Boundary-plane projection (length/width on plane + max height profile)

The functions operate on a WoundModel3DMesh instance produced by
3DWound_Detection_Mesh.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass
class WoundAxis:
	"""Represents one measured axis as a 3D line segment."""

	name: str
	length: float
	start: np.ndarray
	end: np.ndarray


@dataclass
class WoundMeasurementResult:
	"""Container with outputs from all measurement approaches."""

	n_wound_vertices: int
	n_boundary_vertices: int
	n_boundary_edges: int
	approach_1_obb: Dict[str, Any]
	approach_2_plane_profile: Dict[str, Any]
	per_approach_uncertainty: Dict[str, Any]
	uncertainty_estimation: Dict[str, Any]

	def to_dict(self) -> Dict[str, Any]:
		return {
			"n_wound_vertices": self.n_wound_vertices,
			"n_boundary_vertices": self.n_boundary_vertices,
			"n_boundary_edges": self.n_boundary_edges,
			"approach_1_obb": self.approach_1_obb,
			"approach_2_plane_profile": self.approach_2_plane_profile,
			"per_approach_uncertainty": self.per_approach_uncertainty,
			"uncertainty_estimation": self.uncertainty_estimation,
		}


def _safe_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
	n = float(np.linalg.norm(v))
	if n < eps:
		return np.zeros_like(v, dtype=np.float64)
	return np.asarray(v, dtype=np.float64) / n


def _segment_from_center_direction(center: np.ndarray, direction: np.ndarray, length: float) -> Tuple[np.ndarray, np.ndarray]:
	d = _safe_normalize(direction)
	half = 0.5 * float(max(length, 0.0))
	return center - d * half, center + d * half


def _pca_basis(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Return center, principal directions (columns), and eigenvalues (desc)."""
	center = np.mean(points, axis=0)
	centered = points - center
	cov = np.cov(centered.T) if len(points) > 1 else np.eye(3, dtype=np.float64)
	eigvals, eigvecs = np.linalg.eigh(cov)
	order = np.argsort(eigvals)[::-1]
	eigvals = eigvals[order]
	eigvecs = eigvecs[:, order]
	return center, eigvecs, eigvals


def _project_to_axes(points: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
	"""Project Nx3 points into coordinates of orthonormal axes (3x3 columns)."""
	centered = points - origin
	return centered @ axes


def _get_wound_face_indices(wound_model: Any) -> List[int]:
	faces = sorted(list(getattr(wound_model, "current_wound_faces", set()) or []))
	if faces:
		return faces

	# Fallback: use wound edge faces if final set is not persisted.
	edge_faces = sorted(list(getattr(wound_model, "wound_edge_faces", set()) or []))
	return edge_faces


def _collect_wound_vertex_indices(wound_model: Any, wound_face_indices: Sequence[int]) -> np.ndarray:
	if len(wound_face_indices) == 0:
		return np.zeros((0,), dtype=np.int64)

	faces = np.asarray(wound_model.faces, dtype=np.int64)
	selected = faces[np.asarray(wound_face_indices, dtype=np.int64)]
	return np.unique(selected.reshape(-1)).astype(np.int64)


def _collect_boundary_vertices_and_edges(wound_model: Any) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
	boundary_vertices: Set[int] = set(getattr(wound_model, "wound_edge_vertices", set()) or set())
	boundary_edges: List[Tuple[int, int]] = list(getattr(wound_model, "wound_edge_edges", []) or [])

	if boundary_vertices:
		bv = np.array(sorted(boundary_vertices), dtype=np.int64)
	else:
		bv = np.zeros((0,), dtype=np.int64)

	if len(boundary_edges) == 0 and len(bv) >= 2:
		# Fallback for visualization only: consecutive polyline from sorted ids.
		boundary_edges = [(int(bv[i]), int(bv[i + 1])) for i in range(len(bv) - 1)]

	return bv, boundary_edges


def _approach_1_boundary_oriented_obb(boundary_points: np.ndarray, wound_points: np.ndarray) -> Dict[str, Any]:
	"""Approach 1: boundary-oriented 3D bounding box via PCA."""
	ref_points = boundary_points if len(boundary_points) >= 3 else wound_points
	center, axes, eigvals = _pca_basis(ref_points)

	coords = _project_to_axes(wound_points, center, axes)
	mins = np.min(coords, axis=0)
	maxs = np.max(coords, axis=0)
	extents = maxs - mins

	length = float(extents[0])
	width = float(extents[1])
	height = float(extents[2])

	box_center_local = 0.5 * (mins + maxs)
	box_center_world = center + axes @ box_center_local

	seg_l = _segment_from_center_direction(box_center_world, axes[:, 0], length)
	seg_w = _segment_from_center_direction(box_center_world, axes[:, 1], width)
	seg_h = _segment_from_center_direction(box_center_world, axes[:, 2], height)

	return {
		"method": "boundary_oriented_obb",
		"length": length,
		"width": width,
		"height": height,
		"center": box_center_world,
		"axes": axes,
		"eigvals": eigvals,
		"local_min": mins,
		"local_max": maxs,
		"length_segment": np.vstack(seg_l),
		"width_segment": np.vstack(seg_w),
		"height_segment": np.vstack(seg_h),
	}


def _build_plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	n = _safe_normalize(normal)
	ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
	if abs(float(np.dot(ref, n))) > 0.9:
		ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
	u = _safe_normalize(np.cross(n, ref))
	v = _safe_normalize(np.cross(n, u))
	return u, v, n


def _fit_plane_pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	center, eigvecs, eigvals = _pca_basis(points)
	normal = eigvecs[:, 2]  # smallest variance axis
	u, v, n = _build_plane_basis(normal)

	# Improve in-plane orientation using projected covariance.
	proj_u = (points - center) @ u
	proj_v = (points - center) @ v
	uv = np.column_stack([proj_u, proj_v])
	cov2 = np.cov(uv.T) if len(uv) > 1 else np.eye(2, dtype=np.float64)
	eval2, evec2 = np.linalg.eigh(cov2)
	order = np.argsort(eval2)[::-1]
	evec2 = evec2[:, order]

	# Rotate initial in-plane basis to align with principal spread in the plane.
	u2 = _safe_normalize(u * evec2[0, 0] + v * evec2[1, 0])
	v2 = _safe_normalize(u * evec2[0, 1] + v * evec2[1, 1])
	return center, u2, v2, n


def _approach_2_boundary_plane_profile(boundary_points: np.ndarray, wound_points: np.ndarray) -> Dict[str, Any]:
	"""Approach 2: project boundary to fitted plane and derive depth profile."""
	ref_points = boundary_points if len(boundary_points) >= 3 else wound_points
	origin, u, v, n = _fit_plane_pca(ref_points)

	boundary_rel = boundary_points - origin
	boundary_u = boundary_rel @ u
	boundary_v = boundary_rel @ v
	length = float(np.max(boundary_u) - np.min(boundary_u)) if len(boundary_u) > 0 else 0.0
	width = float(np.max(boundary_v) - np.min(boundary_v)) if len(boundary_v) > 0 else 0.0

	wound_rel = wound_points - origin
	signed_h = wound_rel @ n
	abs_h = np.abs(signed_h)

	idx_abs = int(np.argmax(abs_h)) if len(abs_h) > 0 else 0
	max_height = float(abs_h[idx_abs]) if len(abs_h) > 0 else 0.0
	max_height_point = wound_points[idx_abs] if len(wound_points) > 0 else origin.copy()
	max_height_proj = max_height_point - (float(np.dot(max_height_point - origin, n)) * n)

	# Plane center from boundary extents for length/width segment visualization.
	center_u = 0.5 * (float(np.min(boundary_u)) + float(np.max(boundary_u))) if len(boundary_u) > 0 else 0.0
	center_v = 0.5 * (float(np.min(boundary_v)) + float(np.max(boundary_v))) if len(boundary_v) > 0 else 0.0
	plane_center = origin + center_u * u + center_v * v

	seg_l = _segment_from_center_direction(plane_center, u, length)
	seg_w = _segment_from_center_direction(plane_center, v, width)

	return {
		"method": "boundary_plane_profile",
		"plane_origin": origin,
		"plane_normal": n,
		"plane_axis_u": u,
		"plane_axis_v": v,
		"length": length,
		"width": width,
		"max_height": max_height,
		"signed_height_min": float(np.min(signed_h)) if len(signed_h) > 0 else 0.0,
		"signed_height_max": float(np.max(signed_h)) if len(signed_h) > 0 else 0.0,
		"plane_center": plane_center,
		"length_segment": np.vstack(seg_l),
		"width_segment": np.vstack(seg_w),
		"height_segment": np.vstack([max_height_proj, max_height_point]),
	}


def _build_uncertainty_entry(values: List[float], methods: List[str], label: str) -> Dict[str, Any]:
	arr = np.asarray(values, dtype=np.float64)
	if len(arr) == 0:
		return {
			"dimension": label,
			"available": False,
			"reason": "no_values",
		}

	median = float(np.median(arr))
	mean = float(np.mean(arr))
	min_v = float(np.min(arr))
	max_v = float(np.max(arr))
	half_range = 0.5 * (max_v - min_v)
	std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
	mad = float(np.median(np.abs(arr - median))) if len(arr) > 1 else 0.0
	mad_sigma = 1.4826 * mad  # robust sigma estimate for near-normal noise

	# Conservative inter-method uncertainty: spread across different measurement formulations.
	error_abs = float(max(half_range, std, mad_sigma))
	error_pct = float((error_abs / median) * 100.0) if abs(median) > 1e-12 else 0.0

	return {
		"dimension": label,
		"available": True,
		"n_methods": int(len(arr)),
		"methods": methods,
		"estimate": median,
		"mean": mean,
		"error_abs": error_abs,
		"error_percent": error_pct,
		"min": min_v,
		"max": max_v,
		"std": std,
		"mad_sigma": float(mad_sigma),
	}


def _estimate_dimension_uncertainty(
	approach_1: Dict[str, Any],
	approach_2: Dict[str, Any],
) -> Dict[str, Any]:
	"""
	Estimate uncertainty from the spread of different geometric formulations.
	This is an inter-method consistency estimate, not a sensor calibration error.
	"""

	length_values = [float(approach_1["length"]), float(approach_2["length"])]
	length_methods = ["approach_1_obb", "approach_2_plane_profile"]

	width_values = [float(approach_1["width"]), float(approach_2["width"])]
	width_methods = ["approach_1_obb", "approach_2_plane_profile"]

	height_values = [float(approach_1["height"]), float(approach_2["max_height"])]
	height_methods = ["approach_1_obb", "approach_2_plane_profile_max_height"]

	return {
		"method": "inter_method_spread",
		"note": "Error is estimated from disagreement between available measurement approaches.",
		"length": _build_uncertainty_entry(length_values, length_methods, "length"),
		"width": _build_uncertainty_entry(width_values, width_methods, "width"),
		"height": _build_uncertainty_entry(height_values, height_methods, "height"),
	}


def _bootstrap_metric_entry(reference_value: float, samples: Sequence[float], label: str) -> Dict[str, Any]:
	arr = np.asarray(list(samples), dtype=np.float64)
	if len(arr) == 0:
		return {
			"metric": label,
			"available": False,
			"reason": "no_bootstrap_samples",
		}

	std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
	q_low, q_high = np.percentile(arr, [2.5, 97.5]) if len(arr) > 1 else (arr[0], arr[0])
	ci_half = 0.5 * float(q_high - q_low)
	error_abs = float(max(std, ci_half))
	error_pct = float((error_abs / reference_value) * 100.0) if abs(reference_value) > 1e-12 else 0.0

	return {
		"metric": label,
		"available": True,
		"estimate": float(reference_value),
		"error_abs": error_abs,
		"error_percent": error_pct,
		"std": std,
		"ci95_low": float(q_low),
		"ci95_high": float(q_high),
		"n_samples": int(len(arr)),
	}


def _estimate_per_approach_uncertainty(
	boundary_points: np.ndarray,
	wound_points: np.ndarray,
	approach_1: Dict[str, Any],
	approach_2: Dict[str, Any],
	n_bootstrap: int = 80,
	seed: int = 12345,
) -> Dict[str, Any]:
	"""Estimate uncertainty per approach using bootstrap resampling of wound/boundary points."""

	rng = np.random.default_rng(seed)
	n_w = int(len(wound_points))
	n_b = int(len(boundary_points))

	def _resample(points: np.ndarray, n: int) -> np.ndarray:
		idx = rng.integers(0, n, size=n)
		return points[idx]

	a1_len: List[float] = []
	a1_wid: List[float] = []
	a1_hgt: List[float] = []

	a2_len: List[float] = []
	a2_wid: List[float] = []
	a2_hgt: List[float] = []

	max_attempts = int(max(n_bootstrap * 4, n_bootstrap + 10))
	count = 0
	for _ in range(max_attempts):
		if count >= n_bootstrap:
			break
		try:
			w_s = _resample(wound_points, n_w)
			b_s = _resample(boundary_points, n_b) if n_b > 0 else w_s

			a1_s = _approach_1_boundary_oriented_obb(b_s, w_s)
			a2_s = _approach_2_boundary_plane_profile(b_s, w_s)

			a1_len.append(float(a1_s["length"]))
			a1_wid.append(float(a1_s["width"]))
			a1_hgt.append(float(a1_s["height"]))

			a2_len.append(float(a2_s["length"]))
			a2_wid.append(float(a2_s["width"]))
			a2_hgt.append(float(a2_s["max_height"]))

			count += 1
		except Exception:
			continue

	return {
		"method": "bootstrap_resampling",
		"n_bootstrap_target": int(n_bootstrap),
		"n_bootstrap_success": int(count),
		"approach_1_obb": {
			"available": len(a1_len) > 0,
			"metrics": {
				"length": _bootstrap_metric_entry(float(approach_1["length"]), a1_len, "length"),
				"width": _bootstrap_metric_entry(float(approach_1["width"]), a1_wid, "width"),
				"height": _bootstrap_metric_entry(float(approach_1["height"]), a1_hgt, "height"),
			},
		},
		"approach_2_plane_profile": {
			"available": len(a2_len) > 0,
			"metrics": {
				"length": _bootstrap_metric_entry(float(approach_2["length"]), a2_len, "length"),
				"width": _bootstrap_metric_entry(float(approach_2["width"]), a2_wid, "width"),
				"max_height": _bootstrap_metric_entry(float(approach_2["max_height"]), a2_hgt, "max_height"),
			},
		},
	}


def measure_wound_dimensions_from_model(wound_model: Any, visualization: bool = False) -> WoundMeasurementResult:
	"""Compute wound dimensions using multiple geometric approaches."""
	if getattr(wound_model, "vertices", None) is None or getattr(wound_model, "faces", None) is None:
		raise ValueError("Model geometry is not loaded")

	wound_face_indices = _get_wound_face_indices(wound_model)
	if len(wound_face_indices) == 0:
		raise ValueError("No wound faces available. Run get_wound_mesh() first.")

	wound_vertex_idx = _collect_wound_vertex_indices(wound_model, wound_face_indices)
	wound_points = np.asarray(wound_model.vertices[wound_vertex_idx], dtype=np.float64)
	if len(wound_points) < 3:
		raise ValueError("Not enough wound points for measurement")

	boundary_vertex_idx, boundary_edges = _collect_boundary_vertices_and_edges(wound_model)
	if len(boundary_vertex_idx) > 0:
		boundary_points = np.asarray(wound_model.vertices[boundary_vertex_idx], dtype=np.float64)
	else:
		boundary_points = wound_points.copy()

	approach_1 = _approach_1_boundary_oriented_obb(boundary_points, wound_points)
	approach_2 = _approach_2_boundary_plane_profile(boundary_points, wound_points)
	per_approach_uncertainty = _estimate_per_approach_uncertainty(
		boundary_points=boundary_points,
		wound_points=wound_points,
		approach_1=approach_1,
		approach_2=approach_2,
	)
	approach_1["uncertainty"] = per_approach_uncertainty.get("approach_1_obb", {})
	approach_2["uncertainty"] = per_approach_uncertainty.get("approach_2_plane_profile", {})
	uncertainty = _estimate_dimension_uncertainty(approach_1, approach_2)

	result = WoundMeasurementResult(
		n_wound_vertices=int(len(wound_vertex_idx)),
		n_boundary_vertices=int(len(boundary_vertex_idx)),
		n_boundary_edges=int(len(boundary_edges)),
		approach_1_obb=approach_1,
		approach_2_plane_profile=approach_2,
		per_approach_uncertainty=per_approach_uncertainty,
		uncertainty_estimation=uncertainty,
	)

	if visualization:
		print(format_measurement_summary(result))
		visualize_wound_measurements_rerun(wound_model, result)
 
	# Persist on model for downstream access/debugging.
	wound_model.wound_measurement_result = result.to_dict()
	return result


def format_measurement_summary(result: WoundMeasurementResult | Dict[str, Any]) -> str:
	"""Return compact human-readable summary for logs/console."""
	data = result.to_dict() if isinstance(result, WoundMeasurementResult) else result
	a1 = data["approach_1_obb"]
	a2 = data["approach_2_plane_profile"]
	per_unc = data.get("per_approach_uncertainty", {})
	unc = data.get("uncertainty_estimation", {})

	lines = [
		"=== Wound Measurement Summary ===",
		f"Wound vertices: {data['n_wound_vertices']}",
		f"Boundary vertices: {data['n_boundary_vertices']}",
		f"Boundary edges: {data['n_boundary_edges']}",
		"",
		"Approach 1 (Boundary-oriented OBB):",
		f"  Length: {a1['length']:.6f}",
		f"  Width : {a1['width']:.6f}",
		f"  Height: {a1['height']:.6f}",
		"",
		"Approach 2 (Boundary plane + height profile):",
		f"  Length: {a2['length']:.6f}",
		f"  Width : {a2['width']:.6f}",
		f"  Max height (abs distance to plane): {a2['max_height']:.6f}",
		f"  Signed height range: [{a2['signed_height_min']:.6f}, {a2['signed_height_max']:.6f}]",
	]

	if isinstance(per_unc, dict) and per_unc.get("method") == "bootstrap_resampling":
		a1u = per_unc.get("approach_1_obb", {}).get("metrics", {})
		a2u = per_unc.get("approach_2_plane_profile", {}).get("metrics", {})

		def _fmt_u(entry: Dict[str, Any]) -> str:
			if not isinstance(entry, dict) or not entry.get("available", False):
				return "n/a"
			return f"{entry['estimate']:.6f} +/- {entry['error_abs']:.6f} ({entry['error_percent']:.2f}%)"

		lines.extend(
			[
				"",
				"Per-approach uncertainty (bootstrap):",
				f"  Approach 1 length/width/height: {_fmt_u(a1u.get('length', {}))} | {_fmt_u(a1u.get('width', {}))} | {_fmt_u(a1u.get('height', {}))}",
				f"  Approach 2 length/width/max_height: {_fmt_u(a2u.get('length', {}))} | {_fmt_u(a2u.get('width', {}))} | {_fmt_u(a2u.get('max_height', {}))}",
			]
		)

	if isinstance(unc, dict) and unc.get("method") == "inter_method_spread":
		ul = unc.get("length", {})
		uw = unc.get("width", {})
		uh = unc.get("height", {})
		if ul.get("available", False) and uw.get("available", False) and uh.get("available", False):
			lines.extend(
				[
					"",
					"Uncertainty estimation (inter-method spread):",
					f"  Length: {ul['estimate']:.6f} +/- {ul['error_abs']:.6f} ({ul['error_percent']:.2f}%)",
					f"  Width : {uw['estimate']:.6f} +/- {uw['error_abs']:.6f} ({uw['error_percent']:.2f}%)",
					f"  Height: {uh['estimate']:.6f} +/- {uh['error_abs']:.6f} ({uh['error_percent']:.2f}%)",
				]
			)

	return "\n".join(lines)


def visualize_wound_measurements_rerun(
	wound_model: Any,
	measurement_result: Optional[WoundMeasurementResult | Dict[str, Any]] = None,
	session_name: str = "WoundMesh_Measurements",
	spawn: bool = True,
) -> None:
	"""Show measured dimensions in Rerun (boundary, axes, and height profile)."""
	try:
		import rerun as rr
	except Exception:
		print("Rerun not installed. Install with: pip install rerun-sdk")
		return

	if measurement_result is None:
		measurement_result = measure_wound_dimensions_from_model(wound_model)

	data = measurement_result.to_dict() if isinstance(measurement_result, WoundMeasurementResult) else measurement_result

	if getattr(wound_model, "vertices", None) is None or getattr(wound_model, "faces", None) is None:
		print("Measurement visualization skipped: mesh not available")
		return

	try:
		rr.init(session_name, spawn=spawn)
	except Exception as e:
		print(f"Could not launch Rerun viewer directly ({e}); continuing with default init")
		rr.init(session_name)

	rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

	vertices = np.asarray(wound_model.vertices)
	faces = np.asarray(wound_model.faces)
	if getattr(wound_model, "vertex_colors", None) is not None and len(wound_model.vertex_colors) == len(vertices):
		colors = np.clip(np.asarray(wound_model.vertex_colors) * 255.0, 0.0, 255.0).astype(np.uint8)
	else:
		colors = np.full((len(vertices), 3), 140, dtype=np.uint8)

	rr.log(
		"/measurement/base_mesh",
		rr.Mesh3D(vertex_positions=vertices, triangle_indices=faces, vertex_colors=colors),
		static=True,
	)

	wound_faces = sorted(list(getattr(wound_model, "current_wound_faces", set()) or []))
	if len(wound_faces) > 0:
		wound_face_idx = np.array(wound_faces, dtype=np.int64)
		wound_vertex_ids = np.unique(faces[wound_face_idx].reshape(-1))
		wound_vertex_mask = np.full((len(vertices), 3), 70, dtype=np.uint8)
		wound_vertex_mask[wound_vertex_ids] = np.array([235, 40, 40], dtype=np.uint8)
		rr.log(
			"/measurement/wound_region",
			rr.Mesh3D(vertex_positions=vertices, triangle_indices=faces, vertex_colors=wound_vertex_mask),
			static=True,
		)

	boundary_vertex_ids = sorted(list(getattr(wound_model, "wound_edge_vertices", set()) or []))
	if len(boundary_vertex_ids) > 0:
		boundary_pts = vertices[np.array(boundary_vertex_ids, dtype=np.int64)]
		rr.log(
			"/measurement/boundary/points",
			rr.Points3D(boundary_pts, colors=[255, 210, 40], radii=0.0015),
			static=True,
		)

	boundary_edges = list(getattr(wound_model, "wound_edge_edges", []) or [])
	if len(boundary_edges) > 0:
		strips = []
		for v0, v1 in boundary_edges:
			if 0 <= v0 < len(vertices) and 0 <= v1 < len(vertices):
				strips.append(np.array([vertices[v0], vertices[v1]], dtype=np.float64))
		if strips:
			rr.log(
				"/measurement/boundary/edges",
				rr.LineStrips3D(strips, colors=[255, 180, 20]),
				static=True,
			)

	a1 = data["approach_1_obb"]
	a2 = data["approach_2_plane_profile"]

	rr.log(
		"/measurement/approach_1/length",
		rr.LineStrips3D([np.asarray(a1["length_segment"], dtype=np.float64)], colors=[220, 30, 30]),
		static=True,
	)
	rr.log(
		"/measurement/approach_1/width",
		rr.LineStrips3D([np.asarray(a1["width_segment"], dtype=np.float64)], colors=[30, 200, 60]),
		static=True,
	)
	rr.log(
		"/measurement/approach_1/height",
		rr.LineStrips3D([np.asarray(a1["height_segment"], dtype=np.float64)], colors=[40, 120, 240]),
		static=True,
	)

	rr.log(
		"/measurement/approach_2/length_on_plane",
		rr.LineStrips3D([np.asarray(a2["length_segment"], dtype=np.float64)], colors=[250, 100, 0]),
		static=True,
	)
	rr.log(
		"/measurement/approach_2/width_on_plane",
		rr.LineStrips3D([np.asarray(a2["width_segment"], dtype=np.float64)], colors=[0, 170, 255]),
		static=True,
	)
	rr.log(
		"/measurement/approach_2/max_height",
		rr.LineStrips3D([np.asarray(a2["height_segment"], dtype=np.float64)], colors=[255, 0, 200]),
		static=True,
	)

	plane_center = np.asarray(a2["plane_center"], dtype=np.float64)
	plane_normal = np.asarray(a2["plane_normal"], dtype=np.float64)
	rr.log(
		"/measurement/approach_2/plane_normal",
		rr.Arrows3D(origins=[plane_center], vectors=[plane_normal * max(a2["max_height"], 1e-4)]),
		static=True,
	)

	rr.log("/measurement/summary", rr.TextDocument(format_measurement_summary(data)), static=True)
	print("Rerun wound-dimension visualization logged")
