"""
3D Wound Edge Detection - Mesh-Based CRF Version

This module extends the base mesh workflow from `3DWound_Detection_Mesh.py`
and keeps the same camera + mask loading behavior:

	- load_colmap_cameras(...)
	- load_masks(...)
	- load_extruded_masks(...)
	- process_wound_segmentation(...)

The semantic stage is replaced by a CRF-based probability refinement that
implements unary + pairwise terms on mesh faces and mean-field inference.
"""

import os
import sys
import importlib.util
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Path setup (same as base module)
# ---------------------------------------------------------------------------
_current_dir = os.path.dirname(os.path.abspath(__file__))
_pipe_code_dir = os.path.abspath(os.path.join(_current_dir, '..', '..'))
_workspace_root = os.path.abspath(os.path.join(_current_dir, '..', '..', '..'))

if _workspace_root not in sys.path:
	sys.path.insert(0, _workspace_root)
if _pipe_code_dir not in sys.path:
	sys.path.insert(0, _pipe_code_dir)


# ---------------------------------------------------------------------------
# Import base class via importlib (filename starts with a digit)
# ---------------------------------------------------------------------------
_base_path = os.path.join(_current_dir, "3DWound_Detection_Mesh.py")
_spec = importlib.util.spec_from_file_location("wound_mesh_base", _base_path)
_base_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base_module)

WoundModel3DMesh = _base_module.WoundModel3DMesh
Spectator = _base_module.Spectator


class WoundModel3DMeshCRF(WoundModel3DMesh):
	"""Mesh wound segmentation with CRF probability-map refinement.

	Notes
	-----
	- Camera/mask loading is inherited directly from `WoundModel3DMesh`.
	- Labels are fixed to: `wound`, `tissue`, `none`.
	- Unary term comes from either a learned multiclass logistic model
	  or a visibility-based fallback.
	- Pairwise term uses color, normal, centroid and dihedral similarity.
	- Inference uses mean-field updates.
	"""

	def __init__(self):
		super().__init__()

		self.crf_class_names = ["wound", "tissue", "none"]
		self.crf_posteriors: Dict[int, np.ndarray] = {}
		self.crf_labels: Dict[int, str] = {}
		self.crf_confidence: Dict[int, float] = {}
		self.crf_feature_face_indices: List[int] = []
		self.crf_feature_matrix: Optional[np.ndarray] = None
		self.crf_unary_probs: Optional[np.ndarray] = None
		self.crf_classifier = None

	# ------------------------------------------------------------------
	# Utility helpers
	# ------------------------------------------------------------------
	@staticmethod
	def _softmax(logits: np.ndarray) -> np.ndarray:
		mx = np.max(logits, axis=1, keepdims=True)
		ex = np.exp(logits - mx)
		return ex / (np.sum(ex, axis=1, keepdims=True) + 1e-12)

	@staticmethod
	def _rgb_to_lab_batch(rgb: np.ndarray) -> np.ndarray:
		"""Convert RGB [0,1] array of shape (N,3) to LAB."""
		rgb = np.clip(rgb, 0.0, 1.0).astype(np.float64)

		mask = rgb > 0.04045
		linear = np.where(mask, np.power((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)

		m = np.array([
			[0.4124564, 0.3575761, 0.1804375],
			[0.2126729, 0.7151522, 0.0721750],
			[0.0193339, 0.1191920, 0.9503041],
		])
		xyz = linear @ m.T
		xyz /= np.array([0.95047, 1.0, 1.08883])

		delta = 6.0 / 29.0
		big = xyz > delta ** 3
		f = np.where(big, np.cbrt(xyz), xyz / (3.0 * delta ** 2) + 4.0 / 29.0)

		l = 116.0 * f[:, 1] - 16.0
		a = 500.0 * (f[:, 0] - f[:, 1])
		b = 200.0 * (f[:, 1] - f[:, 2])
		return np.column_stack([l, a, b])

	def _compute_face_lab(self) -> np.ndarray:
		if self.vertex_colors is None:
			return np.zeros((len(self.faces), 3), dtype=np.float64)
		v0 = self.vertex_colors[self.faces[:, 0]]
		v1 = self.vertex_colors[self.faces[:, 1]]
		v2 = self.vertex_colors[self.faces[:, 2]]
		rgb = (v0 + v1 + v2) / 3.0
		return self._rgb_to_lab_batch(rgb)

	def _get_ratio_cache(self) -> Dict[int, Tuple[float, float, int]]:
		ratio_cache: Dict[int, Tuple[float, float, int]] = {}
		for face_idx in range(len(self.faces)):
			wound_hits = self.face_visibility_count.get(face_idx, 0)
			tissue_hits = self.tissue_edge_visibility_count.get(face_idx, 0)
			total = self.spectators.__len__()  # total number of views/spectators
			r_w = wound_hits / total if total > 0 else 0.0
			r_t = tissue_hits / total if total > 0 else 0.0
			ratio_cache[face_idx] = (r_w, r_t, total)
		return ratio_cache

	# ------------------------------------------------------------------
	# Unary probabilities
	# ------------------------------------------------------------------
	def _visibility_unary_probs(
		self,
		face_indices: List[int],
		ratio_cache: Dict[int, Tuple[float, float, int]],
		alpha: float = 0.05,
		none_tau: float = 1.5,
	) -> np.ndarray:
		"""Fallback unary term from visibility evidence only."""
		probs = np.zeros((len(face_indices), 3), dtype=np.float64)

		for i, face_idx in enumerate(face_indices):
			r_w, r_t, total = ratio_cache[face_idx]
			vis_conf = 1.0 - np.exp(-total / max(none_tau, 1e-6))

			p_w = max(alpha, r_w * vis_conf)
			p_t = max(alpha, r_t * vis_conf)
			p_n = max(alpha, 1.0 - vis_conf)

			z = p_w + p_t + p_n
			probs[i, 0] = p_w / z
			probs[i, 1] = p_t / z
			probs[i, 2] = p_n / z

		return probs

	def _train_unary_classifier(
		self,
		features: np.ndarray,
		face_indices: List[int],
		ratio_cache: Dict[int, Tuple[float, float, int]],
		wound_ratio_thresh: float = 0.6,
		tissue_ratio_thresh: float = 0.6,
		none_total_views_thresh: int = 0,
	) -> Optional[np.ndarray]:
		"""Train multiclass logistic regression for unary probabilities.

		Classes: wound=0, tissue=1, none=2
		"""
		try:
			import inspect
			from sklearn.linear_model import LogisticRegression
		except Exception:
			return None

		idx_map = {fi: i for i, fi in enumerate(face_indices)}
		x_train = []
		y_train = []

		for fi in face_indices:
			r_w, r_t, total = ratio_cache[fi]

			if r_w >= wound_ratio_thresh and r_w > r_t:
				cls = 0
			elif r_t >= tissue_ratio_thresh and r_t > r_w:
				cls = 1
			elif total <= none_total_views_thresh:
				cls = 2
			else:
				continue

			x_train.append(features[idx_map[fi]])
			y_train.append(cls)

		if len(x_train) < 20:
			return None

		x_train = np.asarray(x_train, dtype=np.float64)
		y_train = np.asarray(y_train, dtype=np.int32)

		if len(np.unique(y_train)) < 2:
			return None

		init_kwargs = {
			"penalty": "l2",
			"class_weight": "balanced",
			"solver": "lbfgs",
			"max_iter": 1000,
		}
		if "multi_class" in inspect.signature(LogisticRegression.__init__).parameters:
			init_kwargs["multi_class"] = "auto"

		model = LogisticRegression(**init_kwargs)
		model.fit(x_train, y_train)
		self.crf_classifier = model

		p = model.predict_proba(features)

		unary = np.full((len(face_indices), 3), 1e-4, dtype=np.float64)
		for col, cls in enumerate(model.classes_):
			unary[:, int(cls)] = p[:, col]
		unary /= np.sum(unary, axis=1, keepdims=True) + 1e-12
		return unary

	# ------------------------------------------------------------------
	# Pairwise kernels / neighborhoods
	# ------------------------------------------------------------------
	def _collect_multiscale_neighbors(
		self,
		active_faces: List[int],
		geodesic_hops: int,
	) -> Tuple[Dict[int, List[int]], Dict[int, List[Tuple[int, int]]]]:
		"""Build immediate (1-ring) and geodesic (<=hops, >1) neighborhoods."""
		active_set = set(active_faces)

		local_nbrs: Dict[int, List[int]] = {}
		geo_nbrs: Dict[int, List[Tuple[int, int]]] = {}

		for fi in active_faces:
			one_ring = [n for n in self.face_adjacency.get(fi, []) if n in active_set]
			local_nbrs[fi] = one_ring

			if geodesic_hops <= 1:
				geo_nbrs[fi] = []
				continue

			visited = {fi}
			q = deque([(fi, 0)])
			collected: List[Tuple[int, int]] = []

			while q:
				node, dist = q.popleft()
				if dist >= geodesic_hops:
					continue
				for nxt in self.face_adjacency.get(node, []):
					if nxt not in active_set or nxt in visited:
						continue
					visited.add(nxt)
					nd = dist + 1
					q.append((nxt, nd))
					if nd > 1:
						collected.append((nxt, nd))

			geo_nbrs[fi] = collected

		return local_nbrs, geo_nbrs

	def _compute_pairwise_weight_params(
		self,
		active_faces: List[int],
		face_lab: np.ndarray,
		centroids: np.ndarray,
	) -> Tuple[float, float, float, float]:
		"""Estimate sigma values from local adjacency statistics."""
		self._ensure_face_normals()

		active_set = set(active_faces)
		color_d = []
		normal_d = []
		pos_d = []
		theta_d = []

		seen = set()
		for fi in active_faces:
			for fj in self.face_adjacency.get(fi, []):
				if fj not in active_set:
					continue
				key = (min(fi, fj), max(fi, fj))
				if key in seen:
					continue
				seen.add(key)

				dc = np.linalg.norm(face_lab[fi] - face_lab[fj])
				dn = np.linalg.norm(self.face_normals[fi] - self.face_normals[fj])
				dp = np.linalg.norm(centroids[fi] - centroids[fj])
				dot = np.clip(np.dot(self.face_normals[fi], self.face_normals[fj]), -1.0, 1.0)
				theta = np.arccos(dot)

				color_d.append(dc)
				normal_d.append(dn)
				pos_d.append(dp)
				theta_d.append(theta)

		if not color_d:
			return 1.0, 1.0, 1.0, 1.0

		sigma_c = float(np.median(color_d) + 1e-9)
		sigma_n = float(np.median(normal_d) + 1e-9)
		sigma_p = float(np.median(pos_d) + 1e-9)
		sigma_t = float(np.median(theta_d) + 1e-9)
		return sigma_c, sigma_n, sigma_p, sigma_t

	# ------------------------------------------------------------------
	# CRF inference
	# ------------------------------------------------------------------
	def run_crf_probability_map(
		self,
		min_total_views: int = 1,
		mean_field_iters: int = 8,
		lambda_local: float = 1.0,
		lambda_geodesic: float = 0.6,
		geodesic_hops: int = 3,
		geodesic_sigma_hops: float = 1.5,
	) -> Dict[int, np.ndarray]:
		"""Compute a CRF-refined per-face probability map.

		Returns
		-------
		dict
			Mapping: face_idx -> np.array([P(wound), P(tissue), P(none)]).
		"""
		if not self.faces_processed:
			raise RuntimeError("Run process_wound_segmentation() before CRF")
		if not self.topology_computed:
			raise RuntimeError("Mesh topology is required before CRF")

		if not any(s.extruded_mask_loaded for s in self.spectators.values()):
			print("Warning: no extruded masks loaded; tissue-edge priors may be weak")
		else:
			self.process_tissue_face_detection(max_mask_rays=200000)

		ratio_cache = self._get_ratio_cache()

		features, face_indices, ratio_cache = self._build_feature_vectors(
			ring_sizes=[1, 3],
			use_ratio_features=False,
			use_height_deviation=True,
			use_normal_deviation=False,
			use_curvature=True,
			color_space="none",
		)

		if features is None or len(face_indices) == 0:
			raise RuntimeError("No valid features available for CRF")

		if min_total_views > 1:
			face_indices = [fi for fi in face_indices if ratio_cache.get(fi, (0.0, 0.0, 0))[2] >= min_total_views]
			if not face_indices:
				raise RuntimeError("No faces satisfy min_total_views")
			all_faces = self.semantic_face_indices
			row_map = {fi: idx for idx, fi in enumerate(all_faces)}
			features = np.asarray([features[row_map[fi]] for fi in face_indices], dtype=np.float64)

		self.crf_feature_face_indices = face_indices
		self.crf_feature_matrix = features

		unary = self._train_unary_classifier(features, face_indices, ratio_cache)
		if unary is None:
			unary = self._visibility_unary_probs(face_indices, ratio_cache)
		unary = np.clip(unary, 1e-8, 1.0)
		unary /= np.sum(unary, axis=1, keepdims=True) + 1e-12
		self.crf_unary_probs = unary

		unary_energy = -np.log(unary)

		self._ensure_face_normals()
		centroids = self._get_face_centroids()
		face_lab = self._compute_face_lab()

		sigma_c, sigma_n, sigma_p, sigma_t = self._compute_pairwise_weight_params(face_indices, face_lab, centroids)
		face_to_row = {fi: i for i, fi in enumerate(face_indices)}

		local_nbrs, geo_nbrs = self._collect_multiscale_neighbors(face_indices, geodesic_hops=geodesic_hops)

		local_edges: Dict[int, List[Tuple[int, float]]] = {}
		geo_edges: Dict[int, List[Tuple[int, float]]] = {}

		for fi in face_indices:
			local_list: List[Tuple[int, float]] = []
			for fj in local_nbrs[fi]:
				dc2 = np.sum((face_lab[fi] - face_lab[fj]) ** 2)
				dn2 = np.sum((self.face_normals[fi] - self.face_normals[fj]) ** 2)
				dp2 = np.sum((centroids[fi] - centroids[fj]) ** 2)
				dot = np.clip(np.dot(self.face_normals[fi], self.face_normals[fj]), -1.0, 1.0)
				theta = np.arccos(dot)
				w = np.exp(-(dc2 / (sigma_c ** 2)) - (dn2 / (sigma_n ** 2)) - (dp2 / (sigma_p ** 2)) - (theta ** 2 / (sigma_t ** 2)))
				local_list.append((face_to_row[fj], float(w)))
			local_edges[face_to_row[fi]] = local_list

			geo_list: List[Tuple[int, float]] = []
			for fj, hops in geo_nbrs[fi]:
				dc2 = np.sum((face_lab[fi] - face_lab[fj]) ** 2)
				dn2 = np.sum((self.face_normals[fi] - self.face_normals[fj]) ** 2)
				dp2 = np.sum((centroids[fi] - centroids[fj]) ** 2)
				geo_decay = np.exp(-((hops - 1) ** 2) / (max(geodesic_sigma_hops, 1e-6) ** 2))
				w = np.exp(-(dc2 / (sigma_c ** 2)) - (dn2 / (sigma_n ** 2)) - (dp2 / (sigma_p ** 2))) * geo_decay
				geo_list.append((face_to_row[fj], float(w)))
			geo_edges[face_to_row[fi]] = geo_list

		compatibility = np.array([
			[0.0, 1.0, 2.0],  # wound vs [wound,tissue,none]
			[1.0, 0.0, 1.0],  # tissue
			[2.0, 1.0, 0.0],  # none
		], dtype=np.float64)

		q = unary.copy()
		n_faces = len(face_indices)

		for _ in range(mean_field_iters):
			logits = np.zeros((n_faces, 3), dtype=np.float64)
			for i in range(n_faces):
				msg_local = np.zeros(3, dtype=np.float64)
				for j, w in local_edges.get(i, []):
					msg_local += w * (compatibility @ q[j])

				msg_geo = np.zeros(3, dtype=np.float64)
				for j, w in geo_edges.get(i, []):
					msg_geo += w * (compatibility @ q[j])

				logits[i] = -unary_energy[i] - lambda_local * msg_local - lambda_geodesic * msg_geo

			q = self._softmax(logits)

		self.crf_posteriors = {}
		self.crf_labels = {}
		self.crf_confidence = {}

		for row, face_idx in enumerate(face_indices):
			p = q[row]
			self.crf_posteriors[face_idx] = p
			lbl_idx = int(np.argmax(p))
			self.crf_labels[face_idx] = self.crf_class_names[lbl_idx]
			self.crf_confidence[face_idx] = float(np.max(p))

		# Also expose through base semantic maps for easier downstream reuse
		self.semantic_posteriors = {
			fi: np.array([p[1], p[0]], dtype=np.float64)
			for fi, p in self.crf_posteriors.items()
		}
		self.semantic_labels = {
			fi: ("wound" if lbl == "wound" else "tissue")
			for fi, lbl in self.crf_labels.items()
			if lbl in {"wound", "tissue"}
		}
		self.semantic_confidence = {
			fi: conf for fi, conf in self.crf_confidence.items()
		}

		print("CRF probability-map inference complete")
		print(f"  Faces inferred: {len(self.crf_posteriors)}")
		print(f"  Mean confidence: {np.mean(list(self.crf_confidence.values())):.3f}")

		return self.crf_posteriors

	# ------------------------------------------------------------------
	# Visualization
	# ------------------------------------------------------------------
	def visualize_crf_probability_map(self, session_name: str = "WoundCRF_ProbabilityMap"):
		"""Visualize CRF probability maps using rerun.
		
		Creates multiple views:
		- Original mesh with vertex colors
		- Wound probability heatmap
		- Tissue probability heatmap
		- None probability heatmap
		- Final classification (wound=red, tissue=green, none=blue)
		- Confidence heatmap
		"""
		if not self.crf_posteriors:
			print("Error: No CRF posteriors available. Run run_crf_probability_map() first.")
			return
		
		try:
			import rerun as rr
		except ImportError:
			print("Error: rerun package not installed. Install with: pip install rerun-sdk")
			return
		
		print(f"\n=== Visualizing CRF Probability Map with Rerun ===")
		rr.init(session_name, spawn=True)
		
		# Log original mesh with vertex colors
		if self.vertex_colors is not None:
			rr.log(
				"mesh/original",
				rr.Mesh3D(
					vertex_positions=self.vertices,
					triangle_indices=self.faces,
					vertex_colors=(self.vertex_colors * 255).astype(np.uint8),
				),
			)
		
		# Create probability colormaps
		wound_colormap = self._create_probability_colormap(class_idx=0)  # wound
		tissue_colormap = self._create_probability_colormap(class_idx=1)  # tissue
		none_colormap = self._create_probability_colormap(class_idx=2)  # none
		classification_colormap = self._create_classification_colormap()
		confidence_colormap = self._create_confidence_colormap_crf()
		
		# Log wound probability heatmap
		self._log_mesh_with_face_colors(
			rr, "mesh/probability/wound", wound_colormap,
			"Wound Probability (Red=High, Blue=Low)"
		)
		
		# Log tissue probability heatmap
		self._log_mesh_with_face_colors(
			rr, "mesh/probability/tissue", tissue_colormap,
			"Tissue Probability (Green=High, Blue=Low)"
		)
		
		# Log none probability heatmap
		self._log_mesh_with_face_colors(
			rr, "mesh/probability/none", none_colormap,
			"None Probability (Gray=High, White=Low)"
		)
		
		# Log final classification
		self._log_mesh_with_face_colors(
			rr, "mesh/classification", classification_colormap,
			"Classification (Red=Wound, Green=Tissue, Blue=None)"
		)
		
		# Log confidence heatmap
		self._log_mesh_with_face_colors(
			rr, "mesh/confidence", confidence_colormap,
			"Confidence (Yellow=High, Purple=Low)"
		)
		
		# Log statistics
		stats_text = self._get_crf_stats_text()
		rr.log("stats/crf_summary", rr.TextDocument(stats_text, media_type=rr.MediaType.MARKDOWN))
		
		print("Visualization complete. Check the Rerun viewer.")
	
	def _create_probability_colormap(self, class_idx: int) -> Dict[int, np.ndarray]:
		"""Create colormap for a specific class probability.
		
		Args:
			class_idx: 0=wound (red scale), 1=tissue (green scale), 2=none (gray scale)
		"""
		colormap: Dict[int, np.ndarray] = {}
		
		for face_idx, posteriors in self.crf_posteriors.items():
			prob = posteriors[class_idx]
			
			if class_idx == 0:  # wound - red to blue gradient
				r = int(255 * prob)
				g = 0
				b = int(255 * (1.0 - prob))
			elif class_idx == 1:  # tissue - green to blue gradient
				r = 0
				g = int(255 * prob)
				b = int(255 * (1.0 - prob))
			else:  # none - white to gray gradient
				val = int(255 * (1.0 - prob * 0.7))
				r = g = b = val
			
			colormap[face_idx] = np.array([r, g, b], dtype=np.uint8)
		
		return colormap
	
	def _create_classification_colormap(self) -> Dict[int, np.ndarray]:
		"""Create colormap for final classification labels."""
		colormap: Dict[int, np.ndarray] = {}
		
		label_colors = {
			"wound": np.array([255, 0, 0], dtype=np.uint8),    # red
			"tissue": np.array([0, 255, 0], dtype=np.uint8),   # green
			"none": np.array([0, 0, 255], dtype=np.uint8),     # blue
		}
		
		for face_idx, label in self.crf_labels.items():
			colormap[face_idx] = label_colors[label]
		
		return colormap
	
	def _create_confidence_colormap_crf(self) -> Dict[int, np.ndarray]:
		"""Create colormap for classification confidence (yellow=high, purple=low)."""
		colormap: Dict[int, np.ndarray] = {}
		
		for face_idx, conf in self.crf_confidence.items():
			# Viridis-like: purple (low) -> yellow (high)
			if conf < 0.5:
				t = conf / 0.5
				r = int(68 + t * (31 - 68))
				g = int(1 + t * (120 - 1))
				b = int(84 + t * (180 - 84))
			else:
				t = (conf - 0.5) / 0.5
				r = int(31 + t * (253 - 31))
				g = int(120 + t * (231 - 120))
				b = int(180 + t * (37 - 180))
			
			colormap[face_idx] = np.array([r, g, b], dtype=np.uint8)
		
		return colormap
	
	def _log_mesh_with_face_colors(
		self,
		rr,
		path: str,
		face_colormap: Dict[int, np.ndarray],
		description: str = "",
	):
		"""Log mesh with per-face colors to rerun."""
		# Convert face colors to vertex colors by averaging
		vertex_colors = np.zeros((len(self.vertices), 3), dtype=np.float32)
		vertex_counts = np.zeros(len(self.vertices), dtype=np.float32)
		
		for face_idx in range(len(self.faces)):
			if face_idx in face_colormap:
				color = face_colormap[face_idx].astype(np.float32)
			else:
				color = np.array([128, 128, 128], dtype=np.float32)  # gray for missing
			
			for vertex_idx in self.faces[face_idx]:
				vertex_colors[vertex_idx] += color
				vertex_counts[vertex_idx] += 1.0
		
		# Average colors
		mask = vertex_counts > 0
		vertex_colors[mask] /= vertex_counts[mask, np.newaxis]
		vertex_colors = np.clip(vertex_colors, 0, 255).astype(np.uint8)
		
		rr.log(
			path,
			rr.Mesh3D(
				vertex_positions=self.vertices,
				triangle_indices=self.faces,
				vertex_colors=vertex_colors,
			),
		)
		
		if description:
			rr.log(f"{path}/description", rr.TextDocument(description))
	
	def _get_crf_stats_text(self) -> str:
		"""Generate statistics summary for CRF results."""
		if not self.crf_posteriors:
			return "No CRF results available."
		
		# Count labels
		label_counts = {"wound": 0, "tissue": 0, "none": 0}
		for label in self.crf_labels.values():
			label_counts[label] += 1
		
		total = len(self.crf_labels)
		
		# Confidence statistics
		confidences = list(self.crf_confidence.values())
		mean_conf = np.mean(confidences)
		std_conf = np.std(confidences)
		min_conf = np.min(confidences)
		max_conf = np.max(confidences)
		
		# Probability statistics per class
		wound_probs = [p[0] for p in self.crf_posteriors.values()]
		tissue_probs = [p[1] for p in self.crf_posteriors.values()]
		none_probs = [p[2] for p in self.crf_posteriors.values()]
		
		text = f"""# CRF Probability Map Statistics

## Classification Distribution
- **Total Faces**: {total}
- **Wound**: {label_counts['wound']} ({100*label_counts['wound']/total:.1f}%)
- **Tissue**: {label_counts['tissue']} ({100*label_counts['tissue']/total:.1f}%)
- **None**: {label_counts['none']} ({100*label_counts['none']/total:.1f}%)

## Confidence Statistics
- **Mean**: {mean_conf:.3f}
- **Std Dev**: {std_conf:.3f}
- **Min**: {min_conf:.3f}
- **Max**: {max_conf:.3f}

## Probability Statistics
### Wound Class
- Mean: {np.mean(wound_probs):.3f}
- Std: {np.std(wound_probs):.3f}

### Tissue Class
- Mean: {np.mean(tissue_probs):.3f}
- Std: {np.std(tissue_probs):.3f}

### None Class
- Mean: {np.mean(none_probs):.3f}
- Std: {np.std(none_probs):.3f}

## CRF Parameters
- **Mean-field iterations**: {8}
- **Local lambda**: {1.0}
- **Geodesic lambda**: {0.6}
- **Class names**: {', '.join(self.crf_class_names)}
"""
		return text


def main():
	"""Example usage for the CRF variant (same loading workflow as base)."""
	mesh_path = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse\dense\poisson_smoothed.ply")
	model_path = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse\dense")
	mask_dir = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse\dense\masks_wound_only_undistorted")
	extruded_mask_dir = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse\dense\masks_extruded_wound_only_undistorted")

	try:
		model = WoundModel3DMeshCRF()
		model.load_mesh(mesh_path)
		model.load_colmap_cameras(model_path)
		model.load_masks(mask_dir)
		model.load_extruded_masks(extruded_mask_dir)
		model.process_wound_segmentation(filter_mode="open3d")
		model.run_crf_probability_map()
		
		# Visualize the probability map
		model.visualize_crf_probability_map()
		
	except Exception as e:
		print(f"Error in main: {e}")
		import traceback
		traceback.print_exc()


if __name__ == "__main__":
	main()
