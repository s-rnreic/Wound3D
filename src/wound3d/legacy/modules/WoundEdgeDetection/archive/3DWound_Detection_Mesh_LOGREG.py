"""
3D Wound Edge Detection - Mesh-Based Logistic Regression Version

This module provides mesh-based 3D wound segmentation using Logistic Regression
for semantic classification and Graph-Cut for spatial regularization.

Pipeline:
    1. Feature Engineering — rich per-face feature vectors including LAB color,
       face normals, curvature, and 1-ring neighborhood statistics.
    2. Feature Normalization — z-score standardization so all dimensions contribute
       equally to the logistic regression decision boundary.
    3. Logistic Regression Training — L2-regularised binary classifier with
       class balancing, trained on faces with strong wound / tissue visibility
       evidence from mask projections.
    4. Probability Map — P(wound | x_i) computed for every face in the mesh.
    5. Spatial Regularization via Graph-Cut — energy minimisation that enforces
       neighbourhood consistency using per-edge weights derived from normal
       similarity, colour difference, and dihedral angle.

Extends:
    WoundModel3DMesh (from 3DWound_Detection_Mesh.py)

Author: Auto-generated
Date: March 2026
"""

import os
import sys
import importlib.util
import numpy as np
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any

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
# Import base class via importlib (module filename starts with a digit)
# ---------------------------------------------------------------------------
_base_path = os.path.join(_current_dir, "3DWound_Detection_Mesh.py")
_spec = importlib.util.spec_from_file_location("wound_mesh_base", _base_path)
_base_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base_module)

WoundModel3DMesh = _base_module.WoundModel3DMesh
Spectator = _base_module.Spectator


# ============================================================================
# WoundModel3DMeshLogReg
# ============================================================================

class WoundModel3DMeshLogReg(WoundModel3DMesh):
    """
    Mesh-based 3D wound segmentation using Logistic Regression + Graph-Cut.

    Inherits all mesh loading, camera handling, mask projection, raycasting,
    topology computation, submesh extraction, and hole-filling from
    ``WoundModel3DMesh``.  Replaces the GMM-based semantic classification
    with:

    * Rich per-face feature vectors (LAB colour, normals, curvature,
      neighbourhood statistics).
    * Logistic Regression (L2 regularised, class-balanced) trained on faces
      with clear wound / tissue visibility evidence.
    * Graph-Cut (s–t min-cut) spatial regularisation with adaptive per-edge
      weights derived from colour difference, normal similarity, and dihedral
      angle.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def __init__(self):
        super().__init__()

        # ---- Logistic Regression model & scaler --------------------
        self.logreg_model = None            # sklearn LogisticRegression
        self.feature_scaler_mean = None     # (D,)
        self.feature_scaler_std = None      # (D,)

        # ---- Pre-computed per-face geometric quantities ------------
        self.face_curvature = None          # (M,)  mean dihedral angle
        self.face_lab_colors = None         # (M, 3) LAB per face

        # ---- Interpretability --------------------------------------
        self.logreg_coefficients = None     # (D,) weights
        self.logreg_intercept = None        # scalar bias
        self.logreg_feature_importance = {} # name -> |weight|

        # ---- Graph-Cut diagnostics ---------------------------------
        self.graph_cut_edge_weights = {}    # (i, j) -> w_ij
        self.graph_cut_sigma_color = None   # adaptive σ_c
        self.graph_cut_sigma_normal = None  # adaptive σ_n
        self.graph_cut_energy = None        # min-cut flow value

    # ==================================================================
    # Vectorised colour conversion
    # ==================================================================
    def _rgb_to_lab_batch(self, rgb: np.ndarray) -> np.ndarray:
        """Convert (N, 3) RGB values in [0, 1] to CIE-LAB.

        Returns
        -------
        lab : ndarray, shape (N, 3)
            Columns are L*, a*, b*.
        """
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float64)

        # Inverse sRGB companding
        mask = rgb > 0.04045
        linear = np.where(mask,
                          np.power((rgb + 0.055) / 1.055, 2.4),
                          rgb / 12.92)

        # RGB → XYZ  (D65)
        M = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
        xyz = linear @ M.T

        # Normalise by D65 white-point
        xyz /= np.array([0.95047, 1.0, 1.08883])

        # f(t) for XYZ → LAB
        delta = 6.0 / 29.0
        big = xyz > delta ** 3
        f = np.where(big, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4.0 / 29.0)

        L = 116.0 * f[:, 1] - 16.0
        a = 500.0 * (f[:, 0] - f[:, 1])
        b = 200.0 * (f[:, 1] - f[:, 2])

        return np.column_stack([L, a, b])

    # ==================================================================
    # Per-face LAB colours (vectorised)
    # ==================================================================
    def _compute_face_lab_colors(self) -> np.ndarray:
        """Compute per-face LAB colours as the mean of vertex colours."""
        if self.vertex_colors is None:
            self.face_lab_colors = np.zeros((len(self.faces), 3), dtype=np.float64)
            return self.face_lab_colors

        v0 = self.vertex_colors[self.faces[:, 0]]
        v1 = self.vertex_colors[self.faces[:, 1]]
        v2 = self.vertex_colors[self.faces[:, 2]]
        mean_rgb = (v0 + v1 + v2) / 3.0

        self.face_lab_colors = self._rgb_to_lab_batch(mean_rgb)
        return self.face_lab_colors

    # ==================================================================
    # Per-face discrete curvature (mean dihedral angle)
    # ==================================================================
    def _compute_face_curvature(self) -> np.ndarray:
        """Compute per-face curvature proxy via mean dihedral angle.

        Uses face adjacency pairs; vectorised via NumPy ``add.at``.
        """
        self._ensure_face_normals()
        n_faces = len(self.faces)

        # Collect adjacency pairs -----------------------------------
        if self._trimesh_cache is not None:
            try:
                adj_pairs = np.asarray(self._trimesh_cache.face_adjacency)
            except Exception:
                adj_pairs = None
        else:
            adj_pairs = None

        if adj_pairs is None:
            pairs = []
            seen = set()
            for f, neighbours in self.face_adjacency.items():
                for n in neighbours:
                    key = (min(f, n), max(f, n))
                    if key not in seen:
                        seen.add(key)
                        pairs.append(key)
            adj_pairs = np.array(pairs, dtype=np.int64) if pairs else np.empty((0, 2), dtype=np.int64)

        if len(adj_pairs) == 0:
            self.face_curvature = np.zeros(n_faces, dtype=np.float64)
            return self.face_curvature

        n0 = self.face_normals[adj_pairs[:, 0]]
        n1 = self.face_normals[adj_pairs[:, 1]]
        dots = np.clip(np.sum(n0 * n1, axis=1), -1.0, 1.0)
        dihedral = np.arccos(dots)

        curv_sum = np.zeros(n_faces, dtype=np.float64)
        curv_cnt = np.zeros(n_faces, dtype=np.float64)
        np.add.at(curv_sum, adj_pairs[:, 0], dihedral)
        np.add.at(curv_sum, adj_pairs[:, 1], dihedral)
        np.add.at(curv_cnt, adj_pairs[:, 0], 1.0)
        np.add.at(curv_cnt, adj_pairs[:, 1], 1.0)

        curv_cnt = np.maximum(curv_cnt, 1.0)
        self.face_curvature = curv_sum / curv_cnt
        return self.face_curvature

    # ==================================================================
    # Feature engineering
    # ==================================================================
    def _build_feature_vectors_logreg(
        self,
        min_total_views: int = 1,
        tissue_candidate_tissue_thresh: float = 0.5,
        tissue_candidate_wound_thresh: float = 0.2,
        tissue_visibility_percentile: float = 70.0,
        ring_sizes: Optional[List[int]] = None,
        use_visibility_ratios: bool = True,
        use_height_deviation: bool = True,
        use_normal_deviation: bool = True,
    ) -> Tuple[np.ndarray, List[int], Dict[int, Tuple[float, float, int]]]:
        """Build rich feature vectors for logistic-regression classification.

        Per-face features (≈18 dimensions):

        ======  ==========================================
        Dim     Feature
        ======  ==========================================
        0-1     r_wound, r_tissue  (visibility ratios)
        2-3     LAB  a*,  b*  (face mean colour)
        4-6     nx, ny, nz  (face normal components)
        7       curvature  (mean dihedral angle)
        8-9     height deviation from tissue plane (ring 1, 3)
        10      normal deviation from tissue plane
        11-12   1-ring mean  LAB a*, b*
        13-15   1-ring variance  LAB L*, a*, b*
        16      1-ring normal deviation (mean angle to neighbours)
        17      local curvature contrast
        ======  ==========================================

        Returns
        -------
        features : ndarray  (N, D)
            Standardised (z-scored) feature matrix.
        face_indices : list[int]
            Global face indices corresponding to rows of *features*.
        ratio_cache : dict
            face_idx → (r_wound, r_tissue, total_views)
        """
        if ring_sizes is None:
            ring_sizes = [1, 3]

        # ----- Geometry pre-computation ----------------------------
        self._ensure_face_normals()
        centroids = self._get_face_centroids()
        lab_colors = self._compute_face_lab_colors()
        curvature = self._compute_face_curvature()

        # ----- Visibility ratios -----------------------------------
        face_indices = []
        ratio_cache: Dict[int, Tuple[float, float, int]] = {}
        for fi in range(len(self.faces)):
            w_vis = self.face_visibility_count.get(fi, 0)
            t_vis = self.tissue_edge_visibility_count.get(fi, 0)
            total = w_vis + t_vis
            r_w = w_vis / total if total > 0 else 0.0
            r_t = t_vis / total if total > 0 else 0.0
            ratio_cache[fi] = (r_w, r_t, total)
            if total >= min_total_views:
                face_indices.append(fi)

        if not face_indices:
            print("Warning: no faces meet the minimum-views criterion")
            return np.empty((0, 0)), [], ratio_cache

        # ----- Tissue plane fit ------------------------------------
        tissue_cands = [
            f for f in face_indices
            if ratio_cache[f][1] > tissue_candidate_tissue_thresh
            and ratio_cache[f][0] < tissue_candidate_wound_thresh
        ]
        if tissue_cands:
            tv = np.array([ratio_cache[f][2] for f in tissue_cands])
            thr = np.percentile(tv, tissue_visibility_percentile)
            hi_vis = [f for f in tissue_cands if ratio_cache[f][2] >= thr]
            if not hi_vis:
                hi_vis = tissue_cands
        else:
            hi_vis = face_indices
        plane_pts = centroids[hi_vis] if hi_vis else centroids[face_indices]
        plane_normal, plane_d = self._fit_plane(plane_pts)

        # ----- Height deviation ------------------------------------
        height_devs: Optional[Dict[int, List[float]]] = None
        if use_height_deviation:
            height_devs = self._compute_height_deviation_multiscale(
                face_indices, centroids, plane_normal, plane_d, ring_sizes
            )

        # ----- Normal deviation from plane -------------------------
        normal_dev: Optional[np.ndarray] = None
        if use_normal_deviation:
            normal_dev = self._compute_normal_deviation(self.face_normals, plane_normal)

        # ----- 1-ring neighbourhood statistics ---------------------
        #   Computed for *all* faces in face_indices in a single pass
        ring1_mean_a = np.zeros(len(self.faces), dtype=np.float64)
        ring1_mean_b = np.zeros(len(self.faces), dtype=np.float64)
        ring1_var_L = np.zeros(len(self.faces), dtype=np.float64)
        ring1_var_a = np.zeros(len(self.faces), dtype=np.float64)
        ring1_var_b = np.zeros(len(self.faces), dtype=np.float64)
        ring1_normal_dev = np.zeros(len(self.faces), dtype=np.float64)
        ring1_curv_contrast = np.zeros(len(self.faces), dtype=np.float64)

        for fi in face_indices:
            nbrs = self.face_adjacency.get(fi, [])
            if not nbrs:
                continue
            nbr_idx = np.array(nbrs, dtype=np.int64)

            # Colour statistics in 1-ring
            nbr_lab = lab_colors[nbr_idx]                # (K, 3)
            ring1_mean_a[fi] = np.mean(nbr_lab[:, 1])
            ring1_mean_b[fi] = np.mean(nbr_lab[:, 2])
            ring1_var_L[fi] = np.var(nbr_lab[:, 0])
            ring1_var_a[fi] = np.var(nbr_lab[:, 1])
            ring1_var_b[fi] = np.var(nbr_lab[:, 2])

            # Normal deviation from neighbours
            n_i = self.face_normals[fi]
            nbr_normals = self.face_normals[nbr_idx]
            dots = np.clip(nbr_normals @ n_i, -1.0, 1.0)
            ring1_normal_dev[fi] = np.mean(np.arccos(np.abs(dots)))

            # Curvature contrast
            ring1_curv_contrast[fi] = curvature[fi] - np.mean(curvature[nbr_idx])

        # ----- Assemble feature matrix -----------------------------
        feature_names: List[str] = []
        cols: List[np.ndarray] = []

        def _add(name: str, arr: np.ndarray):
            feature_names.append(name)
            cols.append(arr)

        if use_visibility_ratios:
            _add("r_wound",  np.array([ratio_cache[f][0] for f in face_indices]))
            _add("r_tissue", np.array([ratio_cache[f][1] for f in face_indices]))

        _add("lab_a", lab_colors[face_indices, 1])
        _add("lab_b", lab_colors[face_indices, 2])
        _add("nx", self.face_normals[face_indices, 0])
        _add("ny", self.face_normals[face_indices, 1])
        _add("nz", self.face_normals[face_indices, 2])
        _add("curvature", curvature[face_indices])

        if use_height_deviation and height_devs is not None:
            for k, ring in enumerate(ring_sizes):
                _add(f"height_dev_r{ring}",
                     np.array([height_devs[f][k] for f in face_indices]))

        if use_normal_deviation and normal_dev is not None:
            _add("normal_dev", normal_dev[face_indices])

        _add("ring1_mean_a", ring1_mean_a[face_indices])
        _add("ring1_mean_b", ring1_mean_b[face_indices])
        _add("ring1_var_L",  ring1_var_L[face_indices])
        _add("ring1_var_a",  ring1_var_a[face_indices])
        _add("ring1_var_b",  ring1_var_b[face_indices])
        _add("ring1_normal_dev", ring1_normal_dev[face_indices])
        _add("ring1_curv_contrast", ring1_curv_contrast[face_indices])

        features = np.column_stack(cols)  # (N, D)
        print(f"Feature matrix: {features.shape[0]} faces × {features.shape[1]} features")
        print(f"Features: {feature_names}")

        # ----- Z-score normalisation --------------------------------
        self.feature_scaler_mean = np.mean(features, axis=0)
        self.feature_scaler_std = np.std(features, axis=0) + 1e-9
        features = (features - self.feature_scaler_mean) / self.feature_scaler_std

        # ----- Store bookkeeping ------------------------------------
        self.semantic_feature_names = feature_names
        self.semantic_face_indices = face_indices
        self.semantic_features = features
        self.semantic_fitted_plane = (plane_normal, plane_d)

        # ----- Feature colour-maps for visualisation ----------------
        self._build_feature_colormaps(features, face_indices, feature_names)

        return features, face_indices, ratio_cache

    # ------------------------------------------------------------------
    def _build_feature_colormaps(self, features: np.ndarray,
                                  face_indices: List[int],
                                  feature_names: List[str]) -> None:
        """Create per-feature colour-maps (viridis-like) for Rerun visualisation."""
        self.semantic_feature_colormaps = {}
        for feat_idx, fname in enumerate(feature_names):
            vals = features[:, feat_idx]
            lo, hi = np.percentile(vals, [5, 95])
            rng = hi - lo if hi > lo else 1.0
            cmap: Dict[int, np.ndarray] = {}
            for i, fi in enumerate(face_indices):
                t = np.clip((vals[i] - lo) / rng, 0.0, 1.0)
                if t < 0.33:
                    r, g, b = int(148 * (1 - t / 0.33)), 0, int(211 * (1 - t / 0.33))
                elif t < 0.67:
                    r, g, b = 0, int(255 * (t - 0.33) / 0.34), 211
                else:
                    r, g, b = int(255 * (t - 0.67) / 0.33), 255, 0
                cmap[fi] = np.array([r, g, b], dtype=np.uint8)
            self.semantic_feature_colormaps[fname] = cmap

    # ==================================================================
    # Logistic Regression training
    # ==================================================================
    def _train_logistic_regression(
        self,
        features: np.ndarray,
        face_indices: List[int],
        ratio_cache: Dict[int, Tuple[float, float, int]],
        wound_ratio_thresh: float = 0.3,
        tissue_ratio_thresh: float = 0.5,
        C: float = 1.0,
        max_iter: int = 1000,
    ) -> bool:
        """Train a binary logistic-regression classifier.

        Training labels come from mask-projection visibility ratios:

        * **wound** (y = 1):  r_wound > *wound_ratio_thresh*
        * **tissue** (y = 0): r_tissue > *tissue_ratio_thresh*
        * Faces meeting *both* thresholds are excluded (ambiguous).

        Parameters
        ----------
        C : float
            Inverse regularisation strength (smaller → stronger L2).
        max_iter : int
            Maximum solver iterations.

        Returns
        -------
        success : bool
        """
        from sklearn.linear_model import LogisticRegression

        idx_map = {fi: i for i, fi in enumerate(face_indices)}
        X_train, y_train = [], []
        wound_sample_indices: List[int] = []
        tissue_sample_indices: List[int] = []

        for fi in face_indices:
            r_w, r_t, _ = ratio_cache[fi]
            is_wound = r_w > wound_ratio_thresh
            is_tissue = r_t > tissue_ratio_thresh
            if is_wound and is_tissue:
                continue  # ambiguous — skip
            if is_wound:
                X_train.append(features[idx_map[fi]])
                y_train.append(1)
                wound_sample_indices.append(fi)
            elif is_tissue:
                X_train.append(features[idx_map[fi]])
                y_train.append(0)
                tissue_sample_indices.append(fi)

        if not X_train:
            print("Error: no training samples found for logistic regression")
            return False

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        n_wound = int(np.sum(y_train == 1))
        n_tissue = int(np.sum(y_train == 0))
        print(f"Logistic Regression training set: {n_tissue} tissue, {n_wound} wound")

        if n_wound == 0 or n_tissue == 0:
            print("Error: need at least one sample per class")
            return False

        # Store class sample indices for visualisation
        self.semantic_class_samples = {
            "tissue": tissue_sample_indices,
            "wound": wound_sample_indices,
        }

        # Fit model ------------------------------------------------
        clf = LogisticRegression(
            penalty="l2",
            C=C,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=max_iter,
        )
        clf.fit(X_train, y_train)

        self.logreg_model = clf
        self.logreg_coefficients = clf.coef_.flatten()
        self.logreg_intercept = float(clf.intercept_[0])

        # Feature importance — absolute weight per normalised feature
        self.logreg_feature_importance = {}
        for i, fname in enumerate(self.semantic_feature_names):
            self.logreg_feature_importance[fname] = float(np.abs(self.logreg_coefficients[i]))

        # Sort and print
        sorted_imp = sorted(self.logreg_feature_importance.items(), key=lambda x: -x[1])
        print("Feature importances (|w|):")
        for name, imp in sorted_imp:
            print(f"  {name:25s} {imp:.4f}")

        return True

    # ==================================================================
    # Predict probabilities for ALL faces
    # ==================================================================
    def _predict_probabilities(
        self,
        features: np.ndarray,
        face_indices: List[int],
    ) -> np.ndarray:
        """Predict P(wound | x_i) for every face.

        Returns
        -------
        posteriors : ndarray, shape (N, 2)
            Column 0 = P(tissue), column 1 = P(wound).
        """
        if self.logreg_model is None:
            raise RuntimeError("Logistic regression model not trained yet")

        posteriors = self.logreg_model.predict_proba(features)  # (N, 2)

        # sklearn orders classes numerically: 0=tissue, 1=wound
        # Ensure column order matches self.semantic_class_names = ["tissue", "wound"]
        classes = self.logreg_model.classes_
        if len(classes) == 2 and classes[0] == 1:
            # Flip columns if sklearn put wound first
            posteriors = posteriors[:, ::-1]

        return posteriors

    # ==================================================================
    # Graph-Cut edge weights
    # ==================================================================
    def _compute_edge_weights(
        self,
        face_indices: List[int],
        sigma_color: Optional[float] = None,
        sigma_normal: Optional[float] = None,
    ) -> Dict[Tuple[int, int], float]:
        """Compute adaptive per-edge pairwise weights for graph-cut.

        .. math::

            w_{ij} = \\exp\\!\\left(-\\frac{\\|c_i-c_j\\|^2}{2\\sigma_c^2}\\right)
                     \\cdot \\exp\\!\\left(-\\frac{\\theta_{ij}^2}{2\\sigma_n^2}\\right)

        where :math:`c` is CIE-LAB colour and :math:`\\theta_{ij}` is the
        dihedral angle between adjacent faces.

        *σ_c* and *σ_n* default to the **median** of the respective quantities
        across all mesh edges (adaptive).
        """
        face_set = set(face_indices)
        lab = self.face_lab_colors  # (M, 3)
        self._ensure_face_normals()
        normals = self.face_normals

        # Collect (i, j, colour_dist, dihedral_angle) for edges inside face_set
        edges: List[Tuple[int, int]] = []
        color_diffs: List[float] = []
        dihedral_angles: List[float] = []

        seen: Set[Tuple[int, int]] = set()
        for fi in face_indices:
            for adj in self.face_adjacency.get(fi, []):
                if adj not in face_set:
                    continue
                key = (min(fi, adj), max(fi, adj))
                if key in seen:
                    continue
                seen.add(key)

                cdiff = float(np.linalg.norm(lab[fi] - lab[adj]))
                dot = np.clip(np.dot(normals[fi], normals[adj]), -1.0, 1.0)
                theta = float(np.arccos(dot))

                edges.append(key)
                color_diffs.append(cdiff)
                dihedral_angles.append(theta)

        if not edges:
            return {}

        color_diffs = np.array(color_diffs)
        dihedral_angles = np.array(dihedral_angles)

        # Adaptive sigmas (median heuristic)
        sc = sigma_color if sigma_color else float(np.median(color_diffs) + 1e-9)
        sn = sigma_normal if sigma_normal else float(np.median(dihedral_angles) + 1e-9)
        self.graph_cut_sigma_color = sc
        self.graph_cut_sigma_normal = sn

        weights: Dict[Tuple[int, int], float] = {}
        w_color = np.exp(-color_diffs ** 2 / (2.0 * sc ** 2))
        w_normal = np.exp(-dihedral_angles ** 2 / (2.0 * sn ** 2))
        w_all = w_color * w_normal

        for idx, key in enumerate(edges):
            weights[key] = float(w_all[idx])

        self.graph_cut_edge_weights = weights
        print(f"Graph-cut edge weights: {len(weights)} edges, "
              f"σ_color={sc:.4f}, σ_normal={sn:.4f}")

        return weights

    # ==================================================================
    # Graph-Cut spatial regularisation
    # ==================================================================
    def _apply_graph_cut(
        self,
        posteriors: np.ndarray,
        face_indices: List[int],
        edge_weights: Dict[Tuple[int, int], float],
        lambda_smooth: float = 1.0,
    ) -> np.ndarray:
        """Min-cut / max-flow spatial regularisation.

        Energy:

        .. math::

            E = \\sum_i -\\log P(y_i|x_i) + \\lambda \\sum_{(i,j)} w_{ij}\\,[y_i \\neq y_j]

        Parameters
        ----------
        posteriors : ndarray (N, 2)
            Column 0 = P(tissue), column 1 = P(wound).
        face_indices : list[int]
            Global face indices (rows of *posteriors*).
        edge_weights : dict
            (i, j) → w_ij  for adjacent face pairs.
        lambda_smooth : float
            Global smoothness weight.

        Returns
        -------
        labels : ndarray of int, shape (N,)
            0 = tissue, 1 = wound.
        """
        n = len(face_indices)
        fi_to_idx = {fi: i for i, fi in enumerate(face_indices)}

        # ---- Unary costs ------------------------------------------
        p_tissue = np.clip(posteriors[:, 0], 1e-12, 1.0)
        p_wound  = np.clip(posteriors[:, 1], 1e-12, 1.0)
        cost_tissue = -np.log(p_tissue)   # cost of labelling tissue
        cost_wound  = -np.log(p_wound)    # cost of labelling wound

        # ---- Try PyMaxflow (graph-based min-cut) ------------------
        try:
            import maxflow

            # Build graph
            n_edges = len(edge_weights)
            g = maxflow.Graph[float](n, n_edges)
            nodes = g.add_nodes(n)

            # Terminal edges (source = tissue, sink = wound)
            for i in range(n):
                # cap_source = cost of assigning wound (cutting source→i)
                # cap_sink   = cost of assigning tissue (cutting i→sink)
                g.add_tedge(nodes[i], cost_wound[i], cost_tissue[i])

            # Neighbour edges with per-edge weights
            for (fi, fj), w in edge_weights.items():
                if fi not in fi_to_idx or fj not in fi_to_idx:
                    continue
                i, j = fi_to_idx[fi], fi_to_idx[fj]
                cap = lambda_smooth * w
                g.add_edge(nodes[i], nodes[j], cap, cap)

            flow = g.maxflow()
            self.graph_cut_energy = float(flow)
            labels = np.array([g.get_segment(nodes[i]) for i in range(n)],
                              dtype=np.int32)
            print(f"Graph-cut solved: flow={flow:.2f}, "
                  f"tissue={int(np.sum(labels == 0))}, wound={int(np.sum(labels == 1))}")
            return labels

        except Exception as e:
            print(f"PyMaxflow unavailable ({e}), falling back to ICM")

        # ---- Fallback: ICM (Iterated Conditional Modes) -----------
        labels = np.argmax(posteriors, axis=1).astype(np.int32)
        for _iter in range(10):
            changed = 0
            for fi in face_indices:
                i = fi_to_idx[fi]
                best_label = labels[i]
                best_cost = float("inf")
                for cand in (0, 1):
                    ucost = cost_tissue[i] if cand == 0 else cost_wound[i]
                    pcost = 0.0
                    for adj in self.face_adjacency.get(fi, []):
                        if adj not in fi_to_idx:
                            continue
                        j = fi_to_idx[adj]
                        if labels[j] != cand:
                            key = (min(fi, adj), max(fi, adj))
                            pcost += lambda_smooth * edge_weights.get(key, 1.0)
                    total = ucost + pcost
                    if total < best_cost:
                        best_cost = total
                        best_label = cand
                if best_label != labels[i]:
                    labels[i] = best_label
                    changed += 1
            if changed == 0:
                break
        print(f"ICM converged: tissue={int(np.sum(labels == 0))}, wound={int(np.sum(labels == 1))}")
        return labels

    # ==================================================================
    # Main classification pipeline (OVERRIDE)
    # ==================================================================
    def run_semantic_classification(
        self,
        # Label thresholds
        wound_ratio_thresh: float = 0.3,
        tissue_ratio_thresh: float = 0.5,
        # Feature building
        min_total_views: int = 1,
        tissue_visibility_percentile: float = 70.0,
        ring_sizes: Optional[List[int]] = None,
        use_visibility_ratios: bool = True,
        use_height_deviation: bool = True,
        use_normal_deviation: bool = True,
        # Logistic regression
        logreg_C: float = 1.0,
        logreg_max_iter: int = 1000,
        # Graph-cut
        lambda_smooth: float = 1.0,
        sigma_color: Optional[float] = None,
        sigma_normal: Optional[float] = None,
        # Misc
        confidence_method: str = "entropy",
        # Unused — kept for API compatibility with base class calls
        color_space: str = "lab_b",
        **kwargs,
    ) -> None:
        """Run the full Logistic-Regression + Graph-Cut classification pipeline.

        Steps:
            1. Detect tissue faces from mask differences.
            2. Build rich feature vectors.
            3. Train logistic regression on high-confidence faces.
            4. Predict P(wound | x) for all faces.
            5. Apply graph-cut spatial regularisation.
            6. Store per-face labels, posteriors, confidence.
            7. Compute wound-edge boundary.
        """
        if not self.faces_processed:
            raise ValueError("Must process wound segmentation first")
        if not self.topology_computed:
            self._compute_mesh_topology()

        # 1. Detect initial tissue faces from mask difference
        self.process_tissue_face_detection(max_mask_rays=200000)

        # 2. Feature engineering
        print("\n--- Feature Engineering ---")
        features, face_indices, ratio_cache = self._build_feature_vectors_logreg(
            min_total_views=min_total_views,
            tissue_visibility_percentile=tissue_visibility_percentile,
            ring_sizes=ring_sizes or [1, 3],
            use_visibility_ratios=use_visibility_ratios,
            use_height_deviation=use_height_deviation,
            use_normal_deviation=use_normal_deviation,
        )
        if features.size == 0:
            print("No features available for classification")
            return

        # 3. Train Logistic Regression
        print("\n--- Logistic Regression Training ---")
        ok = self._train_logistic_regression(
            features, face_indices, ratio_cache,
            wound_ratio_thresh=wound_ratio_thresh,
            tissue_ratio_thresh=tissue_ratio_thresh,
            C=logreg_C, max_iter=logreg_max_iter,
        )
        if not ok:
            print("Logistic Regression training failed")
            return

        # 4. Predict probability map for ALL faces
        print("\n--- Probability Map ---")
        posteriors = self._predict_probabilities(features, face_indices)

        # Store per-face log-likelihoods (log-odds) for visualisation
        self.semantic_log_likelihoods = np.log(posteriors + 1e-12)
        self.semantic_wound_loglik = {}
        self.semantic_tissue_loglik = {}
        for i, fi in enumerate(face_indices):
            self.semantic_tissue_loglik[fi] = float(self.semantic_log_likelihoods[i, 0])
            self.semantic_wound_loglik[fi] = float(self.semantic_log_likelihoods[i, 1])

        # Store posteriors before spatial regularisation
        self.semantic_posteriors_before_mrf = {}
        for i, fi in enumerate(face_indices):
            self.semantic_posteriors_before_mrf[fi] = posteriors[i].copy()

        # 5. Graph-Cut spatial regularisation
        print("\n--- Graph-Cut Spatial Regularisation ---")
        edge_weights = self._compute_edge_weights(
            face_indices,
            sigma_color=sigma_color,
            sigma_normal=sigma_normal,
        )
        labels = self._apply_graph_cut(
            posteriors, face_indices, edge_weights,
            lambda_smooth=lambda_smooth,
        )

        # 6. Compute confidence
        confidence = self._compute_confidence_from_posteriors(
            posteriors, method=confidence_method
        )

        # 7. Store per-face results
        self.semantic_posteriors = {}
        self.semantic_labels = {}
        self.semantic_confidence = {}
        self.semantic_wound_prob = {}
        self.semantic_tissue_prob = {}

        for i, fi in enumerate(face_indices):
            probs = posteriors[i]
            label_name = self.semantic_class_names[int(labels[i])]
            self.semantic_posteriors[fi] = probs
            self.semantic_labels[fi] = label_name
            self.semantic_confidence[fi] = float(confidence[i])
            self.semantic_tissue_prob[fi] = float(probs[0])
            self.semantic_wound_prob[fi] = float(probs[1])

        # 8. Compute wound-edge boundary
        self._compute_wound_edge_boundary()

        # Summary
        n_tissue = sum(1 for v in self.semantic_labels.values() if v == "tissue")
        n_wound = sum(1 for v in self.semantic_labels.values() if v == "wound")
        mean_conf = float(np.mean(list(self.semantic_confidence.values())))
        print(f"\nClassification complete:")
        print(f"  Faces classified: {len(self.semantic_labels)}")
        print(f"  Tissue: {n_tissue}  |  Wound: {n_wound}")
        print(f"  Mean confidence: {mean_conf:.3f}")

    # ==================================================================
    # Rerun visualisation (OVERRIDE)
    # ==================================================================
    def visualize_with_rerun(self, session_name: str = "WoundLogReg",
                             min_tissue_edge_visibility: int = 2):
        """Visualise logistic-regression wound segmentation with Rerun.

        Creates tabs for each stage of the pipeline:
        Full Mesh, Wound Region, Tissue Edge, Features,
        LogReg Classification, Graph-Cut, Confidence Maps, Statistics.
        """
        if not self.model_loaded:
            raise ValueError("Mesh must be loaded first")

        try:
            import rerun as rr
            import rerun.blueprint as rrb
        except ImportError:
            print("Rerun not available for visualisation")
            return

        rr.init(session_name, spawn=True)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

        blueprint = rrb.Blueprint(rrb.Tabs(
            rrb.Spatial3DView(origin="/world",      name="Full Mesh & Tissue Edge"),
            rrb.Spatial3DView(origin="/wound",      name="Wound Region"),
            rrb.Spatial3DView(origin="/tissue_edge", name="Tissue Edge Details"),
            rrb.Spatial3DView(origin="/features",   name="Feature Vectors"),
            rrb.Spatial3DView(origin="/logreg",     name="LogReg Classification"),
            rrb.Spatial3DView(origin="/graphcut",   name="Graph-Cut Regularisation"),
            rrb.Spatial3DView(origin="/confidence",  name="Confidence Maps"),
            rrb.TextDocumentView(origin="/stats",   name="Statistics"),
        ))
        rr.send_blueprint(blueprint)

        # ── Full mesh ───────────────────────────────────────────────
        vp = self.vertices.astype(np.float32)
        if self.vertex_colors is not None:
            vc = (self.vertex_colors * 255).astype(np.uint8) \
                if np.max(self.vertex_colors) <= 1.0 else self.vertex_colors.astype(np.uint8)
        else:
            vc = np.full((len(self.vertices), 3), 128, dtype=np.uint8)

        rr.log("/world/full_mesh", rr.Mesh3D(
            vertex_positions=vp, triangle_indices=self.faces, vertex_colors=vc
        ), static=True)

        # ── Segmented wound overlay on full mesh ─────────────────
        if self.last_wound_vertices is not None:
            wc = self.last_wound_colors
            if wc is not None:
                wc = (wc * 255).astype(np.uint8) if np.max(wc) <= 1.0 else wc.astype(np.uint8)
            else:
                wc = np.zeros((len(self.last_wound_vertices), 3), dtype=np.uint8)
                wc[:, 1] = 255
            rr.log("/world/segmented_wound", rr.Mesh3D(
                vertex_positions=self.last_wound_vertices.astype(np.float32),
                triangle_indices=self.last_wound_faces, vertex_colors=wc
            ), static=True)

        # ── Wound region detail ─────────────────────────────────
        if self.faces_processed and self.wound_face_indices:
            wv, wf, wco = self.extract_submesh_from_faces(self.wound_face_indices)
            if len(wv) > 0:
                if wco is None:
                    wco = np.full((len(wv), 3), 200, dtype=np.uint8)
                else:
                    wco = (wco * 255).astype(np.uint8) if np.max(wco) <= 1.0 else wco.astype(np.uint8)
                rr.log("/wound/initial", rr.Mesh3D(
                    vertex_positions=wv.astype(np.float32),
                    triangle_indices=wf, vertex_colors=wco
                ), static=True)

            self._log_faces_mesh_rr(rr, "/wound/removed/visibility",
                                    self.removed_faces_visibility, [255, 0, 0])
            self._log_faces_mesh_rr(rr, "/wound/removed/connectivity",
                                    self.removed_faces_connectivity, [255, 128, 0])
            self._log_faces_mesh_rr(rr, "/wound/added/enclosed",
                                    self.added_faces_enclosed, [0, 255, 0])
            self._log_faces_mesh_rr(rr, "/wound/added/closing",
                                    self.added_faces_closing, [0, 200, 0])

        # Wound-edge geometry
        if self.wound_outerboundary_vertices:
            ep = self.vertices[list(self.wound_outerboundary_vertices)]
            rr.log("/wound/outerboundary_vertices", rr.Points3D(
                positions=ep, colors=[[255, 255, 0]] * len(ep), radii=0.003
            ), static=True)
        if self.wound_outerboundary_edges:
            strips = [np.array([self.vertices[v0], self.vertices[v1]], dtype=np.float32)
                      for v0, v1 in self.wound_outerboundary_edges]
            rr.log("/wound/outerboundary_edges", rr.LineStrips3D(
                strips=strips, colors=[[255, 255, 0]] * len(strips), radii=0.002
            ), static=True)

        # ── Tissue edge ─────────────────────────────────────────
        if self.tissue_edge_faces:
            te_filt = self.filter_tissue_edge_by_visibility(min_tissue_edge_visibility)
            if te_filt:
                tv, tf, tc = self.extract_submesh_from_faces(te_filt)
                if len(tv) > 0:
                    if tc is None:
                        tc = np.full((len(tv), 3), 200, dtype=np.uint8); tc[:, 0] = 200; tc[:, 1] = 200
                    else:
                        tc = (tc * 255).astype(np.uint8) if np.max(tc) <= 1.0 else tc.astype(np.uint8)
                    rr.log("/world/tissue_edge", rr.Mesh3D(
                        vertex_positions=tv.astype(np.float32),
                        triangle_indices=tf, vertex_colors=tc
                    ), static=True)
                    rr.log("/tissue_edge/mesh_original_colors", rr.Mesh3D(
                        vertex_positions=tv.astype(np.float32),
                        triangle_indices=tf, vertex_colors=tc
                    ), static=True)

                vis_cm = self._create_visibility_colormap(
                    {f: self.tissue_edge_visibility_count[f]
                     for f in te_filt if f in self.tissue_edge_visibility_count})
                self._log_faces_mesh_rr_with_colors(
                    rr, "/tissue_edge/by_visibility", te_filt, vis_cm)

        # ── Feature vector visualisation ─────────────────────────
        if self.semantic_feature_colormaps:
            for fname, cmap in self.semantic_feature_colormaps.items():
                if cmap:
                    self._log_faces_mesh_rr_with_colors(
                        rr, f"/features/{fname}", set(cmap.keys()), cmap)

        # ── Logistic Regression visualisation ────────────────────
        if self.semantic_class_samples:
            class_cols = {"tissue": [0, 150, 255], "wound": [255, 0, 0]}
            for cname, idxs in self.semantic_class_samples.items():
                if idxs:
                    self._log_faces_mesh_rr(
                        rr, f"/logreg/{cname}_training_samples",
                        set(idxs), class_cols.get(cname, [128, 128, 128]))

            # Log model info
            model_txt = "Logistic Regression Model\n" + "=" * 40 + "\n\n"
            if self.logreg_coefficients is not None:
                model_txt += "Feature weights (w):\n"
                sorted_imp = sorted(self.logreg_feature_importance.items(),
                                    key=lambda x: -x[1])
                for name, imp in sorted_imp:
                    idx = self.semantic_feature_names.index(name)
                    w = self.logreg_coefficients[idx]
                    model_txt += f"  {name:25s}  w={w:+.4f}  |w|={imp:.4f}\n"
                model_txt += f"\nIntercept (b): {self.logreg_intercept:+.4f}\n"
                model_txt += f"\nP(wound|x) = σ(w^T x + b)\n"
                model_txt += f"σ(z) = 1 / (1 + exp(-z))\n"

            n_tissue_train = len(self.semantic_class_samples.get("tissue", []))
            n_wound_train = len(self.semantic_class_samples.get("wound", []))
            model_txt += f"\nTraining samples:  tissue={n_tissue_train}  wound={n_wound_train}\n"
            rr.log("/logreg/model_info", rr.TextDocument(model_txt), static=True)

        # Log probability maps from LogReg (before graph-cut)
        if self.semantic_wound_loglik and self.semantic_tissue_loglik:
            for cls_name, lmap in [("tissue", self.semantic_tissue_loglik),
                                    ("wound", self.semantic_wound_loglik)]:
                vals = np.array(list(lmap.values()))
                lo, hi = np.percentile(vals, [2, 98])
                rng = hi - lo if hi > lo else 1.0
                cmap = {}
                for fi, v in lmap.items():
                    t = np.clip((v - lo) / rng, 0.0, 1.0)
                    cmap[fi] = self._confidence_to_color(t)
                self._log_faces_mesh_rr_with_colors(
                    rr, f"/logreg/{cls_name}_log_prob", set(cmap.keys()), cmap)

        # ── Graph-Cut visualisation ──────────────────────────────
        if self.semantic_posteriors_before_mrf:
            # Before graph-cut
            wound_before = {}
            for fi, post in self.semantic_posteriors_before_mrf.items():
                wound_before[fi] = float(post[1])

            wb_colors = self._create_confidence_colormap(wound_before)
            self._log_faces_mesh_rr_with_colors(
                rr, "/graphcut/wound_prob_before", set(wound_before.keys()), wb_colors)

            # After graph-cut
            if self.semantic_wound_prob:
                wa_colors = self._create_confidence_colormap(self.semantic_wound_prob)
                self._log_faces_mesh_rr_with_colors(
                    rr, "/graphcut/wound_prob_after", set(self.semantic_wound_prob.keys()), wa_colors)

            if self.semantic_tissue_prob:
                ta_colors = self._create_confidence_colormap(self.semantic_tissue_prob)
                self._log_faces_mesh_rr_with_colors(
                    rr, "/graphcut/tissue_prob_after", set(self.semantic_tissue_prob.keys()), ta_colors)

            # Label changes
            label_changes = 0
            for fi in self.semantic_face_indices:
                if fi in self.semantic_posteriors_before_mrf and fi in self.semantic_labels:
                    before = int(np.argmax(self.semantic_posteriors_before_mrf[fi]))
                    after = 0 if self.semantic_labels[fi] == "tissue" else 1
                    if before != after:
                        label_changes += 1

            gc_stats = f"Graph-Cut Regularisation\n{'=' * 40}\n\n"
            gc_stats += f"Lambda (smoothness): {getattr(self, '_gc_lambda', 'N/A')}\n"
            gc_stats += f"σ_color: {self.graph_cut_sigma_color}\n"
            gc_stats += f"σ_normal: {self.graph_cut_sigma_normal}\n"
            gc_stats += f"Edge count: {len(self.graph_cut_edge_weights)}\n"
            if self.graph_cut_energy is not None:
                gc_stats += f"Min-cut flow: {self.graph_cut_energy:.2f}\n"
            gc_stats += f"\nFaces relabeled: {label_changes}\n"
            gc_stats += (f"Relabel %: "
                         f"{100 * label_changes / max(1, len(self.semantic_face_indices)):.2f}%\n")
            rr.log("/graphcut/stats", rr.TextDocument(gc_stats), static=True)

        # ── Confidence maps ──────────────────────────────────────
        if self.semantic_confidence:
            sc_colors = self._create_confidence_colormap(self.semantic_confidence)
            self._log_faces_mesh_rr_with_colors(
                rr, "/confidence/semantic_confidence",
                set(self.semantic_confidence.keys()), sc_colors)

            if self.semantic_wound_prob:
                wp_colors = self._create_confidence_colormap(self.semantic_wound_prob)
                self._log_faces_mesh_rr_with_colors(
                    rr, "/confidence/wound_probability",
                    set(self.semantic_wound_prob.keys()), wp_colors)

            if self.semantic_tissue_prob:
                tp_colors = self._create_confidence_colormap(self.semantic_tissue_prob)
                self._log_faces_mesh_rr_with_colors(
                    rr, "/confidence/tissue_probability",
                    set(self.semantic_tissue_prob.keys()), tp_colors)

            # Label overlay
            tissue_set = {f for f, l in self.semantic_labels.items() if l == "tissue"}
            wound_set = {f for f, l in self.semantic_labels.items() if l == "wound"}
            self._log_faces_mesh_rr(rr, "/confidence/labels/tissue", tissue_set, [0, 150, 255])
            self._log_faces_mesh_rr(rr, "/confidence/labels/wound", wound_set, [255, 0, 0])

            if self.wound_edge_face_indices:
                self._log_faces_mesh_rr(
                    rr, "/confidence/wound_edge_boundary",
                    self.wound_edge_face_indices, [255, 200, 0])

        # ── Statistics text ──────────────────────────────────────
        stats_txt = f"""
Mesh Statistics (Logistic Regression Pipeline)
{'=' * 50}
Total Vertices: {len(self.vertices):,}
Total Faces:    {len(self.faces):,}

Wound Faces (mask projection): {len(self.wound_face_indices):,}
Tissue Edge Faces:              {len(self.tissue_edge_faces):,}

Spectators:       {len(self.spectators)}
Masks Loaded:     {sum(1 for s in self.spectators.values() if s.mask_loaded)}
Extruded Loaded:  {sum(1 for s in self.spectators.values() if s.extruded_mask_loaded)}
"""
        if self.semantic_labels:
            n_tissue = sum(1 for v in self.semantic_labels.values() if v == "tissue")
            n_wound = sum(1 for v in self.semantic_labels.values() if v == "wound")
            mean_conf = np.mean(list(self.semantic_confidence.values()))
            stats_txt += f"""
Classification Results:
  Classified faces: {len(self.semantic_labels):,}
  Tissue: {n_tissue:,}
  Wound:  {n_wound:,}
  Mean confidence: {mean_conf:.3f}
  Wound-edge boundary faces: {len(self.wound_edge_face_indices)}
"""
        if self.logreg_feature_importance:
            stats_txt += "\nFeature Importances (|w|):\n"
            sorted_imp = sorted(self.logreg_feature_importance.items(), key=lambda x: -x[1])
            for name, imp in sorted_imp:
                stats_txt += f"  {name:25s} {imp:.4f}\n"

        rr.log("/stats", rr.TextDocument(stats_txt), static=True)
        print(f"Rerun visualisation started: {session_name}")

    # ==================================================================
    # Statistics (extended)
    # ==================================================================
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics including logistic regression diagnostics."""
        stats = super().get_statistics()

        if self.logreg_model is not None:
            stats["logreg_C"] = float(self.logreg_model.C)
            stats["logreg_intercept"] = self.logreg_intercept
            stats["logreg_n_features"] = len(self.semantic_feature_names)
            stats["logreg_feature_importance"] = dict(self.logreg_feature_importance)

        if self.graph_cut_sigma_color is not None:
            stats["graph_cut"] = {
                "sigma_color": self.graph_cut_sigma_color,
                "sigma_normal": self.graph_cut_sigma_normal,
                "n_edges": len(self.graph_cut_edge_weights),
                "min_cut_flow": self.graph_cut_energy,
            }

        return stats


# =========================================================================
# Main
# =========================================================================
def main():
    """
    Example usage of mesh-based wound detection with Logistic Regression
    + Graph-Cut classification.
    """
    # Example paths — adjust to your data
    mesh_path = Path(
        r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse\dense\poisson.ply")
    model_path = Path(
        r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse\dense")
    mask_dir = Path(
        r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse\dense\masks_wound_only_undistorted")
    extruded_mask_dir = Path(
        r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse\dense\masks_extruded_wound_only_undistorted")
    output_dir = Path(
        r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse\dense")

    try:
        # ---- Initialise ----
        wound_model = WoundModel3DMeshLogReg()

        # ---- Load mesh ----
        print("=== Loading Mesh ===")
        wound_model.load_mesh(mesh_path, compute_normals=True)

        # ---- Load COLMAP cameras ----
        print("\n=== Loading COLMAP Cameras ===")
        wound_model.load_colmap_cameras(model_path)

        # ---- Load masks ----
        print("\n=== Loading Wound Masks ===")
        wound_model.load_masks(mask_dir)

        print("\n=== Loading Extruded Masks ===")
        wound_model.load_extruded_masks(extruded_mask_dir)

        # ---- Process wound segmentation (raycasting) ----
        print("\n=== Processing Wound Segmentation ===")
        wound_model.process_wound_segmentation()

        # ---- Logistic Regression + Graph-Cut classification ----
        print("\n=== Semantic Classification (LogReg + Graph-Cut) ===")
        wound_model.run_semantic_classification(
            # Label thresholds for training
            wound_ratio_thresh=0.3,
            tissue_ratio_thresh=0.5,
            # Feature flags
            min_total_views=1,
            tissue_visibility_percentile=70.0,
            ring_sizes=[1, 3],
            use_visibility_ratios=True,
            use_height_deviation=True,
            use_normal_deviation=True,
            # Logistic regression
            logreg_C=1.0,
            logreg_max_iter=1000,
            # Graph-cut
            lambda_smooth=1.0,
            sigma_color=None,   # adaptive (median heuristic)
            sigma_normal=None,  # adaptive
            confidence_method="entropy",
        )

        # ---- Visibility histograms ----
        print("\n=== Plotting Visibility Histograms ===")
        wound_model.plot_visibility_histograms(
            output_dir=output_dir, show_plot=False)

        # ---- Statistics ----
        print("\n=== Statistics ===")
        stats = wound_model.get_statistics()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")

        # ---- Visualisation ----
        print("\n=== Launching Rerun Visualisation ===")
        wound_model.visualize_with_rerun(
            "WoundLogReg_Ferse", min_tissue_edge_visibility=10)

        print("\n=== Processing Complete ===")

    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
