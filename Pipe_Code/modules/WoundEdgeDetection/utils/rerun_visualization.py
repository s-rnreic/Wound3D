import numpy as np
from typing import Any, Optional, Set, Tuple


def _create_confidence_colormap_on_mesh(wound_model: Any, cmap_name: str = "viridis") -> Optional[np.ndarray]:
    """Build per-vertex confidence colors from semantic confidence values."""
    if wound_model.vertices is None or wound_model.faces is None or len(wound_model.vertices) == 0:
        print("Confidence colormap skipped: mesh geometry not loaded")
        return None

    confidence_face = np.zeros(len(wound_model.faces), dtype=np.float64)
    available = np.zeros(len(wound_model.faces), dtype=bool)

    for face_idx in range(len(wound_model.faces)):
        if face_idx >= len(wound_model.face_objects):
            continue
        face_obj = wound_model.face_objects[face_idx]
        if face_obj.confidence is not None:
            confidence_face[face_idx] = float(face_obj.confidence)
            available[face_idx] = True
            continue
        if face_obj.wound_prob is not None:
            confidence_face[face_idx] = float(face_obj.wound_prob)
            available[face_idx] = True

    if not np.any(available):
        print("Confidence colormap skipped: no semantic confidence available")
        return None

    vertex_sum = np.zeros(len(wound_model.vertices), dtype=np.float64)
    vertex_count = np.zeros(len(wound_model.vertices), dtype=np.float64)
    for face_idx, tri in enumerate(wound_model.faces):
        if not available[face_idx]:
            continue
        val = confidence_face[face_idx]
        vertex_sum[tri] += val
        vertex_count[tri] += 1.0

    vertex_conf = np.zeros(len(wound_model.vertices), dtype=np.float64)
    valid_vertex = vertex_count > 0
    vertex_conf[valid_vertex] = vertex_sum[valid_vertex] / vertex_count[valid_vertex]

    conf_min = float(np.min(vertex_conf[valid_vertex])) if np.any(valid_vertex) else 0.0
    conf_max = float(np.max(vertex_conf[valid_vertex])) if np.any(valid_vertex) else 1.0
    if conf_max - conf_min < 1e-9:
        normalized = np.zeros_like(vertex_conf)
    else:
        normalized = (vertex_conf - conf_min) / (conf_max - conf_min)

    try:
        import matplotlib.cm as cm

        cmap = cm.get_cmap(cmap_name)
        vertex_colors = cmap(np.clip(normalized, 0.0, 1.0))[:, :3]
    except Exception as e:
        print(f"Warning: matplotlib colormap failed ({e}), using grayscale")
        vertex_colors = np.repeat(normalized[:, None], 3, axis=1)

    print(
        f"Confidence colormap ready: vertices={len(vertex_colors)}, "
        f"range=[{conf_min:.4f}, {conf_max:.4f}]"
    )
    return vertex_colors


def _compute_face_wound_tissue_posteriors(wound_model: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-face wound/tissue posterior arrays with robust fallbacks."""
    if wound_model.faces is None:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )

    n_faces = len(wound_model.faces)
    face_ids = np.arange(n_faces, dtype=np.int64)
    wound_probs = np.zeros(n_faces, dtype=np.float64)
    tissue_probs = np.zeros(n_faces, dtype=np.float64)

    wound_idx = None
    tissue_idx = None
    if wound_model.semantic_class_names:
        if "wound" in wound_model.semantic_class_names:
            wound_idx = wound_model.semantic_class_names.index("wound")
        if "tissue" in wound_model.semantic_class_names:
            tissue_idx = wound_model.semantic_class_names.index("tissue")

    if wound_idx is None and len(wound_model.semantic_class_names) >= 2:
        wound_idx = 1

    for face_idx in range(n_faces):
        wound_val = 0.0
        tissue_val = 0.0

        probs = wound_model.semantic_posteriors.get(face_idx) if wound_model.semantic_posteriors else None
        if probs is not None and len(probs) > 0:
            if wound_idx is not None and wound_idx < len(probs):
                wound_val = float(probs[wound_idx])
            elif len(probs) >= 2:
                wound_val = float(probs[1])

            if tissue_idx is not None and tissue_idx < len(probs):
                tissue_val = float(probs[tissue_idx])
            else:
                tissue_val = 1.0 - wound_val
        elif face_idx < len(wound_model.face_objects) and wound_model.face_objects[face_idx].wound_prob is not None:
            wound_val = float(wound_model.face_objects[face_idx].wound_prob)
            tissue_val = 1.0 - wound_val

        wound_probs[face_idx] = np.clip(wound_val, 0.0, 1.0)
        tissue_probs[face_idx] = np.clip(tissue_val, 0.0, 1.0)

    return face_ids, wound_probs, tissue_probs


def _compute_face_wound_periwound_posteriors(wound_model: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-face wound/periwound posterior arrays with robust fallbacks."""
    if wound_model.faces is None:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )

    n_faces = len(wound_model.faces)
    face_ids = np.arange(n_faces, dtype=np.int64)
    wound_probs = np.zeros(n_faces, dtype=np.float64)
    periwound_probs = np.zeros(n_faces, dtype=np.float64)

    periwound_idx = None
    wound_idx = None
    if wound_model.semantic_class_names:
        if "periwound" in wound_model.semantic_class_names:
            periwound_idx = wound_model.semantic_class_names.index("periwound")
        if "wound" in wound_model.semantic_class_names:
            wound_idx = wound_model.semantic_class_names.index("wound")

    if periwound_idx is None and len(wound_model.semantic_class_names) >= 1:
        periwound_idx = 0
    if wound_idx is None and len(wound_model.semantic_class_names) >= 2:
        wound_idx = 1

    for face_idx in range(n_faces):
        wound_val = 0.0
        periwound_val = 0.0

        probs = wound_model.semantic_posteriors.get(face_idx) if wound_model.semantic_posteriors else None
        if probs is not None and len(probs) > 0:
            if wound_idx is not None and wound_idx < len(probs):
                wound_val = float(probs[wound_idx])
            elif len(probs) >= 2:
                wound_val = float(probs[1])

            if periwound_idx is not None and periwound_idx < len(probs):
                periwound_val = float(probs[periwound_idx])
            elif len(probs) >= 1:
                periwound_val = float(probs[0])
        elif face_idx < len(wound_model.face_objects):
            face_obj = wound_model.face_objects[face_idx]
            if face_obj.wound_prob is not None:
                wound_val = float(face_obj.wound_prob)
            if face_obj.periwound_prob is not None:
                periwound_val = float(face_obj.periwound_prob)

        wound_probs[face_idx] = np.clip(wound_val, 0.0, 1.0)
        periwound_probs[face_idx] = np.clip(periwound_val, 0.0, 1.0)

    return face_ids, wound_probs, periwound_probs


def _face_values_to_vertex_colors(wound_model: Any, face_values: np.ndarray, cmap_name: str = "viridis") -> Optional[np.ndarray]:
    """Convert per-face scalar values in [0,1] to per-vertex RGB colors in [0,1]."""
    if (
        wound_model.vertices is None
        or wound_model.faces is None
        or len(wound_model.vertices) == 0
        or len(wound_model.faces) == 0
    ):
        return None

    vertex_sum = np.zeros(len(wound_model.vertices), dtype=np.float64)
    vertex_count = np.zeros(len(wound_model.vertices), dtype=np.float64)
    for face_idx, tri in enumerate(wound_model.faces):
        val = float(face_values[face_idx]) if face_idx < len(face_values) else 0.0
        vertex_sum[tri] += val
        vertex_count[tri] += 1.0

    vertex_vals = np.zeros(len(wound_model.vertices), dtype=np.float64)
    valid = vertex_count > 0
    vertex_vals[valid] = vertex_sum[valid] / vertex_count[valid]

    try:
        import matplotlib.cm as cm

        cmap = cm.get_cmap(cmap_name)
        return cmap(np.clip(vertex_vals, 0.0, 1.0))[:, :3]
    except Exception:
        return np.repeat(np.clip(vertex_vals, 0.0, 1.0)[:, None], 3, axis=1)


def debug_visualize_gmm_samples_and_confidence(
    wound_model: Any,
    show_gmm_samples: bool = True,
    show_confidence_mesh: bool = True,
    colormap: str = "viridis",
    point_size: float = 8.0,
    session_name: str = "WoundMesh_DebugViz",
    spawn: bool = True,
) -> None:
    """Log selected GMM samples and confidence map to a Rerun viewer session."""
    try:
        import rerun as rr
    except Exception:
        print("Rerun not installed. Install with: pip install rerun-sdk")
        return

    try:
        rr.init(session_name, spawn=spawn)
    except Exception as e:
        print(f"Could not launch Rerun viewer directly ({e}); continuing with default init")
        rr.init(session_name)

    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rr.log(
        "/debug/info",
        rr.TextDocument(
            f"GMM sample + confidence visualization\n"
            f"show_gmm_samples={show_gmm_samples}\n"
            f"show_confidence_mesh={show_confidence_mesh}\n"
            f"colormap={colormap}"
        ),
        static=True,
    )

    base_radius = max(5e-4, float(point_size) * 3.5e-4)

    if show_gmm_samples:
        try:
            centroids = wound_model._get_face_centroids()
            base_face_indices = wound_model.debug_gmm_face_indices or list(range(len(centroids)))
            base_pts = centroids[base_face_indices] if len(base_face_indices) > 0 else np.zeros((0, 3))

            periwound_ids = [
                i for i in wound_model.debug_gmm_sample_indices.get("periwound", []) if 0 <= i < len(centroids)
            ]
            wound_ids = [
                i for i in wound_model.debug_gmm_sample_indices.get("wound", []) if 0 <= i < len(centroids)
            ]
            periwound_pts = centroids[periwound_ids] if periwound_ids else np.zeros((0, 3))
            wound_pts = centroids[wound_ids] if wound_ids else np.zeros((0, 3))

            if len(base_pts) > 0:
                rr.log(
                    "/debug/gmm_samples/all_faces",
                    rr.Points3D(base_pts, colors=np.full((len(base_pts), 3), 170, dtype=np.uint8), radii=base_radius * 0.45),
                    static=True,
                )
            if len(periwound_pts) > 0:
                rr.log(
                    "/debug/gmm_samples/periwound",
                    rr.Points3D(periwound_pts, colors=np.tile(np.array([[40, 80, 230]], dtype=np.uint8), (len(periwound_pts), 1)), radii=base_radius),
                    static=True,
                )
            if len(wound_pts) > 0:
                rr.log(
                    "/debug/gmm_samples/wound",
                    rr.Points3D(wound_pts, colors=np.tile(np.array([[220, 40, 40]], dtype=np.uint8), (len(wound_pts), 1)), radii=base_radius),
                    static=True,
                )

            rr.log(
                "/debug/gmm_samples/counts",
                rr.TextDocument(
                    f"periwound={len(periwound_ids)}\n"
                    f"wound={len(wound_ids)}\n"
                    f"point_size={point_size}"
                ),
                static=True,
            )
        except Exception as e:
            print(f"GMM sample visualization failed: {e}")

    if show_confidence_mesh:
        try:
            if wound_model.vertices is None or wound_model.faces is None or len(wound_model.vertices) == 0:
                print("Confidence mesh visualization skipped: mesh not loaded")
                return

            vertex_colors = _create_confidence_colormap_on_mesh(wound_model, cmap_name=colormap)
            if vertex_colors is None:
                return

            rr.log(
                "/debug/confidence/mesh",
                rr.Mesh3D(
                    vertex_positions=wound_model.vertices,
                    triangle_indices=wound_model.faces,
                    vertex_colors=np.clip(vertex_colors * 255.0, 0.0, 255.0).astype(np.uint8),
                ),
                static=True,
            )

            face_ids, wound_probs, tissue_probs = _compute_face_wound_tissue_posteriors(wound_model)
            centroids = wound_model._get_face_centroids()
            if len(face_ids) > 0 and len(centroids) == len(face_ids):
                try:
                    import matplotlib.cm as cm

                    cmap = cm.get_cmap(colormap)
                    wound_face_colors = (cmap(np.clip(wound_probs, 0.0, 1.0))[:, :3] * 255.0).astype(np.uint8)
                    tissue_face_colors = (cmap(np.clip(tissue_probs, 0.0, 1.0))[:, :3] * 255.0).astype(np.uint8)
                except Exception:
                    wound_face_colors = np.repeat((np.clip(wound_probs, 0.0, 1.0) * 255.0)[:, None], 3, axis=1).astype(np.uint8)
                    tissue_face_colors = np.repeat((np.clip(tissue_probs, 0.0, 1.0) * 255.0)[:, None], 3, axis=1).astype(np.uint8)

                rr.log(
                    "/debug/posterior/wound_faces",
                    rr.Points3D(centroids, colors=wound_face_colors, radii=base_radius * 0.5),
                    static=True,
                )
                rr.log(
                    "/debug/posterior/tissue_faces",
                    rr.Points3D(centroids, colors=tissue_face_colors, radii=base_radius * 0.5),
                    static=True,
                )
                rr.log("/debug/posterior/wound_values", rr.Scalars(wound_probs), static=True)
                rr.log("/debug/posterior/tissue_values", rr.Scalars(tissue_probs), static=True)

                wound_vertex_colors = _face_values_to_vertex_colors(wound_model, wound_probs, cmap_name=colormap)
                tissue_vertex_colors = _face_values_to_vertex_colors(wound_model, tissue_probs, cmap_name=colormap)

                if wound_vertex_colors is not None:
                    rr.log(
                        "/debug/posterior/wound_mesh",
                        rr.Mesh3D(
                            vertex_positions=wound_model.vertices,
                            triangle_indices=wound_model.faces,
                            vertex_colors=np.clip(wound_vertex_colors * 255.0, 0.0, 255.0).astype(np.uint8),
                        ),
                        static=True,
                    )
                if tissue_vertex_colors is not None:
                    rr.log(
                        "/debug/posterior/tissue_mesh",
                        rr.Mesh3D(
                            vertex_positions=wound_model.vertices,
                            triangle_indices=wound_model.faces,
                            vertex_colors=np.clip(tissue_vertex_colors * 255.0, 0.0, 255.0).astype(np.uint8),
                        ),
                        static=True,
                    )
        except Exception as e:
            print(f"Confidence mesh visualization failed: {e}")


def debug_visualize_get_wound_mesh_process_rerun(
    wound_model: Any,
    session_name: str = "WoundMesh_GetWoundMesh_Process",
    colormap: str = "turbo",
    point_size: float = 6.0,
    spawn: bool = True,
) -> None:
    """Visualize get_wound_mesh process stages and posteriors in Rerun."""
    try:
        import rerun as rr
    except Exception:
        print("Rerun not installed. Install with: pip install rerun-sdk")
        return

    if wound_model.vertices is None or wound_model.faces is None or len(wound_model.faces) == 0:
        print("get_wound_mesh process visualization skipped: mesh geometry not loaded")
        return

    try:
        rr.init(session_name, spawn=spawn)
    except Exception as e:
        print(f"Could not launch Rerun viewer directly ({e}); continuing with default init")
        rr.init(session_name)

    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    if wound_model.vertex_colors is not None and len(wound_model.vertex_colors) == len(wound_model.vertices):
        original_vertex_colors = np.clip(wound_model.vertex_colors * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        original_vertex_colors = np.full((len(wound_model.vertices), 3), 150, dtype=np.uint8)

    rr.log(
        "/debug/get_wound_mesh/original_mesh",
        rr.Mesh3D(
            vertex_positions=wound_model.vertices,
            triangle_indices=wound_model.faces,
            vertex_colors=original_vertex_colors,
        ),
        static=True,
    )

    base_radius = max(5e-4, float(point_size) * 3.5e-4)
    centroids = wound_model._get_face_centroids()

    vertex_colors = _create_confidence_colormap_on_mesh(wound_model, cmap_name=colormap)
    if vertex_colors is not None:
        rr.log(
            "/debug/get_wound_mesh/confidence_mesh",
            rr.Mesh3D(
                vertex_positions=wound_model.vertices,
                triangle_indices=wound_model.faces,
                vertex_colors=np.clip(vertex_colors * 255.0, 0.0, 255.0).astype(np.uint8),
            ),
            static=True,
        )

    face_ids, wound_probs, periwound_probs = _compute_face_wound_periwound_posteriors(wound_model)
    if len(face_ids) > 0 and len(centroids) == len(face_ids):
        try:
            import matplotlib.cm as cm

            cmap = cm.get_cmap(colormap)
            wound_face_colors = (cmap(np.clip(wound_probs, 0.0, 1.0))[:, :3] * 255.0).astype(np.uint8)
            periwound_face_colors = (cmap(np.clip(periwound_probs, 0.0, 1.0))[:, :3] * 255.0).astype(np.uint8)
        except Exception:
            wound_face_colors = np.repeat((np.clip(wound_probs, 0.0, 1.0) * 255.0)[:, None], 3, axis=1).astype(np.uint8)
            periwound_face_colors = np.repeat((np.clip(periwound_probs, 0.0, 1.0) * 255.0)[:, None], 3, axis=1).astype(np.uint8)

        rr.log(
            "/debug/get_wound_mesh/posterior/wound_faces",
            rr.Points3D(centroids, colors=wound_face_colors, radii=base_radius * 0.5),
            static=True,
        )
        rr.log(
            "/debug/get_wound_mesh/posterior/periwound_faces",
            rr.Points3D(centroids, colors=periwound_face_colors, radii=base_radius * 0.5),
            static=True,
        )

    def _log_stage_mesh(path: str, face_set: Set[int], color: np.ndarray) -> None:
        if not face_set:
            rr.log(path + "/count", rr.TextDocument("count=0"), static=True)
            return

        idx = np.array(sorted(face_set), dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < len(wound_model.faces))]
        if idx.size == 0:
            rr.log(path + "/count", rr.TextDocument("count=0(valid)"), static=True)
            return

        stage_vertex_colors = np.full((len(wound_model.vertices), 3), 90, dtype=np.uint8)
        stage_vertex_ids = np.unique(wound_model.faces[idx].reshape(-1))
        stage_vertex_colors[stage_vertex_ids] = color.astype(np.uint8)

        rr.log(
            path + "/mesh",
            rr.Mesh3D(
                vertex_positions=wound_model.vertices,
                triangle_indices=wound_model.faces,
                vertex_colors=stage_vertex_colors,
            ),
            static=True,
        )

    _log_stage_mesh(
        "/debug/get_wound_mesh/stages/initial_wound_faces",
        set(getattr(wound_model, "initial_wound_faces_stage", set())),
        np.array([255, 200, 0], dtype=np.uint8),
    )
    _log_stage_mesh(
        "/debug/get_wound_mesh/stages/selected_after_select_wound_faces",
        set(getattr(wound_model, "probability_filtered_wound_faces_stage", set())),
        np.array([230, 70, 30], dtype=np.uint8),
    )
    _log_stage_mesh(
        "/debug/get_wound_mesh/stages/after_refine_wound_faces",
        set(getattr(wound_model, "post_closing_wound_faces_stage", set())),
        np.array([20, 220, 80], dtype=np.uint8),
    )
    _log_stage_mesh(
        "/debug/get_wound_mesh/stages/final_current_wound_faces",
        set(getattr(wound_model, "current_wound_faces", set())),
        np.array([250, 30, 30], dtype=np.uint8),
    )

    final_wound_face_set = set(getattr(wound_model, "current_wound_faces", set()))
    if final_wound_face_set:
        wound_vertices, wound_faces, wound_colors = wound_model.extract_submesh_from_faces(final_wound_face_set)
        if len(wound_vertices) > 0 and len(wound_faces) > 0:
            if wound_colors is not None and len(wound_colors) == len(wound_vertices):
                wound_vertex_colors = np.clip(wound_colors * 255.0, 0.0, 255.0).astype(np.uint8)
            else:
                wound_vertex_colors = np.tile(np.array([[250, 30, 30]], dtype=np.uint8), (len(wound_vertices), 1))

            rr.log(
                "/debug/get_wound_mesh/final_wound_mesh",
                rr.Mesh3D(
                    vertex_positions=wound_vertices,
                    triangle_indices=wound_faces,
                    vertex_colors=wound_vertex_colors,
                ),
                static=True,
            )
    else:
        rr.log(
            "/debug/get_wound_mesh/final_wound_mesh_info",
            rr.TextDocument("final_wound_mesh unavailable: current_wound_faces is empty"),
            static=True,
        )

    print("Rerun get_wound_mesh process visualization logged")
