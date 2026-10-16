from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import rerun as rr
import rerun.blueprint as rrb
import numpy as np





def _safe_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    finite_mask = np.isfinite(array)
    if not np.any(finite_mask):
        return np.zeros_like(array, dtype=np.float64)
    finite_values = array[finite_mask]
    min_value = float(np.min(finite_values))
    max_value = float(np.max(finite_values))
    if abs(max_value - min_value) < 1e-12:
        return np.zeros_like(array, dtype=np.float64)
    result = np.zeros_like(array, dtype=np.float64)
    result[finite_mask] = (array[finite_mask] - min_value) / (max_value - min_value)
    return np.clip(result, 0.0, 1.0)


def _viridis_like_colormap(values: np.ndarray) -> np.ndarray:
    """Map scalar values in [0, 1] to viridis RGB colors using matplotlib."""
    value_array = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    
    try:
        from matplotlib import cm
        cmap = cm.get_cmap('viridis')
        # Apply colormap and convert to uint8 RGB
        rgba_colors = cmap(value_array)  # Returns (N, 4) with RGBA in [0, 1]
        rgb_colors = (rgba_colors[:, :3] * 255.0).astype(np.uint8)
        return rgb_colors
    except Exception:
        # Fallback: use hardcoded viridis stops
        stops = np.array([
            [0.267, 0.005, 0.329],
            [0.283, 0.141, 0.458],
            [0.254, 0.265, 0.530],
            [0.207, 0.372, 0.553],
            [0.164, 0.471, 0.558],
            [0.128, 0.567, 0.551],
            [0.135, 0.659, 0.518],
            [0.267, 0.749, 0.441],
            [0.478, 0.821, 0.318],
            [0.741, 0.873, 0.150],
        ], dtype=np.float64)
        positions = np.linspace(0.0, 1.0, len(stops))
        colors = np.zeros((len(value_array), 3), dtype=np.uint8)
        for channel in range(3):
            colors[:, channel] = np.clip(
                np.interp(value_array, positions, stops[:, channel]) * 255.0,
                0.0,
                255.0,
            ).astype(np.uint8)
        return colors




def _load_rgb_image_from_common_locations(base_dir: Optional[Path], image_name: str) -> Optional[np.ndarray]:
    """Load a matching RGB image from common experiment folders if available."""
    candidate_paths: List[Path] = []
    if base_dir is not None:
        base_dir = Path(base_dir)
        candidate_paths.extend([
            base_dir / image_name,
            base_dir.parent / "raw_images" / image_name,
            base_dir.parent / "sharp_images" / image_name,
            base_dir.parent / "images" / image_name,
            base_dir / "raw_images" / image_name,
            base_dir / "sharp_images" / image_name,
        ])

    for path in candidate_paths:
        if path.exists():
            try:
                import cv2
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except Exception:
                continue
    return None


def _rotmat_to_quat_xyzw(rotation_matrix: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion in xyzw format."""
    R = np.asarray(rotation_matrix, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0:
        S = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return (float(x), float(y), float(z), float(w))







def _face_values_to_mesh_colormap(face_values: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Convert per-face scalar values to per-vertex colors using viridis colormap."""
    normalized_values = _safe_normalize(face_values)
    face_colors = _viridis_like_colormap(normalized_values)
    n_vertices = int(np.max(faces)) + 1 if len(faces) > 0 else 0
    vertex_color_sum = np.zeros((n_vertices, 3), dtype=np.float64)
    vertex_count = np.zeros(n_vertices, dtype=np.int32)
    for face_idx, face in enumerate(faces):
        color = face_colors[face_idx].astype(np.float64)
        for v in face:
            vertex_color_sum[v] += color
            vertex_count[v] += 1
    vertex_colors = np.zeros((n_vertices, 3), dtype=np.uint8)
    for v in range(n_vertices):
        if vertex_count[v] > 0:
            vertex_colors[v] = np.clip(vertex_color_sum[v] / vertex_count[v], 0, 255).astype(np.uint8)
    return vertex_colors


def visualize_wound_mesh_and_cameras(
    wound_model: Any,
    session_name: str = "WoundModel3D_Mesh",
    spawn: bool = True,
    image_root: Optional[Path] = None,
    max_spectators: Optional[int] = None,
) -> None:
    """Log the base wound mesh and camera instances to a dedicated Rerun session."""
   

    try:
        rr.init(session_name, spawn=spawn)
    except Exception as e:
        print(f"Could not launch Rerun viewer directly ({e}); continuing with default init")
        rr.init(session_name)

    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)


    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/world",
                name="Wound Mesh",
                contents=[
                    "+ cameras/**",
                    "+ $origin/**",
                ],
            ),
        ),
    )
    rr.send_blueprint(blueprint)


    vertices = np.asarray(wound_model.vertices, dtype=np.float64)
    faces = np.asarray(wound_model.faces, dtype=np.int32)
    vertex_colors = getattr(wound_model, "vertex_colors", None)
    base_colors = (
        np.clip(np.asarray(vertex_colors, dtype=np.float64), 0.0, 1.0)
        if vertex_colors is not None and len(vertex_colors) == len(vertices)
        else None
    )
    mesh_colors = (
        (np.clip(base_colors * 255.0, 0.0, 255.0).astype(np.uint8))
        if base_colors is not None
        else np.full((len(vertices), 3), 140, dtype=np.uint8)
    )

    rr.log(
        "/world/base_mesh",
        rr.Mesh3D(vertex_positions=vertices, triangle_indices=faces, vertex_colors=mesh_colors),
        static=True,
    )

    # Log camera poses and images (optional)
    spectators = list(getattr(wound_model, "spectators", {}).values())
    if max_spectators is not None:
        spectators = spectators[:max_spectators]

    for step_index, spectator in enumerate(spectators):
        if spectator is None or not getattr(spectator, "camera_data_loaded", False):
            continue

        fx = float(spectator.intrinsics["fx"])
        fy = float(spectator.intrinsics["fy"])
        cx = float(spectator.intrinsics["cx"])
        cy = float(spectator.intrinsics["cy"])

        R_cam_to_world = spectator.rotation_matrix.T
        t_cam_to_world = -R_cam_to_world @ spectator.translation_vector.reshape(3, 1)

        camera_path = f"/cameras/camera_{step_index:03d}"

        try:
            quat_xyzw = _rotmat_to_quat_xyzw(R_cam_to_world)
            rr.log(
                camera_path,
                rr.Transform3D(
                    translation=t_cam_to_world.reshape(3),
                    rotation=rr.Quaternion(xyzw=quat_xyzw),
                ),
                static=True,
            )
        except Exception:
            pass

        rr.log(
            f"{camera_path}",
            rr.Pinhole(
                resolution=[int(spectator.image_width), int(spectator.image_height)],
                focal_length=[fx, fy],
                principal_point=[cx, cy],
            ),
        )

        colmap_image = getattr(wound_model, "images", {}).get(spectator.image_id)
        image_name = getattr(colmap_image, "name", spectator.image_name)
        image_data = getattr(colmap_image, "data", None)
        if image_data is None:
            image_data = _load_rgb_image_from_common_locations(image_root, image_name)


        if image_data is not None:
            downsized_image = image_data
         
            import cv2

            target_width = max(1, int(round(int(spectator.image_width) * 0.1)))
            target_height = max(1, int(round(int(spectator.image_height) * 0.1)))
            downsized_image = cv2.resize(
                np.asarray(image_data),
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )

            rr.log(
                f"{camera_path}/image",
                rr.Image(image=downsized_image),
            )
        else:
            print(f"Spectator {step_index} has no image data to log")
            

def visualize_r_values_and_posteriors(
    wound_model: Any,
    session_name: str = "WoundModel3D_Posteriors",
    spawn: bool = True,
) -> None:
    """Log r-values, posterior maps, and log-ratio colormaps (including Tau variants)."""
    try:
        import rerun as rr
    except Exception:
        print("Rerun not installed. Install with: pip install rerun-sdk")
        return

    vertices = np.asarray(wound_model.vertices, dtype=np.float64)
    faces = np.asarray(wound_model.faces, dtype=np.int32)

    
    rr.init(session_name, spawn=spawn)
    blueprint = rrb.Blueprint(
        rrb.Tabs(
            rrb.Spatial3DView(
                origin="/",
                name="r-values",
                contents=[
                    "+ /analysis/r_values/**",
                ],
            ),
            rrb.Spatial3DView(
                origin="/",
                name="propabilities and log-ratios",
                contents=[
                    "+ /analysis/posteriors/**",
                    "+ /analysis/log_ratio_values/**",
                ],
            )
        ),
    )
    rr.send_blueprint(blueprint)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    face_count = len(getattr(wound_model, "face_objects", []))
    r_wound_values = np.fromiter((float(face_obj.r_wound or 0.0) for face_obj in wound_model.face_objects), dtype=np.float64, count=face_count)
    r_periwound_values = np.fromiter((float(face_obj.r_periwound or 0.0) for face_obj in wound_model.face_objects), dtype=np.float64, count=face_count)

    # Log r-value colormaps
    wound_colors = _face_values_to_mesh_colormap(r_wound_values, faces)
    rr.log("/analysis/r_values/wound", rr.Mesh3D(vertex_positions=vertices, triangle_indices=faces, vertex_colors=wound_colors), static=True)
    periwound_colors = _face_values_to_mesh_colormap(r_periwound_values, faces)
    rr.log("/analysis/r_values/periwound", rr.Mesh3D(vertex_positions=vertices, triangle_indices=faces, vertex_colors=periwound_colors), static=True)

    # Posteriors and log-ratio
    semantic_posteriors = getattr(wound_model, "semantic_posteriors", {}) or {}
    semantic_log_ratio_values = getattr(wound_model, "log_ratio_values", {}) or getattr(wound_model, "semantic_log_ratio_values", {}) or {}
    semantic_log_ratio_strength = getattr(wound_model, "log_ratio_strength_values", {}) or getattr(wound_model, "semantic_log_ratio_strength_values", {}) or {}

    # Build per-face arrays if necessary
    if isinstance(semantic_log_ratio_values, dict):
        log_ratio_values = np.fromiter((float(semantic_log_ratio_values.get(face_obj.face_index, 0.0)) for face_obj in wound_model.face_objects), dtype=np.float64, count=face_count)
    else:
        log_ratio_values = np.asarray(semantic_log_ratio_values, dtype=np.float64).reshape(-1)

    if isinstance(semantic_log_ratio_strength, dict):
        strength_vals = np.fromiter((float(semantic_log_ratio_strength.get(face_obj.face_index, abs(float(semantic_log_ratio_values.get(face_obj.face_index, 0.0)) if isinstance(semantic_log_ratio_values, dict) else 0.0))) for face_obj in wound_model.face_objects), dtype=np.float64, count=face_count)
    else:
        strength_vals = np.asarray(semantic_log_ratio_strength, dtype=np.float64).reshape(-1)

    if log_ratio_values.size == len(faces):
        normalized_log_ratio = _safe_normalize(log_ratio_values)
        face_colors = _viridis_like_colormap(normalized_log_ratio)
        flat_vertex_positions = vertices[faces].reshape(-1, 3)
        flat_triangle_indices = np.arange(len(faces) * 3, dtype=np.int32).reshape(-1, 3)
        flat_vertex_colors = np.repeat(face_colors, 3, axis=0)

        rr.log("/analysis/posteriors/log_ratio_values_mesh", rr.Mesh3D(vertex_positions=np.ascontiguousarray(flat_vertex_positions, dtype=np.float64), triangle_indices=np.ascontiguousarray(flat_triangle_indices, dtype=np.int32), vertex_colors=np.ascontiguousarray(flat_vertex_colors, dtype=np.uint8)), static=True)

        # Uncertainty band variants
        Tau_list = [2.0, 5.0, 10.0]
        strength_arr = strength_vals
        if strength_arr.size == len(faces):
            gray_color = np.array([200, 200, 200], dtype=np.uint8)
            for tau in Tau_list:
                face_colors_masked = np.tile(gray_color, (len(faces), 1))
                strong_mask = strength_arr >= float(tau)
                if np.any(strong_mask):
                    face_colors_masked[strong_mask] = face_colors[strong_mask]
                flat_vertex_colors_uncert = np.repeat(face_colors_masked, 3, axis=0).astype(np.uint8)
                rr.log(f"/analysis/posteriors/uncertainty_band/log_ratio_values_mesh_Tau{tau}", rr.Mesh3D(vertex_positions=np.ascontiguousarray(flat_vertex_positions, dtype=np.float64), triangle_indices=np.ascontiguousarray(flat_triangle_indices, dtype=np.int32), vertex_colors=np.ascontiguousarray(flat_vertex_colors_uncert, dtype=np.uint8)), static=True)


def visualize_final_wound_and_measurements(
    wound_model: Any,
    session_name: str = "WoundModel3D_Final",
    spawn: bool = True,
) -> None:
    """Log the final segmented wound mesh and simple measurement overlays."""
    


    rr.init(session_name, spawn=spawn)
 

    # Attempt to extract final wound submesh
    wound_faces = getattr(wound_model, "current_wound_faces", None)
    if not wound_faces:
        print("No final wound faces available to log")
        return

    if hasattr(wound_model, "extract_submesh_from_faces"):
        w_verts, w_faces, w_colors = wound_model.extract_submesh_from_faces(wound_faces)
    else:
        print("extract_submesh_from_faces not available on wound_model")
        return

    if w_verts is None or len(w_verts) == 0:
        print("Extracted wound submesh is empty")
        return

    rr.log(
        "/analysis/final_wound_mesh",
        rr.Mesh3D(vertex_positions=np.ascontiguousarray(w_verts, dtype=np.float64), triangle_indices=np.ascontiguousarray(w_faces.astype(np.int32), dtype=np.int32), vertex_colors=np.ascontiguousarray((w_colors if w_colors is not None else np.full((len(w_verts), 3), 214, dtype=np.uint8)), dtype=np.uint8)),
        static=True,
    )

    # Optionally log wound outerboundary vertices as points
    edge_vertices = getattr(wound_model, "wound_outerboundary_vertices", None)
    if edge_vertices:
        ev = np.asarray(list(edge_vertices), dtype=np.int64)
        if ev.size > 0:
            points = np.asarray(wound_model.vertices)[ev]
            rr.log("/analysis/final_wound_outerboundary_vertices", rr.Points3D(positions=np.ascontiguousarray(points, dtype=np.float64), colors=np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (len(points), 1))), static=True)


def visualize_wound_assessment_rerun(
    wound_model: Any,
    session_name: str = "WoundModel3D_Assessment",
    spawn: bool = True,
    image_root: Optional[Path] = None,
    max_spectators: Optional[int] = None,
) -> None:
    """Backward-compatible wrapper: launches the three focused visualizers as separate sessions."""
    # Mesh + cameras
    #visualize_wound_mesh_and_cameras(wound_model, session_name=f"{session_name}_mesh", spawn=spawn, image_root=image_root, max_spectators=max_spectators)
    # r-values and posteriors
    visualize_r_values_and_posteriors(wound_model, session_name=f"{session_name}_posteriors", spawn=spawn)
    # final wound and measurements
    visualize_final_wound_and_measurements(wound_model, session_name=f"{session_name}_final", spawn=spawn)


    
    

    print(f"Rerun wound-assessment visualization logged to session '{session_name}'")


def debug_visualize_gmm_samples_and_confidence(*args: Any, **kwargs: Any) -> None:
    """Backward-compatible alias for the previous import path."""
    visualize_wound_assessment_rerun(*args, **kwargs)
