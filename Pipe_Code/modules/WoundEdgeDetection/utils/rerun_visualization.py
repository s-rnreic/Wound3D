from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an image-like array to uint8 RGB for rerun logging."""
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)

    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            max_value = float(np.nanmax(array)) if array.size else 0.0
            if max_value <= 1.0:
                array = np.clip(array * 255.0, 0.0, 255.0)
            else:
                array = np.clip(array, 0.0, 255.0)
        else:
            array = np.clip(array, 0, 255)
        array = array.astype(np.uint8)

    if array.shape[-1] > 3:
        array = array[..., :3]
    return np.ascontiguousarray(array)


def _downscale_image_for_rerun(image: np.ndarray, scale: float = 0.1) -> np.ndarray:
    """Create a low-detail image while preserving original resolution."""
    array = _as_uint8_rgb(image)
    if array.size == 0:
        return array

    height, width = array.shape[:2]
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    if target_width == width and target_height == height:
        return array

    try:
        import cv2

        # Downsample to reduce detail, then upscale back to preserve display dimensions.
        downscaled = cv2.resize(array, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return cv2.resize(downscaled, (width, height), interpolation=cv2.INTER_NEAREST)
    except Exception:
        step_y = max(1, int(round(1.0 / scale)))
        step_x = max(1, int(round(1.0 / scale)))
        sampled = array[::step_y, ::step_x]
        return np.repeat(np.repeat(sampled, step_y, axis=0), step_x, axis=1)[:height, :width]


def _grayscale_to_rgb(mask: Optional[np.ndarray], color: Tuple[int, int, int]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    mask_array = np.asarray(mask)
    if mask_array.size == 0:
        return None
    mask_rgb = np.zeros((*mask_array.shape[:2], 3), dtype=np.uint8)
    active = mask_array > 0
    mask_rgb[active] = np.asarray(color, dtype=np.uint8)
    return mask_rgb


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
    """Map scalar values in [0, 1] to a simple viridis-like RGB palette."""
    value_array = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
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


def _face_values_to_vertex_colors(
    face_indices: Sequence[int],
    face_values: Dict[int, float],
    faces: np.ndarray,
    base_vertex_colors: Optional[np.ndarray],
    highlight_color: Tuple[int, int, int],
    background_color: Tuple[int, int, int] = (40, 40, 40),
) -> np.ndarray:
    vertex_colors = (
        np.clip(np.asarray(base_vertex_colors) * 255.0, 0.0, 255.0).astype(np.uint8)
        if base_vertex_colors is not None and len(base_vertex_colors) > 0
        else np.full((int(np.max(faces)) + 1 if len(faces) else 0, 3), background_color, dtype=np.uint8)
    )
    if vertex_colors.ndim != 2 or vertex_colors.shape[1] != 3:
        vertex_colors = np.full((int(np.max(faces)) + 1 if len(faces) else 0, 3), background_color, dtype=np.uint8)
    if len(face_indices) == 0 or len(faces) == 0:
        return vertex_colors

    valid_face_indices = [int(face_idx) for face_idx in face_indices if 0 <= int(face_idx) < len(faces)]
    if not valid_face_indices:
        return vertex_colors

    selected_vertices = np.unique(faces[np.asarray(valid_face_indices, dtype=np.int64)].reshape(-1))
    vertex_colors = vertex_colors.copy()
    vertex_colors[selected_vertices] = np.asarray(highlight_color, dtype=np.uint8)
    return vertex_colors


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


def _camera_center_and_axes(rotation_matrix: np.ndarray, translation_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return camera center and the camera's right/up/forward axes in world space."""
    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64)
    translation_vector = np.asarray(translation_vector, dtype=np.float64).reshape(3, 1)
    camera_center = (-rotation_matrix.T @ translation_vector).reshape(3)
    forward = rotation_matrix.T @ np.array([0.0, 0.0, 1.0])
    up = rotation_matrix.T @ np.array([0.0, -1.0, 0.0])
    right = np.cross(forward, up)
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    up = up / (np.linalg.norm(up) + 1e-9)
    right = right / (np.linalg.norm(right) + 1e-9)
    return camera_center, right, up, forward


def _log_camera_frustum(rr: Any, entity_path: str, spectator: Any, size: float = 0.05) -> None:
    """Log a small pinhole frustum to represent the spectator in 3D."""
    try:
        camera_center, right, up, forward = _camera_center_and_axes(
            spectator.rotation_matrix,
            spectator.translation_vector,
        )
        width = float(spectator.image_width or 1)
        height = float(spectator.image_height or 1)
        fx = float(spectator.intrinsics["fx"])
        fov_vertical = 2.0 * np.arctan(height / (2.0 * max(fx, 1e-9)))
        aspect = width / max(height, 1.0)
        half_height = size * np.tan(fov_vertical / 2.0)
        half_width = half_height * aspect
        forward_pt = camera_center + forward * size
        top_left = forward_pt + up * half_height - right * half_width
        top_right = forward_pt + up * half_height + right * half_width
        bottom_left = forward_pt - up * half_height - right * half_width
        bottom_right = forward_pt - up * half_height + right * half_width
        frustum_lines = np.array([
            [camera_center, top_left],
            [camera_center, top_right],
            [camera_center, bottom_left],
            [camera_center, bottom_right],
            [top_left, top_right],
            [top_right, bottom_right],
            [bottom_right, bottom_left],
            [bottom_left, top_left],
        ], dtype=np.float64)
        rr.log(
            entity_path,
            rr.Arrows3D(
                origins=frustum_lines[:, 0],
                vectors=frustum_lines[:, 1] - frustum_lines[:, 0],
                colors=np.full((len(frustum_lines), 3), [235, 235, 235], dtype=np.uint8),
            ),
        )
    except Exception:
        pass


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



def _create_colorscale_bar(min_val: float = 0.0, max_val: float = 1.0, width: int = 80, height: int = 400) -> np.ndarray:
    """Create a vertical colorscale bar image with labeled value range."""
    try:
        import cv2
    except ImportError:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create vertical colorscale (top=max, bottom=min)
    colorscale_data = np.linspace(0.0, 1.0, height)[::-1].reshape(-1, 1)
    colors = _viridis_like_colormap(colorscale_data.reshape(-1)).reshape(height, 1, 3)
    
    # Repeat to fill width
    colorbar = np.tile(colors, (1, width, 1))
    
    # Add a white label margin to the right
    label_space = np.full((height, width, 3), 255, dtype=np.uint8)
    colorbar_with_labels = np.hstack([colorbar, label_space])
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 0, 0)
    thickness = 1
    
    # Max value label (top)
    max_text = f"{max_val:.2f}"
    text_size = cv2.getTextSize(max_text, font, font_scale, thickness)[0]
    cv2.putText(
        colorbar_with_labels,
        max_text,
        (width + 5, 10 + text_size[1]),
        font,
        font_scale,
        font_color,
        thickness,
    )
    
    # Mid value label (center)
    mid_val = (min_val + max_val) / 2.0
    mid_text = f"{mid_val:.2f}"
    text_size_mid = cv2.getTextSize(mid_text, font, font_scale, thickness)[0]
    cv2.putText(
        colorbar_with_labels,
        mid_text,
        (width + 5, height // 2 + text_size_mid[1] // 2),
        font,
        font_scale,
        font_color,
        thickness,
    )
    
    # Min value label (bottom)
    min_text = f"{min_val:.2f}"
    text_size_min = cv2.getTextSize(min_text, font, font_scale, thickness)[0]
    cv2.putText(
        colorbar_with_labels,
        min_text,
        (width + 5, height - 5),
        font,
        font_scale,
        font_color,
        thickness,
    )
    
    return colorbar_with_labels.astype(np.uint8)


def visualize_wound_assessment_rerun(
    wound_model: Any,
    session_name: str = "WoundModel3D_Assessment",
    spawn: bool = True,
    image_root: Optional[Path] = None,
    max_spectators: Optional[int] = None,
) -> None:
    """Log the wound-assessment pipeline to Rerun.

    Logs:
    - the original mesh as a static 3D scene
    - a time series of camera/mask projections for each spectator
    - wound/periwound r-value colormaps on face centroids
    """
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except Exception:
        print("Rerun not installed. Install with: pip install rerun-sdk")
        return

    if getattr(wound_model, "mesh", None) is None or getattr(wound_model, "faces", None) is None or getattr(wound_model, "vertices", None) is None:
        print("Rerun visualization skipped: mesh data is not available")
        return

    try:
        rr.init(session_name, spawn=spawn)
    except Exception as e:
        print(f"Could not launch Rerun viewer directly ({e}); continuing with default init")
        rr.init(session_name)

    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    try:
        # Create a blueprint with the world view and a tabbed analysis container
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
                # Group analysis views into tabs so user can switch between them
                rrb.Tabs(
                    rrb.Spatial3DView(origin="/analysis/r_values", name="Face r-value Clouds"),
                    rrb.Spatial3DView(origin="/analysis/posteriors/wound", name="Wound Posterior"),
                    rrb.Spatial3DView(origin="/analysis/posteriors/periwound", name="Periwound Posterior"),
                ),
            ),
        )
        rr.send_blueprint(blueprint)
    except Exception:
        pass

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

    centroids = np.asarray([face_obj.centroid for face_obj in getattr(wound_model, "face_objects", [])], dtype=np.float64)
    face_count = len(getattr(wound_model, "face_objects", []))
    r_wound_values = np.fromiter((float(face_obj.r_wound or 0.0) for face_obj in wound_model.face_objects), dtype=np.float64, count=face_count)
    r_periwound_values = np.fromiter((float(face_obj.r_periwound or 0.0) for face_obj in wound_model.face_objects), dtype=np.float64, count=face_count)
    semantic_posteriors = getattr(wound_model, "semantic_posteriors", {}) or {}
    wound_posterior_values = np.fromiter(
        (
            float(np.asarray(semantic_posteriors.get(face_obj.face_index, [0.0, 0.0]), dtype=np.float64)[1])
            for face_obj in wound_model.face_objects
        ),
        dtype=np.float64,
        count=face_count,
    )
    periwound_posterior_values = np.fromiter(
        (
            float(np.asarray(semantic_posteriors.get(face_obj.face_index, [0.0, 0.0]), dtype=np.float64)[0])
            for face_obj in wound_model.face_objects
        ),
        dtype=np.float64,
        count=face_count,
    )

    if face_count > 0:
        wound_colors = _face_values_to_mesh_colormap(r_wound_values, faces)
        rr.log(
            "/analysis/r_values/wound",
            rr.Mesh3D(vertex_positions=vertices, triangle_indices=faces, vertex_colors=wound_colors),
            static=True,
        )
        periwound_colors = _face_values_to_mesh_colormap(r_periwound_values, faces)
        rr.log(
            "/analysis/r_values/periwound",
            rr.Mesh3D(vertex_positions=vertices, triangle_indices=faces, vertex_colors=periwound_colors),
            static=True,
        )
        
        # Log colorscale bars with value ranges
        try:
            wound_min = float(np.nanmin(r_wound_values)) if np.any(np.isfinite(r_wound_values)) else 0.0
            wound_max = float(np.nanmax(r_wound_values)) if np.any(np.isfinite(r_wound_values)) else 1.0
            periwound_min = float(np.nanmin(r_periwound_values)) if np.any(np.isfinite(r_periwound_values)) else 0.0
            periwound_max = float(np.nanmax(r_periwound_values)) if np.any(np.isfinite(r_periwound_values)) else 1.0
            
            wound_colorscale = _create_colorscale_bar(wound_min, wound_max, width=80, height=400)
            rr.log(
                "/analysis/r_values/wound_colorscale",
                rr.Image(image=wound_colorscale),
                static=True,
            )
            
            periwound_colorscale = _create_colorscale_bar(periwound_min, periwound_max, width=80, height=400)
            rr.log(
                "/analysis/r_values/periwound_colorscale",
                rr.Image(image=periwound_colorscale),
                static=True,
            )
        except Exception as e:
            print(f"Could not log colorscale bars: {e}")

        # Also log final posterior colormaps (face-level) under /analysis/posteriors
        try:
            wound_posterior_colors = _face_values_to_mesh_colormap(wound_posterior_values, faces)
            rr.log(
                "/analysis/posteriors/wound",
                rr.Mesh3D(vertex_positions=vertices, triangle_indices=faces, vertex_colors=wound_posterior_colors),
                static=True,
            )

            periwound_posterior_colors = _face_values_to_mesh_colormap(periwound_posterior_values, faces)
            rr.log(
                "/analysis/posteriors/periwound",
                rr.Mesh3D(vertex_positions=vertices, triangle_indices=faces, vertex_colors=periwound_posterior_colors),
                static=True,
            )

            # Log colorbars for posteriors
            wound_post_min = float(np.nanmin(wound_posterior_values)) if np.any(np.isfinite(wound_posterior_values)) else 0.0
            wound_post_max = float(np.nanmax(wound_posterior_values)) if np.any(np.isfinite(wound_posterior_values)) else 1.0
            periwound_post_min = float(np.nanmin(periwound_posterior_values)) if np.any(np.isfinite(periwound_posterior_values)) else 0.0
            periwound_post_max = float(np.nanmax(periwound_posterior_values)) if np.any(np.isfinite(periwound_posterior_values)) else 1.0

            wound_post_colorscale = _create_colorscale_bar(wound_post_min, wound_post_max, width=80, height=400)
            rr.log(
                "/analysis/posteriors/wound_colorscale",
                rr.Image(image=wound_post_colorscale),
                static=True,
            )

            periwound_post_colorscale = _create_colorscale_bar(periwound_post_min, periwound_post_max, width=80, height=400)
            rr.log(
                "/analysis/posteriors/periwound_colorscale",
                rr.Image(image=periwound_post_colorscale),
                static=True,
            )
        except Exception as e:
            print(f"Could not log posterior colormaps: {e}")








    spectators = list(getattr(wound_model, "spectators", {}).values())
    if max_spectators is not None:
        spectators = spectators[:max_spectators]

    # First pass: log static camera poses and frustums
    for step_index, spectator in enumerate(spectators):
        if spectator is None or not getattr(spectator, "camera_data_loaded", False):
            continue

        fx = float(spectator.intrinsics["fx"])
        fy = float(spectator.intrinsics["fy"])
        cx = float(spectator.intrinsics["cx"])
        cy = float(spectator.intrinsics["cy"])

        # COLMAP stores rotation_matrix as world->camera and translation_vector as world->camera
        # For Rerun, we need camera->world, so we compute the inverse
        R_cam_to_world = spectator.rotation_matrix.T
        t_cam_to_world = -R_cam_to_world @ spectator.translation_vector.reshape(3, 1)

        camera_path = f"/cameras/camera_{step_index:03d}"

        # Log camera transform in world space
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

        # Log pinhole intrinsics
        rr.log(
            f"{camera_path}/image",
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
            rr.log(
                f"{camera_path}/image",
                rr.Image(image=_downscale_image_for_rerun(image_data, scale=0.1)),
            )
        else:
            print(f"Spectator {step_index} has no image data to log")
    

    print(f"Rerun wound-assessment visualization logged to session '{session_name}'")


def debug_visualize_gmm_samples_and_confidence(*args: Any, **kwargs: Any) -> None:
    """Backward-compatible alias for the previous import path."""
    visualize_wound_assessment_rerun(*args, **kwargs)
