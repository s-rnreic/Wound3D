from collections import defaultdict, deque
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

import numpy as np
from scipy import sparse


    
def fit_constrained_healthy_skin_surface(self,
                                         alpha: float = 0.2,
                                         beta: float = 0.8,
                                         gamma: float = 0.8,
                                         periwound_data_weight: float = 1.0,
                                         regularization: float = 1e-8,
                                         enforce_wound_height_floor: bool = True,
                                         enforce_boundary_height_match: bool = True,
                                         use_cvxpy: bool = False,
                                         ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Fit a healthy-skin reference surface on the wound/periwound support region.

    The optimization keeps the PCA-plane coordinates fixed and solves only for the
    local height of each support vertex:

        min  alpha * sum_{i in P} w_i (h_i - z_i)^2
             + beta * h^T L^T L h
             + gamma * sum_{i in W} (h_i - h_i_prior)^2

    where P is the periwound vertex set, wound vertices are excluded from the data
    term, both wound and periwound faces define the smoothness support region,
    and h_prior is sampled from fit_periwound_surface.
    """
    if self.vertices is None or self.faces is None:
        raise ValueError('Mesh must be loaded')

    vertices = np.asarray(self.vertices, dtype=np.float64)
    faces = np.asarray(self.faces, dtype=np.int64)
    face_count = len(faces)

    wound_faces = {
        int(face_id)
        for face_id in (getattr(self, "current_wound_faces", set()) or set())
        if 0 <= int(face_id) < face_count
    }
    periwound_faces = {
        int(face_id)
        for face_id in (getattr(self, "current_periwound_faces", set()) or set())
        if 0 <= int(face_id) < face_count
    }


    pw_faces = set(periwound_faces)
    w_faces = set(wound_faces)
    support_face_ids = np.asarray(sorted(pw_faces | w_faces), dtype=np.int64)


    support_faces = faces[support_face_ids]
    support_vertex_ids = np.unique(support_faces.reshape(-1)).astype(np.int64)
    vertex_to_local = {int(vertex_id): int(local_idx) for local_idx, vertex_id in enumerate(support_vertex_ids.tolist())}
    support_vertices = vertices[support_vertex_ids]
    support_faces_local = np.asarray(
        [[vertex_to_local[int(vertex_id)] for vertex_id in tri] for tri in support_faces],
        dtype=np.int32,
    )
    nv = len(support_vertices)

    pw_mask = np.zeros(nv, dtype=bool)
    wound_mask = np.zeros(nv, dtype=bool)
    for local_face_idx, source_face_id in enumerate(support_face_ids):
        if source_face_id in pw_faces:
            pw_mask[support_faces_local[local_face_idx]] = True
        if source_face_id in w_faces:
            wound_mask[support_faces_local[local_face_idx]] = True

    if not np.any(pw_mask):
        raise ValueError('No periwound vertices found')

    ref_pts = support_vertices[pw_mask]

    ref_normals = None
    if getattr(self, 'face_normals', None) is not None and len(self.face_normals) > 0:
        pw_face_normals = [
            np.asarray(self.face_normals[int(source_face_id)], dtype=np.float64)
            for local_face_idx, source_face_id in enumerate(support_face_ids)
            if source_face_id in pw_faces
        ]
        if pw_face_normals:
            ref_normals = np.asarray(pw_face_normals, dtype=np.float64)

    origin, basis, ev = _compute_local_pca_frame(self, ref_pts, reference_normals=ref_normals)
    lc = _project_points_to_local_frame(support_vertices, origin, basis)
    uv = lc[:, :2]
    h0 = lc[:, 2].copy()

    d3d = _reconstruct_surface_from_heights(origin, basis, lc, h0)
    L = _build_cotangent_laplacian(self, d3d, support_faces_local)

    
    
    data_weights = np.zeros(nv, dtype=np.float64)
    data_weights[pw_mask] = float(periwound_data_weight)
    measured = np.zeros(nv, dtype=np.float64)
    measured[pw_mask] = h0[pw_mask]

    prior_heights, prior_weights = _extract_wound_height_priors_from_periwound_surface(
        self,
        support_vertices=support_vertices,
        support_vertex_ids=support_vertex_ids,
        wound_mask=wound_mask,
        origin=origin,
        basis=basis,
    )
    
    import rerun as rr

    rr.log(
                    '/analysis/all_points',
                    rr.Points3D(
                        positions=np.ascontiguousarray(vertices, dtype=np.float64),
                        colors= [255,0,0],
                        radii=0.0005,
                    ),
                    static=True,
                )
    rr.log(
                        '/analysis/plane_points',
                        rr.Points3D(
                            positions=np.ascontiguousarray(np.stack([lc[:, 0], lc[:, 1], h0], axis=1), dtype=np.float64),
                            colors= [255,0,0],
                            radii=0.0005,
                        ),
                        static=True,
                    )
    rr.log(
                        '/analysis/reconstructed_points',
                        rr.Points3D(
                            positions=np.ascontiguousarray(d3d, dtype=np.float64),
                            colors= [0,255,0],
                            radii=0.0005,
                        ),
                        static=True,
                    )
    rr.log(
        '/analysis/prior_heights',
        rr.Points3D(
            positions=np.ascontiguousarray(
                np.column_stack([
                    lc[prior_weights > 0, 0],
                    lc[prior_weights > 0, 1],
                    prior_heights[prior_weights > 0],
                ]),
                dtype=np.float64,
            ),
            colors=[0, 0, 255],
            radii=0.0005,
        ),
        static=True,
    )

    lower_bounds = None
    if enforce_wound_height_floor:
        lower_bounds = np.full(nv, -np.inf, dtype=np.float64)
        lower_bounds[wound_mask] = h0[wound_mask]

    fixed_heights = None
    border_mask = pw_mask & wound_mask
    if enforce_boundary_height_match and np.any(border_mask):
        fixed_heights = np.full(nv, np.nan, dtype=np.float64)
        fixed_heights[border_mask] = h0[border_mask]

    W = sparse.diags(data_weights, format='csr')
    WPrior = sparse.diags(prior_weights, format='csr')
    data_Q = W.T @ W
    smooth_Q = L.T @ L
    prior_Q = WPrior.T @ WPrior

    def _quadratic_energy_scale(term: sparse.spmatrix) -> float:
        diag = np.asarray(term.diagonal(), dtype=np.float64).ravel()
        scale = float(np.mean(diag[np.isfinite(diag) & (diag > 0.0)])) if diag.size > 0 else 1.0
        if not np.isfinite(scale) or scale <= 1e-12:
            return 1.0
        return scale

    data_scale = _quadratic_energy_scale(data_Q)
    smooth_scale = _quadratic_energy_scale(smooth_Q)
    prior_scale = _quadratic_energy_scale(prior_Q)

    A = (
        alpha * (data_Q / data_scale)
        + beta * (smooth_Q / smooth_scale)
        + gamma * (prior_Q / prior_scale)
        + regularization * sparse.identity(nv, format='csr')
    )
    b = (
        alpha * ((data_weights * measured) / data_scale)
        + gamma * ((prior_weights * prior_heights) / prior_scale)
    )

    print(
        f'Healthy-skin fit: support_faces={len(support_face_ids)}, '
        f'support_vertices={nv}, periwound_vertices={int(pw_mask.sum())}, '
        f'wound_vertices={int(wound_mask.sum())}, '
        f'boundary_vertices={int(np.count_nonzero(border_mask))}, '
        f'prior_vertices={int(np.count_nonzero(prior_weights > 0.0))}, '
        f'data_scale={data_scale:.3e}, smooth_scale={smooth_scale:.3e}, prior_scale={prior_scale:.3e}, '
        f'wound_floor={bool(enforce_wound_height_floor)}, '
        f'boundary_match={bool(enforce_boundary_height_match)}'
    )
    h_opt = _solve_quadratic_height_system(
        self,
        A,
        b,
        use_cvxpy=use_cvxpy,
        lower_bounds=lower_bounds,
        fixed_values=fixed_heights,
    )
    rec = _reconstruct_surface_from_heights(origin, basis, lc, h_opt)

    
    rr.log(
        '/analysis/optimized_surface',
        rr.Points3D(
            positions=np.ascontiguousarray(rec, dtype=np.float64),
            colors=[255, 255, 0],
            radii=0.0005,
        ),
        static=True,
    )
    height_delta = np.asarray(h_opt, dtype=np.float64).ravel() - h0
    height_epsilon = max(1e-6, 1e-6 * float(np.max(np.abs(h0)) if h0.size > 0 else 1.0))
    positive_delta = np.where(height_delta > height_epsilon, height_delta, 0.0)

    edge_01 = uv[support_faces_local[:, 1]] - uv[support_faces_local[:, 0]]
    edge_02 = uv[support_faces_local[:, 2]] - uv[support_faces_local[:, 0]]
    face_areas = 0.5 * np.abs(edge_01[:, 0] * edge_02[:, 1] - edge_01[:, 1] * edge_02[:, 0])
    vertex_area_weights = np.zeros(nv, dtype=np.float64)
    area_share = face_areas / 3.0
    np.add.at(vertex_area_weights, support_faces_local[:, 0], area_share)
    np.add.at(vertex_area_weights, support_faces_local[:, 1], area_share)
    np.add.at(vertex_area_weights, support_faces_local[:, 2], area_share)
    surface_volume = float(np.sum(vertex_area_weights * positive_delta))
    fallback_volume = _estimate_wound_convex_hull_volume(vertices, faces, wound_faces)

    signed_triangle_volume_meshes, signed_triangle_volumes = _build_signed_triangle_volume_meshes(
        support_vertices=support_vertices,
        measured_surface=d3d,
        optimized_surface=rec,
        faces=support_faces_local[np.asarray(
            [int(face_id) in w_faces for face_id in support_face_ids],
            dtype=bool,
        )],
        origin=origin,
        basis=basis,
    )
    convex_hull_mesh = _build_wound_convex_hull_mesh(vertices, faces, wound_faces)

    volume = surface_volume
    volume_ratio = np.nan
    volume_relative_difference = np.nan
    volume_warning = None
    if not np.isfinite(volume) or volume <= 0.0:
        if np.isfinite(fallback_volume) and fallback_volume > 0.0:
            volume = float(fallback_volume)
            volume_warning = (
                'Warning: surface volume estimate was invalid; '
                f'using wound convex-hull fallback volume {fallback_volume:.6f}.'
            )
            print(volume_warning)
    elif np.isfinite(fallback_volume) and fallback_volume > 0.0:
        volume_ratio = float(volume / fallback_volume)
        volume_relative_difference = float(abs(volume - fallback_volume) / max(volume, fallback_volume))
        if volume_relative_difference > 0.5:
            volume_warning = (
                'Warning: surface volume and wound convex-hull volume differ by '
                f'{volume_relative_difference * 100.0:.1f}% (ratio={volume_ratio:.3f}); '
                'the volume calculation might not be suitable for this kind of wound.'
            )
            print(volume_warning)

    self.constrained_healthy_skin_surface_mesh = (
        np.asarray(rec, dtype=np.float64),
        np.asarray(support_faces_local, dtype=np.int32),
    )
    self.constrained_healthy_skin_surface_solution = np.asarray(h_opt, dtype=np.float64).ravel()
    self.constrained_healthy_skin_surface_frame = (
        np.asarray(origin, dtype=np.float64),
        np.asarray(basis, dtype=np.float64),
        np.asarray(ev, dtype=np.float64),
    )
    self.constrained_healthy_skin_surface_volume = volume
    self.constrained_healthy_skin_surface_surface_volume = surface_volume
    self.constrained_healthy_skin_surface_fallback_volume = fallback_volume
    self.constrained_healthy_skin_surface_volume_ratio = volume_ratio
    self.constrained_healthy_skin_surface_volume_relative_difference = volume_relative_difference
    self.constrained_healthy_skin_surface_volume_warning = volume_warning
    self.constrained_healthy_skin_surface_volume_mesh = self.constrained_healthy_skin_surface_mesh
    self.constrained_healthy_skin_surface_volume_meshes = [self.constrained_healthy_skin_surface_mesh]
    self.constrained_healthy_skin_surface_signed_triangle_volume_meshes = signed_triangle_volume_meshes
    self.constrained_healthy_skin_surface_signed_triangle_volumes = signed_triangle_volumes
    self.constrained_healthy_skin_surface_convex_hull_mesh = convex_hull_mesh
    self.constrained_healthy_skin_surface_volume_components = [
        {
            'component_index': 0,
            'vertex_count': int(len(rec)),
            'face_count': int(len(support_faces_local)),
            'volume': volume,
            'surface_volume': surface_volume,
            'fallback_volume': fallback_volume,
            'volume_ratio': volume_ratio,
            'volume_relative_difference': volume_relative_difference,
        }
    ]
    self.constrained_healthy_skin_surface_data = {
        'support_vertex_ids': np.asarray(support_vertex_ids, dtype=np.int64),
        'support_face_ids': np.asarray(support_face_ids, dtype=np.int64),
        'support_faces': np.asarray(support_faces_local, dtype=np.int32),
        'wound_mask': np.asarray(wound_mask, dtype=bool),
        'periwound_mask': np.asarray(pw_mask, dtype=bool),
        'height_epsilon': float(height_epsilon),
        'volume': float(volume),
        'surface_volume': float(surface_volume),
        'fallback_volume': float(fallback_volume),
        'volume_ratio': float(volume_ratio) if np.isfinite(volume_ratio) else np.nan,
        'volume_relative_difference': float(volume_relative_difference) if np.isfinite(volume_relative_difference) else np.nan,
        'volume_warning': volume_warning,
        'signed_triangle_volumes': np.asarray(signed_triangle_volumes, dtype=np.float64),
    }

    _log_signed_volume_visualization(
        rr,
        signed_triangle_volume_meshes,
        signed_triangle_volumes,
        convex_hull_mesh,
        session_path='/analysis/healthy_skin_surface_signed_volume',
    )
    if volume_warning is None:
        print(
            'Positive-height volume estimate: '
            f'{surface_volume:.6f} (fallback={fallback_volume:.6f}, epsilon={height_epsilon:.3e})'
        )
    else:
        print(
            'Positive-height volume estimate: '
            f'{surface_volume:.6f} (fallback={fallback_volume:.6f}, epsilon={height_epsilon:.3e})'
        )
    
    
    
    return rec, support_faces_local
    


def _project_points_to_local_frame(points, origin, basis):
    return (np.asarray(points, dtype=np.float64) -
            np.asarray(origin, dtype=np.float64)[None, :]) @ np.asarray(basis, dtype=np.float64)


def _reconstruct_surface_from_heights(origin, basis, lc, h):
    origin = np.asarray(origin, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    lc = np.asarray(lc, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64).ravel()

    rec_lc = lc.copy()
    rec_lc[:, 2] = h
    return origin[None, :] + rec_lc @ basis.T


def _as_rgba_color(color, default_alpha=255):
    rgba = np.asarray(color, dtype=np.uint8).ravel()
    if rgba.size >= 4:
        return rgba[:4]
    if rgba.size == 3:
        return np.asarray([rgba[0], rgba[1], rgba[2], default_alpha], dtype=np.uint8)
    if rgba.size == 1:
        value = int(rgba[0])
        return np.asarray([value, value, value, default_alpha], dtype=np.uint8)
    return np.asarray([120, 180, 240, default_alpha], dtype=np.uint8)


def _compute_vertex_normals(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    normals = np.zeros_like(vertices, dtype=np.float64)
    if len(vertices) == 0 or len(faces) == 0:
        return normals

    tri_vertices = vertices[faces]
    face_normals = np.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0])
    face_lengths = np.linalg.norm(face_normals, axis=1)
    valid = face_lengths > 1e-12
    if not np.any(valid):
        return normals

    face_normals[valid] /= face_lengths[valid][:, None]
    for tri, normal in zip(faces, face_normals):
        normals[tri] += normal

    normal_lengths = np.linalg.norm(normals, axis=1)
    nonzero = normal_lengths > 1e-12
    normals[nonzero] /= normal_lengths[nonzero][:, None]
    return normals


def _shade_mesh_colors(vertices, faces, base_color, shading_mode='smooth', light_direction=(0.35, 0.45, 0.82)):
    base_rgba = _as_rgba_color(base_color, default_alpha=160).astype(np.float64)
    normals = _compute_vertex_normals(vertices, faces)
    if len(normals) == 0:
        return np.tile(base_rgba[None, :], (len(np.asarray(vertices)), 1)).astype(np.uint8), normals

    light_dir = np.asarray(light_direction, dtype=np.float64)
    light_norm = np.linalg.norm(light_dir)
    if light_norm <= 1e-12:
        light_dir = np.array([0.35, 0.45, 0.82], dtype=np.float64)
        light_norm = np.linalg.norm(light_dir)
    light_dir = light_dir / light_norm

    diffuse = np.clip(normals @ light_dir, 0.0, 1.0)
    ambient = 0.20
    if shading_mode == 'flat':
        intensity = np.full(len(normals), 0.85, dtype=np.float64)
    elif shading_mode == 'phong':
        view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        half_vec = light_dir + view_dir
        half_vec /= (np.linalg.norm(half_vec) + 1e-12)
        specular = np.clip(normals @ half_vec, 0.0, 1.0) ** 24
        intensity = ambient + 0.72 * diffuse + 0.28 * specular
    elif shading_mode == 'pbr':
        intensity = ambient + 0.62 * diffuse
        intensity = 0.86 * intensity + 0.14 * np.clip((normals[:, 2] + 1.0) * 0.5, 0.0, 1.0)
    else:  # smooth
        intensity = ambient + 0.75 * diffuse

    centroid = np.mean(np.asarray(vertices, dtype=np.float64), axis=0) if len(vertices) > 0 else np.zeros(3, dtype=np.float64)
    depth_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    depth = np.asarray(vertices, dtype=np.float64) - centroid[None, :]
    depth_weight = np.clip(0.5 + 0.5 * (depth @ depth_axis) / (np.linalg.norm(depth, axis=1).max() + 1e-12), 0.2, 1.0)
    intensity = np.clip(intensity * depth_weight, 0.18, 1.0)

    rgb = np.clip(base_rgba[:3][None, :] * intensity[:, None], 0.0, 255.0)
    rgba = np.concatenate([rgb, np.full((len(rgb), 1), base_rgba[3], dtype=np.float64)], axis=1)
    return rgba.astype(np.uint8), normals.astype(np.float32)


def _mesh_chunk_to_colored_mesh(vertices, faces, base_color, shading_mode='phong'):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        return None
    colors, normals = _shade_mesh_colors(vertices, faces, base_color, shading_mode=shading_mode)
    return {
        'vertex_positions': np.ascontiguousarray(vertices, dtype=np.float64),
        'triangle_indices': np.ascontiguousarray(faces, dtype=np.int32),
        'vertex_normals': np.ascontiguousarray(normals, dtype=np.float32),
        'vertex_colors': np.ascontiguousarray(colors, dtype=np.uint8),
        'albedo_factor': _as_rgba_color(base_color, default_alpha=160),
    }


def _extract_wound_height_priors_from_periwound_surface(
                                                        self,
                                                        support_vertices,
                                                        support_vertex_ids,
                                                        wound_mask,
                                                        origin,
                                                        basis):
    
    prior_heights = np.zeros(len(support_vertices), dtype=np.float64)
    prior_weights = np.zeros(len(support_vertices), dtype=np.float64)

    wound_mask = np.asarray(wound_mask, dtype=bool)

    boundary_vertices = None
    wound_outerboundary_vertices = getattr(self, 'wound_outerboundary_vertices', None)
    if wound_outerboundary_vertices is not None and len(wound_outerboundary_vertices) > 0:
        boundary_vertices = np.asarray(sorted(int(v) for v in wound_outerboundary_vertices), dtype=np.int64)
    else:
        boundary_vertices_value = getattr(self, 'boundary_vertices', None)
        if boundary_vertices_value is not None and len(boundary_vertices_value) > 0:
            boundary_vertices = np.asarray(sorted(int(v) for v in boundary_vertices_value), dtype=np.int64)

    if boundary_vertices is None or np.asarray(boundary_vertices).size == 0:
        boundary_faces = getattr(self, 'wound_outerboundary_faces', None)
        if boundary_faces is not None and len(boundary_faces) > 0:
            boundary_face_ids = sorted(
                int(face_id)
                for face_id in boundary_faces
                if 0 <= int(face_id) < len(getattr(self, 'faces', []))
            )
            if boundary_face_ids and getattr(self, 'faces', None) is not None:
                boundary_vertices = np.unique(np.asarray(self.faces[np.asarray(boundary_face_ids, dtype=np.int64)], dtype=np.int64).reshape(-1))

    if boundary_vertices is None or np.asarray(boundary_vertices).size == 0:
        print('No wound boundary cache available; disabling prior term')
        return prior_heights, prior_weights

    boundary_vertices = np.asarray(sorted({int(v) for v in np.asarray(boundary_vertices, dtype=np.int64).tolist()}), dtype=np.int64)
    if boundary_vertices.size == 0:
        print('No wound boundary cache available; disabling prior term')
        return prior_heights, prior_weights

    support_vertex_ids = np.asarray(support_vertex_ids, dtype=np.int64)
    if support_vertex_ids.size > 0:
        local_lookup = {int(vertex_id): int(local_idx) for local_idx, vertex_id in enumerate(support_vertex_ids.tolist())}
        boundary_local = np.asarray(
            [local_lookup[int(vertex_id)] for vertex_id in boundary_vertices if int(vertex_id) in local_lookup],
            dtype=np.int64,
        )
    else:
        boundary_local = np.asarray(boundary_vertices, dtype=np.int64)

    boundary_uv = np.asarray(
        _project_points_to_local_frame(support_vertices[boundary_local], origin, basis)[:, :2],
        dtype=np.float64,
    )
    boundary_uv = np.unique(boundary_uv[np.isfinite(boundary_uv).all(axis=1)], axis=0)


    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from scipy.spatial import ConvexHull


    hull = ConvexHull(boundary_uv)
    boundary_polygon = boundary_uv[np.asarray(hull.vertices, dtype=np.int64)]


    def _points_in_polygon(points, polygon):
        points = np.asarray(points, dtype=np.float64)
        polygon = np.asarray(polygon, dtype=np.float64)
        if points.size == 0 or polygon.shape[0] < 3:
            return np.zeros(len(points), dtype=bool)

        x = points[:, 0]
        y = points[:, 1]
        inside = np.zeros(len(points), dtype=bool)
        x0, y0 = polygon[-1]
        for x1, y1 in polygon:
            y_between = ((y1 > y) != (y0 > y))
            x_intersect = (x0 - x1) * (y - y1) / (y0 - y1 + 1e-12) + x1
            inside ^= y_between & (x < x_intersect)
            x0, y0 = x1, y1
        return inside

    prior_fit = self.fit_periwound_surface(method='rbf', function_name='thin_plate', outer_weighting_exponent=0.1)
    if not prior_fit or prior_fit[0] is None:
        print('No periwound prior surface available; disabling prior term')
        return prior_heights, prior_weights

    prior_vertices = np.asarray(prior_fit[0], dtype=np.float64)
    prior_local = _project_points_to_local_frame(prior_vertices, origin, basis)
    prior_uv = np.asarray(prior_local[:, :2], dtype=np.float64)
    prior_z = np.asarray(prior_local[:, 2], dtype=np.float64)

    finite_prior = np.isfinite(prior_uv).all(axis=1) & np.isfinite(prior_z)

    prior_uv = prior_uv[finite_prior]
    prior_z = prior_z[finite_prior]


    prior_inside = _points_in_polygon(prior_uv, boundary_polygon)

    prior_uv_inside = prior_uv[prior_inside]
    prior_z_inside = prior_z[prior_inside]

    target_mask = wound_mask.copy()
    target_uv = np.asarray(_project_points_to_local_frame(support_vertices[target_mask], origin, basis)[:, :2], dtype=np.float64)
    target_inside = _points_in_polygon(target_uv, boundary_polygon)


    target_indices = np.flatnonzero(target_mask)[target_inside]
    target_uv_inside = target_uv[target_inside]

    try:
        interpolator = LinearNDInterpolator(prior_uv_inside, prior_z_inside, fill_value=np.nan)
        sampled_prior = np.asarray(interpolator(target_uv_inside), dtype=np.float64).ravel()
    except Exception:
        interpolator = NearestNDInterpolator(prior_uv_inside, prior_z_inside)
        sampled_prior = np.asarray(interpolator(target_uv_inside), dtype=np.float64).ravel()

    invalid = ~np.isfinite(sampled_prior)
    if np.any(invalid):
        fallback = NearestNDInterpolator(prior_uv_inside, prior_z_inside)
        sampled_prior[invalid] = np.asarray(fallback(target_uv_inside[invalid]), dtype=np.float64).ravel()

    valid = np.isfinite(sampled_prior)

    prior_heights[target_indices[valid]] = sampled_prior[valid]
    prior_weights[target_indices[valid]] = 1.0

    return prior_heights, prior_weights


def _compute_local_pca_frame(self, points, reference_normals=None):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 3:
        raise ValueError('Need >=3 3D points')
    origin = np.mean(points, axis=0)
    centered = points - origin[None, :]
    cov = (centered.T @ centered) / float(len(points))
    ev, evec = np.linalg.eigh(cov)
    o = np.argsort(ev)[::-1]
    ev, basis = ev[o], evec[:, o]
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1.0
    if reference_normals is not None and len(reference_normals) > 0:
        mn = np.mean(np.asarray(reference_normals, dtype=np.float64), axis=0)
        mnn = np.linalg.norm(mn)
        if mnn > 1e-12 and float(np.dot(basis[:, 2], mn / mnn)) < 0.0:
            basis[:, 2] *= -1.0; basis[:, 0] *= -1.0
    for i in range(3):
        basis[:, i] /= np.linalg.norm(basis[:, i]) + 1e-12
    return origin, basis, ev



def _cotangent(a, b):
    cn = np.linalg.norm(np.cross(a, b))
    return 0.0 if cn <= 1e-12 else float(np.dot(a, b) / cn)


def _build_cotangent_laplacian(self, vertices, faces):
    import scipy
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    nv = len(v)
    if nv == 0:
        return scipy.sparse.csr_matrix((0, 0), dtype=np.float64)
    ew = defaultdict(float)
    for tri in f:
        i, j, k = map(int, tri)
        pi, pj, pk = v[i], v[j], v[k]
        ct = [_cotangent(pj - pi, pk - pi),
              _cotangent(pk - pj, pi - pj),
              _cotangent(pi - pk, pj - pk)]
        ew[tuple(sorted((j, k)))] += 0.5 * ct[0]
        ew[tuple(sorted((k, i)))] += 0.5 * ct[1]
        ew[tuple(sorted((i, j)))] += 0.5 * ct[2]
    rows, cols, data = [], [], []
    rs = np.zeros(nv, dtype=np.float64)
    for (i, j), w in ew.items():
        if not np.isfinite(w) or abs(w) <= 1e-12:
            continue
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([-w, -w])
        rs[i] += w
        rs[j] += w
    rows.extend(range(nv))
    cols.extend(range(nv))
    data.extend(rs.tolist())
    return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(nv, nv), dtype=np.float64)


def _estimate_wound_convex_hull_volume(vertices, faces, wound_face_ids):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    if wound_face_ids is None:
        candidate_face_ids = []
    else:
        candidate_face_ids = wound_face_ids

    valid_face_ids = sorted(
        int(face_id)
        for face_id in candidate_face_ids
        if 0 <= int(face_id) < len(faces)
    )
    if len(valid_face_ids) == 0:
        return 0.0

    wound_vertices = np.unique(faces[np.asarray(valid_face_ids, dtype=np.int64)].reshape(-1))
    if wound_vertices.size < 4:
        return 0.0

    points = np.asarray(vertices[wound_vertices], dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.shape[0] < 4:
        return 0.0

    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(points)
        volume = float(getattr(hull, 'volume', 0.0))
        if not np.isfinite(volume) or volume < 0.0:
            return 0.0
        return volume
    except Exception:
        return 0.0


def _solve_quadratic_height_system(self, Q, b, use_cvxpy=False, lower_bounds=None, upper_bounds=None, fixed_values=None):
    Q = Q.tocsr()
    b = np.asarray(b, dtype=np.float64).ravel()
    has_bounds = lower_bounds is not None or upper_bounds is not None or fixed_values is not None
    try:
        if use_cvxpy or has_bounds:
            import importlib
            cp = importlib.import_module('cvxpy')
            h = cp.Variable(Q.shape[0])
            constraints = []
            if lower_bounds is not None:
                lower_bounds = np.asarray(lower_bounds, dtype=np.float64).ravel()
                finite_lower = np.isfinite(lower_bounds)
                if np.any(finite_lower):
                    constraints.append(h[finite_lower] >= lower_bounds[finite_lower])
            if upper_bounds is not None:
                upper_bounds = np.asarray(upper_bounds, dtype=np.float64).ravel()
                finite_upper = np.isfinite(upper_bounds)
                if np.any(finite_upper):
                    constraints.append(h[finite_upper] <= upper_bounds[finite_upper])
            if fixed_values is not None:
                fixed_values = np.asarray(fixed_values, dtype=np.float64).ravel()
                finite_fixed = np.isfinite(fixed_values)
                if np.any(finite_fixed):
                    constraints.append(h[finite_fixed] == fixed_values[finite_fixed])
            cp.Problem(cp.Minimize(0.5 * cp.quad_form(h, Q) - b @ h), constraints).solve(solver=cp.OSQP, verbose=False)
            if h.value is not None:
                return np.asarray(h.value, dtype=np.float64).ravel()
    except Exception:
        pass

    if has_bounds:
        try:
            from scipy.optimize import minimize

            def objective(x):
                return 0.5 * float(x @ (Q @ x)) - float(b @ x)

            def gradient(x):
                return np.asarray(Q @ x - b, dtype=np.float64)

            bounds = None
            if lower_bounds is not None or upper_bounds is not None:
                lb = np.full(Q.shape[0], -np.inf, dtype=np.float64) if lower_bounds is None else np.asarray(lower_bounds, dtype=np.float64).ravel()
                ub = np.full(Q.shape[0], np.inf, dtype=np.float64) if upper_bounds is None else np.asarray(upper_bounds, dtype=np.float64).ravel()
                if fixed_values is not None:
                    fixed_values = np.asarray(fixed_values, dtype=np.float64).ravel()
                    finite_fixed = np.isfinite(fixed_values)
                    lb[finite_fixed] = fixed_values[finite_fixed]
                    ub[finite_fixed] = fixed_values[finite_fixed]
                bounds = [(float(lb[i]), float(ub[i])) for i in range(Q.shape[0])]
            elif fixed_values is not None:
                fixed_values = np.asarray(fixed_values, dtype=np.float64).ravel()
                finite_fixed = np.isfinite(fixed_values)
                lb = np.full(Q.shape[0], -np.inf, dtype=np.float64)
                ub = np.full(Q.shape[0], np.inf, dtype=np.float64)
                lb[finite_fixed] = fixed_values[finite_fixed]
                ub[finite_fixed] = fixed_values[finite_fixed]
                bounds = [(float(lb[i]), float(ub[i])) for i in range(Q.shape[0])]

            x0 = np.zeros(Q.shape[0], dtype=np.float64)
            if lower_bounds is not None:
                finite_lower = np.isfinite(lower_bounds)
                x0[finite_lower] = np.maximum(x0[finite_lower], np.asarray(lower_bounds, dtype=np.float64).ravel()[finite_lower])
            if fixed_values is not None:
                finite_fixed = np.isfinite(np.asarray(fixed_values, dtype=np.float64).ravel())
                x0[finite_fixed] = np.asarray(fixed_values, dtype=np.float64).ravel()[finite_fixed]

            result = minimize(objective, x0, jac=gradient, bounds=bounds, method='L-BFGS-B')
            if result.success:
                return np.asarray(result.x, dtype=np.float64).ravel()
        except Exception:
            pass

    from scipy.sparse import linalg as spla
    return np.asarray(spla.spsolve(Q.tocsc(), np.asarray(b, dtype=np.float64)), dtype=np.float64).ravel()


def _build_signed_triangle_volume_meshes(support_vertices, measured_surface, optimized_surface,
                                         faces, origin, basis):
    """Build signed triangular prisms for the supplied wound faces only."""
    support_vertices = np.asarray(support_vertices, dtype=np.float64)
    measured_surface = np.asarray(measured_surface, dtype=np.float64)
    optimized_surface = np.asarray(optimized_surface, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return [], np.empty(0, dtype=np.float64)
    faces = faces.reshape(-1, 3)
    local_measured = _project_points_to_local_frame(measured_surface, origin, basis)
    local_optimized = _project_points_to_local_frame(optimized_surface, origin, basis)
    meshes = []
    volumes = []
    for tri in faces:
        bottom = measured_surface[tri]
        top = optimized_surface[tri]
        tri_local = local_measured[tri]
        signed_area = 0.5 * float(np.cross(tri_local[1, :2] - tri_local[0, :2], tri_local[2, :2] - tri_local[0, :2]))
        mean_delta = float(np.mean(local_optimized[tri, 2] - tri_local[:, 2]))
        volumes.append(signed_area * mean_delta)
        verts = np.vstack([bottom, top])
        prism_faces = np.asarray([
            [0, 2, 1], [3, 4, 5],
            [0, 1, 4], [0, 4, 3],
            [1, 2, 5], [1, 5, 4],
            [2, 0, 3], [2, 3, 5],
        ], dtype=np.int32)
        meshes.append((np.ascontiguousarray(verts), prism_faces))
    return meshes, np.asarray(volumes, dtype=np.float64)


def _build_wound_convex_hull_mesh(vertices, faces, wound_face_ids):
    """Return the 3-D convex-hull boundary mesh for wound vertices."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    ids = sorted(int(i) for i in (wound_face_ids or set()) if 0 <= int(i) < len(faces))
    if not ids:
        return None
    vertex_ids = np.unique(faces[np.asarray(ids, dtype=np.int64)].reshape(-1))
    points = vertices[vertex_ids]
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) < 4:
        return None
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        return np.ascontiguousarray(points), np.ascontiguousarray(hull.simplices, dtype=np.int32)
    except Exception:
        return None


def _log_signed_volume_visualization(rr, triangle_meshes, signed_triangle_volumes, convex_hull_mesh,
                                     session_path='/analysis/healthy_skin_surface_signed_volume'):
    if not triangle_meshes:
        return
    values = np.asarray(signed_triangle_volumes, dtype=np.float64).ravel()
    max_abs = max(float(np.max(np.abs(values))), 1e-12)
    combined_vertices = []
    combined_faces = []
    combined_colors = []
    vertex_offset = 0
    for mesh, value in zip(triangle_meshes, values):
        magnitude = min(abs(float(value)) / max_abs, 1.0)
        color = np.asarray([255, int(70 + 150 * (1.0 - magnitude)), 50, 170], dtype=np.uint8) if value >= 0 else np.asarray([50, int(70 + 150 * (1.0 - magnitude)), 255, 170], dtype=np.uint8)
        verts, faces = mesh
        combined_vertices.append(verts)
        combined_faces.append(np.asarray(faces, dtype=np.int32) + vertex_offset)
        combined_colors.append(np.tile(color[None, :], (len(verts), 1)))
        vertex_offset += len(verts)

    combined_color = np.concatenate(combined_colors, axis=0)
    rr.log(f'{session_path}/triangle_volume_parts', rr.Mesh3D(
        vertex_positions=np.ascontiguousarray(np.concatenate(combined_vertices, axis=0), dtype=np.float64),
        triangle_indices=np.ascontiguousarray(np.concatenate(combined_faces, axis=0), dtype=np.int32),
        vertex_colors=np.ascontiguousarray(combined_color, dtype=np.uint8),
        albedo_factor=np.asarray([255, 140, 80, 170], dtype=np.uint8),
    ), static=True)
    if convex_hull_mesh is not None:
        verts, faces = convex_hull_mesh
        color = np.asarray([180, 180, 180, 70], dtype=np.uint8)
        rr.log(f'{session_path}/convex_hull', rr.Mesh3D(
            vertex_positions=verts, triangle_indices=faces,
            vertex_colors=np.tile(color[None, :], (len(verts), 1)), albedo_factor=color,
        ), static=True)


def visualize_healthy_skin_surface(self, session_path='/analysis/healthy_skin_surface', spawn=True, color=(70, 150, 255, 120), shading_mode='phong'):
    try:
        import rerun as rr
    except Exception:
        print('rerun unavailable')
        return
    m = getattr(self, 'constrained_healthy_skin_surface_mesh', None)
    if m is None:
        print('no surface')
        return
    verts, faces = m
    if verts is None or faces is None or len(verts) == 0 or len(faces) == 0:
        print('empty')
        return
    try:
        rr.init('WoundModel3D_Assessment_final', spawn=spawn)
        rgba = _as_rgba_color(color, default_alpha=120)
        shaded_colors, vertex_normals = _shade_mesh_colors(verts, faces, rgba, shading_mode=shading_mode)
        rr.log(session_path, rr.Mesh3D(
            vertex_positions=np.ascontiguousarray(verts, dtype=np.float64),
            triangle_indices=np.ascontiguousarray(faces, dtype=np.int32),
            vertex_normals=np.ascontiguousarray(vertex_normals, dtype=np.float32),
            vertex_colors=np.ascontiguousarray(shaded_colors, dtype=np.uint8),
            albedo_factor=rgba,
        ), static=True)
        print(f'Logged to {session_path}')
    except Exception as e:
        print(f'Failed: {e}')


def visualize_healthy_skin_surface_volumes(self,
                                           session_path: str = '/analysis/healthy_skin_surface_volumes',
                                           spawn: bool = True,
                                           color=None,
                                           visualize_all: bool = True,
                                           shading_mode: str = 'phong') -> None:
    try:
        import rerun as rr
    except Exception:
        print('rerun unavailable')
        return

    volume_meshes = getattr(self, 'constrained_healthy_skin_surface_volume_meshes', None)
    combined_mesh = getattr(self, 'constrained_healthy_skin_surface_volume_mesh', None)
    if not volume_meshes and combined_mesh is not None:
        volume_meshes = [combined_mesh]

    if not volume_meshes:
        print('no volume meshes')
        return

    try:
        rr.init('WoundModel3D_Assessment_final', spawn=spawn)
        meshes_to_log = volume_meshes if visualize_all else [combined_mesh or volume_meshes[0]]
        palette = np.asarray([
            _as_rgba_color(color, default_alpha=165) if color is not None else [255, 106, 93, 165],
            [73, 140, 255, 165],
            [72, 201, 176, 165],
            [214, 93, 177, 165],
            [255, 190, 92, 165],
            [150, 122, 255, 165],
        ], dtype=np.uint8)

        for idx, mesh_data in enumerate(meshes_to_log):
            if mesh_data is None:
                continue
            verts, faces = mesh_data
            if verts is None or faces is None or len(verts) == 0 or len(faces) == 0:
                continue
            mesh_color = palette[idx % len(palette)]
            shaded_colors, vertex_normals = _shade_mesh_colors(verts, faces, mesh_color, shading_mode=shading_mode)
            rr.log(
                f'{session_path}/component_{idx:03d}',
                rr.Mesh3D(
                    vertex_positions=np.ascontiguousarray(verts, dtype=np.float64),
                    triangle_indices=np.ascontiguousarray(faces, dtype=np.int32),
                    vertex_normals=np.ascontiguousarray(vertex_normals, dtype=np.float32),
                    vertex_colors=np.ascontiguousarray(shaded_colors, dtype=np.uint8),
                    albedo_factor=np.asarray(mesh_color, dtype=np.uint8),
                ),
                static=True,
            )
        print(f'Logged {len(meshes_to_log)} volume mesh component(s) to {session_path}')
    except Exception as e:
        print(f'visualize_healthy_skin_surface_volumes failed: {e}')


def visualize_healthy_skin_surface_signed_volume(
        self,
        session_path: str = '/analysis/healthy_skin_surface_signed_volume',
        spawn: bool = True) -> None:
    """Visualize the combined signed triangle prisms and the wound convex hull in Rerun."""
    try:
        import rerun as rr
    except Exception:
        print('rerun unavailable')
        return
    meshes = getattr(self, 'constrained_healthy_skin_surface_signed_triangle_volume_meshes', None)
    values = getattr(self, 'constrained_healthy_skin_surface_signed_triangle_volumes', None)
    hull = getattr(self, 'constrained_healthy_skin_surface_convex_hull_mesh', None)
    if not meshes or values is None:
        print('no signed volume data; run fit_constrained_healthy_skin_surface first')
        return
    try:
        rr.init('WoundModel3D_Assessment_final', spawn=spawn)
        _log_signed_volume_visualization(rr, meshes, values, hull, session_path=session_path)
        print(f'Logged one combined mesh containing {len(meshes)} signed triangle volume part(s) and the convex hull to {session_path}')
    except Exception as e:
        print(f'visualize_healthy_skin_surface_signed_volume failed: {e}')


def visualize_healthy_skin_surface_shell_parts(self,
                                       session_path: str = '/analysis/healthy_skin_surface_shell_parts',
                                       spawn: bool = True,
                                       shading_mode: str = 'phong') -> None:
    try:
        import rerun as rr
    except Exception:
        print('rerun unavailable')
        return

    shell_parts = getattr(self, 'constrained_healthy_skin_surface_volume_shell_parts', None)
    if not shell_parts:
        print('no shell parts')
        return

    try:
        rr.init('WoundModel3D_Assessment_final', spawn=spawn)
        group_colors = {
            'bottom_faces': np.asarray([250, 120, 90, 180], dtype=np.uint8),
            'top_faces': np.asarray([90, 165, 255, 180], dtype=np.uint8),
            'side_faces': np.asarray([90, 220, 160, 180], dtype=np.uint8),
        }

        for part_index, part in enumerate(shell_parts):
            vertices = np.asarray(part['vertices'], dtype=np.float64)
            component_name = f'component_{int(part["component_index"]):03d}'
            for group_name, base_color in group_colors.items():
                faces = np.asarray(part.get(group_name), dtype=np.int64)
                if faces.size == 0:
                    continue
                mesh_data = _mesh_chunk_to_colored_mesh(vertices, faces, base_color, shading_mode=shading_mode)
                if mesh_data is None:
                    continue
                rr.log(
                    f'{session_path}/{component_name}/{group_name}',
                    rr.Mesh3D(**mesh_data),
                    static=True,
                )

        print(f'Logged shell face groups for {len(shell_parts)} component(s) to {session_path}')
    except Exception as e:
        print(f'visualize_healthy_skin_surface_shell_parts failed: {e}')


######### Second Approach


def fit_periwound_surface(self,
                            method: str = "rbf",
                            voxel_size: Optional[float] = None,
                            grid_resolution: int = 128,
                            smoothing: float = 0.0,
                            spline_degree: int = 3,
                            return_points: bool = False,
                            outer_weighting_exponent: float = 0.1,
                            function_name: str = 'thin_plate',
                            weighting_mode: str = "linear_floor",
                            inner_weight: float = 0.80,
                            ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Fit a smooth surface through the current periwound faces and return a mesh or sample points.

    Strategy:
    - Project periwound face centroids into a fitted local tissue plane (u,v coords).
    - Fit either an RBF (recommended) or a B-spline-like smooth bivariate surface z(u,v).
    - Return a dense grid-mesh sampled from the fitted function (vertices, faces) or points.

    Args:
        method: 'rbf' or 'spline' (falls back to a simple poly fit if SciPy not available)
        voxel_size: Optional base voxel size for aggregating normals/points before fitting.
            When wound boundary geometry is available, the effective voxel size
            is scaled from that boundary size so the aggregation stays proportional
            to the wound extent.
        grid_resolution: Grid resolution for sampled output (per axis).
        smoothing: Regularization/smoothing parameter for spline/rbf where applicable.
        spline_degree: Degree for bivariate spline when used.
        return_points: If True, return sampled points instead of a triangulated mesh.
        outer_weighting_exponent: Controls how much outer periwound faces dominate
            the surface fit. 0.0 = all faces weighted equally (default).
            > 0.0 = faces farther from the wound boundary get more influence.
            The exact mapping depends on `weighting_mode`.
        function_name: RBF kernel function name passed to scipy RBFInterpolator/Rbf.
            Options: 'gaussian', 'multiquadric', 'inverse_multiquadric',
            'thin_plate', 'linear', 'cubic', 'quintic'.
        weighting_mode: How to convert topological distance to per-face weight.
            Options:
            - "power" (default): weight = (d / max_d) ** exponent.
                Inner-boundary faces get weight ≈ 0 (clipped to 1e-6).
            - "linear_floor": weight = inner_weight + (1 - inner_weight) * (d / max_d).
                Inner-boundary faces get weight = inner_weight (default 0.5),
                outermost faces get weight = 1.0. The exponent parameter is ignored.
            - "sigmoid": weight = 1 / (1 + exp(-steepness * (d / max_d - midpoint))).
                Uses exponent as steepness (default 5.0) and inner_weight as midpoint
                (default 0.5). Creates an S-curve transition.
            - "linear": weight = d / max_d (same as power with exponent=1.0).
                Inner-boundary faces get weight ≈ 0.
        inner_weight: Minimum weight for inner-boundary faces when using
            "linear_floor" or "sigmoid" modes. Default 0.7 means inner-boundary
            faces are 70% as important as outer faces.

    Returns:
        (vertices, faces) mesh tuple or (points, None) if return_points=True. Returns None on failure.
    """
    # Validate periwound data
    periwound_faces = getattr(self, "current_periwound_faces", None)
    if not periwound_faces:
        print("fit_periwound_surface: no periwound faces available (run get_wound_mesh first)")
        return None

    periwound_faces = set(periwound_faces)
    wound_faces = set(getattr(self, "current_wound_faces", set()) or set())

    # Collect centroids and normals
    sorted_pw = sorted(periwound_faces)
    pts = np.asarray([self.face_objects[f].centroid for f in sorted_pw], dtype=np.float64)
    normals = np.asarray([self.face_objects[f].face_normal for f in sorted_pw if self.face_objects[f].face_normal is not None], dtype=np.float64)
    if pts.size == 0:
        print("fit_periwound_surface: no centroid points available")
        return None
    
    # ---- Compute per-face weights from topological distance to wound boundary ----
    weights = None
    if outer_weighting_exponent > 0.0:
        if wound_faces is not None and len(wound_faces) > 0:
            # Inner boundary: periwound faces that share an edge with a wound face
            inner_boundary = set()
            for f in periwound_faces:
                for adj in self.face_adjacency.get(f, []):
                    if adj in wound_faces:
                        inner_boundary.add(f)
                        break

            if inner_boundary:
                # BFS from inner boundary through periwound faces to compute topological distance
                distance = {f: 0 for f in inner_boundary}
                queue = deque(inner_boundary)
                while queue:
                    current = queue.popleft()
                    for adj in self.face_adjacency.get(current, []):
                        if adj in periwound_faces and adj not in distance:
                            distance[adj] = distance[current] + 1
                            queue.append(adj)

                # Convert distance to weight using the selected weighting mode
                max_dist = max(distance.values()) if distance else 1.0
                if max_dist > 0:
                    raw_d = np.array([
                        float(distance.get(f, 0)) / max_dist
                        for f in sorted_pw
                    ], dtype=np.float64)

                    if weighting_mode == "linear_floor":
                        # Linear from inner_weight at d=0 to 1.0 at d=1.0
                        # inner_boundary faces get weight = inner_weight (e.g. 0.5 = 50% importance)
                        inner_weight = float(np.clip(inner_weight, 0.01, 0.99))
                        weights = inner_weight + (1.0 - inner_weight) * raw_d
                        print(
                            f"  Outer weighting: mode='linear_floor', "
                            f"inner_weight={inner_weight:.2f}, "
                            f"inner_boundary_faces={len(inner_boundary)}, "
                            f"max_distance={max_dist}, "
                            f"weight_range=[{weights.min():.4f}, {weights.max():.4f}]"
                        )

                    elif weighting_mode == "sigmoid":
                        # S-curve: smooth transition from low to high weight
                        steepness = float(outer_weighting_exponent) if outer_weighting_exponent > 0 else 5.0
                        midpoint = float(np.clip(inner_weight, 0.1, 0.9))
                        weights = 1.0 / (1.0 + np.exp(-steepness * (raw_d - midpoint)))
                        # Normalize so max weight = 1.0
                        weights = weights / (weights.max() + 1e-9)
                        print(
                            f"  Outer weighting: mode='sigmoid', "
                            f"steepness={steepness:.1f}, midpoint={midpoint:.2f}, "
                            f"inner_boundary_faces={len(inner_boundary)}, "
                            f"max_distance={max_dist}, "
                            f"weight_range=[{weights.min():.4f}, {weights.max():.4f}]"
                        )

                    elif weighting_mode == "linear":
                        # Simple linear: weight = d / max_d
                        weights = raw_d.copy()
                        weights = np.clip(weights, 1e-6, 1.0)
                        print(
                            f"  Outer weighting: mode='linear', "
                            f"inner_boundary_faces={len(inner_boundary)}, "
                            f"max_distance={max_dist}, "
                            f"weight_range=[{weights.min():.4f}, {weights.max():.4f}]"
                        )

                    else:  # "power" (default)
                        # Power scaling: weight = (d / max_d) ** exponent
                        weights = raw_d ** outer_weighting_exponent
                        weights = np.clip(weights, 1e-6, 1.0)
                        print(
                            f"  Outer weighting: mode='power', "
                            f"exponent={outer_weighting_exponent}, "
                            f"inner_boundary_faces={len(inner_boundary)}, "
                            f"max_distance={max_dist}, "
                            f"weight_range=[{weights.min():.4f}, {weights.max():.4f}]"
                        )
            else:
                print("  Warning: no inner boundary faces found, using uniform weights")
        else:
            print("  Warning: no wound faces available for distance weighting, using uniform weights")

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

    effective_voxel_size, wound_diameter = _estimate_periwound_voxel_size(self,
        periwound_points=pts,
        plane_origin=origin,
        plane_u=u,
        plane_v=v,
        voxel_size=voxel_size,
    )
    if wound_diameter is not None:
        print(
            f"fit_periwound_surface: wound diameter ~ {wound_diameter:.4f}, "
            f"voxel_size -> {effective_voxel_size:.4f}"
        )

    # Optional voxelization/aggregation of points and normals
    if effective_voxel_size is not None and effective_voxel_size > 0.0:
        # voxelize in uv plane
        vox_idx = np.floor(uv / float(effective_voxel_size)).astype(np.int64)
        keys = [tuple(k) for k in vox_idx.tolist()]
        agg = {}
        for i, k in enumerate(keys):
            if k not in agg:
                agg[k] = {'uv': [], 'z': [], 'n': [], 'w': []}
            agg[k]['uv'].append(uv[i])
            agg[k]['z'].append(z[i])
            if i < len(normals):
                agg[k]['n'].append(normals[i])
            if weights is not None:
                agg[k]['w'].append(weights[i])

        uv_list = []
        z_list = []
        weights_list = []
        for k, vdata in agg.items():
            uv_list.append(np.mean(vdata['uv'], axis=0))
            z_list.append(np.mean(vdata['z'], axis=0))
            if vdata['w']:
                weights_list.append(np.mean(vdata['w'], axis=0))
        uv = np.asarray(uv_list, dtype=np.float64)
        z = np.asarray(z_list, dtype=np.float64)
        if weights_list:
            weights = np.asarray(weights_list, dtype=np.float64)
            print(f"  Voxelization aggregated weights: {len(weights)} points, range=[{weights.min():.4f}, {weights.max():.4f}]")

    # Choose fitting backend: try SciPy RBF / SmoothBivariateSpline
    fitted = None
    function_name = function_name
    try:
        if method == 'rbf':
            if weights is not None and outer_weighting_exponent > 0.0:
                # Use RBFInterpolator which supports per-point smoothing (weights)
                try:
                    from scipy.interpolate import RBFInterpolator
                    # Convert weights to smoothing: higher weight = less smoothing (trust more)
                    # Invert so outer faces (high weight) get low smoothing, inner faces get high smoothing
                    per_point_smoothing = float(smoothing) * (1.0 - weights) + 1e-9
                    rbf = RBFInterpolator(
                        uv, z[:, None],
                        kernel=function_name,
                        smoothing=per_point_smoothing,
                    )
                    def evaluator(xi, yi):
                        xi_arr = np.asarray(xi).ravel()
                        yi_arr = np.asarray(yi).ravel()
                        pts = np.column_stack([xi_arr, yi_arr])
                        return rbf(pts).ravel().reshape(xi.shape)
                    fitted = evaluator
                    print(f"  Using RBFInterpolator with per-point smoothing (weight range: [{weights.min():.4f}, {weights.max():.4f}])")
                except (ImportError, AttributeError):
                    # Fall back to legacy Rbf (no per-point weights)
                    from scipy.interpolate import Rbf
                    rbf = Rbf(uv[:, 0], uv[:, 1], z, function=function_name, smooth=float(smoothing))
                    def evaluator(xi, yi):
                        return rbf(xi, yi)
                    fitted = evaluator
                    print("  RBFInterpolator not available, using legacy Rbf (weights not applied)")
            else:
                from scipy.interpolate import Rbf
                rbf = Rbf(uv[:, 0], uv[:, 1], z, function=function_name, smooth=float(smoothing))
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
        # SciPy not available or failed -> fallback to weighted quadratic polynomial least-squares
        A = np.vstack([np.ones(len(uv)), uv[:, 0], uv[:, 1], uv[:, 0] ** 2, uv[:, 0] * uv[:, 1], uv[:, 1] ** 2]).T
        if weights is not None and outer_weighting_exponent > 0.0:
            # Weighted least-squares: minimize sum(w_i * (A_i @ coeffs - z_i)^2)
            W = np.diag(weights)
            coeffs = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ z, rcond=None)[0]
            print(f"  Using weighted polynomial least-squares")
        else:
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

    try:
        import rerun as rr
        rr.init('WoundModel3D_Assessment_final', spawn=True)
        rr.log(
            '/analysis/periwound_surface/original_samples',
            rr.Points3D(
                positions=np.ascontiguousarray(pts, dtype=np.float64),
                colors=np.tile(np.array([240, 160, 50], dtype=np.uint8)[None, :], (len(pts), 1)),
                radii=0.002,
            ),
            static=True,
        )
        rr.log(
            '/analysis/periwound_surface/surface_mesh',
            rr.Mesh3D(
                vertex_positions=np.ascontiguousarray(verts, dtype=np.float64),
                triangle_indices=np.ascontiguousarray(faces_idx, dtype=np.int32),
                vertex_colors=np.tile(np.array([240, 160, 50], dtype=np.uint8)[None, :], (len(verts), 1)),
            ),
            static=True,
        )
        print('Logged periwound surface to Rerun')
    except Exception as e:
        print(f'Rerun logging for periwound surface failed: {e}')

    return verts, faces_idx

def _estimate_periwound_voxel_size(self,
                                    periwound_points: np.ndarray,
                                    plane_origin: np.ndarray,
                                    plane_u: np.ndarray,
                                    plane_v: np.ndarray,
                                    voxel_size: Optional[float]) -> Tuple[float, Optional[float]]:
        """Estimate an aggregation voxel size from the wound boundary scale."""
        boundary_points = None

        if getattr(self, "vertices", None) is not None:
            if getattr(self, "wound_outerboundary_vertices", None):
                vertex_ids = sorted(v for v in self.wound_outerboundary_vertices if 0 <= int(v) < len(self.vertices))
                if vertex_ids:
                    boundary_points = np.asarray(self.vertices[vertex_ids], dtype=np.float64)

            if boundary_points is None and getattr(self, "boundary_vertices", None):
                vertex_ids = sorted(v for v in self.boundary_vertices if 0 <= int(v) < len(self.vertices))
                if vertex_ids:
                    boundary_points = np.asarray(self.vertices[vertex_ids], dtype=np.float64)

            if boundary_points is None and getattr(self, "wound_outerboundary_faces", None):
                face_ids = sorted(f for f in self.wound_outerboundary_faces if 0 <= int(f) < len(self.face_objects))
                if face_ids:
                    boundary_points = np.asarray([self.face_objects[f].centroid for f in face_ids], dtype=np.float64)

            if boundary_points is None and getattr(self, "wound_outerboundary_faces", None):
                face_ids = sorted(f for f in self.wound_outerboundary_faces if 0 <= int(f) < len(self.face_objects))
                if face_ids:
                    boundary_points = np.asarray([self.face_objects[f].centroid for f in face_ids], dtype=np.float64)

        if boundary_points is None and getattr(self, "current_periwound_faces", None):
            periwound_faces = set(self.current_periwound_faces)
            boundary_face_ids = [
                face_idx
                for face_idx in periwound_faces
                if any(adj not in periwound_faces for adj in self.face_adjacency.get(face_idx, []))
            ]
            if boundary_face_ids:
                boundary_points = np.asarray([self.face_objects[f].centroid for f in boundary_face_ids], dtype=np.float64)

        if boundary_points is None or len(boundary_points) < 2:
            boundary_points = periwound_points

        if boundary_points is None or len(boundary_points) < 2:
            base_voxel_size = float(voxel_size) if voxel_size is not None and voxel_size > 0.0 else 1.0
            return base_voxel_size, None

        rel = boundary_points - plane_origin[None, :]
        boundary_uv = np.vstack([rel @ plane_u, rel @ plane_v]).T
        uv_span = np.ptp(boundary_uv, axis=0)
        wound_diameter = float(np.linalg.norm(uv_span))

        if not np.isfinite(wound_diameter) or wound_diameter <= 0.0:
            base_voxel_size = float(voxel_size) if voxel_size is not None and voxel_size > 0.0 else 1.0
            return base_voxel_size, None

        base_voxel_size = float(voxel_size) if voxel_size is not None and voxel_size > 0.0 else 1.0
        target_voxels_across_wound = 32.0
        scaled_voxel_size = max((wound_diameter / target_voxels_across_wound) * base_voxel_size, 1e-6)
        return scaled_voxel_size, wound_diameter

  