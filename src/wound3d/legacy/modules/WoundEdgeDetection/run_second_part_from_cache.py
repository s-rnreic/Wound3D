"""Load cached preprocessed data and run split second-stage analysis.

This script supports a two-step workflow:

1. Load the heavy preprocessing cache, run wound refinement only until
   `_segment_periwound_faces(...)`, then save a new cache containing
   `current_wound_faces` and `current_periwound_faces`.
2. Later load that periwound-ready cache and run
   `fit_constrained_healthy_skin_surface(...)` without repeating wound/periwound
   segmentation.
"""
from pathlib import Path
import sys
import numpy as np

# Ensure workspace root is importable when launched as a direct script.
_workspace_root = Path(__file__).resolve().parents[3]
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from wound3d.legacy.modules.WoundEdgeDetection.cache_helper import load_preprocessed, save_preprocessed


DEFAULT_INPUT_CACHE = Path("C:/Users/Reich/LocalCoding/COLMAP_Pipline/cache/wound_model_cache_Ferse.joblib")
DEFAULT_PERIWOUND_CACHE = Path("C:/Users/Reich/LocalCoding/COLMAP_Pipline/cache/wound_model_cache_Ferse_periwound.joblib")


def run_until_periwound_and_cache(
    input_cache_path: Path = DEFAULT_INPUT_CACHE,
    output_cache_path: Path = DEFAULT_PERIWOUND_CACHE,
    min_component_faces: int = 100,
    closing_iterations: int = 5,
    closing_kernel_size: int = 2,
    boundary_growth_max_iterations: int = 25,
    periwound_thickness: int = 30,
    periwound_use_semantic_candidates: bool = True,
    visualization: bool = True,
):
    """Run `get_wound_mesh` logic only up to `_segment_periwound_faces` and cache it.

    This intentionally stops before `fit_constrained_healthy_skin_surface(...)` so
    healthy-skin fitting can be developed/re-run later from `output_cache_path`.
    """
    wm = load_preprocessed(input_cache_path)

    print("=== Running second part until periwound segmentation ===")
    if not wm.faces_processed:
        raise ValueError("Cached model is not ready: `faces_processed` is False")

    selected_faces = wm._select_wound_faces()
    filtered_faces = wm._refine_wound_faces(
        face_set=selected_faces,
        min_component_faces=min_component_faces,
        closing_iterations=closing_iterations,
        closing_kernel_size=closing_kernel_size,
        boundary_growth_max_iterations=boundary_growth_max_iterations,
    )

    #wm.log_wound(filtered_faces, stage_name="refined")
    wm.current_wound_faces = set(filtered_faces)
    wm.periwound_thickness = int(periwound_thickness)

    periwound_faces = wm._segment_periwound_faces(
        wound_faces=filtered_faces,
        thickness=periwound_thickness,
        use_semantic_candidates=periwound_use_semantic_candidates,
    )
    wm.current_periwound_faces = set(periwound_faces)
    wm.periwound_edge_faces = set(periwound_faces)
    #wm.log_wound(periwound_faces, stage_name="periwound_region")

    wm.compute_wound_edge(filtered_faces, periwound_faces=periwound_faces)
    #wm.log_wound(wm.wound_outerboundary_faces, stage_name="wound_outerboundary")
    
    
    wound_vertices, wound_faces, wound_colors = wm.extract_submesh_from_faces(filtered_faces)
    #wm.save_wound_mesh(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse_Phantom_arUco\dense\wound.ply", wound_vertices, wound_faces, wound_colors)
    _visualize_wound_periwound_overlays(wm)
    
    print(
        f"Periwound segmentation complete: wound={len(filtered_faces)} faces, "
        f"periwound={len(periwound_faces)} faces, thickness={wm.periwound_thickness}, "
        f"semantic_support={periwound_use_semantic_candidates}"
    )

    print(f"Saving periwound-ready cache to: {output_cache_path}")
    save_preprocessed(wm, output_cache_path)

    # if visualization:
    #     visualize_from_model(wm)

    return wm


def run_fit_from_periwound_cache(
    periwound_cache_path: Path = DEFAULT_PERIWOUND_CACHE,
    output_cache_path: Path | None = None,
    alpha: float = 0.1,
    beta: float = 0.1,
    gamma: float = 0.8,
    periwound_data_weight: float = 1.0,
    use_cvxpy: bool = False,
    visualization: bool = True,
):
    """Load a periwound-ready cache and run healthy-skin surface fitting."""
    wm = load_preprocessed(periwound_cache_path)
    
    healthy_skin_surface = None

    print("=== Running healthy-skin surface fit from periwound cache ===")
    healthy_skin_surface = wm.fit_constrained_healthy_skin_surface(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        periwound_data_weight=periwound_data_weight,
        use_cvxpy=use_cvxpy,
    )

    if healthy_skin_surface is not None:
        print("Healthy-skin reference surface reconstructed from cached wound/periwound labels")
        volume = getattr(wm, "constrained_healthy_skin_surface_volume", None)
        if volume is not None:
            print(f"Largest enclosed volume: {float(volume):.6f}")
        if visualization:
            wm.visualize_healthy_skin_surface(session_path="/analysis/healthy_skin_surface", shading_mode="phong")
            wm.visualize_healthy_skin_surface_volumes(session_path="/analysis/healthy_skin_surface_volumes", visualize_all=True, shading_mode="phong")
            wm.visualize_healthy_skin_surface_shell_parts(session_path="/analysis/healthy_skin_surface_shell_parts", shading_mode="phong")

    # if visualization:
    #     _visualize_wound_periwound_overlays(wm)

    # if output_cache_path is not None:
    #     print(f"Saving healthy-skin-fit cache to: {output_cache_path}")
    #     save_preprocessed(wm, output_cache_path)

    return wm


def visualize_from_model(wm, session_name: str = "WoundModel3D_Assessment"):
    """Run the existing Rerun visualization for an already loaded model."""
    try:
        from wound3d.legacy.modules.WoundEdgeDetection.utils.rerun_visualization import (
            visualize_wound_assessment_rerun,
        )

        visualize_wound_assessment_rerun(
            wm,
            session_name=session_name,
            spawn=True,
            image_root=Path.cwd(),
        )
    except Exception as e:
        print(f"Visualization failed: {e}")


def _visualize_wound_periwound_overlays(wm, session_name: str = "WoundModel3D_Assessment", spawn: bool = True):
    """Log wound, periwound, and wound-edge overlays to Rerun."""
    try:
        import rerun as rr
    except Exception as e:
        print(f"Visualization failed: {e}")
        return

    vertices = np.asarray(wm.vertices, dtype=np.float64)
    faces = np.asarray(wm.faces, dtype=np.int32)

    def _to_uint8_colors(colors, expected_length: int, fallback_color: tuple[int, int, int]) -> np.ndarray:
        if colors is None:
            return np.full((expected_length, 3), fallback_color, dtype=np.uint8)

        color_array = np.asarray(colors)
        if len(color_array) != expected_length:
            return np.full((expected_length, 3), fallback_color, dtype=np.uint8)

        if color_array.dtype.kind in "fc" and color_array.size > 0 and float(np.nanmax(color_array)) <= 1.0:
            color_array = color_array * 255.0
        return np.clip(color_array, 0.0, 255.0).astype(np.uint8)

    def _log_submesh(path: str, face_indices, fallback_color: tuple[int, int, int]) -> None:
        face_set = set(int(face_idx) for face_idx in (face_indices or []))
        if not face_set:
            return

        sub_vertices, sub_faces, sub_colors = wm.extract_submesh_from_faces(face_set)
        if sub_vertices is None or len(sub_vertices) == 0 or sub_faces is None or len(sub_faces) == 0:
            return

        rr.log(
            path,
            rr.Mesh3D(
                vertex_positions=np.ascontiguousarray(sub_vertices, dtype=np.float64),
                triangle_indices=np.ascontiguousarray(sub_faces, dtype=np.int32),
                vertex_colors=np.ascontiguousarray(
                    _to_uint8_colors(sub_colors, len(sub_vertices), fallback_color),
                    dtype=np.uint8,
                ),
            ),
            static=True,
        )

    rr.init(session_name, spawn=spawn)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    base_vertex_colors = _to_uint8_colors(getattr(wm, "vertex_colors", None), len(vertices), (140, 140, 140))
    rr.log(
        "/analysis/base_mesh",
        rr.Mesh3D(
            vertex_positions=np.ascontiguousarray(vertices, dtype=np.float64),
            triangle_indices=np.ascontiguousarray(faces, dtype=np.int32),
            vertex_colors=np.ascontiguousarray(base_vertex_colors, dtype=np.uint8),
        ),
        static=True,
    )

    _log_submesh("/analysis/wound", getattr(wm, "current_wound_faces", set()), (235, 40, 40))
    _log_submesh("/analysis/periwound", getattr(wm, "current_periwound_faces", set()), (40, 140, 235))
    _log_submesh("/analysis/wound_outerboundary_faces", getattr(wm, "wound_outerboundary_faces", set()), (255, 120, 0))

    edge_vertices = getattr(wm, "wound_outerboundary_vertices", None)
    if edge_vertices:
        edge_vertex_ids = sorted(int(vertex_id) for vertex_id in edge_vertices)
        if edge_vertex_ids:
            boundary_points = vertices[np.asarray(edge_vertex_ids, dtype=np.int64)]
            rr.log(
                "/analysis/wound_outerboundary_vertices",
                rr.Points3D(
                    positions=np.ascontiguousarray(boundary_points, dtype=np.float64),
                    colors=np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (len(boundary_points), 1)),
                    radii=0.003,
                ),
                static=True,
            )

    boundary_edges = list(getattr(wm, "wound_outerboundary_edges", []) or [])
    if boundary_edges:
        strip_vertex_ids = [int(boundary_edges[0][0])]
        strip_vertex_ids.extend(int(b) for _, b in boundary_edges)
        strip_vertex_ids = [vid for vid in strip_vertex_ids if 0 <= vid < len(vertices)]
        if len(strip_vertex_ids) >= 2:
            strip_points = vertices[np.asarray(strip_vertex_ids, dtype=np.int64)]
            if len(strip_points) > 0 and not np.allclose(strip_points[0], strip_points[-1]):
                strip_points = np.vstack([strip_points, strip_points[0]])
            rr.log(
                "/analysis/wound_outerboundary_edges",
                rr.LineStrips3D([np.asarray(strip_points, dtype=np.float64)], colors=[255, 180, 20]),
                static=True,
            )

def run_from_cache(cache_path: Path = Path("cache/wound_model_cache.joblib"), visualization: bool = True):
    """Legacy one-shot path: runs the original full `get_wound_mesh(...)`."""
    wm = load_preprocessed(cache_path)

    print("=== Running second part: Extracting wound mesh ===")
    wm.get_wound_mesh(
            min_component_faces=100,  # Minimum size for valid component
            closing_iterations=5,  # Number of closing iterations
            closing_kernel_size=2,  # Number of adjacency layers per iteration (1-3 typical)
        )


    if visualization:
        visualize_from_model(wm)


if __name__ == "__main__":
    # Step 1: create a cache that stops after `_segment_periwound_faces(...)`.
    DEFAULT_INPUT_CACHE = Path("C:\\Users\\Reich\\LocalCoding\\COLMAP_Pipline\\cache\\wound_model_cache_TestModel1.joblib")
    DEFAULT_PERIWOUND_CACHE = Path("C:/Users/Reich/LocalCoding/COLMAP_Pipline/cache/wound_model_cache_TestModel1_periwound.joblib")

    # run_until_periwound_and_cache(
    #     input_cache_path=DEFAULT_INPUT_CACHE,
    #     output_cache_path=DEFAULT_PERIWOUND_CACHE,
    # )

    # Step 2, later when desired: comment Step 1 and run this from the saved cache.
    run_fit_from_periwound_cache(
        periwound_cache_path=DEFAULT_PERIWOUND_CACHE,
        #output_cache_path=Path("C:/Users/Reich/LocalCoding/COLMAP_Pipline/cache/wound_model_cache_Fuss_healthy_fit.joblib"),
    )
