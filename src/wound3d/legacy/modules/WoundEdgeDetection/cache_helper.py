"""Small utilities to save/load preprocessed WoundModel3DMesh state to disk.

This avoids re-running expensive preprocessing steps (mesh load, raycasts,
semantic classification) when you only change the downstream analysis code.

Usage:
  from wound3d.legacy.modules.WoundEdgeDetection.cache_helper import save_preprocessed, load_preprocessed
  save_preprocessed(wound_model, Path("cache/wound_model_cache.joblib"))
  wm = load_preprocessed(Path("cache/wound_model_cache.joblib"))
"""
from pathlib import Path
import sys
import traceback
import importlib
try:
    import joblib
except Exception:
    joblib = None
import numpy as np

# Ensure workspace root is importable when helper is used via direct scripts.
_workspace_root = Path(__file__).resolve().parents[3]
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))


def _dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if joblib is not None:
        joblib.dump(obj, str(path))
    else:
        # fallback to numpy savez for main arrays; but prefer joblib when available
        np.savez_compressed(str(path) + ".npz", **obj)


def _load(path: Path):
    if joblib is not None and path.exists():
        return joblib.load(str(path))
    npz = np.load(str(path) + ".npz", allow_pickle=True)
    return dict(npz)


def save_preprocessed(wound_model, out_path: Path):
    """Save essential preprocessed arrays and small python structures.

    Avoids serializing Open3D objects. We save numpy arrays and simple dicts
    needed to reconstruct a usable WoundModel3DMesh for analysis.
    """
    out_path = Path(out_path)
    try:
        data = {
            "vertices": wound_model.vertices,
            "faces": wound_model.faces,
            "vertex_colors": wound_model.vertex_colors,
            "vertex_normals": wound_model.vertex_normals,
            "face_normals": wound_model.face_normals,
            "face_adjacency": wound_model.face_adjacency,
            "edge_adjacency": wound_model.edge_adjacency,
            "spectators": {k: {"image_name": s.image_name, "mask_loaded": s.mask_loaded, "visible_face_indices": s.visible_face_indices} for k, s in wound_model.spectators.items()},
            "semantic_posteriors": wound_model.semantic_posteriors,
            "semantic_log_ratio_values": wound_model.semantic_log_ratio_values,
            "semantic_log_ratio_strength_values": wound_model.semantic_log_ratio_strength_values,
            "faces_processed": wound_model.faces_processed,
            "topology_computed": wound_model.topology_computed,
            "current_wound_faces": sorted(int(f) for f in getattr(wound_model, "current_wound_faces", set()) or set()),
            "current_periwound_faces": sorted(int(f) for f in getattr(wound_model, "current_periwound_faces", set()) or set()),
            "periwound_edge_faces": sorted(int(f) for f in getattr(wound_model, "periwound_edge_faces", set()) or set()),
            "wound_outerboundary_faces": sorted(int(f) for f in getattr(wound_model, "wound_outerboundary_faces", set()) or set()),
            "wound_outerboundary_edges": [tuple(int(v) for v in edge) for edge in getattr(wound_model, "wound_outerboundary_edges", []) or []],
            "wound_outerboundary_vertices": sorted(int(v) for v in getattr(wound_model, "wound_outerboundary_vertices", set()) or set()),
            "wound_outerboundary_bridge_edges": [tuple(int(v) for v in edge) for edge in getattr(wound_model, "wound_outerboundary_bridge_edges", []) or []],
            "wound_outerboundary_is_closed": bool(getattr(wound_model, "wound_outerboundary_is_closed", False)),
            "periwound_thickness": int(getattr(wound_model, "periwound_thickness", 0) or 0),
            "semantic_periwound_candidates": sorted(int(f) for f in getattr(wound_model, "semantic_periwound_candidates", set()) or set()),
            "periwound_boundary_normal_reference": getattr(wound_model, "periwound_boundary_normal_reference", None),
            "periwound_face_normal_reference": getattr(wound_model, "periwound_face_normal_reference", None),
            "periwound_boundary_alignment": getattr(wound_model, "periwound_boundary_alignment", None),
            "constrained_healthy_skin_surface_mesh": getattr(wound_model, "constrained_healthy_skin_surface_mesh", None),
            "constrained_healthy_skin_surface_frame": getattr(wound_model, "constrained_healthy_skin_surface_frame", None),
            "constrained_healthy_skin_surface_solution": getattr(wound_model, "constrained_healthy_skin_surface_solution", None),
            "constrained_healthy_skin_surface_data": getattr(wound_model, "constrained_healthy_skin_surface_data", None),
            "constrained_healthy_skin_surface_combined_mesh": getattr(wound_model, "constrained_healthy_skin_surface_combined_mesh", None),
            "constrained_healthy_skin_surface_volume": getattr(wound_model, "constrained_healthy_skin_surface_volume", None),
            "constrained_healthy_skin_surface_volume_components": getattr(wound_model, "constrained_healthy_skin_surface_volume_components", None),
        }

        # Save a compact version of face_objects (primitives only).
        # Using tuples avoids the overhead of serializing a dict for every face.
        fo_list = []
        for fo in wound_model.face_objects:
            fo_list.append((
                int(fo.face_index),
                tuple(int(v) for v in fo.vertex_indices),
                np.asarray(fo.centroid, dtype=np.float64),
                float(fo.r_wound or 0.0),
                float(fo.r_periwound or 0.0),
                int(fo.visibility_wound) if fo.visibility_wound is not None else 0,
                float(fo.wound_prob or 0.0),
                float(fo.periwound_prob or 0.0),
            ))
        data["face_objects_simple"] = fo_list

        _dump(data, out_path)
        print(f"Saved preprocessed data to: {out_path}")
    except Exception as e:
        print(f"Failed to save preprocessed data: {e}")
        traceback.print_exc()


def load_preprocessed(in_path: Path):
    """Load a cached preprocessed dict and reconstruct a minimal WoundModel3DMesh.

    Returns an instance of WoundModel3DMesh with arrays and simple fields filled.
    """
    mesh_mod = importlib.import_module("wound3d.legacy.modules.WoundEdgeDetection.Wound_Detection_Mesh")

    in_path = Path(in_path)
    if not in_path.exists() and not (str(in_path) + ".npz"):
        raise FileNotFoundError(f"Cache not found: {in_path}")

    try:
        data = _load(in_path)
    except Exception as e:
        print(f"Failed to load cached data: {e}")
        traceback.print_exc()
        raise

    wm = mesh_mod.WoundModel3DMesh()

    # Assign arrays and simple maps
    wm.vertices = data.get("vertices")
    wm.faces = data.get("faces")
    wm.vertex_colors = data.get("vertex_colors")
    wm.vertex_normals = data.get("vertex_normals")
    wm.face_normals = data.get("face_normals")
    wm.face_adjacency = data.get("face_adjacency", {})
    wm.edge_adjacency = data.get("edge_adjacency", {})
    wm.spectators = {}
    for k, v in (data.get("spectators") or {}).items():
        sp = mesh_mod.Spectator(image_id=-1, image_name=v.get("image_name", k), camera_id=-1)
        sp.mask_loaded = bool(v.get("mask_loaded", False))
        sp.visible_face_indices = list(v.get("visible_face_indices") or [])
        wm.spectators[k] = sp

    wm.semantic_posteriors = data.get("semantic_posteriors", {}) or {}
    wm.semantic_log_ratio_values = data.get("semantic_log_ratio_values", {}) or {}
    wm.semantic_log_ratio_strength_values = data.get("semantic_log_ratio_strength_values", {}) or {}
    wm.faces_processed = bool(data.get("faces_processed", False))
    wm.topology_computed = bool(data.get("topology_computed", False))

    # Rebuild face_objects from compact representation.
    # This is the heaviest step during cache loading, so keep it lightweight
    # and avoid the full dataclass constructor path when possible.
    fo_simple = data.get("face_objects_simple", []) or []
    face_objects = []
    face_adj = wm.face_adjacency if wm.face_adjacency else {}
    face_normals = wm.face_normals
    for fo in fo_simple:
        if isinstance(fo, dict):
            face_idx = int(fo.get("face_index"))
            vertex_indices = tuple(int(v) for v in (fo.get("vertex_indices") or ()))
            centroid = np.asarray(fo.get("centroid"), dtype=np.float64)
            r_wound = float(fo.get("r_wound", 0.0))
            r_periwound = float(fo.get("r_periwound", 0.0))
            visibility_wound = int(fo.get("visibility_wound", 0))
            wound_prob = float(fo.get("wound_prob", 0.0))
            periwound_prob = float(fo.get("periwound_prob", 0.0))
        else:
            face_idx, vertex_indices, centroid, r_wound, r_periwound, visibility_wound, wound_prob, periwound_prob = fo
            vertex_indices = tuple(int(v) for v in (vertex_indices or ()))
            centroid = np.asarray(centroid, dtype=np.float64)

        face_normal = None
        if face_normals is not None and 0 <= face_idx < len(face_normals):
            face_normal = face_normals[face_idx]

        neighbor_face_indices = set(face_adj.get(face_idx, [])) if face_adj else set()

        face_obj = mesh_mod.Faces.__new__(mesh_mod.Faces)
        face_obj.face_index = face_idx
        face_obj.vertex_indices = vertex_indices
        face_obj.centroid = centroid
        face_obj.face_normal = face_normal
        face_obj.color_vector = None
        face_obj.neighbor_face_indices = neighbor_face_indices
        face_obj.visibility_wound = None
        face_obj.visibility_periwound = None
        face_obj.visibility_tissue = None
        face_obj.r_wound = r_wound
        face_obj.r_periwound = r_periwound
        face_obj.r_tissue = None
        face_obj.wound_prob = wound_prob
        face_obj.periwound_prob = periwound_prob
        face_obj.posterior_log_ratio = None
        face_obj.posterior_log_ratio_strength = None
        face_obj.label = None
        face_obj.confidence = None
        face_obj.visibility_wound = visibility_wound if visibility_wound is not None else None

        face_objects.append(face_obj)

    wm.face_objects = face_objects

    wm.current_wound_faces = set(int(f) for f in data.get("current_wound_faces", []) or [])
    wm.current_periwound_faces = set(int(f) for f in data.get("current_periwound_faces", []) or [])
    wm.periwound_edge_faces = set(int(f) for f in data.get("periwound_edge_faces", []) or [])
    wm.wound_outerboundary_faces = set(int(f) for f in data.get("wound_outerboundary_faces", []) or [])
    wm.wound_outerboundary_edges = [tuple(int(v) for v in edge) for edge in data.get("wound_outerboundary_edges", []) or []]
    wm.wound_outerboundary_vertices = set(int(v) for v in data.get("wound_outerboundary_vertices", []) or [])
    wm.wound_outerboundary_bridge_edges = [tuple(int(v) for v in edge) for edge in data.get("wound_outerboundary_bridge_edges", []) or []]
    wm.wound_outerboundary_is_closed = bool(data.get("wound_outerboundary_is_closed", False))
    if not wm.wound_outerboundary_edges and wm.wound_outerboundary_faces and wm.edge_adjacency and wm.faces is not None:
        wound_faces_set = set(int(f) for f in wm.wound_outerboundary_faces)
        periwound_faces_set = set(int(f) for f in getattr(wm, "current_periwound_faces", set()) or set())
        reconstructed_edges = []
        reconstructed_vertices = set()
        for edge, face_list in wm.edge_adjacency.items():
            wound_count = sum(1 for f in face_list if f in wound_faces_set)
            periwound_count = sum(1 for f in face_list if f in periwound_faces_set)
            if wound_count > 0 and periwound_count > 0:
                edge_tuple = tuple(int(v) for v in edge)
                reconstructed_edges.append(edge_tuple)
                reconstructed_vertices.update(edge_tuple)
        wm.wound_outerboundary_edges = reconstructed_edges
        wm.wound_outerboundary_vertices = reconstructed_vertices

    wm.periwound_thickness = int(data.get("periwound_thickness", 0) or 0)
    wm.semantic_periwound_candidates = set(int(f) for f in data.get("semantic_periwound_candidates", []) or [])
    wm.periwound_boundary_normal_reference = data.get("periwound_boundary_normal_reference", None)
    wm.periwound_face_normal_reference = data.get("periwound_face_normal_reference", None)
    wm.periwound_boundary_alignment = data.get("periwound_boundary_alignment", None)
    wm.constrained_healthy_skin_surface_mesh = data.get("constrained_healthy_skin_surface_mesh", None)
    wm.constrained_healthy_skin_surface_frame = data.get("constrained_healthy_skin_surface_frame", None)
    wm.constrained_healthy_skin_surface_solution = data.get("constrained_healthy_skin_surface_solution", None)
    wm.constrained_healthy_skin_surface_data = data.get("constrained_healthy_skin_surface_data", None)
    wm.constrained_healthy_skin_surface_combined_mesh = data.get("constrained_healthy_skin_surface_combined_mesh", None)
    wm.constrained_healthy_skin_surface_volume = data.get("constrained_healthy_skin_surface_volume", None)
    wm.constrained_healthy_skin_surface_volume_components = data.get("constrained_healthy_skin_surface_volume_components", None)
    wm.constrained_healthy_skin_surface_volume_mesh = None

    print(f"Reconstructed WoundModel3DMesh from cache: {in_path}")
    return wm
