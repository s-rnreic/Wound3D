"""Run the heavy first part of the pipeline and cache results.

Run this once and then edit the second-part scripts freely; use
`run_second_part_from_cache.py` to run the analysis repeatedly without
recomputing the heavy steps.
"""
from pathlib import Path
import sys
import importlib

# Ensure workspace root is importable when launched as a direct script.
_workspace_root = Path(__file__).resolve().parents[3]
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from wound3d.legacy.modules.WoundEdgeDetection.cache_helper import save_preprocessed

def run_and_cache(cache_path: Path = Path("cache/wound_model_cache_TestModel2.joblib")):
    mesh_mod = importlib.import_module("wound3d.legacy.modules.WoundEdgeDetection.Wound_Detection_Mesh")
    # Paths (same defaults as the example main)
    mesh_path = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\WoundModelTrials\V015\Roman\dense\poisson.ply")
    model_path = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\WoundModelTrials\V015\Roman\dense")
    mask_dir = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\WoundModelTrials\V015\Roman\dense\masks_wound_only_undistorted")
    extruded_mask_dir = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\WoundModelTrials\V015\Roman\dense\masks_extruded_wound_only_undistorted")

    

    mesh_path = mesh_mod.MeshReduction(mesh_path, model_path, mask_dir, extruded_mask_dir) #Creates a periwound-trimmed mesh based on mask projection. Basically removes everything outside the periwound outer boundary. It then simplifies the mesh to reduce face count while preserving detail.
    wound_model = mesh_mod.WoundModel3DMesh()
    
    wound_model.load_mesh(mesh_path, compute_normals=True)
    # Load COLMAP cameras
    print("\n=== Loading COLMAP Cameras ===")
    wound_model.load_colmap_cameras(model_path)
    
    # Load wound masks
    print("\n=== Loading Wound Masks ===")
    wound_model.load_masks(mask_dir)
    
    # Load extruded masks for periwound detection
    print("\n=== Loading Extruded Masks ===")
    wound_model.load_extruded_masks(extruded_mask_dir)
    
    # Process wound segmentation
    print("\n=== Processing Wound Segmentation ===")
    wound_model.process_wound_segmentation()
    
    # Detect initial periwound faces from mask difference
    wound_model.process_periwound_face_detection()
    #Run semantic classification (feature-based, 2-class: periwound, wound)
    print("\n=== Semantic Classification (Gaussian, MRF optional) ===")
    wound_model.run_semantic_classification(
        r_wound_thresh=0.3,
        r_periwound_thresh=0.5,
        ring_sizes=[1, 3],
        confidence_method="entropy",
        lambda_smooth=1.0,
        use_ratio_features=False,
        use_centroid_features=True,
        use_height_deviation=True,
        use_normal_deviation=False,
        use_curvature=True,
        use_mrf=True,  # Set to True to enable MRF smoothing
        mrf_max_iters=10,
        color_space="rgb",
        class_prior_mode="sample_proportional",
        mrf_method="mean_field"
    )

       
      

    print(f"Saving cache to: {cache_path}")
    save_preprocessed(wound_model, cache_path)


if __name__ == "__main__":
    run_and_cache()
