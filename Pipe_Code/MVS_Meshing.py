import os
import subprocess
import time
import threading
import shutil
import trimesh
from pathlib import Path
from run_command import run_command
from Pipe_Code.modules.masking.mask_undistortion import undistort_masks_with_colmap

def execute_poisson_meshing(folder_path, input_ply=None):
    dataset_path = os.path.abspath(folder_path)
    dense_path = os.path.join(dataset_path, 'dense')
    
    # Check if dense directory already exists
    if not os.path.exists(dense_path):
        print("Dense reconstruction does not exist, skipping...")
        return
    
    start_time = time.time()
    
    try:
        # Run poisson meshing
        input_path = input_ply if input_ply is not None else os.path.join(dense_path, 'fused.ply')
        run_command(f"colmap poisson_mesher --input_path '{input_path}' --output_path '{os.path.join(dense_path, 'poisson.ply')}' --PoissonMeshing.trim 4")
    except Exception as e:
        print("Poisson meshing failed: ", e)
        return
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return elapsed_time
    # Log the execution time and folder path
    # with open("mesh_elapse_time.txt", "a") as log_file:
    #     log_file.write(f"{folder_path}: {elapsed_time} seconds\n")

def execute_dense_reconstruction(folder_path):
    dataset_path = os.path.abspath(folder_path)
    dense_path = os.path.join(dataset_path, 'dense')
    
    # Check if dense directory already exists
    # if os.path.exists(dense_path):
    #     print("Dense reconstruction already exists, skipping...")
    #     return
    
    # Create dense directory
    os.makedirs(dense_path, exist_ok=True)
    
    start_time = time.time()
    
    try:
        # Run image undistorter
        run_command(f"colmap image_undistorter --image_path '{os.path.join(dataset_path, 'sharp_images')}' --input_path '{os.path.join(dataset_path, 'sparse', '0')}' --output_path '{dense_path}' --output_type COLMAP")

        # Undistort masks before patch match stereo
        sparse_model_dir = Path(dataset_path) / "sparse" / "0"
        mask_candidates = [
            "mask",
            "masks",
            "mask_wound_only",
            "masks_wound_only",
            "mask_extruded_wound_only",
            "masks_extruded_wound_only",
        ]
        undistorted_mask_paths = {}
        for mask_name in mask_candidates:
            mask_dir = Path(folder_path) / mask_name
            if not mask_dir.exists():
                continue
            output_dir = Path(dense_path) / f"{mask_name}_undistorted"
            undistort_masks_with_colmap(
                mask_dir=mask_dir,
                sparse_model_dir=sparse_model_dir,
                output_dir=output_dir,
                colmap_exe="colmap",
                output_type="COLMAP",
                max_image_size=None,
            )
            _flatten_undistorted_masks(output_dir)
            undistorted_mask_paths[mask_name] = output_dir
    
        # Run patch match stereo
        run_command(f"colmap patch_match_stereo --workspace_path '{dense_path}' --workspace_format COLMAP --PatchMatchStereo.geom_consistency true")
        
        # Run stereo fusion
        mask_path = os.path.join(folder_path, "masks")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(folder_path, "mask")
        if os.path.exists(mask_path):
            # Prefer undistorted masks if they were created
            key = "masks" if mask_path.endswith("masks") else "mask"
            mask_path = undistorted_mask_paths.get(key, mask_path)
            mask_path = os.fspath(mask_path)
            run_command(f"colmap stereo_fusion --workspace_path '{dense_path}' --workspace_format COLMAP --input_type geometric --output_path '{os.path.join(dense_path)}' --output_type BIN --StereoFusion.mask_path '{mask_path}'")
            run_command(f"colmap stereo_fusion --workspace_path '{dense_path}' --workspace_format COLMAP --input_type geometric --output_path '{os.path.join(dense_path, 'fused.ply')}' --output_type PLY --StereoFusion.mask_path '{mask_path}'")        
        else:
            run_command(f"colmap stereo_fusion --workspace_path '{dense_path}' --workspace_format COLMAP --input_type geometric --output_path '{os.path.join(dense_path)}' --output_type BIN")
            run_command(f"colmap stereo_fusion --workspace_path '{dense_path}' --workspace_format COLMAP --input_type geometric --output_path '{os.path.join(dense_path, 'fused.ply')}' --output_type PLY")        
        
    except Exception as e:
        print("Reconstruction failed")
        return
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return elapsed_time 


def delete_dense_folders(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if 'dense' in dirnames:
            dense_folder_path = os.path.join(dirpath, 'dense')
            print(f"Deleting folder: {dense_folder_path}")
            shutil.rmtree(dense_folder_path)


def _flatten_undistorted_masks(output_dir: Path) -> None:
    """
    Keep only undistorted masks in output_dir and remove COLMAP artifacts.
    Note: With the new workflow, masks should already be in output_dir with .png extension.
    This function now just cleans up any remaining subdirectories.
    """
    output_dir = Path(output_dir)

    # Clean up any subdirectories that might have been created
    for item in output_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        elif item.is_file() and item.suffix.lower() != ".png":
            # Remove non-png files (shouldn't exist, but clean up if present)
            item.unlink()

if __name__ == "__main__":
    # Execute dense reconstruction for every folder in the Experiments directory
    experiments_path = r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\SALVE_Trials"
    execute_poisson_meshing(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\SALVE_Trials\SD_2")
    # for folder_name in os.listdir(experiments_path):
    #     folder_path = os.path.join(experiments_path, folder_name)
    #     if os.path.isdir(folder_path):
    #         print(f"Processing folder: {folder_path}")
    #         try:
    #             execute_poisson_meshing(folder_path)
    #         except ValueError as e:
    #             print(f"Error processing {folder_path}: {e}")

