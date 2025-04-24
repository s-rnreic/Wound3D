import os
import subprocess
import time
import threading
import shutil
import trimesh
from run_command import run_command

def execute_poisson_meshing(folder_path):
    dataset_path = os.path.abspath(folder_path)
    dense_path = os.path.join(dataset_path, 'dense')
    
    # Check if dense directory already exists
    if not os.path.exists(dense_path):
        print("Dense reconstruction does not exist, skipping...")
        return
    
    start_time = time.time()
    
    try:
        # Run poisson meshing
        run_command(f"colmap poisson_mesher --input_path '{os.path.join(dense_path, 'fused.ply')}' --output_path '{os.path.join(dense_path, 'poisson.ply')}' --PoissonMeshing.trim 7")
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
    if os.path.exists(dense_path):
        print("Dense reconstruction already exists, skipping...")
        return
    
    # Create dense directory
    os.makedirs(dense_path, exist_ok=True)
    
    start_time = time.time()
    
    try:
        # Run image undistorter
        run_command(f"colmap image_undistorter --image_path '{os.path.join(dataset_path, 'sharp_images')}' --input_path '{os.path.join(dataset_path, 'sparse', '0')}' --output_path '{dense_path}' --output_type COLMAP")
        
        # Run patch match stereo
        run_command(f"colmap patch_match_stereo --workspace_path '{dense_path}' --workspace_format COLMAP --PatchMatchStereo.geom_consistency true")
        
        # Run stereo fusion
        mask_path = os.path.join(folder_path, "masks")
        if os.path.exists(mask_path):
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
    # # Log the execution time and folder path
    # with open("dense_elapse_time.txt", "a") as log_file:
    #     log_file.write(f"{folder_path}: {elapsed_time} seconds\n")

def delete_dense_folders(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if 'dense' in dirnames:
            dense_folder_path = os.path.join(dirpath, 'dense')
            print(f"Deleting folder: {dense_folder_path}")
            shutil.rmtree(dense_folder_path)

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

