import sys
import os
import prepare_vid_for_colmap_v5
import create_temp_project_ini as ctpi
import create_temp_glomap_ini as ctgi
import sfm_database
import subprocess
import tkinter as tk
from tkinter import filedialog, simpledialog  # Add simpledialog for user input
import build_masks as masking
import rerun_viewer
import time  # Add import for time module
from read_write_model import write_model
import MVS_Meshing as mvs
import SfM as sfm
from pathlib import Path
# user inserts a input_path and a workspace_path
# workspace_path : path to a folder where all images (raw,sharp), the colmap data will be saved
#
# structure of the workspace_path (goal):
# -- workspace_path
#   |_ raw_images
#   |_ sharp_images
#   |_ sparse
#   |_ dense
#   |_ database.db
#   |_ project.ini 
#

def log_time(log_file, step, elapsed_time):
    if elapsed_time is not None:
        with open(log_file, 'a') as f:
            f.write(f"{step}: {elapsed_time:.2f} seconds\n")
    else:
        with open(log_file, 'a') as f:
            f.write(f"{step}: elapsed time is None\n")

def calculate_total_elapsed_time(log_file):
    total_time = 0.0
    with open(log_file, 'r') as f:
        for line in f:
            time_str = line.split(':')[-1].strip().split()[0]
            total_time += float(time_str)
    return total_time

def create_workspace(workspace_path:str)->None:
    global raw_images_path, sharp_image_path, image_mask_path
    os.makedirs(workspace_path, exist_ok=True)
    raw_images_path = os.path.join(workspace_path, 'raw_images')
    os.makedirs(raw_images_path, exist_ok=True)
    sharp_image_path = os.path.join(workspace_path, 'sharp_images')
    os.makedirs(sharp_image_path, exist_ok=True)
    image_mask_path=os.path.join(workspace_path, 'masks')



#1. Analyse Video and extract image
def Video_Pipeline( input_path:str, image_output_path:str, log_file:str, segments:float, image_options:dict , Masking:bool=True)->str:
    start_time = time.time()
    prepare_vid_for_colmap_v5.get_IMG_from_Video(input_path, raw_images_path)
    
    sharpness_values = prepare_vid_for_colmap_v5.get_image_sharpness(raw_images_path, image_options)
    selected_indices = prepare_vid_for_colmap_v5.select_sharp_images(sharpness_values, segments, min_sharpness_percentile=50)
    prepare_vid_for_colmap_v5.copy_selected_images(selected_indices, raw_images_path, sharp_image_path, image_options)
    if Masking:
        masking.process_folder(sharp_image_path, image_mask_path)
    elapsed_time = time.time() - start_time
    log_time(log_file, "Video_Pipeline", elapsed_time)
    return sharp_image_path, image_mask_path

# 2. COLMAP Pipeline

def SfM_Pipeline(image_output_path: str, workspace_path: str, image_mask_path: str, log_file: str) -> str:
    start_time = time.time()
    global sparse_path
    total_ini_path = os.path.join(workspace_path, 'total_project.ini')
    # Version of PIPLINE which doesnt work with a bat file anymore
    # Each colmap command is executed with a tempProject.ini file --> file contains the parameter of each colmap command
    # construct the command to execute the batch file for creating a .ini file
    
    #Feature Extraction and Matching
    features, matches, pairs = sfm.LightGlue(Path(image_output_path), Path(workspace_path))
    
    #Create Database and import features and matches
    sfm.Stream_into_db(Path(workspace_path), Path(image_output_path), pairs=pairs, features=features, matches=matches)
    
    #Global Mapping trough the GLOMAP pipeline
    sparse_path = os.path.join(workspace_path, 'sparse')
    os.makedirs(sparse_path, exist_ok=True)
    ctgi.create_glomap_ini(workspace_path + '\\glomap_config.ini')
    #open_ini_in_editor(workspace_path + '\\glomap_config.ini')
    ctgi.execute_glomap(workspace_path + '\\glomap_config.ini', database_path, sparse_path, sharp_image_path)
    ctpi.append_to_total_project_ini(workspace_path + '\\glomap_config.ini', total_ini_path)
    elapsed_time = time.time() - start_time
    log_time(log_file, "SfM_Pipeline", elapsed_time)

    
    # #3. COLMAP RERUN Streaming
def COLMAP_RERUN(workspace_path:str, log_file:str) -> None:
    start_time = time.time()
    cameras, images, filtered_points3D=rerun_viewer.Rerun_Colmap_Pipeline(workspace_path)
    elapsed_time = time.time() - start_time
    log_time(log_file, "COLMAP_RERUN", elapsed_time)
    return cameras, images, filtered_points3D

def ANALYSE_COLMAP_OUTPUT(workspace_path: str, sparse_path:str , log_file: str = None) -> None:
    start_time = time.time()
    # Analyse the output of the COLMAP pipeline
    # Analyse the sparse folder
    subfolders = [os.path.join(sparse_path, subfolder) for subfolder in os.listdir(sparse_path) if os.path.isdir(os.path.join(sparse_path, subfolder))]
    
    for subfolder in subfolders:
        result = subprocess.run(f"colmap model_analyzer --path {subfolder}", shell=True, capture_output=True, text=True)
        print(result.stderr)
    
    elapsed_time = time.time() - start_time
    if log_file and os.path.exists(log_file):
        log_time(log_file, "ANALYSE_COLMAP_OUTPUT", elapsed_time)

def MVS(workspace_path:str, log_file:str)->None:
    
    elapse_time= mvs.execute_dense_reconstruction(workspace_path)
    log_time(log_file, "MVS", elapse_time)


def Meshing(workspace_path:str, log_file:str)->None:
    elapse_time= mvs.execute_poisson_meshing(workspace_path)
    log_time(log_file, "Meshing", elapse_time)

def open_ini_in_editor(ini_path: str) -> None:
    editor = os.getenv('EDITOR', 'notepad' if os.name == 'nt' else 'nano')
    subprocess.call([editor, ini_path])
       

def run_colmap_process(video_path, segments, image_options):
    global ini_path, database_path
    
    log_file = os.path.join(workspace_path, 'pipeline_times.log')
    create_workspace(workspace_path)
    ini_path, database_path=ctpi.standard_ini_pipline(workspace_path, sharp_image_path, image_mask_path)
    #open_ini_in_editor(ini_path)
    
    #Video_Pipeline(video_path, workspace_path, log_file, segments, image_options, Masking=True)
    
    SfM_Pipeline(sharp_image_path, workspace_path,image_mask_path, log_file)
    
    ANALYSE_COLMAP_OUTPUT(workspace_path, sparse_path , log_file)
    
    cameras, images, filtered_points3D=COLMAP_RERUN(workspace_path, log_file)
    
    #write_model(cameras, images, filtered_points3D, os.path.join(sparse_path, '0'), ext=".bin")
    
    MVS(workspace_path, log_file)
    
    #Meshing(workspace_path, log_file)
    
    
    total_elapsed_time = calculate_total_elapsed_time(log_file)
    print(f"Total elapsed time: {total_elapsed_time:.2f} seconds")
    return total_elapsed_time        

def main()->None:
    TEST = True
    global workspace_path, ini_path, sharp_image_path, raw_images_path, database_path
    
    if TEST:
        video_path = r"C:\Users\Reich\LocalCoding\COLMAP_WUNDE\COLMAP\VID\Ferse_3+4.mp4"
        workspace_path = r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\test\Test"
        trial_database_path = r'C:\Users\Reich\LocalCoding\COLMAP_Pipline\test'
        experiment_description = ' Test for new sfm'
    
    #segments, image_options = get_user_input()
    
    resize_factor = 1.0
    segments = 0.1
    image_options = {
        'reduce_size': resize_factor,
        #'square_crop': True,
        #'crop': 0.8,
    }
    total_elapsed_time = run_colmap_process(video_path, segments, image_options)
    
    
    sfm_database.load_existing_experiments(trial_database_path, workspace_path, experiment_description=experiment_description, resize_factor=resize_factor, fraction_of_images=segments, elapsed_time=total_elapsed_time)

    

if __name__ == '__main__':

    main()

