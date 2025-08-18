import sys
import os
import prepare_vid_for_colmap_v5
import create_temp_project_ini as ctpi
import create_temp_glomap_ini as ctgi
import sfm_database
import subprocess
from run_command import run_command
# import tkinter as tk
# from tkinter import filedialog, simpledialog  # Add simpledialog for user input
import build_masks as masking
import rerun_viewer
import time  # Add import for time module
from Pipe_Code.utils.read_write_model import write_model
import MVS_Meshing as mvs
from itertools import product
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
            f.write(f'{step}: elapsed time is None\n')

def calculate_total_elapsed_time(log_file):
    total_time = 0.0
    with open(log_file, 'r') as f:
        for line in f:
            time_str = line.split(':')[-1].strip().split()[0]
            if time_str == 'elapsed':
                continue
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
def Video_Pipeline( input_path:str, workspace_path:str, log_file:str, segments:float, image_options:dict, min_sharpness_percentile:float=25.0, enhance_option:str=None , Masking:bool=True)->str:
    start_time = time.time()
    prepare_vid_for_colmap_v5.extract_images_from_video(input_path, raw_images_path)
    
    sharpness_values = prepare_vid_for_colmap_v5.get_image_sharpness(raw_images_path,mask_folder_path=None, image_options=image_options)
    selected_indices = prepare_vid_for_colmap_v5.select_sharp_images(sharpness_values, segments, absolut_min=min_sharpness_percentile, input_path=workspace_path)
    prepare_vid_for_colmap_v5.copy_selected_images(selected_indices, raw_images_path, sharp_image_path, image_options)
    if Masking:
        masking.process_folder(sharp_image_path, image_mask_path)
    
    if enhance_option is not None:
        prepare_vid_for_colmap_v5.enhance_img(sharp_image_path,sharp_image_path ,enhance_option)
    
    elapsed_time = time.time() - start_time
    log_time(log_file, "Video_Pipeline", elapsed_time)
    return sharp_image_path, image_mask_path

# 2. COLMAP Pipeline
def COLMAP_Pipeline(image_output_path: str, workspace_path: str, image_mask_path: str, log_file: str) -> str:
    start_time = time.time()
    global sparse_path
    total_ini_path = os.path.join(workspace_path, 'total_project.ini')
    # Version of PIPLINE which doesnt work with a bat file anymore
    # Each colmap command is executed with a tempProject.ini file --> file contains the parameter of each colmap command
    # construct the command to execute the batch file for creating a .ini file
    
    #Feature Extraction
    keywords = ['HEADER','ImageReader','SiftExtraction']
    temp_ini_path=ctpi.create_temp_project_ini(ini_path, *keywords)
    feature_extractor_command = f"colmap feature_extractor --project_path \"{temp_ini_path}\""
    run_command(feature_extractor_command)
    ctpi.append_to_total_project_ini(temp_ini_path, total_ini_path)
    
    #Feature Matching
    keywords = ['HEADER','SiftMatching','TwoViewGeometry','ExhaustiveMatching']
    temp_ini_path=ctpi.create_temp_project_ini(ini_path, *keywords)
    ctpi.delete_keywords_from_project_ini(temp_ini_path, 'HEADER','image_path')
    feature_matching_command = f"colmap exhaustive_matcher --project_path \"{temp_ini_path}\""
    run_command(feature_matching_command)
    ctpi.append_to_total_project_ini(temp_ini_path, total_ini_path)
    elapsed_time = time.time() - start_time
    log_time(log_file, "COLMAP_Pipeline", elapsed_time)
    

def GLOMAP_Pipeline(workspace_path, glomap_ini_path, log_file: str):
    start_time = time.time()
    total_ini_path = os.path.join(workspace_path, 'total_project.ini')
    
    #open_ini_in_editor(workspace_path + '\\glomap_config.ini')
    ctgi.execute_glomap(glomap_ini_path, database_path, sparse_path, sharp_image_path)
    ctpi.append_to_total_project_ini(glomap_ini_path, total_ini_path)

    elapsed_time = time.time() - start_time
    log_time(log_file, "GLOMAP_Pipeline", elapsed_time)


    
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
        result = subprocess.run(f"colmap model_analyzer --path \"{subfolder}\"", shell=True, capture_output=True, text=True)
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
       

def run_colmap_process(video_path, segments, image_options, enhance_option=None, custom_parameters=None, sharpness_value=25)->float:
    global ini_path, database_path , sparse_path
    
    log_file = os.path.join(workspace_path, 'pipeline_times.log')
    create_workspace(workspace_path)
    ini_path, database_path=ctpi.standard_ini_pipline(workspace_path, sharp_image_path, image_mask_path)
    #open_ini_in_editor(ini_path)
    
    # if custom_parameters:
    #     for header, param, value in custom_parameters:
    #         ctpi.insert_keywords_to_project_ini(ini_path, header, param, str(value))

    Video_Pipeline(video_path, workspace_path, log_file, segments, image_options, min_sharpness_percentile=sharpness_value, enhance_option=enhance_option, Masking=True)

    COLMAP_Pipeline(sharp_image_path, workspace_path,image_mask_path, log_file)
    
    sparse_path = os.path.join(workspace_path, 'sparse')
    os.makedirs(sparse_path, exist_ok=True)
    glomap_ini_path=os.path.join(workspace_path ,'glomap_config.ini')
    ctgi.create_glomap_ini(glomap_ini_path)

    if custom_parameters:
        for header, param, value in custom_parameters:
            ctpi.insert_keywords_to_project_ini(glomap_ini_path, header, param, str(value))

    GLOMAP_Pipeline(workspace_path, glomap_ini_path, log_file)

    ANALYSE_COLMAP_OUTPUT(workspace_path, sparse_path , log_file)
    
    cameras, images, filtered_points3D=COLMAP_RERUN(workspace_path, log_file)
    
    #write_model(cameras, images, filtered_points3D, os.path.join(sparse_path, '0'), ext=".bin")
    
    MVS(workspace_path, log_file)
    
    Meshing(workspace_path, log_file)
    
    
    total_elapsed_time = calculate_total_elapsed_time(log_file)
    # print(f"Total elapsed time: {total_elapsed_time:.2f} seconds")
    return total_elapsed_time        


def main()->None:
    TEST = True
    global workspace_path, ini_path, sharp_image_path, raw_images_path, database_path
    
    if TEST:
        video_path = r"C:\Users\Reich\Interface\Videos\Wound_Recordings\Ferse\Ferse_3+4.mp4"
        workspace_path = r"C:\Users\Reich\LocalCoding\SfM_Pipeline_alternative\wundpro_vid2cloud\test\t2"
        trial_database_path = r'C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_Mask_Sparse_Dense'
        experiment_description = 'Reco Mandarin Orbit2, 1.0 resize factor, segments = 40, sharpness is calculated for whole image. Masks are applied only for Dense Reconstruction'
    
    
    resize_factor = 1.0
    segments = 50
    image_options = {
        'reduce_size': resize_factor,
        #'square_crop': True,
        #'crop': 0.8,
    }
    total_elapsed_time = run_colmap_process(video_path, segments, image_options)

    #sfm_database.insert_experiment2db(trial_database_path, workspace_path, experiment_description=experiment_description, resize_factor=resize_factor, fraction_of_images=segments, elapsed_time=total_elapsed_time)


if __name__ == '__main__':
    global workspace_path, ini_path, sharp_image_path, raw_images_path, database_path
    # video_path = r"C:\Users\Reich\Interface\Videos\Wound_Recordings\Ferse\Ferse_3+4.mp4"
    base_workspace_path = r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\test"
    #video_dir = r"C:\Users\Reich\Interface\Videos\Package_Recordings\Roll_labled"
    video_path = r"c:\Users\Reich\Interface\Videos\Wound_Recordings\Ferse\Ferse_3+4.mp4"
    workspace_path = base_workspace_path
    resize_factor = [1.0]  # You can adjust or make this a list if needed
    segments = [50]        # You can adjust or make this a list if needed  # Percentile values for sharpness selection
   
    run_colmap_process(video_path, segments[0], {'reduce_size': resize_factor[0]}, enhance_option=None, custom_parameters=None, sharpness_value=25.0)
