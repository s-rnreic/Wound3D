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

'''
def get_user_input():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    segments = 0.15
    reduce_size = 0.75
    square_crop = True
    crop = 0.8

    def on_submit():
        nonlocal segments, reduce_size, square_crop, crop
        segments = float(segments_var.get())
        reduce_size = float(reduce_size_var.get()) if reduce_size_check_var.get() else None
        square_crop = square_crop_var.get() == 1
        crop = float(crop_var.get()) if crop_check_var.get() else None
        input_window.destroy()

    input_window = tk.Toplevel(root)
    input_window.title("Input Options")

    tk.Label(input_window, text="Segments:").grid(row=0, column=0)
    segments_var = tk.StringVar(value="0.15")
    tk.Entry(input_window, textvariable=segments_var).grid(row=0, column=1)

    reduce_size_check_var = tk.IntVar(value=1)
    tk.Checkbutton(input_window, text="Use Reduce Size", variable=reduce_size_check_var).grid(row=1, column=0)
    reduce_size_var = tk.StringVar(value="0.75")
    tk.Entry(input_window, textvariable=reduce_size_var).grid(row=1, column=1)

    tk.Label(input_window, text="Square Crop:").grid(row=2, column=0)
    square_crop_var = tk.IntVar(value=1)
    tk.Checkbutton(input_window, variable=square_crop_var).grid(row=2, column=1)

    crop_check_var = tk.IntVar(value=1)
    tk.Checkbutton(input_window, text="Use Crop", variable=crop_check_var).grid(row=3, column=0)
    crop_var = tk.StringVar(value="0.8")
    tk.Entry(input_window, textvariable=crop_var).grid(row=3, column=1)

    tk.Button(input_window, text="Submit", command=on_submit).grid(row=4, columnspan=2)

    root.wait_window(input_window)

    image_options = {
        'reduce_size': float(reduce_size_var.get()) if reduce_size_check_var.get() else None,
        'square_crop': square_crop_var.get() == 1,
        'crop': float(crop_var.get()) if crop_check_var.get() else None,
    }

    return segments, image_options
'''
def log_time(log_file, step, elapsed_time):
    with open(log_file, 'a') as f:
        f.write(f"{step}: {elapsed_time:.2f} seconds\n")

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
def Video_Pipeline( input_path:str, image_output_path:str, log_file:str, segments:float, image_options:dict)->str:
    start_time = time.time()
    prepare_vid_for_colmap_v5.get_IMG_from_Video(input_path, raw_images_path)
    
    sharpness_values = prepare_vid_for_colmap_v5.get_image_sharpness(raw_images_path, image_options)
    selected_indices = prepare_vid_for_colmap_v5.select_sharp_images(sharpness_values, segments, min_sharpness_percentile=50)
    prepare_vid_for_colmap_v5.copy_selected_images(selected_indices, raw_images_path, sharp_image_path, image_options)
    masking.process_folder(sharp_image_path, image_mask_path)
    elapsed_time = time.time() - start_time
    log_time(log_file, "Video_Pipeline", elapsed_time)
    return sharp_image_path, image_mask_path

# 2. COLMAP Pipeline
def COLMAP_Pipeline(image_output_path: str, workspace_path: str, image_mask_path: str, log_file: str) -> str:
    start_time = time.time()
    global sparse_path
    # Version of PIPLINE which doesnt work with a bat file anymore
    # Each colmap command is executed with a tempProject.ini file --> file contains the parameter of each colmap command
    # construct the command to execute the batch file for creating a .ini file
    
    #Feature Extraction
    keywords = ['HEADER','ImageReader','SiftExtraction']
    temp_ini_path=ctpi.create_temp_project_ini(ini_path, *keywords)
    os.system(f'call colmap feature_extractor --project_path "{temp_ini_path}"')
    
    #Feature Matching
    keywords = ['HEADER','SiftMatching','TwoViewGeometry','ExhaustiveMatching']
    temp_ini_path=ctpi.create_temp_project_ini(ini_path, *keywords)
    ctpi.delete_keywords_from_project_ini(temp_ini_path, 'HEADER','image_path')
    os.system(f'call colmap exhaustive_matcher --project_path "{temp_ini_path}"')
    
    
    #Mapping
    sparse_path = os.path.join(workspace_path, 'sparse')
    os.makedirs(sparse_path, exist_ok=True)
    ctgi.create_glomap_ini(workspace_path + '\\glomap_config.ini')
    time.sleep(2)
    open_ini_in_editor(workspace_path + '\\glomap_config.ini')
    time.sleep(2)
    ctgi.execute_glomap(workspace_path + '\\glomap_config.ini', database_path, sparse_path, sharp_image_path)
    elapsed_time = time.time() - start_time
    log_time(log_file, "COLMAP_Pipeline", elapsed_time)

    
    # #3. COLMAP RERUN Streaming
def COLMAP_RERUN(workspace_path:str, log_file:str) -> None:
    start_time = time.time()
    rerun_viewer.Rerun_Colmap_Pipeline(workspace_path)
    elapsed_time = time.time() - start_time
    log_time(log_file, "COLMAP_RERUN", elapsed_time)

def ANALYSE_COLMAP_OUTPUT(workspace_path: str, log_file: str = None) -> None:
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

def Database_Pipeline(workspace_path:str, log_file:str)->None:
    pass

def open_ini_in_editor(ini_path: str) -> None:
    editor = os.getenv('EDITOR', 'notepad' if os.name == 'nt' else 'nano')
    subprocess.call([editor, ini_path])
        

def main()->None:
    TEST = True
    global workspace_path, ini_path, sharp_image_path, raw_images_path, database_path
    
    if TEST:
        video_path = r"C:\Users\Reich\LocalCoding\COLMAP_WUNDE\COLMAP\VID\Ferse_3.mp4"
        workspace_path = r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\benchmark\test"
    else:
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        workspace_path = filedialog.askdirectory(title="Select Workspace Directory")
    
    log_file = os.path.join(workspace_path, 'pipeline_times.log')
    create_workspace(workspace_path)
    
    #segments, image_options = get_user_input()
    segments = 0.5
    resize_factor = 0.5
    image_options = {
        'reduce_size': resize_factor,
        #'square_crop': True,
        #'crop': 0.8,
    }
    
    ini_path, database_path=ctpi.standard_ini_pipline(workspace_path, sharp_image_path, image_mask_path)
    open_ini_in_editor(ini_path)
    
    Video_Pipeline(video_path, workspace_path, log_file, segments, image_options)
    
    COLMAP_Pipeline(sharp_image_path, workspace_path,image_mask_path, log_file)
    
    ANALYSE_COLMAP_OUTPUT(workspace_path, log_file)
    
    COLMAP_RERUN(workspace_path, log_file)
    
    total_elapsed_time = calculate_total_elapsed_time(log_file)
    print(f"Total elapsed time: {total_elapsed_time:.2f} seconds")
    
    trial_database_path = r'C:\Users\Reich\LocalCoding\COLMAP_Pipline\benchmark'
    
    sfm_database.load_existing_experiments(trial_database_path, workspace_path, experiment_description='Trails with 20% resolution and 50% of the number of images', resize_factor=resize_factor, fraction_of_images=segments, elapsed_time=total_elapsed_time)
    
    return 0 

if __name__ == '__main__':

    main()

