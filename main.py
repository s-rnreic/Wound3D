import sys
import os
import prepare_vid_for_colmap_v5
import create_temp_project_ini as ctpi
import create_temp_glomap_ini as ctgi
import subprocess
import tkinter as tk
from tkinter import filedialog
import build_masks as masking
# import pycolmap 

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




def create_workspace(workspace_path:str)->None:
    global raw_images_path, sharp_image_path, image_mask_path
    os.makedirs(workspace_path, exist_ok=True)
    raw_images_path = os.path.join(workspace_path, 'raw_images')
    os.makedirs(raw_images_path, exist_ok=True)
    sharp_image_path = os.path.join(workspace_path, 'sharp_images')
    os.makedirs(sharp_image_path, exist_ok=True)
    image_mask_path=os.path.join(workspace_path, 'masks')

#1. Analyse Video and extract image
def Video_Pipeline( input_path:str, image_output_path:str)->str:
    # input_path : path to the video 
    # output_path: path where the extracted images are saved at --> when it doesnt exist --> created

    prepare_vid_for_colmap_v5.get_IMG_from_Video(input_path, raw_images_path)
    
    # Extract sharp and equidistant images
    segments=0.15
    image_options = {
        #'reduce_size': 0.75,
        #'square_crop': True,
        # 'crop' : .8,
    }
    sharpness_values = prepare_vid_for_colmap_v5.get_image_sharpness(raw_images_path, image_options)
    selected_indices = prepare_vid_for_colmap_v5.select_sharp_images(sharpness_values, segments, min_sharpness_percentile=50)
    prepare_vid_for_colmap_v5.copy_selected_images(selected_indices, raw_images_path, sharp_image_path, image_options)
    masking.process_folder(sharp_image_path, image_mask_path)

    return sharp_image_path, image_mask_path

# 2. COLMAP Pipeline
def COLMAP_Pipeline(image_output_path: str, workspace_path: str,image_mask_path: str) -> str:
    
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
    
    open_ini_in_editor(workspace_path + '\\glomap_config.ini')
    ctgi.execute_glomap(workspace_path + '\\glomap_config.ini', database_path, sparse_path, sharp_image_path)

    
    # #3. COLMAP RERUN Streaming
    # def COLMAP_RERUN(image_output_path: str, workspace_path: str, image_mask_path: str) -> str:

def ANALYSE_COLMAP_OUTPUT(workspace_path: str) -> None:
    # Analyse the output of the COLMAP pipeline
    
    # Analyse the sparse folder
    subfolders = [os.path.join(sparse_path, subfolder) for subfolder in os.listdir(sparse_path) if os.path.isdir(os.path.join(sparse_path, subfolder))]
    
    for subfolder in subfolders:
        result = subprocess.run(f"colmap model_analyzer --path {subfolder}", shell=True, capture_output=True, text=True)
        print(result.stderr)

def open_ini_in_editor(ini_path: str) -> None:
    editor = os.getenv('EDITOR', 'notepad' if os.name == 'nt' else 'nano')
    subprocess.call([editor, ini_path])
        

def main()->None:
    TEST = True
    global workspace_path, ini_path, sharp_image_path, raw_images_path, database_path
    
    if TEST:
        video_path = r"C:\Users\Reich\LocalCoding\COLMAP_WUNDE\COLMAP\VID\Ferse_3.mp4"
        workspace_path = r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\test_glomap"
    else:
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        workspace_path = filedialog.askdirectory(title="Select Workspace Directory")
    
    create_workspace(workspace_path)
    
    ini_path, database_path=ctpi.standard_ini_pipline(workspace_path, sharp_image_path, image_mask_path)
    open_ini_in_editor(ini_path)
    
    Video_Pipeline(video_path, workspace_path)
    
    COLMAP_Pipeline(sharp_image_path, workspace_path,image_mask_path)
    
    ANALYSE_COLMAP_OUTPUT(workspace_path)
    
    return 0 

if __name__ == '__main__':
    video_path = r"C:\Users\Reich\LocalCoding\Wundbilder\fuss\Loch_vorne_v1.mp4"
    workspace_path = r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\test"
    #Video_Pipeline(video_path, workspace_path)
    ANALYSE_COLMAP_OUTPUT(workspace_path)
    #main()
    