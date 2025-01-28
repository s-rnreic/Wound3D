import sys
import os
sys.path.append(r'C:\Users\Reich\LocalCoding\COLMAP')
sys.path.append(r'C:\Users\Reich\LocalCoding\API_POINTCLOUD_TEST\Mattern_Prog')
import Get_IMG_from_VIDEO 
import prepare_vid_for_colmap_v4
import os
import tkinter as tk
from tkinter import filedialog
#import pycolmap 

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

#1. Analyse Video and extract image
def Video_Pipeline( input_path:str, image_output_path:str)->str:
    # input_path : path to the video 
    # output_path: path where the extracted images are saved at --> when it doesnt exist --> created
    
    
    # Create images out of Video
    raw_images_path = os.path.join(image_output_path, 'raw_images')
    os.makedirs(raw_images_path, exist_ok=True)
    Get_IMG_from_VIDEO.get_IMG_from_Video(input_path, raw_images_path)
    
    selected_image_path = os.path.join(image_output_path, 'sharp_images')
    os.makedirs(selected_image_path, exist_ok=True)
    
    # Extract sharp and equidistant images
    segments=0.15
    image_options = {
        #'reduce_size': 0.75,
        #'square_crop': True,
        # 'crop' : .8,
    }
    sharpness_values = prepare_vid_for_colmap_v4.get_image_sharpness(raw_images_path, image_options)
    selected_indices = prepare_vid_for_colmap_v4.select_sharp_images(sharpness_values, segments, min_sharpness_percentile=50)
    prepare_vid_for_colmap_v4.copy_selected_images(selected_indices, raw_images_path, selected_image_path, image_options)

    return selected_image_path

def COLMAP_Pipeline(image_output_path: str, workspace_path: str) -> str:
    # Construct the command to execute the batch file
    batch_file_path = os.path.join(os.path.dirname(__file__), 'SIMPLE_COLMAP_PIPLINE.bat')
    command = f'call "{batch_file_path}" "{workspace_path}" "{image_output_path}"'
    
    # Execute the batch file
    os.system(command)

def main():
    TEST = True
    if TEST:
        video_path = r"C:\Users\Reich\LocalCoding\COLMAP\VID\Ferse_3.mp4"
        workspace_path = r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\test"
    else:
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        workspace_path = filedialog.askdirectory(title="Select Workspace Directory")
    
    selected_image_path = Video_Pipeline(video_path, workspace_path)
    COLMAP_Pipeline(selected_image_path, workspace_path)
    return 0 

if __name__ == '__main__':
    main()
