import sys
import os
from PIL import Image
sys.path.append(os.path.abspath(os.path.join('..')))  # Adjust based on your path structure
#from PointCloudTest.Functions.VID_to_IMG import VID
import cv2



def resize_images_in_folder(input_folder, output_folder):
        # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        # Process only image files (you can customize the extensions)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the image
            with Image.open(input_path) as img:
                # Get original dimensions
                original_width, original_height = img.size
                
                # Calculate new height
                new_height = int(original_height * 0.6)
                
                # Calculate the cropping box (left, upper, right, lower)
                left = 0
                upper = int(original_height * 0.2)  # 20% from the top
                right = original_width
                lower = upper + new_height
                
                # Crop the image
                cropped_img = img.crop((left, upper, right, lower))
                
                # Save the cropped image
                cropped_img.save(output_path)
                print(f"Resized image saved as: {output_path}")

def get_IMG_from_Video(video_path, output_folder, subset_folder:bool=False):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create a directory for the video
    if subset_folder:
        video_output_folder = os.path.join(output_folder, video_name)
        os.makedirs(video_output_folder, exist_ok=True)
    else:
        video_output_folder = output_folder
    
    # Capture the video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    # Extract frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = f"{video_name}_frame_{frame_count:04d}.jpg"
        cv2.imwrite(os.path.join(video_output_folder, frame_filename), frame)
        frame_count += 1
    
    # Release the video capture object
    cap.release()
    
    print("Frames extracted successfully from the video!")

def get_IMG_from_Videos(input_path, output_path):
    input_folder = input_path
    output_folder = output_path

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List all video files in the input folder
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

    # Process each video
    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        
        # Create a directory for each video
        video_output_folder = os.path.join(output_folder, video_name)
        os.makedirs(video_output_folder, exist_ok=True)
        
        # Capture the video
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        # Extract frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = f"{video_name}_frame_{frame_count:04d}.jpg"
            cv2.imwrite(os.path.join(video_output_folder, frame_filename), frame)
            frame_count += 1
        
        # Release the video capture object
        cap.release()
        
    print("Frames extracted successfully!")

