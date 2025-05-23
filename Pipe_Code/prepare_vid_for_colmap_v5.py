import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
import ffmpeg  # Ensure this is from the ffmpeg-python library
import pandas as pd
from utils import image_enhancement as ie
from utils import CLAHE as clahe


def extract_images_from_video(input_video, output_path, skip_frames=0):
    """Extract images from a video."""
    
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        
    os.makedirs(output_path)
    new_file_name_stub = os.path.basename(os.path.dirname(input_video))
    
    stream = ffmpeg.input(input_video)
    
    if skip_frames:
        stream = ffmpeg.filter(stream, "select", f"not(mod(n,{skip_frames}))")
        stream = ffmpeg.filter(stream, "setpts", "N/FRAME_RATE/TB")
        
    output_pattern = os.path.join(output_path, f"%05d.jpg")
    
    try:
        ffmpeg.output(stream, output_pattern, qscale=3).run()
    except ffmpeg.Error as e:
        print(e.stderr)


def get_image_sharpness(input_folder, image_options=None, percentage_to_analyze=1):
    """Calculate sharpness for images in a folder and return as a DataFrame."""

    csv_path = os.path.join(input_folder, 'sharpness.csv')

    # Check if the CSV exists and use_csv is True, then load and return the DataFrame
    # if use_csv and os.path.exists(csv_path):
    #     return pd.read_csv(csv_path, header=None, names=['Sharpness'])

    # List image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]
    subset_size = int(len(image_files) * percentage_to_analyze)
    selected_indices = np.linspace(0, len(image_files) - 1, subset_size).astype(int)

    sharpness_values = []

    for idx in selected_indices:
        img = cv2.imread(os.path.join(input_folder, image_files[idx]))

        # Apply image options if provided
        if image_options:
            if 'reduce_size' in image_options:
                resize_factor = image_options['reduce_size']
                img = cv2.resize(img, None, fx=resize_factor, fy=resize_factor)

            if image_options.get('square_crop', False):
                h, w = img.shape[:2]
                min_dim = min(h, w)
                img = img[0:min_dim, int((w - min_dim) / 2):int((w + min_dim) / 2)]  # Crop to square and middle part of width

            if 'crop' in image_options:
                crop_factor = image_options['crop']
                h, w = img.shape[:2]
                crop_start_h = int(h * (1 - crop_factor) / 2)
                crop_end_h = int(h - h * (1 - crop_factor) / 2)
                crop_start_w = int(w * (1 - crop_factor) / 2)
                crop_end_w = int(w - w * (1 - crop_factor) / 2)
                img = img[crop_start_h:crop_end_h, crop_start_w:crop_end_w]


        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_values.append(variance)

    # Normalize sharpness values
    sharpness_values = np.array(sharpness_values) / np.max(sharpness_values)
    
    # Convert the list of sharpness values into a DataFrame
    sharpness_df = pd.DataFrame(sharpness_values, columns=['Sharpness'])
    
    # Save the DataFrame to CSV if use_csv is True
    sharpness_df.to_csv(csv_path, index=False, header=False)
    
    return sharpness_df


def select_sharp_images(sharpness_df, segments, min_sharpness_percentile=50, input_path=None):
    """Select indices of sharpest images, ensuring they are nearly equidistant and above a sharpness threshold."""
    
    # Convert percentage to a decimal if necessary
    if segments < 1:
        segments = int(len(sharpness_df) * segments)
    else:
        segments = int(segments)
    
    if len(sharpness_df) < segments:
        print("Too few images. Algorithm takes now the maximal number of images.")
        segments = len(sharpness_df)
        
    
    print(f"Selecting {segments} images from {len(sharpness_df)} images.")
    
    # Calculate the minimum sharpness threshold
    min_sharpness = sharpness_df['Sharpness'].quantile(min_sharpness_percentile / 100.0)
    
    selected_indices = []
    for segment in range(segments):
        # Calculate the start and end indices for this segment
        start_idx = (len(sharpness_df) * segment) // segments
        end_idx = (len(sharpness_df) * (segment + 1)) // segments
        
        # Filter images in this segment that are above the sharpness threshold
        segment_df = sharpness_df.iloc[start_idx:end_idx]
        above_threshold_df = segment_df[segment_df['Sharpness'] >= min_sharpness]
        
        if not above_threshold_df.empty:
            # Find the index closest to the middle of the original segment range
            middle_idx = start_idx + (end_idx - start_idx) // 2
            
            # Manual approach to find the closest index
            closest_idx_diff = abs(above_threshold_df.index - middle_idx)
            closest_idx = closest_idx_diff.argmin()
            
            selected_indices.append(above_threshold_df.iloc[closest_idx].name)
        else:
            highest_threshold_df = segment_df.loc[segment_df['Sharpness'].idxmax()]
            if highest_threshold_df['Sharpness'] < sharpness_df['Sharpness'].quantile(95.0 / 100.0):
                selected_indices.append(highest_threshold_df.name)
            else:
                pass

    if input_path is not None:
        selected_image_path = os.path.join(input_path, f"Subset_{len(selected_indices)}/Images")
        plot_sharpness_results(sharpness_df, selected_indices, min_sharpness, segments, selected_image_path)

    return selected_indices


def plot_sharpness_results(sharpness_df, selected_indices, min_sharpness, num_blocks, input_path):
    sharpness_values = sharpness_df['Sharpness'].values
    image_indices = np.arange(len(sharpness_values))
    
    plt.figure(figsize=(12, 6))

    # Plot all sharpness values
    plt.plot(image_indices, sharpness_values, label='Sharpness')

    # Highlight selected images
    plt.scatter(selected_indices, sharpness_values[selected_indices], color='g', label='Selected Images', zorder=5)

    # Draw horizontal line for minimum sharpness threshold
    plt.axhline(min_sharpness, color='r', linestyle='--', label=f'Min Sharpness: {min_sharpness:.2f}')

    # Draw vertical lines to separate segments
    for i in range(1, num_blocks):
        plt.axvline(i * len(sharpness_values) / num_blocks, color='k', linestyle='--', alpha=0.4, linewidth=0.5)

    # Make title of the plot: - final number of selected images, - total number of images - percentage of selected images
    plt.title(f'Sharpness of Images with Selected Images and Thresholds\n'
                f'Final Number of Selected Images: {len(selected_indices)}\n'
                f'Total Number of Images: {len(sharpness_values)}\n'
                f'Percentage of Selected Images: {len(selected_indices) / len(sharpness_values) * 100:.2f}%')

    plt.xlabel('Image Index')
    plt.ylabel('Sharpness')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.minorticks_on()

    # save the plot in the same folder as the images:
    # create the folder if it does not exist
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    plt.savefig(os.path.join(os.path.dirname(input_path), 'sharpness_plot.png'))


def copy_selected_images(selected_indices, src_folder, dest_folder, image_options=None):
    """Copy selected images to a destination folder, optionally applying image options."""

    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    os.makedirs(dest_folder)

    all_files = [f for f in os.listdir(src_folder) if f.endswith(('.jpg', '.png'))]

    for idx in selected_indices:
        img = cv2.imread(os.path.join(src_folder, all_files[idx]))

        # Apply image options if provided
        if image_options:
            if 'crop' in image_options:  # Regular crop first
                crop_factor = image_options['crop']
                h, w = img.shape[:2]
                crop_start_h = int(h * (1 - crop_factor) / 2)
                crop_end_h = int(h - h * (1 - crop_factor) / 2)
                crop_start_w = int(w * (1 - crop_factor) / 2)
                crop_end_w = int(w - w * (1 - crop_factor) / 2)
                img = img[crop_start_h:crop_end_h, crop_start_w:crop_end_w]

            if image_options.get('square_crop', False):  # Square crop 
                h, w = img.shape[:2]
                min_dim = min(h, w)
                img = img[0:min_dim, int((w - min_dim) / 2):int((w + min_dim) / 2)] 

            if 'reduce_size' in image_options:  # Resize at the end
                resize_factor = image_options['reduce_size']
                img = cv2.resize(img, None, fx=resize_factor, fy=resize_factor)

        # Save the modified image
        cv2.imwrite(os.path.join(dest_folder, f"{idx:04d}.jpg"), img)

def enhance_img(input_folder, destination_folder, method=None, image_options=None):
    """
    Enhance images in a folder using a specified method and save them to a destination folder.

    Parameters:
        input_folder (str): Path to the folder containing input images.
        destination_folder (str): Path to the folder to save enhanced images.
        method (str, optional): Enhancement method to apply (e.g., 'clahe', 'stretching', 'gamma_correction', 'histogram_equalization').
                                If None, no enhancement is applied.
        image_options (dict, optional): Additional options for the enhancement method.
    """
    from utils.image_enhancement import histogram_equalization, gamma_correction, contrast_stretching, adaptive_histogram_equalization
    from utils.CLAHE import apply_clahe

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for file in os.listdir(input_folder):
        if file.endswith(('.jpg', '.png')):
            img_path = os.path.join(input_folder, file)
            img = cv2.imread(img_path)

            # Apply the specified enhancement method if provided
            if method == 'clahe':
                img = apply_clahe(img_path)
            elif method == 'stretching':
                img = contrast_stretching(img)
            elif method == 'gamma_correction':
                img = gamma_correction(img)
            elif method == 'histogram_equalization':
                img = histogram_equalization(img)
            elif method == 'adaptive_histogram_equalization':
                img = adaptive_histogram_equalization(img)
            elif method is None:
                pass  # No enhancement applied
            else:
                raise ValueError(f"Unsupported enhancement method: {method}")

            # Save the (possibly enhanced) image
            cv2.imwrite(os.path.join(destination_folder, file), img)

def main():

    input_path = r"C:\Users\Reich\Interface\Videos\Wound_Recordings\Ferse\Ferse_3+4.mp4"  # or input_path = r"D:\3D-Recon\Yoda\Images_RAW"
    segments = 10  # or segments = 0.1
    image_options = {
        # 'reduce_size': 0.5,
        # 'square_crop': True,
        # 'crop' : .8,
    }

    base_dir = r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Test\T2"
    os.makedirs(base_dir, exist_ok=True)
    raw_images_path = os.path.join(base_dir, 'raw_images')
    os.makedirs(raw_images_path, exist_ok=True)
    sharp_image_path = os.path.join(base_dir, 'sharp_images')
    os.makedirs(sharp_image_path, exist_ok=True)
    extract_images_from_video(input_path, raw_images_path)
    
    sharpness_values = get_image_sharpness(raw_images_path, image_options)
    selected_indices = select_sharp_images(sharpness_values, segments, min_sharpness_percentile=50, input_path=base_dir)
    copy_selected_images(selected_indices, raw_images_path, sharp_image_path, image_options)



if __name__ == "__main__":
    #main()
    # plt.show()
    #enhance_img(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments_Enhancement_histogram_equalization\raw_images", r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments_Enhancement_histogram_equalization\sharp_images", method='histogram_equalization')
    video_path=r"C:\Users\Reich\Interface\Videos\Package_Recordings\Tee\Tee_Orbit2_5sek.mp4"
    extract_images_from_video(video_path, r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Test\T2", skip_frames=0)