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
    """
        Extract images from a video. The Video is given by an input_path and all images are extracted into the output_path.
        All Images are named after "%05d.jpg" style (ex. 00001.jpg). 
    """
    
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


def get_image_sharpness(input_folder, mask_folder_path=None, image_options=None, percentage_to_analyze=1):
    """
    Calculate sharpness for images in a folder and return as a DataFrame. If mask_folder_path is given, only calculate sharpness in masked area for each image using corresponding mask.
    The Sharpness is calculated by the use of the Laplacian Variance and either calculated for the area inside the mask or the whole area. 
    """
    
    csv_path = os.path.join(input_folder, 'sharpness.csv')

    # List image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]
    subset_size = int(len(image_files) * percentage_to_analyze)
    selected_indices = np.linspace(0, len(image_files) - 1, subset_size).astype(int)

    sharpness_values = []
    if mask_folder_path is not None:
        print(f"Using masks from: {mask_folder_path}")
    else:
        print("No mask folder provided, calculating sharpness for entire images.")
        
    for idx in selected_indices:
        img_name = image_files[idx]
        img = cv2.imread(os.path.join(input_folder, img_name))

        # Apply image options if provided
        if image_options:
            if 'reduce_size' in image_options:
                resize_factor = image_options['reduce_size']
                img = cv2.resize(img, None, fx=resize_factor, fy=resize_factor)

            if image_options.get('square_crop', False):
                h, w = img.shape[:2]
                min_dim = min(h, w)
                img = img[0:min_dim, int((w - min_dim) / 2):int((w + min_dim) / 2)]

            if 'crop' in image_options:
                crop_factor = image_options['crop']
                h, w = img.shape[:2]
                crop_start_h = int(h * (1 - crop_factor) / 2)
                crop_end_h = int(h - h * (1 - crop_factor) / 2)
                crop_start_w = int(w * (1 - crop_factor) / 2)
                crop_end_w = int(w - w * (1 - crop_factor) / 2)
                img = img[crop_start_h:crop_end_h, crop_start_w:crop_end_w]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # If mask_folder_path is given, load the corresponding mask for this image
        if mask_folder_path is not None:
            # Mask file is named like 0001.jpg.png for image 0001.jpg
            mask_name = img_name + '.png'
            mask_path = os.path.join(mask_folder_path, mask_name)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    if mask.shape != gray.shape:
                        mask = cv2.resize(mask.astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask_bool = mask > 0
                    masked_gray = np.where(mask_bool, gray, 0)
                    laplacian = cv2.Laplacian(masked_gray, cv2.CV_64F)
                    variance = laplacian[mask_bool].var() if np.any(mask_bool) else 0
                else:
                    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            else:
                variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        else:
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_values.append(variance)

    # Normalize sharpness values
    #sharpness_values = np.array(sharpness_values) / np.max(sharpness_values) if len(sharpness_values) > 0 and np.max(sharpness_values) > 0 else np.array(sharpness_values)
    
    # Convert the list of sharpness values into a DataFrame
    sharpness_df = pd.DataFrame(np.array(sharpness_values), columns=['Sharpness'])
    
    # Save the DataFrame to CSV if use_csv is True
    sharpness_df.to_csv(csv_path, index=False, header=False)
    
    return sharpness_df


def select_sharp_images(sharpness_df, segments, min_sharpness_percentile=50.0, absolut_min=25, input_path=None):
    """
    Select indices of sharpest images, ensuring they are nearly equidistant and above a sharpness threshold.
    "absolut_min": if the image-sharpness is below this value then a certain area around this image is not considered for image selection
    
    """
    
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
    if min_sharpness < absolut_min:
        print("The Sharpness of the whole Dataset seems to be to low or to blurry to be reconstructed.")
    
    
    
    
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
        plot_sharpness_results(sharpness_df, selected_indices, min_sharpness, segments, input_path)

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
    plt.savefig(os.path.join(input_path, 'sharpness_plot.png'))


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
        cv2.imwrite(os.path.join(dest_folder, f"{idx:05d}.jpg"), img)



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

    input_path = r"C:\Users\Reich\Interface\Videos\Package_Recordings\old_Roll\Roll_Orbit3.mp4" # or input_path = r"D:\3D-Recon\Yoda\Images_RAW"
    segments = 50  # or segments = 0.1
    image_options = {
        # 'reduce_size': 0.5,
        # 'square_crop': True,
        # 'crop' : .8,
    }
    mask_folder_path=r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_Sharpness_Thresh\global_sharp\Trial1\masks"
    base_dir = r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_Sharpness_Thresh\global_sharp\Trial1"
    # os.makedirs(base_dir, exist_ok=True)
    raw_images_path = os.path.join(base_dir, 'raw_images')
    # os.makedirs(raw_images_path, exist_ok=True)
    sharp_image_path = os.path.join(base_dir, 'sharp_images')
    # os.makedirs(sharp_image_path, exist_ok=True)
    #extract_images_from_video(input_path, raw_images_path)
    # Store sharpness and selected indices for both cases
    sharpness_results = {}
    selected_indices_results = {}

    for mask_option in [mask_folder_path, None]:
        print(f"\n--- Calculating sharpness with mask_folder_path={mask_option} ---")
        sharpness_values = get_image_sharpness(
            raw_images_path,
            mask_folder_path=mask_option,
            image_options=image_options
        )
        sharpness_results[str(mask_option)] = sharpness_values
        selected_indices = select_sharp_images(
            sharpness_values,
            segments,
            min_sharpness_percentile=90,
            input_path=None  # We'll plot manually below
        )
        selected_indices_results[str(mask_option)] = selected_indices

    # Use sharpness values from the iteration without mask for the plot
    sharpness_no_mask = sharpness_results["None"]['Sharpness'].values
    indices_no_mask = selected_indices_results["None"]
    indices_with_mask = selected_indices_results[str(mask_folder_path)]

    # Save paired indices to a .txt file
    indices_txt_path = os.path.join(base_dir, 'selected_indices_pairs.txt')
    with open(indices_txt_path, 'w') as f:
        for idx_no_mask, idx_with_mask in zip(indices_no_mask, indices_with_mask):
            f.write(f"{idx_no_mask}\t{idx_with_mask}\n")

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(sharpness_no_mask)), sharpness_no_mask, label='Sharpness (No Mask)', color='blue')
    plt.scatter(indices_no_mask, sharpness_no_mask[indices_no_mask], color='green', label='Selected (No Mask)', zorder=5)
    plt.scatter(indices_with_mask, sharpness_no_mask[indices_with_mask], color='red', label='Selected (With Mask)', marker='x', zorder=6)
    plt.xlabel('Image Index')
    plt.ylabel('Sharpness')
    plt.title('Sharpness and Selected Indices (With/Without Mask)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'sharpness_selected_indices_comparison.png'))
    plt.show()
    # selected_indices = select_sharp_images(sharpness_values, segments, min_sharpness_percentile=50, input_path=base_dir)
    # copy_selected_images(selected_indices, raw_images_path, sharp_image_path, image_options)



if __name__ == "__main__":
    from build_masks import process_folder
    #enhance_img(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments_Enhancement_histogram_equalization\raw_images", r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments_Enhancement_histogram_equalization\sharp_images", method='histogram_equalization')
    base_dir = r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Test\Test_Foot"
    raw_images_path=base_dir + "\\raw_images"
    sharp_images_path=base_dir + "\\local_sharp_images"
    mask_path=base_dir + "\\masks"
    video_path=r"C:\Users\Reich\Interface\Videos\Wound_Recordings\Fuß\Fuss_combined.mp4"
    # extract_images_from_video(video_path, raw_images_path, skip_frames=0)
    # process_folder(raw_images_path, mask_path)
    #sharp_df=get_image_sharpness(raw_images_path, mask_folder_path=mask_path)
    #indices=select_sharp_images(sharp_df, segments=50, min_sharpness_percentile=50, input_path=base_dir)
    #copy_selected_images(indices, raw_images_path, sharp_images_path)
    view_masks_and_images(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Test\Test_Foot",plot=False, save_gif=True, gif_name="masked_images.gif", gif_duration=50, all_images=True)
     #view_masks_and_images(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_Sharpness_Thresh\global_sharp\Trial3",plot=False, save_gif=True, gif_name="masked_images.gif", gif_duration=0.5)
    #view_masks_and_images(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_Sharpness_Thresh\local_sharp\Trial3",plot=False, save_gif=True, gif_name="masked_images.gif", gif_duration=0.5)
    #plot_sharpness(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_Sharpness_Thresh\global_sharp\Trial3\sharp_images", save_gif=True, gif_name="sharpness_per_pixel.gif", gif_duration=0.5)
    #plot_sharpness(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_Sharpness_Thresh\local_sharp\Trial3\sharp_images", save_gif=True, gif_name="sharpness_per_pixel.gif", gif_duration=0.5)
    #plot_sharpness_histo(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_SharpnessMask\t2\sharp_images", save_gif=True, gif_name="sharpness_histograms.gif", gif_duration=0.5)
    #main()
    # plt.show()
