import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
import ffmpeg  # Ensure this is from the ffmpeg-python library
import pandas as pd
from wound3d.legacy.utils import CLAHE as clahe
from wound3d.legacy.utils import image_enhancement as ie
from wound3d.legacy.utils.SavitzkyGolay import Savitzky_Golay


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


def select_sharp_images(sharpness_df, segments,  absolut_min=25, PLOT=False, input_path=None):
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
    
    # Calculate segment size
    segment_size = len(sharpness_df) // segments
    
    # Get filtered indices using Savitzky-Golay filter
    potential_indices = Savitzky_Golay(sharpness_df, window_length=100, polyorder=3, PLOT=False, tolerance=0.05)
    
    # Filter indices that meet the threshold
    above_threshold_indices = [idx for idx in potential_indices if sharpness_df.iloc[idx]['Sharpness'] >= absolut_min]
    
    # Initialize selected indices
    selected_indices = []
    
    for segment in range(segments):
        # Calculate the start and end indices for this segment
        start_idx = segment_size * segment
        end_idx = segment_size * (segment + 1) if segment < segments - 1 else len(sharpness_df)
        
        # Filter images in this segment that are above the sharpness threshold
        segment_potential_indices = [idx for idx in above_threshold_indices if start_idx <= idx < end_idx]
        
        if segment_potential_indices:
            # Find the index closest to the middle of the segment
            middle_idx = start_idx + (end_idx - start_idx) // 2
            
            # Find the closest index to the middle
            closest_idx = min(segment_potential_indices, key=lambda idx: abs(idx - middle_idx))
            selected_indices.append(closest_idx)
        else:
            # If no images in this segment are above the threshold, mark as None for now
            selected_indices.append(None)
    
    # Plot the selected images after the first pass
    if input_path or PLOT:
        plt.figure(figsize=(14, 7))
        sharpness_values = sharpness_df['Sharpness'].values
        plt.plot(sharpness_values, label='Sharpness', color='blue')
        
        # Add vertical lines to separate segments
        for i in range(1, segments):
            segment_boundary = i * segment_size
            plt.axvline(x=segment_boundary, color='gray', linestyle='--', alpha=0.5)
        
        # Highlight selected images
        valid_indices = [idx for idx in selected_indices if idx is not None]
        if valid_indices:
            plt.scatter(valid_indices, sharpness_values[valid_indices], color='green', 
                       s=80, label='Initial Selected Images', zorder=5)
            
        # Highlight gaps (None values)
        none_positions = [i for i, idx in enumerate(selected_indices) if idx is None]
        if none_positions:
            for pos in none_positions:
                center = pos * segment_size + segment_size // 2
                plt.axvspan(pos * segment_size, (pos + 1) * segment_size, 
                           alpha=0.2, color='red', label='_nolegend_')
                plt.text(center, max(sharpness_values) * 0.9, "Gap", 
                        ha='center', color='red', fontweight='bold')
        
        plt.xlabel('Image Index')
        plt.ylabel('Sharpness')
        plt.title('Initial Image Selection - Before Filling Gaps')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        # Save the first pass plot
        if not os.path.exists(input_path):
            os.makedirs(input_path)
        
        if input_path:
            plt.savefig(os.path.join(input_path, 'initial_selection.png'))
        elif PLOT:
            plt.show()
    
    # Fill in segments with no selected images
    filled_indices = []
    for seg in range(segments):
        if selected_indices[seg] is not None:
            continue
        print("found empty segment:", seg)
        # Find left and right neighbors with a selected image
        left = seg - 1
        while left >= 0 and selected_indices[left] is None:
            left -= 1
            
        right = seg + 1
        while right < segments and selected_indices[right] is None:
            right += 1
            
        # Define search range based on segment boundaries, not just neighboring images
        left_segment_edge = seg * segment_size
        right_segment_edge = (seg + 1) * segment_size if seg < segments - 1 else len(sharpness_df)
        
        # Expand search range to include neighboring segments
        if left >= 0:
            left_segment_edge = left * segment_size
        if right < segments:
            right_segment_edge = (right + 1) * segment_size if right < segments - 1 else len(sharpness_df)
        
        # Find potential candidates in the combined range that are above threshold
        candidates = [idx for idx in above_threshold_indices 
                     if left_segment_edge <= idx <= right_segment_edge]

        if not candidates:
            # if no candidates, dont select any image in this segment
            print(f"Segment {seg}: No candidates found, skipping this segment.")
            continue
            
        if len(candidates) < 3:
            # Not enough candidates for a triplet, pick no image in this segment
            continue
        
        # Find triplet with most even spacing relative to segment edges
        best_triplet = None
        best_spacing_score = float('inf')
        
        # Try all possible triplets
        for i in range(len(candidates) - 2):
            for j in range(i + 1, len(candidates) - 1):
                for k in range(j + 1, len(candidates)):
                    triplet = (candidates[i], candidates[j], candidates[k])
                    
                    # Create a list of all points including segment boundaries
                    all_points = [left_segment_edge] + list(triplet) + [right_segment_edge]
                    
                    # Calculate spacings between adjacent points
                    spacings = [all_points[i+1] - all_points[i] for i in range(len(all_points)-1)]
                    
                    # Calculate variance of spacings (lower is better - more even)
                    spacing_variance = np.var(spacings)
                    
                    # Consider both variance and the average distance (prefer larger distances)
                    # This helps avoid clumping in one area
                    avg_spacing = np.mean(spacings)
                    combined_score = spacing_variance / avg_spacing
                    
                    if combined_score < best_spacing_score:
                        best_spacing_score = combined_score
                        best_triplet = triplet
        
        if best_triplet:
            # Assign the triplet to fill the gap(s)
            if left >= 0 and selected_indices[left] is None:
                selected_indices[left] = best_triplet[0]
                filled_indices.append(best_triplet[0])
                
            selected_indices[seg] = best_triplet[1]
            filled_indices.append(best_triplet[1])
            
            if right < segments and selected_indices[right] is None:
                selected_indices[right] = best_triplet[2]
                filled_indices.append(best_triplet[2])
        
        print(f"Segment {seg}: Filled with triplet {best_triplet} (spacing score: {best_spacing_score:.2f})")
        
        # Plot after each triplet is added
        if input_path or PLOT and best_triplet:
            plt.figure(figsize=(14, 7))
            sharpness_values = sharpness_df['Sharpness'].values
            plt.plot(sharpness_values, label='Sharpness', color='blue')
            
            # Add vertical lines to separate segments
            for i in range(1, segments):
                segment_boundary = i * segment_size
                plt.axvline(x=segment_boundary, color='gray', linestyle='--', alpha=0.5)
            
            # Add vertical lines for the segment range we're working with
            plt.axvline(x=left_segment_edge, color='purple', linestyle='-', alpha=0.7, 
                      linewidth=2, label='Selection Range')
            plt.axvline(x=right_segment_edge, color='purple', linestyle='-', alpha=0.7, linewidth=2)
            
            # Highlight all selected images so far
            valid_indices = [idx for idx in selected_indices if idx is not None]
            if valid_indices:
                plt.scatter(valid_indices, sharpness_values[valid_indices], color='green', 
                           s=60, label='Selected Images', zorder=5)
                
            # Highlight newly added images from this triplet
            new_indices = [idx for idx in filled_indices if idx in best_triplet]
            if new_indices:
                plt.scatter(new_indices, sharpness_values[new_indices], color='red', 
                           s=100, marker='*', label='Newly Added Images', zorder=6)
            
            # Draw the triplet connection
            if best_triplet:
                triplet_x = list(best_triplet)
                triplet_y = [sharpness_values[i] for i in best_triplet]
                plt.plot(triplet_x, triplet_y, 'r--', alpha=0.7, linewidth=2, label='Triplet')
                
                # Show the spacing values
                all_points = [left_segment_edge] + list(best_triplet) + [right_segment_edge]
                spacings = [all_points[i+1] - all_points[i] for i in range(len(all_points)-1)]
                
                for i, spacing in enumerate(spacings):
                    x_pos = all_points[i] + spacing/2
                    y_pos = max(sharpness_values) * 0.8
                    plt.text(x_pos, y_pos, f"d={spacing}", ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.7))
            
            plt.xlabel('Image Index')
            plt.ylabel('Sharpness')
            plt.title(f'Image Selection After Filling Gap in Segment {seg}\n'
                      f'Spacing Score: {best_spacing_score:.2f}')
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            
            # Save each triplet addition plot
            if input_path:
                plt.savefig(os.path.join(input_path, f'gap_filling_segment_{seg}.png'))
            elif PLOT:
                plt.show()
    
    # Final pass - make sure no None values remain
    for i in range(segments):
        if selected_indices[i] is None:
            # Use middle of segment as last resort
            selected_indices[i] = (i * segment_size) + (segment_size // 2)
    
    # Create final plot with all selected images
    if input_path or PLOT:
        plt.figure(figsize=(14, 7))
        sharpness_values = sharpness_df['Sharpness'].values
        plt.plot(sharpness_values, label='Sharpness', color='blue')
        
        # Add vertical lines to separate segments
        for i in range(1, segments):
            segment_boundary = i * segment_size
            plt.axvline(x=segment_boundary, color='gray', linestyle='--', alpha=0.5)
        
        # Highlight all selected images
        plt.scatter(selected_indices, sharpness_values[selected_indices], color='green', 
                   s=80, label='Final Selected Images', zorder=5)
        
        # Highlight filled images
        if filled_indices:
            plt.scatter(filled_indices, sharpness_values[filled_indices], color='orange', 
                       marker='*', s=100, label='Gap-Filled Images', zorder=6)
        
        plt.xlabel('Image Index')
        plt.ylabel('Sharpness')
        plt.title('Final Image Selection')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        # Save the final selection plot
        if input_path or PLOT:
            plt.savefig(os.path.join(input_path, 'final_selection.png'))
        elif PLOT:
            plt.show()
    
    return [idx for idx in selected_indices if idx is not None]



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
    from wound3d.legacy.utils.CLAHE import apply_clahe
    from wound3d.legacy.utils.image_enhancement import (
        adaptive_histogram_equalization,
        contrast_stretching,
        gamma_correction,
        histogram_equalization,
    )

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
            elif method is not None:
                raise ValueError(f"Unsupported enhancement method: {method}")

            cv2.imwrite(os.path.join(destination_folder, file), img)
