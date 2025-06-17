import cv2
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

def generate_mask(image_path, blur_ksize=33, exp_factor=20, PLOT=False):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Extract A and B channels
    a_channel, b_channel = image[:, :, 1], image[:, :, 2]
    
    # Apply Gaussian Blur to A and B channels
    a_blur = cv2.GaussianBlur(a_channel, (blur_ksize, blur_ksize), 0)
    b_blur = cv2.GaussianBlur(b_channel, (blur_ksize, blur_ksize), 0)
    
    # Add A and B channels into a single grayscale image
    ab_image = np.maximum(a_blur, b_blur)

    if PLOT:
        plt.figure(figsize=(6, 6))
        plt.title('ab_image (max of blurred A and B channels)')
        im = plt.imshow(ab_image, cmap='gray')
        plt.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()
    
    # Compute the average value of the center circle (5% of the image)
    h, w = ab_image.shape
    center_x, center_y = w // 2, h // 2
    radius = int(0.1 * min(h, w) / 2)
    y, x = np.ogrid[:h, :w]
    mask_circle = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    threshold = np.mean(ab_image[mask_circle])
    deviation = 2*np.std(ab_image[mask_circle])

    if PLOT:
        plt.figure(figsize=(6, 6))
        plt.title('Center Circle Area for Mean/Deviation')
        im = plt.imshow(ab_image, cmap='gray')
        circle_overlay = np.zeros_like(ab_image, dtype=np.uint8)
        circle_overlay[mask_circle] = 255
        plt.contour(circle_overlay, levels=[127], colors='r', linewidths=2)
        plt.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        # Add mean and std text
        plt.text(
            0.05, 0.95,
            f"Mean: {threshold:.2f}\nStd: {np.std(ab_image[mask_circle]):.2f}",
            color='white', fontsize=12, ha='left', va='top',
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3')
        )
        plt.show()

    # Compute importance weighting (radial gradient)
    y, x = np.indices((h, w))
    dist_map = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    dist_map = dist_map ** exp_factor  # Make the gradient steeper
    dist_map = 1 - (dist_map / np.max(dist_map))  # Normalize to [0,1]

    if PLOT:
        plt.figure(figsize=(6, 6))
        plt.title('Radial Distance Map (dist_map)')
        im = plt.imshow(dist_map, cmap='viridis')
        plt.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()

    # Apply radial gradient to enhance central importance
    weighted_ab = ab_image * dist_map

    if PLOT:
        plt.figure(figsize=(6, 6))
        plt.title('Weighted ab_image (weighted_ab)')
        im = plt.imshow(weighted_ab, cmap='gray')
        plt.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()

    # Mask is everything above 
    mask = (weighted_ab + deviation > threshold).astype(np.uint8) * 255

    if PLOT:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Final Mask')
        im1 = plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.colorbar(im1, fraction=0.046, pad=0.04)

        # Show the original image where mask is true, black elsewhere
        # Reload original image in RGB for display
        orig_img = cv2.imread(image_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        masked_img = np.zeros_like(orig_img)
        mask_bool = mask.astype(bool)
        for c in range(3):
            masked_img[..., c][mask_bool] = orig_img[..., c][mask_bool]

        plt.subplot(1, 2, 2)
        plt.title('Original Image (Masked)')
        im2 = plt.imshow(masked_img)
        plt.axis('off')
        plt.show()
    # # Apply morphological closing to fill in holes
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel
    #                         , iterations=2)
    
    return mask 

def moving_average_mask(image_folder_path, blur_ksize=33, exp_factor=2, window_size=50, PLOT=False, output_folder=None):
    """
    Generate masks for all images in a folder using a moving average threshold for the center circle mean.
    Additionally, calculate the mean and standard deviation of the ab-channel image for the whole dataset,
    and compute moving averages for both means and deviations.
    """
    import collections
    import sys
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
    image_paths = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if f.endswith(('.jpg', '.png'))]
    image_paths.sort()  # Ensure consistent order
    masks = []
    all_means = []
    all_devs = []
    ab_images = []
    # First pass: calculate all means and deviations for ab-channel images
    
    #first image 
    first_image_path = image_paths[0]
    image_first = cv2.imread(first_image_path)
    h, w = image_first.shape[0:2]
    center_x, center_y = w // 2, h // 2
    radius = int(0.1 * min(h, w) / 2)
    y, x = np.ogrid[:h, :w]
    mask_circle = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            all_means.append(np.nan)
            all_devs.append(np.nan)
            ab_images.append(None)
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        a_channel, b_channel = image[:, :, 1], image[:, :, 2]
        a_blur = cv2.GaussianBlur(a_channel, (blur_ksize, blur_ksize), 0)
        b_blur = cv2.GaussianBlur(b_channel, (blur_ksize, blur_ksize), 0)
        ab_image = np.maximum(a_blur, b_blur)
        ab_images.append(ab_image)
        center_mean = np.mean(ab_image[mask_circle])
        center_dev = np.std(ab_image[mask_circle])
        all_means.append(center_mean)
        all_devs.append(center_dev)
    all_means = np.array(all_means)
    all_devs = np.array(all_devs)
    # Compute moving averages
    mov_avg_means = np.convolve(all_means, np.ones(window_size)/window_size, mode='valid')
    mov_avg_devs = np.convolve(all_devs, np.ones(window_size)/window_size, mode='valid')
    # Second pass: generate masks using moving averages
    
    if PLOT:
        import matplotlib.pyplot as plt

        # Plot the means and deviations of the whole dataset
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.title("Mean of ab_image (whole dataset)")
        plt.plot([np.mean(ab[mask_circle]) if ab is not None else np.nan for ab in ab_images], label="Mean (whole image)")
        plt.xlabel("Image Index")
        plt.ylabel("Mean Value")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.title("StdDev of ab_image (whole dataset)")
        plt.plot([np.std(ab[mask_circle]) if ab is not None else np.nan for ab in ab_images], label="StdDev (whole image)")
        plt.xlabel("Image Index")
        plt.ylabel("StdDev Value")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Plot means
        axs[0].plot(all_means, label='Center Means (per image)')
        axs[0].plot(np.arange(window_size//2, window_size//2+len(mov_avg_means)), mov_avg_means, label=f'Moving Avg Means (window={window_size})')
        axs[0].set_xlabel('Image Index')
        axs[0].set_ylabel('Mean Value')
        axs[0].set_title('Center Means and Moving Average')
        axs[0].legend()
        axs[0].grid(True)

        # Plot deviations
        axs[1].plot(all_devs, label='Center StdDev (per image)')
        axs[1].plot(np.arange(window_size//2, window_size//2+len(mov_avg_devs)), mov_avg_devs, label=f'Moving Avg StdDev (window={window_size})')
        axs[1].set_xlabel('Image Index')
        axs[1].set_ylabel('StdDev Value')
        axs[1].set_title('Center StdDev and Moving Average')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()
        #sys.exit(0)
    

    for idx, ab_image in enumerate(ab_images):
        if ab_image is None:
            masks.append(None)
            continue
        h, w = ab_image.shape
        center_x, center_y = w // 2, h // 2
        radius = int(0.1 * min(h, w) / 2)
        y, x = np.ogrid[:h, :w]
        mask_circle = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        # Use moving average for threshold and deviation (align index for 'valid' mode)
        mov_idx = min(max(idx - window_size//2, 0), len(mov_avg_means)-1)
        threshold = mov_avg_means[mov_idx]
        deviation = 2 * mov_avg_devs[mov_idx]
        y_idx, x_idx = np.indices((h, w))
        dist_map = np.sqrt((x_idx - center_x) ** 2 + (y_idx - center_y) ** 2)
        dist_map = dist_map ** exp_factor
        dist_map = 1 - (dist_map / np.max(dist_map))
        weighted_ab = ab_image * dist_map
        mask = (weighted_ab + deviation > threshold).astype(np.uint8) * 255
        masks.append(mask)
        if output_folder is not None:
            out_path = os.path.join(output_folder, os.path.basename(image_paths[idx]) + '.png')
            cv2.imwrite(out_path, mask)
    if PLOT:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(all_means, label='Center Means (per image)')
        plt.plot(np.arange(window_size//2, window_size//2+len(mov_avg_means)), mov_avg_means, label=f'Moving Avg Means (window={window_size})')
        plt.plot(all_devs, label='Center StdDev (per image)')
        plt.plot(np.arange(window_size//2, window_size//2+len(mov_avg_devs)), mov_avg_devs, label=f'Moving Avg StdDev (window={window_size})')
        plt.xlabel('Image Index')
        plt.ylabel('Value')
        plt.title('Center Means/StdDev and Moving Averages')
        plt.legend()
        plt.tight_layout()
        plt.show()
    return masks

def process_image(image_path, output_folder):
    mask = generate_mask(image_path)
    if mask is not None:
        output_path = os.path.join(output_folder, os.path.basename(image_path) +'.png')
        cv2.imwrite(output_path, mask)

def process_folder(input_folder, output_folder, num_workers=4):
    os.makedirs(output_folder, exist_ok=True)
    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_image, image_paths, [output_folder] * len(image_paths))

if __name__ == '__main__':
    # Your multiprocessing code here
    # Example usage
    input_folder = r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Test\Masking_moving_average\images_Ferse"  # Change to your folder path
    test_img=r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_Resolution\Trial2\raw_images\Ferse_3+4_frame_0114.jpg"
    # get one folder up:
    # output_folder = os.path.join(input_folder, "", "Masks")
    # process_folder(input_folder, output_folder)
    #for images in os.listdir(input_folder):
    #generate_mask(test_img,exp_factor=2,blur_ksize=33, PLOT=True)
    # Test single image
    # image_path = r"C:\DATEN\wounds\Tumor\Run1\Subset_418\Images\Tumor_0279.jpg"
    # mask = process_image(image_path, output_folder)
    #process_folder(input_folder, r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Test\Masking_moving_average\normal")
    moving_average_mask(input_folder, PLOT=True, output_folder=r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Test\Masking_moving_average\moving_average")#r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Test\Masking_moving_average\moving_average")

