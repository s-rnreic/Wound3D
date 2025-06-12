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
    input_folder = r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\test\raw_images"  # Change to your folder path
    test_img=r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_Resolution\Trial2\raw_images\Ferse_3+4_frame_0114.jpg"
    # get one folder up:
    # output_folder = os.path.join(input_folder, "", "Masks")
    # process_folder(input_folder, output_folder)
    #for images in os.listdir(input_folder):
    generate_mask(test_img,exp_factor=2,blur_ksize=33, PLOT=True)
    # Test single image
    # image_path = r"C:\DATEN\wounds\Tumor\Run1\Subset_418\Images\Tumor_0279.jpg"
    # mask = process_image(image_path, output_folder)

# get center region of ALL images
# Decide which color to use based on that.