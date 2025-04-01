import cv2
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor

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
    
    # Compute the average value of the center circle (5% of the image)
    h, w = ab_image.shape
    center_x, center_y = w // 2, h // 2
    radius = int(0.1 * min(h, w) / 2)
    y, x = np.ogrid[:h, :w]
    mask_circle = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    threshold = np.mean(ab_image[mask_circle])
    deviation = 2*np.std(ab_image[mask_circle])
    
    # Compute importance weighting (radial gradient)
    y, x = np.indices((h, w))
    dist_map = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    dist_map = dist_map ** exp_factor  # Make the gradient steeper
    dist_map = 1 - (dist_map / np.max(dist_map))  # Normalize to [0,1]
    
    # Apply radial gradient to enhance central importance
    weighted_ab = ab_image * dist_map
    
    # Mask is everything above 
    mask = (weighted_ab + deviation> threshold).astype(np.uint8) * 255
    
    if PLOT:
        cv2.imshow('Mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
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
    test_img=r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\test\raw_images\Ferse_3_frame_0053.jpg"
    # get one folder up:
    # output_folder = os.path.join(input_folder, "", "Masks")
    # process_folder(input_folder, output_folder)
    #for images in os.listdir(input_folder):
    generate_mask(test_img,exp_factor=25,blur_ksize=33, PLOT=True)
    # Test single image
    # image_path = r"C:\DATEN\wounds\Tumor\Run1\Subset_418\Images\Tumor_0279.jpg"
    # mask = process_image(image_path, output_folder)

# get center region of ALL images
# Decide which color to use based on that.