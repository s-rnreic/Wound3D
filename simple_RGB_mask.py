import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_masks(folder_path, dest_folder_path, r_thresh, g_thresh, b_thresh):
    if not os.path.exists(dest_folder_path):
        os.makedirs(dest_folder_path)
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Apply thresholding
            r_mask = img[:, :, 2] > r_thresh
            g_mask = img[:, :, 1] > g_thresh
            b_mask = img[:, :, 0] > b_thresh
            
            # Combine masks
            mask = r_mask & g_mask & b_mask
            kernel = np.ones((17, 17), np.uint8)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            # Save mask
            mask_path = os.path.join(dest_folder_path, f'{os.path.splitext(filename)[0]}.jpg.png')
            cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)

def plot_first_image_mask(folder_path, r_thresh, g_thresh, b_thresh):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Apply thresholding
            r_mask = img[:, :, 2] > r_thresh
            g_mask = img[:, :, 1] > g_thresh
            b_mask = img[:, :, 0] > b_thresh
            
            # Combine masks
            mask = r_mask & g_mask & b_mask
            
            # Apply morphological operations to remove small black spots
            kernel = np.ones((17, 17), np.uint8)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

            # Plot the mask
            plt.imshow(mask, cmap='gray')
            plt.title(f'Mask for {filename}')
            plt.show()
            break

if __name__ == "__main__":
    folder_path = r'C:\Users\Reich\LocalCoding\COLMAP_Pipline\test\raw_images'
    dest_folder_path = r'C:\Users\Reich\LocalCoding\COLMAP_Pipline\test\masks'
    r_thresh = 125
    g_thresh = 0
    b_thresh = 0
    
    create_masks(folder_path, dest_folder_path, r_thresh, g_thresh, b_thresh)
    
    TEST = False
    if TEST:
        plot_first_image_mask(folder_path, r_thresh, g_thresh, b_thresh)
