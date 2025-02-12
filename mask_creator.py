import cv2
import numpy as np
import sys, os
sys.path.append(r'C:\Users\Reich\LocalCoding\IMG_FILTER_PLOT_CONV')
import IMG_ColorConv as ICC
from concurrent.futures import ProcessPoolExecutor


def create_mask(image_path,colorspace='HSV',lower_bound=(50, 100, 100), upper_bound=(70, 255, 255))->np.ndarray:
    # Load an image
    image = cv2.imread(image_path)
    if image is None:
        print("Invalid input image. Please check the image path or ensure it is loaded correctly.")
        return None
    # Convert to HSV color space

    conv_image = ICC.convert_to(image, colorspace)[0]
    if conv_image is None or not isinstance(conv_image, np.ndarray):
        print("Conversion to specified color space failed.")
        return None
    # Define thresholds for masking
    # lower_bound = np.array(lower_bound)
    # upper_bound = np.array(upper_bound)
    # Create a mask
    mask = cv2.inRange(conv_image, lower_bound, upper_bound)
    # Apply the mask to get the result
    result = cv2.bitwise_and(image, image, mask=mask)

    return mask
# Load an image

def process_image(image_path, output_folder):
    
    mask = create_mask(image_path, colorspace='HSV', lower_bound=(0, 0, 115), upper_bound=(75, 255, 255))
    
    if mask is not None:
        output_path = os.path.join(output_folder, os.path.basename(image_path) +'.png')
        cv2.imwrite(output_path, mask)

def process_folder(input_folder, output_folder, num_workers=4):
    os.makedirs(output_folder, exist_ok=True)
    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]
    for image_path in image_paths:
        process_image(image_path, output_folder)
    # with ProcessPoolExecutor(max_workers=num_workers) as executor:
    #     executor.map(process_image, image_paths, [output_folder] * len(image_paths))

def main():
    input_folder = r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\MASSEKLASSE\images"  # Change to your folder path
    output_folder = r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\MASSEKLASSE\mask"
    process_folder(input_folder, output_folder)
    # image_path = r'C:\Users\Reich\LocalCoding\Wundbilder\small_dataset\image_1.jpg'
    # result = create_mask(image_path)
    # cv2.imshow('Result', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()