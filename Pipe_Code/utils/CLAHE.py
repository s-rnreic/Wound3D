import cv2
import numpy as np

def apply_clahe(image_path: str, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.

    Parameters:
        image_path (str): Path to the input image.
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (tuple): Size of the grid for histogram equalization.

    Returns:
        np.ndarray: Image with enhanced contrast.
    """
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Convert the image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into its channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel with the a and b channels
    lab_clahe = cv2.merge((l_clahe, a, b))

    # Convert the LAB image back to BGR color space
    enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return enhanced_image

if __name__ == "__main__":
    # Example usage
    input_image_path = r"C:\Users\Reich\LocalCoding\COLMAP_WUNDE\IMG_Data\images_Ferse\Ferse_3_frame_0061.jpg"
    output_image_path = r"c:\Users\Reich\LocalCoding\COLMAP_Pipline\Test\test_clahe.jpg"

    try:
        enhanced_image = apply_clahe(input_image_path)
        # Save the enhanced image
        cv2.imwrite(output_image_path, enhanced_image)
        print(f"Enhanced image saved to: {output_image_path}")
    except FileNotFoundError as e:
        print(e)