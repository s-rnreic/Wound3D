import numpy as np
from skimage import io, exposure
from skimage.color import gray2rgb, rgb2gray
import matplotlib.pyplot as plt
import cv2

# Modify functions to work on RGB images

def histogram_equalization(image):
    """
    Apply histogram equalization to an image.

    Parameters:
    - image: Input image (RGB).

    Returns:
    - equalized_image: Image after histogram equalization.
    """
    equalized_image = exposure.equalize_hist(image)
    equalized_image = (equalized_image * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    return equalized_image

def gamma_correction(image, gamma=1.0):
    """
    Apply gamma correction to an image.

    Parameters:
    - image: Input image (RGB).
    - gamma: The gamma value for the correction (default is 1.0).

    Returns:
    - gamma_corrected: Image after gamma correction.
    """
    gamma_corrected = exposure.adjust_gamma(image, gamma)
    return gamma_corrected

def contrast_stretching(image):
    """
    Apply contrast stretching to an image.

    Parameters:
    - image: Input image (RGB).

    Returns:
    - contrast_stretched: Image after contrast stretching.
    """
    p2, p98 = np.percentile(image, (2, 98))  # Use the 2nd and 98th percentiles for clipping
    contrast_stretched = exposure.rescale_intensity(image, in_range=(p2, p98))
    return contrast_stretched

def adaptive_histogram_equalization(image, kernel_size=256):
    """
    Apply Adaptive Histogram Equalization (AHE) to an image.

    Parameters:
    - image: Input image (RGB).
    - kernel_size: Size of the kernel for local contrast enhancement.

    Returns:
    - ahe_image: Image after adaptive histogram equalization.
    """
    ahe_image = exposure.equalize_adapthist(image, kernel_size=kernel_size)
    ahe_image = (ahe_image * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    return ahe_image

# Update main function to load and process RGB images
if __name__ == "__main__":
    # Load the image
    image = cv2.imread(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments_Enhancement_histogram_equalization\raw_images\Ferse_3+4_frame_0005.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    output_path = r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Test"
    
    # Apply different contrast enhancement techniques
    equalized_image = histogram_equalization(image)
    gamma_corrected_image = gamma_correction(image)
    contrast_stretched_image = contrast_stretching(image)
    ahe_image = adaptive_histogram_equalization(image)

    # Save the processed images
    cv2.imwrite(f'{output_path}/equalized_image.jpg', cv2.cvtColor(equalized_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'{output_path}/gamma_corrected_image.jpg', cv2.cvtColor(gamma_corrected_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'{output_path}/contrast_stretched_image.jpg', cv2.cvtColor(contrast_stretched_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'{output_path}/ahe_image.jpg', cv2.cvtColor(ahe_image, cv2.COLOR_RGB2BGR))

    # Plot the results
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(equalized_image)
    axes[1].set_title('Histogram Equalization')
    axes[1].axis('off')

    axes[2].imshow(gamma_corrected_image)
    axes[2].set_title('Gamma Correction')
    axes[2].axis('off')

    axes[3].imshow(contrast_stretched_image)
    axes[3].set_title('Contrast Stretching')
    axes[3].axis('off')

    axes[4].imshow(ahe_image)
    axes[4].set_title('Adaptive Histogram Equalization')
    axes[4].axis('off')

    plt.tight_layout()
    plt.show()