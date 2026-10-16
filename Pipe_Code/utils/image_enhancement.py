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
