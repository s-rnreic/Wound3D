import cv2 
import numpy as np
import os
import glob
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


def Savitzky_Golay(sharpness_df, window_length=11, polyorder=3, PLOT=True, tolerance=0.05):
    """
    Selects images whose sharpness is above the Savitzky-Golay filtered value, or not smaller than 5% below the filter value.
    Args:
        sharpness_df (DataFrame): DataFrame containing sharpness values with a 'Sharpness' column.
        window_length (int): Window length for Savitzky-Golay filter (must be odd).
        polyorder (int): Polynomial order for Savitzky-Golay filter.
        PLOT (bool): Whether to plot the results.
        tolerance (float): Fractional tolerance below the filter value (default 0.05 = 5%).
    Returns:
        selected_indices (list): List of selected image indices.
    """
    # Extract sharpness values as a NumPy array
    sharpness_values = sharpness_df['Sharpness'].to_numpy()
    
    # Adjust window_length if needed to ensure it's valid
    if len(sharpness_values) < window_length:
        window_length = max(3, len(sharpness_values) // 2 * 2 + 1)
    
    # Apply Savitzky-Golay filter
    sg_filtered = savgol_filter(sharpness_values, window_length=window_length, polyorder=polyorder)
    
    # Select indices where sharpness meets criteria
    selected_indices = [i for i, s in enumerate(sharpness_values) if s > sg_filtered[i] or s >= sg_filtered[i] * (1 - tolerance)]
    
    if PLOT:
        plt.figure(figsize=(12, 6))
        plt.plot(sharpness_values, label='Sharpness', color='blue')
        plt.plot(sg_filtered, label=f'Savitzky-Golay (window={window_length}, poly={polyorder})', color='red')
        plt.scatter(selected_indices, sharpness_values[selected_indices], color='green', s=60, label='Selected Images')
        plt.xlabel('Image Index')
        plt.ylabel('Variance of Laplacian (Sharpness)')
        plt.title('Sharpness and Savitzky-Golay Filter Selection (with 5% tolerance)')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return selected_indices

if __name__ == "__main__":
    print("This module is not intended to be run directly. Please import it in your script.")