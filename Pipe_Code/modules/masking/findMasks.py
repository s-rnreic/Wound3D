import os
import glob
from pathlib import Path


def find_corresponding_masks(image_folder, mask_folder):
    """
    Find corresponding mask files for images in the image_folder.
    
    Args:
        image_folder (str): Path to the folder containing images
        mask_folder (str): Path to the folder containing mask files
        
    Returns:
        dict: Dictionary mapping image paths to their corresponding mask paths.
              If no mask is found for an image, the value will be None.
    """
    # Convert to Path objects for better path handling
    image_folder = Path(image_folder)
    mask_folder = Path(mask_folder)
    
    # Get all image files (common image extensions)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_folder.glob(f'*{ext}'))
    
    # Get all mask files (could be any extension)
    all_mask_files = list(mask_folder.glob('*.*'))
    
    # Create a dictionary to store the mapping
    image_to_mask = {}
    
    for image_path in image_files:
        image_name = image_path.stem  # Get filename without extension
        
        # Look for matching mask with any extension
        matching_masks = [
            mask_path for mask_path in all_mask_files
            if mask_path.stem == image_name or mask_path.stem.startswith(f"{image_name}_")
        ]
        
        # Also check for masks that might have different naming patterns but contain the image name
        if not matching_masks:
            matching_masks = [
                mask_path for mask_path in all_mask_files
                if image_name in mask_path.stem
            ]
        
        # Store the result in the dictionary
        if matching_masks:
            image_to_mask[str(image_path)] = str(matching_masks[0])
        else:
            image_to_mask[str(image_path)] = None
    
    return image_to_mask


def find_mask_for_image(image_path, mask_folder):
    """
    Find the corresponding mask file for a single image.
    
    Args:
        image_path (str): Path to the image file
        mask_folder (str): Path to the folder containing mask files
        
    Returns:
        str or None: Path to the corresponding mask file, or None if not found
    """
    image_path = Path(image_path)
    mask_folder = Path(mask_folder)
    
    image_name = image_path.stem  # Get filename without extension
    
    # Look for mask with any extension
    matching_masks = list(mask_folder.glob(f"{image_name}.*"))
    
    # Check for masks with a suffix after the image name
    if not matching_masks:
        matching_masks = list(mask_folder.glob(f"{image_name}_*.*"))
    
    # Try a broader search if still not found
    if not matching_masks:
        all_mask_files = list(mask_folder.glob('*.*'))
        matching_masks = [
            mask_path for mask_path in all_mask_files
            if image_name in mask_path.stem
        ]
    
    # Return the first matching mask if found
    if matching_masks:
        return str(matching_masks[0])
    
    return None


if __name__ == "__main__":
    # Example usage
    image_folder = r"C:\Users\Reich\LocalCoding\Wundbilder\Wundbilder_XotoCAM\Ferse\Sharp_images"
    mask_folder = r"C:\Users\Reich\LocalCoding\Wundbilder\Wundbilder_XotoCAM\Ferse\Maske_CombinedLAB"
    
    # Find masks for all images in a folder
    mapping = find_corresponding_masks(image_folder, mask_folder)
    for img_path, mask_path in mapping.items():
        if mask_path:
            print(f"Image: {img_path} -> Mask: {mask_path}")
        else:
            print(f"Image: {img_path} -> No mask found")
    
    