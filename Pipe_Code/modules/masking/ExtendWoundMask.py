"""
Extended Wound Mask Creator
============================

Creates extended versions of wound masks by adding surrounding tissue.
Extension is adaptive based on wound-to-tissue ratio:
- Small wounds (low ratio) get larger extensions
- Large wounds (high ratio) get smaller extensions
- Minimum extension: 20 pixels
- Extension only includes areas present in the wound+tissue mask
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from Pipe_Code.modules.masking.findMasks import find_corresponding_masks


class ExtendedWoundMaskCreator:
    def __init__(self, wound_mask_folder, tissue_mask_folder, output_folder, extension_percentage=0.15):
        """
        Initialize the Extended Wound Mask Creator.
        
        Extension is adaptive based on wound-to-tissue ratio:
        - Small wounds get larger extensions (more tissue margin)
        - Large wounds get smaller extensions (less tissue margin)
        - Minimum extension is always 20 pixels
        
        Args:
            wound_mask_folder (str): Path to folder containing wound-only masks
            tissue_mask_folder (str): Path to folder containing wound+tissue masks
            output_folder (str): Path to folder for saving extended wound masks
            extension_percentage (float): Base extension percentage (0.1-0.2), adjusted by wound-to-tissue ratio
        """
        self.wound_mask_folder = Path(wound_mask_folder)
        self.tissue_mask_folder = Path(tissue_mask_folder)
        self.output_folder = Path(output_folder)
        self.extension_percentage = extension_percentage
        
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Get wound and tissue mask mappings
        self.wound_masks = self._get_mask_files(self.wound_mask_folder)
        self.tissue_masks = self._get_mask_files(self.tissue_mask_folder)
        
        # Find matching pairs
        self.mask_pairs = self._find_mask_pairs()
        
    def _get_mask_files(self, folder):
        """Get all mask files from a folder."""
        mask_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            mask_files.extend(list(folder.glob(f'*{ext}')))
        return {f.stem: f for f in mask_files}
    
    def _find_mask_pairs(self):
        """Find matching pairs of wound and tissue masks."""
        pairs = []
        for name, wound_path in self.wound_masks.items():
            if name in self.tissue_masks:
                pairs.append((name, wound_path, self.tissue_masks[name]))
            else:
                print(f"Warning: No matching tissue mask for wound mask: {name}")
        return pairs
    
    def create_extended_masks(self, min_extension_px=20):
        """
        Create extended wound masks for all pairs.
        
        Args:
            min_extension_px (int): Minimum extension in pixels (default: 20)
            
        Returns:
            dict: Statistics about the mask creation process
        """
        stats = {
            'total': len(self.mask_pairs),
            'successful': 0,
            'failed': 0,
            'avg_extension_ratio': []
        }
        
        print(f"Creating extended masks for {len(self.mask_pairs)} image pairs...")
        
        for name, wound_path, tissue_path in tqdm(self.mask_pairs, desc="Processing masks"):
            try:
                # Load masks
                wound_mask = cv2.imread(str(wound_path), cv2.IMREAD_GRAYSCALE)
                tissue_mask = cv2.imread(str(tissue_path), cv2.IMREAD_GRAYSCALE)
                
                if wound_mask is None or tissue_mask is None:
                    print(f"Failed to load masks for: {name}")
                    stats['failed'] += 1
                    continue
                
                # Create extended mask
                extended_mask, extension_ratio = self._extend_wound_mask(
                    wound_mask, tissue_mask, min_extension_px
                )
                
                # Save extended mask
                output_path = self.output_folder / f"{name}.png"
                cv2.imwrite(str(output_path), extended_mask)
                
                stats['successful'] += 1
                stats['avg_extension_ratio'].append(extension_ratio)
                
            except Exception as e:
                print(f"Error processing {name}: {e}")
                stats['failed'] += 1
        
        # Calculate average extension ratio
        if stats['avg_extension_ratio']:
            stats['avg_extension_ratio'] = np.mean(stats['avg_extension_ratio'])
        else:
            stats['avg_extension_ratio'] = 0.0
        
        return stats
    
    def _extend_wound_mask(self, wound_mask, tissue_mask, min_extension_px=50):
        """
        Extend wound mask based on wound-to-tissue ratio, constrained by tissue mask.
        Small wound-to-tissue ratio = more extension
        Large wound-to-tissue ratio = less extension
        
        Args:
            wound_mask: Binary wound-only mask
            tissue_mask: Binary wound+tissue mask
            min_extension_px: Minimum extension in pixels (default: 20)
            
        Returns:
            tuple: (extended_mask, extension_ratio)
        """
        # Convert to binary
        wound_binary = (wound_mask > 127).astype(np.uint8)
        tissue_binary = (tissue_mask > 127).astype(np.uint8)
        
        # Calculate wound and tissue areas
        wound_area = np.sum(wound_binary)
        tissue_area = np.sum(tissue_binary)
        
        if wound_area == 0 or tissue_area == 0:
            return wound_mask, 0.0
        
        # Calculate wound-to-tissue ratio
        wound_to_tissue_ratio = wound_area / tissue_area
        
        # Adjust extension based on ratio (inverse relationship)
        # Formula: extension_multiplier = exp(-2 * ratio) + 0.5
        # Behavior:
        #   - ratio = 0.1 (small wound)  → multiplier ≈ 1.32 (132% of base extension)
        #   - ratio = 0.3 (medium wound) → multiplier ≈ 1.05 (105% of base extension)
        #   - ratio = 0.5 (large wound)  → multiplier ≈ 0.87 (87% of base extension)
        # This ensures small wounds get more tissue margin than large wounds
        extension_multiplier = np.exp(-2 * wound_to_tissue_ratio) + 0.6
        adjusted_extension_percentage = self.extension_percentage * extension_multiplier
        
        # Calculate dilation kernel size based on wound size and adjusted extension
        wound_radius = np.sqrt(wound_area / np.pi)
        kernel_size = max(
            min_extension_px,
            int(wound_radius * np.sqrt(adjusted_extension_percentage))
        )
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create kernel for dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Dilate wound mask
        dilated_wound = cv2.dilate(wound_binary, kernel, iterations=1)
        
        # Constrain to tissue mask (extended part must be in tissue area)
        extended_mask = np.bitwise_and(dilated_wound, tissue_binary)
        
        # Calculate actual extension ratio
        extended_area = np.sum(extended_mask)
        extension_ratio = (extended_area - wound_area) / wound_area if wound_area > 0 else 0.0
        
        # Convert back to 0-255 range
        extended_mask = (extended_mask * 255).astype(np.uint8)
        
        return extended_mask, extension_ratio
    
    def visualize_comparison(self, name, save=True, show=False):
        """
        Visualize the comparison of wound, tissue, and extended masks.
        
        Args:
            name (str): Name of the mask pair to visualize
            save (bool): Whether to save the visualization
            show (bool): Whether to display the visualization
        """
        import matplotlib.pyplot as plt
        
        # Find the mask pair
        wound_path = self.wound_masks.get(name)
        tissue_path = self.tissue_masks.get(name)
        extended_path = self.output_folder / f"{name}.png"
        
        if wound_path is None or tissue_path is None or not extended_path.exists():
            print(f"Cannot visualize {name}: masks not found")
            return
        
        # Load masks
        wound_mask = cv2.imread(str(wound_path), cv2.IMREAD_GRAYSCALE)
        tissue_mask = cv2.imread(str(tissue_path), cv2.IMREAD_GRAYSCALE)
        extended_mask = cv2.imread(str(extended_path), cv2.IMREAD_GRAYSCALE)
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(wound_mask, cmap='gray')
        axes[0].set_title('Wound Only')
        axes[0].axis('off')
        
        axes[1].imshow(tissue_mask, cmap='gray')
        axes[1].set_title('Wound + Tissue')
        axes[1].axis('off')
        
        axes[2].imshow(extended_mask, cmap='gray')
        axes[2].set_title(f'Extended Wound ({self.extension_percentage*100:.0f}%)')
        axes[2].axis('off')
        
        # Show difference (extension area)
        wound_binary = (wound_mask > 127).astype(np.uint8)
        extended_binary = (extended_mask > 127).astype(np.uint8)
        difference = extended_binary - wound_binary
        
        axes[3].imshow(difference, cmap='hot')
        axes[3].set_title('Extension Area')
        axes[3].axis('off')
        
        plt.suptitle(f'Mask Extension Visualization: {name}')
        plt.tight_layout()
        
        if save:
            viz_path = self.output_folder / f"{name}_visualization.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {viz_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def print_statistics(self, stats):
        """Print statistics about the mask creation process."""
        print("\n" + "="*60)
        print("EXTENDED WOUND MASK CREATION COMPLETE")
        print("="*60)
        print(f"Total mask pairs: {stats['total']}")
        print(f"Successfully processed: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Average extension ratio: {stats['avg_extension_ratio']*100:.2f}%")
        print(f"Base extension percentage: {self.extension_percentage*100:.0f}%")
        print(f"Extension adapts to wound-to-tissue ratio:")
        print(f"  - Small wounds → larger extension")
        print(f"  - Large wounds → smaller extension")
        print(f"  - Minimum extension: 20 pixels")
        print(f"Output folder: {self.output_folder}")
        print("="*60)


def create_extended_wound_masks(wound_mask_folder, tissue_mask_folder, output_folder, 
                                extension_percentage=0.15, visualize_samples=0):
    """
    Convenience function to create extended wound masks.
    
    Extension adapts to wound-to-tissue ratio - small wounds get larger extensions.
    Minimum extension is always 20 pixels.
    
    Args:
        wound_mask_folder (str): Path to wound-only masks
        tissue_mask_folder (str): Path to wound+tissue masks
        output_folder (str): Path for output extended masks
        extension_percentage (float): Base extension percentage (0.1-0.2), adjusted by ratio
        visualize_samples (int): Number of samples to visualize (0 to skip)
        
    Returns:
        dict: Statistics about the process
    """
    creator = ExtendedWoundMaskCreator(
        wound_mask_folder, 
        tissue_mask_folder, 
        output_folder,
        extension_percentage=extension_percentage
    )
    
    stats = creator.create_extended_masks()
    creator.print_statistics(stats)
    
    # Visualize samples if requested
    if visualize_samples > 0 and stats['successful'] > 0:
        print(f"\nGenerating {visualize_samples} sample visualizations...")
        sample_names = list(creator.wound_masks.keys())[:visualize_samples]
        for name in sample_names:
            creator.visualize_comparison(name, save=True, show=False)
    
    return stats


if __name__ == "__main__":
    # Example usage
    wound_mask_folder = r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\test\masks_wound_only"
    tissue_mask_folder = r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\test\masks"
    output_folder = r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\test\masks_wound_extended"
    
    # Create extended masks with 70% extension
    stats = create_extended_wound_masks(
        wound_mask_folder,
        tissue_mask_folder,
        output_folder,
        extension_percentage=1.0,  # 70% extension
        visualize_samples=3  # Visualize first 3 samples
    )
