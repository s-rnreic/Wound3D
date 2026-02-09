import os
import cv2
from matplotlib import image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import deque
import math
from Pipe_Code.modules.masking.findMasks import find_corresponding_masks
from pathlib import Path
from tqdm import tqdm  # For progress display during loading


class MaskAdjustment:
    def __init__(self, image_folder, mask_folder, output_folder, min_mask_ratio=0.1, max_mask_ratio=0.8, neighbors_count=3):
        """
        Initialize the MaskAdjustment class.
        
        Args:
            image_folder (str): Path to folder containing images
            mask_folder (str): Path to folder containing corresponding masks
            output_folder (str): Path to folder for saving adjusted masks and evaluation results
            min_mask_ratio (float): Minimum acceptable mask area as ratio of image area (default: 0.1)
            max_mask_ratio (float): Maximum acceptable mask area as ratio of image area (default: 0.8)
            neighbors_count (int): Number of neighbors to consider for each mask (default: 5)
        """
        self.image_folder = Path(image_folder)
        self.image_data = {}  # Will store loaded images
        self.mask_folder = Path(mask_folder)
        self.mask_data = {}   # Will store loaded masks
        self.output_folder = Path(output_folder)
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.neighbors_count = neighbors_count
        
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Get image-mask mappings
        self.image_mask_map = find_corresponding_masks(image_folder, mask_folder)
        
        # Sort images by name to establish sequence
        self.sorted_images = sorted(list(self.image_mask_map.keys()))
        
        # Preload all images and masks to avoid repeated disk I/O
        print("Loading all images and masks...")
        self._preload_data()
        
        
        
        # Store mask properties
        self.mask_properties = {}
        
        # Calculate mask properties for all images
        print("Calculating mask properties...")
        self._calculate_all_mask_properties()
        
        self._calculate_solidity()
        
    def _preload_data(self):
        """
        Preload all images and their corresponding masks into memory
        """
        print(f"Loading {len(self.sorted_images)} images and masks...")
        
        for image_path in tqdm(self.sorted_images, desc="Loading data"):
            mask_path = self.image_mask_map[image_path]
            
            # Load image if it exists
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

                combined_ab = cv2.add(img[:, :, 1], img[:, :, 2]).astype(np.float64)  # Combine A and B channels

                # Normalize combined_ab from 0 to 255
                combined_ab = np.clip(combined_ab, combined_ab.min(), combined_ab.max())  # Clip to min/max range
                combined_ab = ((combined_ab - combined_ab.min()) / (combined_ab.max() - combined_ab.min()) * 255).astype(np.uint8)
                img = combined_ab

                if img is not None:
                    self.image_data[image_path] = img
                else:
                    print(f"Warning: Could not load image: {image_path}")
            

            # Load mask if it exists
            if mask_path and os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    self.mask_data[mask_path] = mask
                else:
                    print(f"Warning: Could not load mask: {mask_path}")
        
        print(f"Loaded {len(self.image_data)} images and {len(self.mask_data)} masks")
        
    def _calculate_all_mask_properties(self):
        """Calculate properties for all masks in the dataset"""
        for image_path in tqdm(self.sorted_images, desc="Calculating properties"):
            mask_path = self.image_mask_map[image_path]
            if mask_path:
                # Get image and mask from preloaded data
                image = self.image_data.get(image_path)
                mask = self.mask_data.get(mask_path)
                
                if image is None or mask is None:
                    print(f"Warning: Missing preloaded image or mask: {image_path} or {mask_path}")
                    continue
                
                # Calculate properties
                properties = self._calculate_mask_properties(image, mask)
                self.mask_properties[image_path] = {
                    'mask_path': mask_path,
                    'area': properties['area'],
                    'area_ratio': properties['area_ratio'],
                    'lab_mean': properties['lab_mean'],
                    'lab_std': properties['lab_std'],
                    'position': properties['position'],
                    'is_valid': self.min_mask_ratio <= properties['area_ratio'] <= self.max_mask_ratio,
                    'neighbors': self._find_neighbors(image_path)
                }
    
    def _calculate_mask_properties(self, image, mask):
        """
        Calculate properties of a mask including area, LAB mean and std
        
        Args:
            image: RGB image
            mask: Binary mask (0 for background, 255 for foreground)
        
        Returns:
            dict: Properties of the mask
        """
        # Convert mask to binary (0 and 1)
        binary_mask = mask > 0
        
        # Calculate mask area
        area = np.sum(binary_mask)
        area_ratio = area / (image.shape[0] * image.shape[1])
        
        # Calculate center of mass (position)
        y_indices, x_indices = np.where(binary_mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            position = (np.mean(x_indices), np.mean(y_indices))
        else:
            position = (image.shape[1] // 2, image.shape[0] // 2)

        
        # Calculate mean and std of combined LAB inside the mask
        if area > 0:
            lab_mean = np.mean(image[binary_mask])
            lab_std = np.std(image[binary_mask])
        else:
            lab_mean = 0
            lab_std = 0
        
        return {
            'area': area,
            'area_ratio': area_ratio,
            'lab_mean': lab_mean,
            'lab_std': lab_std,
            'position': position
        }
    
    def _find_neighbors(self, image_path):
        """
        Find n closest neighbors for a given image based on position in sequence
        
        Args:
            image_path: Path of the image
        
        Returns:
            list: Paths of neighboring images
        """
        index = self.sorted_images.index(image_path)
        n = self.neighbors_count
        
        # Get indices of neighbors, ensuring we don't go out of bounds
        neighbor_indices = []
        for i in range(-n, n+1):
            if i != 0:  # Skip the image itself
                neighbor_idx = index + i
                if 0 <= neighbor_idx < len(self.sorted_images):
                    neighbor_indices.append(neighbor_idx)
        
        # Return paths of neighbors
        return [self.sorted_images[i] for i in neighbor_indices]
    
    def _calculate_solidity(self):
        """
        Calculate solidity of masks based on area and convex hull area
        """
        for image_path, mask_props in tqdm(self.mask_properties.items(), desc="Calculating solidity"):
            mask_path = mask_props['mask_path']
            mask = self.mask_data.get(mask_path)
            
            if mask is None:
                continue
            
            # Use scikit-image regionprops to calculate solidity
            import skimage.measure
            region_props = skimage.measure.regionprops(mask.astype(int))
            
            # If no regions found (empty mask), set solidity to 0
            if not region_props:
                mask_props['solidity'] = 0.0
                continue
                
            # Otherwise, get solidity from the first (and usually only) region
            mask_props['solidity'] = region_props[0].solidity

    def evaluate_masks(self, visualize=True):
        """
        Evaluate all masks based on area ratio criteria and calculate moving median
        
        Args:
            visualize (bool): Whether to generate visualization of the evaluation
            
        Returns:
            tuple: (valid_masks, invalid_masks) - lists of image paths with valid and invalid masks
        """
        # Calculate moving median for all masks
        #self._calculate_moving_median()
        
        
        self._calculate_savitky()
        
        for img_path, props in self.mask_properties.items():
            if props['is_valid']:
                if props['area_ratio'] < props['savitzky_area'] - 0.1 or props['area_ratio'] > props['savitzky_area'] + 0.1:
                    props['is_valid'] = False

        valid_masks = []
        invalid_masks = []
        
        for image_path, props in self.mask_properties.items():
            if props['is_valid']:
                valid_masks.append(image_path)
            else:
                invalid_masks.append(image_path)
        
        if visualize:
            self._visualize_evaluation()
            
        return valid_masks, invalid_masks
    
    def _calculate_savitky(self, polynomial_order=2, window_size=15):
        """
        Calculate Savitzky-Golay filter for smoothing mask areas
        Only uses valid masks for filtering, then applies the smoothed values to all masks
        
        Args:
            polynomial_order (int): Order of the polynomial used in the filter
            window_size (int): Size of the window for the filter
        """
        from scipy.signal import savgol_filter
        
        # Get indices and area ratios of valid masks
        valid_indices = []
        valid_areas = []
        
        for i, img_path in enumerate(self.sorted_images):
            if img_path in self.mask_properties:
                props = self.mask_properties[img_path]
                if props['is_valid']:
                    valid_indices.append(i)
                    valid_areas.append(props['area_ratio'])
        
        if len(valid_areas) < window_size:
            # Not enough valid points for the window size
            window_size = max(5, len(valid_areas) - (len(valid_areas) % 2 == 0))  # Ensure odd window
        
        if len(valid_areas) < 3:
            print("Warning: Not enough valid masks for Savitzky-Golay filtering. Using simple averaging.")
            # Use simple averaging instead
            avg_area = np.mean(valid_areas) if valid_areas else 0.5  # Default to 0.5 if no valid areas
            for img_path in self.sorted_images:
                if img_path in self.mask_properties:
                    self.mask_properties[img_path]['savitzky_area'] = avg_area
            return
        
        # Apply filter to valid areas
        smoothed_valid_areas = savgol_filter(valid_areas, window_size, polynomial_order)
        
        # Create a mapping from indices to smoothed values
        index_to_smoothed = {}
        for i, orig_idx in enumerate(valid_indices):
            index_to_smoothed[orig_idx] = smoothed_valid_areas[i]
        
        # Interpolate to get values for invalid masks
        for i, img_path in enumerate(self.sorted_images):
            if img_path in self.mask_properties:
                if i in index_to_smoothed:
                    # Direct mapping for valid masks
                    self.mask_properties[img_path]['savitzky_area'] = index_to_smoothed[i]
                else:
                    # For invalid masks, interpolate from nearest valid neighbors
                    # Find nearest valid indices before and after
                    prev_valid = None
                    next_valid = None
                    
                    for valid_idx in valid_indices:
                        if valid_idx < i:
                            prev_valid = valid_idx
                        elif valid_idx > i:
                            next_valid = valid_idx
                            break
                    
                    if prev_valid is not None and next_valid is not None:
                        # Linear interpolation
                        weight_next = (i - prev_valid) / (next_valid - prev_valid)
                        weight_prev = 1 - weight_next
                        interpolated = (weight_prev * index_to_smoothed[prev_valid] + 
                                       weight_next * index_to_smoothed[next_valid])
                    elif prev_valid is not None:
                        # Extrapolation from previous
                        interpolated = index_to_smoothed[prev_valid]
                    elif next_valid is not None:
                        # Extrapolation from next
                        interpolated = index_to_smoothed[next_valid]
                    else:
                        # No valid masks, use a default value
                        interpolated = 0.5  # Default area ratio
                    
                    self.mask_properties[img_path]['savitzky_area'] = interpolated
    
    def _calculate_moving_median(self, window_size=10):
        """
        Calculate moving median of mask areas
        
        Args:
            window_size (int): Size of the window for median calculation
        """
        # Extract areas in sequence order
        areas = [self.mask_properties[img]['area_ratio'] for img in self.sorted_images 
                 if self.image_mask_map[img] is not None]
        
        # Handle edge cases by mirroring the data at the boundaries
        padded_areas = []
        # Add reversed beginning
        padded_areas.extend(areas[:window_size//2][::-1])
        # Add original data
        padded_areas.extend(areas)
        # Add reversed end
        padded_areas.extend(areas[-(window_size//2):][::-1])
        
        # Calculate moving median
        medians = []
        half_window = window_size // 2
        for i in range(half_window, len(padded_areas) - half_window):
            window = padded_areas[i-half_window:i+half_window+1]
            medians.append(np.median(window))
        
        # Store the median values
        for i, img_path in enumerate(self.sorted_images):
            if self.image_mask_map[img_path] is not None:
                self.mask_properties[img_path]['area_median'] = medians[i]
    
    def _visualize_evaluation(self):
        """Generate visualization of mask areas and their evaluation"""
        areas = []
        medians = []
        statuses = []
        savitzky_areas = []
        for img_path in self.sorted_images:
            if self.image_mask_map[img_path] is not None:
                props = self.mask_properties[img_path]
                areas.append(props['area_ratio'])
                #medians.append(props['area_median'])
                savitzky_areas.append(props['savitzky_area'])
                statuses.append(props['is_valid'])
        
        # Generate plot
        plt.figure(figsize=(12, 6))
        x = range(len(areas))
        
        # Plot areas
        plt.scatter(x, areas, c=['g' if valid else 'r' for valid in statuses], label='Mask Area Ratio')
        #plt.plot(x, medians, 'b-', label='Moving Median')
        plt.plot(x, savitzky_areas, 'm-', label='Savitzky-Golay Smoothed Area')
        # Plot threshold lines
        plt.axhline(y=self.min_mask_ratio, color='r', linestyle='--', label=f'Min Threshold ({self.min_mask_ratio})')
        plt.axhline(y=self.max_mask_ratio, color='r', linestyle='--', label=f'Max Threshold ({self.max_mask_ratio})')
        
        plt.xlabel('Image Index')
        plt.ylabel('Mask Area Ratio')
        plt.title('Mask Area Evaluation')
        plt.legend()
        plt.grid(True)

        plt.show()
        # Save the plot
        #plt.savefig(os.path.join(self.output_folder, 'mask_evaluation.png'))
        plt.close()
    
    def simple_small_mask_adjustment(self):
        """
        Adjust masks that are too small by applying dilation until they meet the minimum size threshold
        
        Args:
            save_adjusted (bool): Whether to save adjusted masks to output folder
            
        Returns:
            dict: Mapping of original mask paths to adjusted mask paths
        """
        adjusted_masks = {}
        
        for img_path in tqdm(self.sorted_images, desc="Adjusting small masks"):
            if img_path not in self.mask_properties:
                continue
                
            props = self.mask_properties[img_path]
            # Only process masks that are valid but too small
            if props['area_ratio'] < self.min_mask_ratio and props['mask_path'] is not None:
                # Get image and mask from preloaded data
                mask_path = props['mask_path']
                image = self.image_data.get(img_path)
                original_mask = self.mask_data.get(mask_path)
                
                if image is None or original_mask is None:
                    print(f"Warning: Missing preloaded image or mask: {img_path} or {mask_path}")
                    continue
                
                # Create a copy for adjustment
                adjusted_mask = original_mask.copy()
                img_area = image.shape[0] * image.shape[1]
                target_area = self.min_mask_ratio * img_area
                
                # Start with a small kernel and gradually increase
                kernel_size = 3
                max_iterations = 50  # Prevent infinite loop
                iterations = 0
                
                while np.sum(adjusted_mask > 0) < target_area and iterations < max_iterations:
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    adjusted_mask = cv2.dilate(adjusted_mask, kernel, iterations=1)
                    iterations += 1
                    
                    # If we're not making enough progress, increase kernel size
                    if iterations % 5 == 0:
                        kernel_size += 2

                new_props = self._calculate_mask_properties(image, adjusted_mask)
                self.mask_properties[img_path].update({
                    'area': new_props['area'],
                    'area_ratio': new_props['area_ratio'],
                    'lab_mean': new_props['lab_mean'],
                    'lab_std': new_props['lab_std'],
                    'position': new_props['position'],
                    'is_valid': self.min_mask_ratio <= new_props['area_ratio'] <= self.max_mask_ratio,
                })
                
                # Update the mask data in our data structure
                self.mask_data[mask_path] = adjusted_mask

                # Create comparison visualization
                #self._visualize_adjustment(image, original_mask, adjusted_mask, img_path)

        return adjusted_masks
        
    def perform_preprocessing(self, kernel_size_ratio=0.05, iterations=1, min_solidity=0.8, PLOT=False):
        """
        Perform morphological opening operation on masks to remove small noise,
        then perform closing if solidity is below threshold.
        If multiple components remain, keep only the one closest to the image center.
        
        Args:
            kernel_size_ratio (float): Ratio of image size to determine kernel size (default: 0.05)
            iterations (int): Number of iterations for the opening operation
            save_adjusted (bool): Whether to save preprocessed masks to output folder
            min_solidity (float): Minimum solidity threshold (default: 0.8)
            
        Returns:
            dict: Mapping of original mask paths to preprocessed mask paths
        """
        preprocessed_masks = {}
        
        for img_path in tqdm(self.sorted_images, desc="Preprocessing masks"):
            if img_path not in self.mask_properties:
                continue
                
            props = self.mask_properties[img_path]
            mask_path = props['mask_path']
            if mask_path is None:
                continue
            
            # Get image and mask from preloaded data
            mask = self.mask_data.get(mask_path)
            image = self.image_data.get(img_path)
            avg_dimension = (image.shape[0] + image.shape[1]) / 2
            if mask is None or image is None:
                print(f"Warning: Missing preloaded image or mask: {img_path} or {mask_path}")
                continue
                
            # Define kernel sizes for morphological operations
            closing_kernel_size = max(3, int(avg_dimension * kernel_size_ratio))
            closing_kernel_size = closing_kernel_size if closing_kernel_size % 2 == 1 else closing_kernel_size + 1  # Ensure odd kernel size
            closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
            
            # First perform closing to connect close parts
            preprocessed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel, iterations=1)
            
            # Then perform opening to remove small noise
            opening_kernel_size_ratio = 0.03
            opening_kernel_size = max(3, int(avg_dimension * opening_kernel_size_ratio))
            opening_kernel_size = opening_kernel_size if opening_kernel_size % 2 == 1 else opening_kernel_size + 1  # Ensure odd kernel size
            opening_kernel = np.ones((opening_kernel_size, opening_kernel_size), np.uint8)
            preprocessed_mask = cv2.morphologyEx(preprocessed_mask, cv2.MORPH_OPEN, opening_kernel, iterations=iterations)

            if PLOT:    
                # Display the original and opened mask
                plt.subplot(1, 3, 1)
                plt.imshow(mask, cmap='gray')
                plt.title('Original Mask')
                
                plt.subplot(1, 3, 2)
                plt.imshow(preprocessed_mask, cmap='gray')
                plt.title('Opened Mask')
            
            # Calculate and improve solidity
            import skimage.measure
            mask_props = skimage.measure.regionprops(mask.astype(int))
            solidity = mask_props[0].solidity if mask_props else 0.0

            # Perform closing operation until solidity reaches threshold
            max_closing_iterations = 20  # Prevent infinite loop
            closing_iterations = 0
            
            while solidity < min_solidity and closing_iterations < max_closing_iterations:
                # Perform closing operation
                preprocessed_mask = cv2.morphologyEx(preprocessed_mask, cv2.MORPH_CLOSE, closing_kernel, iterations=1)

                mask_props = skimage.measure.regionprops(preprocessed_mask.astype(int))
                solidity = mask_props[0].solidity if mask_props else 0.0

                closing_iterations += 1
            
            # Check if the mask has multiple disconnected components
            # Label each connected component with a unique ID
            num_labels, labels = cv2.connectedComponents(preprocessed_mask)
            
            # If we have multiple components (num_labels > 1 because background is also a component)
            if num_labels > 2:
                # Calculate image center
                center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
                
                # Find the component closest to the center
                min_distance = float('inf')
                closest_label = 0
                
                # Analyze each component (skip label 0 which is background)
                for label in range(1, num_labels):
                    # Get coordinates of all pixels in this component
                    component = (labels == label).astype(np.uint8)
                    y_indices, x_indices = np.where(component > 0)
                    
                    if len(y_indices) == 0:
                        continue
                        
                    # Calculate centroid
                    centroid_y = np.mean(y_indices)
                    centroid_x = np.mean(x_indices)
                    
                    # Calculate distance to image center
                    distance = np.sqrt((centroid_y - center_y)**2 + (centroid_x - center_x)**2)
                    
                    # Update closest component
                    if distance < min_distance:
                        min_distance = distance
                        closest_label = label
                
                # Create a mask with only the closest component
                if closest_label > 0:
                    preprocessed_mask = (labels == closest_label).astype(np.uint8) * 255
            
            # Display the final processed mask
            if PLOT:
                plt.subplot(1, 3, 3)
                plt.imshow(preprocessed_mask, cmap='gray')
                plt.title('Final Mask')

                plt.show()
            
            # Recalculate solidity for the final mask (important if we kept only one component)
            mask_props = skimage.measure.regionprops(preprocessed_mask.astype(int))
            solidity = mask_props[0].solidity if mask_props else 0.0
            
            # Update mask properties
            new_props = self._calculate_mask_properties(image, preprocessed_mask)
            self.mask_properties[img_path].update({
                'area': new_props['area'],
                'area_ratio': new_props['area_ratio'],
                'lab_mean': new_props['lab_mean'],
                'lab_std': new_props['lab_std'],
                'position': new_props['position'],
                'solidity': solidity,  # Store the updated solidity
                'is_valid': self.min_mask_ratio <= new_props['area_ratio'] <= self.max_mask_ratio,
            })
            
            # Update the mask data in our data structure
            self.mask_data[mask_path] = preprocessed_mask

        
        return preprocessed_masks
        
    def adjust_masks(self, save_adjusted=False):
        """
        Adjust masks that are outside the acceptable range
        
        Args:
            save_adjusted (bool): Whether to save adjusted masks to output folder
            
        Returns:
            dict: Mapping of original mask paths to adjusted mask paths
        """
        
        # first check the invalid masks and their neighbors
        # calculate for their valid neighbors the mean mean and mean std value and give als the std for the mean
                
                

        adjusted_masks = {}
        
        for img_path in tqdm(self.sorted_images, desc="Adjusting masks"):
            if img_path not in self.mask_properties:
                continue
                
            props = self.mask_properties[img_path]
            mask_path = props['mask_path']
            if not props['is_valid'] and mask_path is not None:
                # Find valid neighbors
                valid_neighbors = [n for n in props['neighbors'] 
                                  if n in self.mask_properties and 
                                  self.mask_properties[n]['is_valid']]
                
                if not valid_neighbors:
                    print(f"Warning: No valid neighbors found for {img_path}. Cannot adjust mask.")
                    continue
                
                # Calculate mean LAB values from valid neighbors
                lab_means = []
                lab_stds = []
                mean_area_ratios = []
                for neighbor in valid_neighbors:
                    neighbor_props = self.mask_properties[neighbor]
                    lab_means.append(neighbor_props['lab_mean'])
                    lab_stds.append(neighbor_props['lab_std'])
                    mean_area_ratios.append(neighbor_props['area_ratio'])

                # Calculate overall mean and std
                props.update({'neighbour_mean_mean_lab': np.mean(lab_means), 'neighbour_mean_std_lab': np.mean(lab_stds), 
                             'mean_area_ratios': np.mean(mean_area_ratios)})
                
                print(f'difference between {img_path} and its valid neighbors:')
                print(f"  mean LAB: {props['lab_mean']} vs {props['neighbour_mean_mean_lab']}")
                print(f"  std LAB: {props['lab_std']} vs {props['neighbour_mean_std_lab']}")
                print(f"  mean area ratio: {props['area_ratio']} vs {props['mean_area_ratios']}")

                # Get image and mask from preloaded data
                image = self.image_data.get(img_path)
                original_mask = self.mask_data.get(mask_path)
                
                if image is None or original_mask is None:
                    print(f"Warning: Missing preloaded image or mask: {img_path} or {mask_path}")
                    continue

                if abs(props['lab_mean'] - props['neighbour_mean_mean_lab']) > 10 or \
                   abs(props['lab_std'] - props['neighbour_mean_std_lab']) > 10 or \
                   abs(props['mean_area_ratios'] - props['area_ratio']) > 0.15 :
                    # Create adjusted mask
                    adjusted_mask = self._interpolate_mask(img_path, valid_neighbors, image.shape[:2])
                    
                    # Update mask in our data structure
                    self.mask_data[mask_path] = adjusted_mask
                    
                    # Update mask properties
                    new_props = self._calculate_mask_properties(image, adjusted_mask)
                    self.mask_properties[img_path].update({
                        'area': new_props['area'],
                        'area_ratio': new_props['area_ratio'],
                        'lab_mean': new_props['lab_mean'],
                        'lab_std': new_props['lab_std'],
                        'position': new_props['position'],
                        'is_valid': self.min_mask_ratio <= new_props['area_ratio'] <= self.max_mask_ratio,
                    })
                    
                    
                    # Create comparison visualization
                    self._visualize_adjustment(image, original_mask, adjusted_mask, img_path)
        
        return adjusted_masks
    
    def _interpolate_mask(self, target_img_path, neighbor_img_paths, img_shape):
        """
        Interpolate a new mask based on neighboring valid masks
        
        Args:
            target_img_path: Path of the image needing mask adjustment
            neighbor_img_paths: List of paths to valid neighbor images
            img_shape: Shape of the target image (height, width)
            
        Returns:
            np.array: Adjusted binary mask
        """
        # Sort neighbor paths by their index in the sorted_images list
        target_idx = self.sorted_images.index(target_img_path)
        neighbor_indices = [self.sorted_images.index(n) for n in neighbor_img_paths]
        
        # Calculate distance to target
        distances = [abs(idx - target_idx) for idx in neighbor_indices]
        
        # Sort neighbors by distance
        neighbor_data = list(zip(distances, neighbor_img_paths, neighbor_indices))
        neighbor_data.sort()  # Sort by distance
        
        # Limit to closest neighbors if we have too many
        if len(neighbor_data) > 5:
            neighbor_data = neighbor_data[:5]
            
        # Get target image from preloaded data
        target_image = self.image_data.get(target_img_path)
        
        # Collect valid neighbor masks and positions
        neighbor_masks = []
        neighbor_positions = []
        neighbor_indices_sorted = []
        
        for _, n_path, n_idx in neighbor_data:
            n_mask_path = self.mask_properties[n_path]['mask_path']
            if n_mask_path:
                mask = self.mask_data.get(n_mask_path)
                if mask is not None:
                    neighbor_masks.append(mask)
                    neighbor_positions.append(self.mask_properties[n_path]['position'])
                    neighbor_indices_sorted.append(n_idx)
        
        if not neighbor_masks:
            return np.zeros(img_shape, dtype=np.uint8)
        
        # Calculate target position based on neighbor positions
        use_position_path = False
        interpolated_position = None
        
        # Only try position interpolation if we have at least 2 neighbors and they're in a sequence
        if len(neighbor_indices_sorted) >= 2:
            # Check if neighbor indices form a sequence (indicating a potential path)
            is_sequence = True
            neighbor_indices_sorted.sort()
            for i in range(1, len(neighbor_indices_sorted)):
                if abs(neighbor_indices_sorted[i] - neighbor_indices_sorted[i-1]) > 2:
                    is_sequence = False
                    break
            
            if is_sequence:
                # Verify if positions form a consistent path
                # Calculate if positions change consistently
                if len(neighbor_positions) >= 3:
                    positions = np.array(neighbor_positions)
                    dx = positions[1:, 0] - positions[:-1, 0]
                    dy = positions[1:, 1] - positions[:-1, 1]
                    
                    # Check if the direction changes consistently
                    consistent_direction = True
                    for i in range(1, len(dx)):
                        if (dx[i] * dx[i-1] < 0) or (dy[i] * dy[i-1] < 0):
                            consistent_direction = False
                            break
                    
                    if consistent_direction:
                        use_position_path = True
                        
                        # Find neighbors before and after target in sequence
                        before_neighbors = [idx for idx in neighbor_indices_sorted if idx < target_idx]
                        after_neighbors = [idx for idx in neighbor_indices_sorted if idx > target_idx]
                        
                        if before_neighbors and after_neighbors:
                            # We have neighbors before and after, interpolate position
                            before_idx = max(before_neighbors)
                            after_idx = min(after_neighbors)
                            
                            # Find the positions corresponding to these indices
                            before_position = None
                            after_position = None
                            
                            for i, n_idx in enumerate(neighbor_indices_sorted):
                                if n_idx == before_idx:
                                    before_position = neighbor_positions[i]
                                if n_idx == after_idx:
                                    after_position = neighbor_positions[i]
                            
                            if before_position and after_position:
                                # Linear interpolation of position
                                t = (target_idx - before_idx) / (after_idx - before_idx)
                                interpolated_position = (
                                    before_position[0] + t * (after_position[0] - before_position[0]),
                                    before_position[1] + t * (after_position[1] - before_position[1])
                                )
                        elif before_neighbors:
                            # Only neighbors before, extrapolate forward
                            interpolated_position = neighbor_positions[-1]
                        elif after_neighbors:
                            # Only neighbors after, extrapolate backward
                            interpolated_position = neighbor_positions[0]
        
        # Calculate average mask area from valid neighbors
        avg_area_ratio = np.mean([self.mask_properties[n]['area_ratio'] for _, n, _ in neighbor_data])
        
        # Access target properties
        target_props = self.mask_properties[target_img_path]
        
        # Start with a combined mask (average of neighbors)
        combined_mask = np.zeros(img_shape, dtype=np.float32)
        for mask in neighbor_masks:
            # Resize mask if needed
            if mask.shape != img_shape:
                mask = cv2.resize(mask, (img_shape[1], img_shape[0]))
            combined_mask += mask / 255.0
        
        combined_mask /= len(neighbor_masks)
        
        # Threshold to create binary mask
        # Adjust threshold to achieve desired area ratio
        for threshold in np.linspace(0.2, 0.8, 20):
            binary_mask = (combined_mask > threshold).astype(np.uint8) * 255
            area_ratio = np.sum(binary_mask > 0) / (img_shape[0] * img_shape[1])
            
            # If we're close enough to desired area ratio, stop
            if abs(area_ratio - avg_area_ratio) < 0.05:
                break
        
        # Apply morphological operations to smooth the mask
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Validation and adjustment based on LAB values
        max_iterations = 10
        iteration = 0
        
        # Get target neighbor values for validation
        target_lab_mean = target_props.get('neighbour_mean_mean_lab')
        target_lab_std = target_props.get('neighbour_mean_std_lab')
        
        if target_lab_mean is not None and target_lab_std is not None:
            # Apply position shift if we have an interpolated position
            if use_position_path and interpolated_position is not None:
                # Create empty mask for shifted position
                shifted_mask = np.zeros(img_shape, dtype=np.uint8)
                
                # Calculate current centroid
                current_mask = binary_mask > 0
                y_indices, x_indices = np.where(current_mask)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    current_centroid = (np.mean(x_indices), np.mean(y_indices))
                    
                    # Calculate shift required
                    shift_x = int(interpolated_position[0] - current_centroid[0])
                    shift_y = int(interpolated_position[1] - current_centroid[1])
                    
                    # Apply shift using translation matrix
                    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                    binary_mask = cv2.warpAffine(binary_mask, M, (img_shape[1], img_shape[0]))
            
            # Iterative adjustment loop
            while iteration < max_iterations:
                # Calculate mask properties
                props = self._calculate_mask_properties(target_image, binary_mask)
                
                # Check if LAB values are close enough
                lab_mean_diff = abs(props['lab_mean'] - target_lab_mean)
                lab_std_diff = abs(props['lab_std'] - target_lab_std)
                
                if lab_mean_diff < 10 and lab_std_diff < 10:
                    # Values are close enough, we're done
                    break
                
                if iteration < 5 and use_position_path:
                    # Try adjusting position first
                    # Get current mask center
                    current_mask = binary_mask > 0
                    y_indices, x_indices = np.where(current_mask)
                    
                    if len(y_indices) > 0 and len(x_indices) > 0:
                        current_centroid = (np.mean(x_indices), np.mean(y_indices))
                        
                        # Try small shifts in each direction
                        shifts = [(0, 10), (10, 0), (0, -10), (-10, 0)]
                        best_diff = lab_mean_diff + lab_std_diff
                        best_mask = binary_mask.copy()
                        
                        for dx, dy in shifts:
                            M = np.float32([[1, 0, dx], [0, 1, dy]])
                            shifted_mask = cv2.warpAffine(binary_mask, M, (img_shape[1], img_shape[0]))
                            
                            test_props = self._calculate_mask_properties(target_image, shifted_mask)
                            test_mean_diff = abs(test_props['lab_mean'] - target_lab_mean)
                            test_std_diff = abs(test_props['lab_std'] - target_lab_std)
                            test_total_diff = test_mean_diff + test_std_diff
                            
                            if test_total_diff < best_diff:
                                best_diff = test_total_diff
                                best_mask = shifted_mask.copy()
                        
                        binary_mask = best_mask
                else:
                    # After position adjustments, try reducing mask size if needed
                    area_ratio = props['area_ratio']
                    if area_ratio > 0.2:  # Don't make mask too small
                        # Erode the mask slightly
                        erosion_kernel = np.ones((3, 3), np.uint8)
                        binary_mask = cv2.erode(binary_mask, erosion_kernel, iterations=1)
                
                iteration += 1
            

        return binary_mask
    
    def _visualize_adjustment(self, image, original_mask, adjusted_mask, img_path):
        """
        Create a visualization comparing original and adjusted masks
        
        Args:
            image: Original image (can be 2D or 3D)
            original_mask: Original binary mask
            adjusted_mask: Adjusted binary mask
            img_path: Path of the image
        """
        img_name = Path(img_path).stem
        
        # Check if image is 2D or 3D
        is_2d = len(image.shape) == 2
        
        # For 2D grayscale images, convert to 3-channel grayscale first
        if is_2d:
            # Convert to 3-channel grayscale image
            display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            display_image = image.copy()
        
        # Create masks for visualization
        h, w = original_mask.shape[:2]
        original_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        original_rgb[original_mask > 0] = [0, 0, 255]  # Red for original mask
        
        adjusted_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        adjusted_rgb[adjusted_mask > 0] = [0, 255, 0]  # Green for adjusted mask
        
        overlap_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        overlap_rgb[(original_mask > 0) & (adjusted_mask > 0)] = [255, 255, 0]  # Yellow for overlap
        
        # Create combined visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Display original image with original mask overlay
        axes[0].imshow(original_mask, cmap='Reds', alpha=0.5)
        axes[0].imshow(image, cmap='gray' if is_2d else None, alpha=0.7)
        axes[0].set_title('Original Mask')
        axes[0].axis('off')
        
        # Display original image with adjusted mask overlay
        axes[1].imshow(adjusted_mask, cmap='Greens', alpha=0.5)
        axes[1].imshow(image, cmap='gray' if is_2d else None, alpha=0.7)
        axes[1].set_title('Adjusted Mask')
        axes[1].axis('off')
        
        # Display overlap
        axes[2].imshow(original_mask, cmap='Reds', alpha=0.3)
        axes[2].imshow(adjusted_mask, cmap='Greens', alpha=0.3)
        axes[2].imshow(image, cmap='gray' if is_2d else None, alpha=0.6)
        axes[2].set_title('Overlap')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        #plt.savefig(os.path.join(self.output_folder, f"{img_name}_mask_adjustment.png"))
        plt.close()

    def visualize_all_properties(self):
        """
        Visualize all mask properties including area, LAB mean, and solidity
        """
        areas = []
        lab_means = []
        lab_stds = []
        solidities = []
        
        for img_path, props in self.mask_properties.items():
            areas.append(props['area_ratio'])
            lab_means.append(props['lab_mean'])
            lab_stds.append(props['lab_std'])
            solidities.append(props['solidity'])
        
        # Create subplots
        fig, axs = plt.subplots(4, 1, figsize=(10, 20))
        
        axs[0].plot(areas, label='Area Ratio')
        axs[0].set_title('Mask Area Ratio')
        axs[0].set_xlabel('Image Index')
        axs[0].set_ylabel('Area Ratio')
        
        axs[1].plot(lab_means, label='LAB Mean', color='orange')
        axs[1].set_title('LAB Mean of Masks')
        axs[1].set_xlabel('Image Index')
        axs[1].set_ylabel('LAB Mean')
        
        axs[2].plot(lab_stds, label='LAB Std', color='green')
        axs[2].set_title('LAB Std of Masks')
        axs[2].set_xlabel('Image Index')
        axs[2].set_ylabel('LAB Std')
        
        axs[3].plot(solidities, label='Solidity', color='red')
        axs[3].set_title('Solidity of Masks')
        axs[3].set_xlabel('Image Index')
        axs[3].set_ylabel('Solidity')

        plt.tight_layout()
        plt.show()

    def _save_all_masks(self):
        """
        Save all masks to the output folder
       
        """
        for mask_path, mask in self.mask_data.items():
            mask_filename = Path(mask_path).name
            save_path = os.path.join(self.output_folder, mask_filename)
            cv2.imwrite(save_path, mask)
            print(f"Saved mask to {save_path}")

if __name__ == "__main__":
    # Example usage
    image_folder = r"c:\Users\Reich\LocalCoding\Wundbilder\Wundbilder_XotoCAM\Ferse\Sharp_images"
    mask_folder = r"C:\Users\Reich\LocalCoding\Wundbilder\Wundbilder_XotoCAM\Ferse\Maske_CombinedLAB"
    output_folder = r"C:\Users\Reich\LocalCoding\Wundbilder\Wundbilder_XotoCAM\Ferse\CombinedLAB_Eval"

    # Create MaskAdjustment instance
    mask_adjuster = MaskAdjustment(image_folder, mask_folder, output_folder)

    # Preprocessing to improve mask quality by handling low-solidity masks
    print("Preprocessing masks to improve quality...")
    
    mask_adjuster.perform_preprocessing(min_solidity=0.7)

    # Visualize mask areas after preprocessing, highlighting invalid ones
    plt.figure(figsize=(12, 6))
    
    # Get mask data in sequence order
    areas = []
    statuses = []
    
    for img_path in mask_adjuster.sorted_images:
        if img_path in mask_adjuster.mask_properties:
            props = mask_adjuster.mask_properties[img_path]
            areas.append(props['area_ratio'])
            statuses.append(props['is_valid'])
    
    # Plot areas with different colors for valid and invalid
    valid_indices = [i for i, valid in enumerate(statuses) if valid]
    invalid_indices = [i for i, valid in enumerate(statuses) if not valid]
    
    plt.scatter(valid_indices, [areas[i] for i in valid_indices], c='g', label='Valid Masks')
    plt.scatter(invalid_indices, [areas[i] for i in invalid_indices], c='r', label='Invalid Masks')
    
    # Add horizontal lines for thresholds
    plt.axhline(y=mask_adjuster.min_mask_ratio, color='r', linestyle='--', 
                label=f'Min Threshold ({mask_adjuster.min_mask_ratio})')
    plt.axhline(y=mask_adjuster.max_mask_ratio, color='r', linestyle='--', 
                label=f'Max Threshold ({mask_adjuster.max_mask_ratio})')
    
    plt.title('Mask Area Ratios After Preprocessing')
    plt.xlabel('Image Index')
    plt.ylabel('Area Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Adjust small masks
    print("Adjusting small masks by making them at least big enough to pass the 10% threshold...")
    mask_adjuster.simple_small_mask_adjustment()

# Visualize mask areas after preprocessing, highlighting invalid ones
    plt.figure(figsize=(12, 6))
    
    # Get mask data in sequence order
    areas = []
    statuses = []
    
    for img_path in mask_adjuster.sorted_images:
        if img_path in mask_adjuster.mask_properties:
            props = mask_adjuster.mask_properties[img_path]
            areas.append(props['area_ratio'])
            statuses.append(props['is_valid'])
    
    # Plot areas with different colors for valid and invalid
    valid_indices = [i for i, valid in enumerate(statuses) if valid]
    invalid_indices = [i for i, valid in enumerate(statuses) if not valid]
    
    plt.scatter(valid_indices, [areas[i] for i in valid_indices], c='g', label='Valid Masks')
    plt.scatter(invalid_indices, [areas[i] for i in invalid_indices], c='r', label='Invalid Masks')
    
    # Add horizontal lines for thresholds
    plt.axhline(y=mask_adjuster.min_mask_ratio, color='r', linestyle='--', 
                label=f'Min Threshold ({mask_adjuster.min_mask_ratio})')
    plt.axhline(y=mask_adjuster.max_mask_ratio, color='r', linestyle='--', 
                label=f'Max Threshold ({mask_adjuster.max_mask_ratio})')
    
    plt.title('Mask Area Ratios After Small Mask Adjustment')
    plt.xlabel('Image Index')
    plt.ylabel('Area Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Evaluate masks
    valid_masks, invalid_masks = mask_adjuster.evaluate_masks()
    print(f"Found {len(valid_masks)} valid masks and {len(invalid_masks)} invalid masks")
    
    # Adjust invalid masks
    adjusted_masks = mask_adjuster.adjust_masks(save_adjusted=False)
    print(f"Adjusted {len(adjusted_masks)} masks")

    # Save all Masks
    #mask_adjuster._save_all_masks()

    # Visualize mask areas after preprocessing, highlighting invalid ones
    plt.figure(figsize=(12, 6))
    
    # Get mask data in sequence order
    areas = []
    statuses = []
    
    for img_path in mask_adjuster.sorted_images:
        if img_path in mask_adjuster.mask_properties:
            props = mask_adjuster.mask_properties[img_path]
            areas.append(props['area_ratio'])
            statuses.append(props['is_valid'])
    
    # Plot areas with different colors for valid and invalid
    valid_indices = [i for i, valid in enumerate(statuses) if valid]
    invalid_indices = [i for i, valid in enumerate(statuses) if not valid]
    
    plt.scatter(valid_indices, [areas[i] for i in valid_indices], c='g', label='Valid Masks')
    plt.scatter(invalid_indices, [areas[i] for i in invalid_indices], c='r', label='Invalid Masks')
    
    # Add horizontal lines for thresholds
    plt.axhline(y=mask_adjuster.min_mask_ratio, color='r', linestyle='--', 
                label=f'Min Threshold ({mask_adjuster.min_mask_ratio})')
    plt.axhline(y=mask_adjuster.max_mask_ratio, color='r', linestyle='--', 
                label=f'Max Threshold ({mask_adjuster.max_mask_ratio})')
    
    plt.title('Mask Area Ratios After mask interpolation')
    plt.xlabel('Image Index')
    plt.ylabel('Area Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Optional: Clear image data from memory if no longer needed
    print("Freeing memory...")
    mask_adjuster.image_data.clear()
    
    # Optional: Only keep adjusted masks in memory
    unchanged_masks = set(mask_adjuster.mask_data.keys()) - set(adjusted_masks.keys())
    for mask_path in unchanged_masks:
        if mask_path in mask_adjuster.mask_data:
            del mask_adjuster.mask_data[mask_path]
    
    print("Processing completed!")
    
  