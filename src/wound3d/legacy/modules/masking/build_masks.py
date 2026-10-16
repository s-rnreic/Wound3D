from pathlib import Path

import cv2
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import torch
import importlib.util

# SAM2 must be installed from the submodule into the active environment, e.g.:
# python -m pip install -e external/sam2
# Do not mutate sys.path because that makes the selected SAM2 version dependent
# on the current working directory and hides installation problems.
SAM2_AVAILABLE = False
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    print("SAM2 is not installed. Install the initialized submodule with: python -m pip install -e external/sam2")

import matplotlib.pyplot as plt


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

    if PLOT:
        plt.figure(figsize=(6, 6))
        plt.title('Center Circle Area for Mean/Deviation')
        im = plt.imshow(ab_image, cmap='gray')
        circle_overlay = np.zeros_like(ab_image, dtype=np.uint8)
        circle_overlay[mask_circle] = 255
        plt.contour(circle_overlay, levels=[127], colors='r', linewidths=2)
        plt.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        # Add mean and std text
        plt.text(
            0.05, 0.95,
            f"Mean: {threshold:.2f}\nStd: {np.std(ab_image[mask_circle]):.2f}",
            color='white', fontsize=12, ha='left', va='top',
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3')
        )
        plt.show()

    # Compute importance weighting (radial gradient)
    y, x = np.indices((h, w))
    dist_map = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    dist_map = dist_map ** exp_factor  # Make the gradient steeper
    dist_map = 1 - (dist_map / np.max(dist_map))  # Normalize to [0,1]

    if PLOT:
        plt.figure(figsize=(6, 6))
        plt.title('Radial Distance Map (dist_map)')
        im = plt.imshow(dist_map, cmap='viridis')
        plt.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()

    # Apply radial gradient to enhance central importance
    weighted_ab = ab_image * dist_map

    if PLOT:
        plt.figure(figsize=(6, 6))
        plt.title('Weighted ab_image (weighted_ab)')
        im = plt.imshow(weighted_ab, cmap='gray')
        plt.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()

    # Mask is everything above 
    mask = (weighted_ab + deviation > threshold).astype(np.uint8) * 255

    if PLOT:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Final Mask')
        im1 = plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.colorbar(im1, fraction=0.046, pad=0.04)

        # Show the original image where mask is true, black elsewhere
        # Reload original image in RGB for display
        orig_img = cv2.imread(image_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        masked_img = np.zeros_like(orig_img)
        mask_bool = mask.astype(bool)
        for c in range(3):
            masked_img[..., c][mask_bool] = orig_img[..., c][mask_bool]

        plt.subplot(1, 2, 2)
        plt.title('Original Image (Masked)')
        im2 = plt.imshow(masked_img)
        plt.axis('off')
        plt.show()
    # # Apply morphological closing to fill in holes
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel
    #                         , iterations=2)
    
    return mask 

def moving_average_mask(image_folder_path, blur_ksize=33, exp_factor=2, window_size=50, PLOT=False, output_folder=None):
    """
    Generate masks for all images in a folder using a moving average threshold for the center circle mean.
    Additionally, calculate the mean and standard deviation of the ab-channel image for the whole dataset,
    and compute moving averages for both means and deviations.
    """
    import collections
    import sys
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
    image_paths = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if f.endswith(('.jpg', '.png'))]
    image_paths.sort()  # Ensure consistent order
    masks = []
    all_means = []
    all_devs = []
    ab_images = []
    # First pass: calculate all means and deviations for ab-channel images
    
    #first image 
    first_image_path = image_paths[0]
    image_first = cv2.imread(first_image_path)
    h, w = image_first.shape[0:2]
    center_x, center_y = w // 2, h // 2
    radius = int(0.1 * min(h, w) / 2)
    y, x = np.ogrid[:h, :w]
    mask_circle = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            all_means.append(np.nan)
            all_devs.append(np.nan)
            ab_images.append(None)
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        a_channel, b_channel = image[:, :, 1], image[:, :, 2]
        a_blur = cv2.GaussianBlur(a_channel, (blur_ksize, blur_ksize), 0)
        b_blur = cv2.GaussianBlur(b_channel, (blur_ksize, blur_ksize), 0)
        ab_image = np.maximum(a_blur, b_blur)
        ab_images.append(ab_image)
        center_mean = np.mean(ab_image[mask_circle])
        center_dev = np.std(ab_image[mask_circle])
        all_means.append(center_mean)
        all_devs.append(center_dev)
    all_means = np.array(all_means)
    all_devs = np.array(all_devs)
    # Compute moving averages
    mov_avg_means = np.convolve(all_means, np.ones(window_size)/window_size, mode='valid')
    mov_avg_devs = np.convolve(all_devs, np.ones(window_size)/window_size, mode='valid')
    # Second pass: generate masks using moving averages
    
    if PLOT:
        import matplotlib.pyplot as plt

        # Plot the means and deviations of the whole dataset
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.title("Mean of ab_image (whole dataset)")
        plt.plot([np.mean(ab[mask_circle]) if ab is not None else np.nan for ab in ab_images], label="Mean (whole image)")
        plt.xlabel("Image Index")
        plt.ylabel("Mean Value")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.title("StdDev of ab_image (whole dataset)")
        plt.plot([np.std(ab[mask_circle]) if ab is not None else np.nan for ab in ab_images], label="StdDev (whole image)")
        plt.xlabel("Image Index")
        plt.ylabel("StdDev Value")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Plot means
        axs[0].plot(all_means, label='Center Means (per image)')
        axs[0].plot(np.arange(window_size//2, window_size//2+len(mov_avg_means)), mov_avg_means, label=f'Moving Avg Means (window={window_size})')
        axs[0].set_xlabel('Image Index')
        axs[0].set_ylabel('Mean Value')
        axs[0].set_title('Center Means and Moving Average')
        axs[0].legend()
        axs[0].grid(True)

        # Plot deviations
        axs[1].plot(all_devs, label='Center StdDev (per image)')
        axs[1].plot(np.arange(window_size//2, window_size//2+len(mov_avg_devs)), mov_avg_devs, label=f'Moving Avg StdDev (window={window_size})')
        axs[1].set_xlabel('Image Index')
        axs[1].set_ylabel('StdDev Value')
        axs[1].set_title('Center StdDev and Moving Average')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()
        #sys.exit(0)
    

    for idx, ab_image in enumerate(ab_images):
        if ab_image is None:
            masks.append(None)
            continue
        h, w = ab_image.shape
        center_x, center_y = w // 2, h // 2
        radius = int(0.1 * min(h, w) / 2)
        y, x = np.ogrid[:h, :w]
        mask_circle = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        # Use moving average for threshold and deviation (align index for 'valid' mode)
        mov_idx = min(max(idx - window_size//2, 0), len(mov_avg_means)-1)
        threshold = mov_avg_means[mov_idx]
        deviation = 2 * mov_avg_devs[mov_idx]
        y_idx, x_idx = np.indices((h, w))
        dist_map = np.sqrt((x_idx - center_x) ** 2 + (y_idx - center_y) ** 2)
        dist_map = dist_map ** exp_factor
        dist_map = 1 - (dist_map / np.max(dist_map))
        weighted_ab = ab_image * dist_map
        mask = (weighted_ab + deviation > threshold).astype(np.uint8) * 255
        masks.append(mask)
        if output_folder is not None:
            out_path = os.path.join(output_folder, os.path.basename(image_paths[idx]) + '.png')
            cv2.imwrite(out_path, mask)
    if PLOT:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(all_means, label='Center Means (per image)')
        plt.plot(np.arange(window_size//2, window_size//2+len(mov_avg_means)), mov_avg_means, label=f'Moving Avg Means (window={window_size})')
        plt.plot(all_devs, label='Center StdDev (per image)')
        plt.plot(np.arange(window_size//2, window_size//2+len(mov_avg_devs)), mov_avg_devs, label=f'Moving Avg StdDev (window={window_size})')
        plt.xlabel('Image Index')
        plt.ylabel('Value')
        plt.title('Center Means/StdDev and Moving Averages')
        plt.legend()
        plt.tight_layout()
        plt.show()
    return masks

def process_image(image_path, output_folder):
    mask = generate_mask(image_path)
    if mask is not None:
        output_path = os.path.join(output_folder, os.path.basename(image_path) +'.png')
        cv2.imwrite(output_path, mask)


def _load_deepskin_segmentation():
    """Load the cached DeepSkin adapter without modifying the active environment."""
    try:
        import deepskin
        from wound3d.legacy.modules.masking.deepskin_wrapper import wound_segmentation
    except ImportError as error:
        raise RuntimeError(
            "DeepSkin is required for this masking workflow but is not installed in the active environment. "
            "Install the checked-out dependency before running: "
            "python -m pip install -e external/Deepskin"
        ) from error
    del deepskin
    return wound_segmentation


def _create_deepskin_masks(input_folder: str, wound_only_folder: str, wound_tissue_folder: str) -> None:
    """Create DeepSkin wound-only and wound+tissue masks for an image folder.

    DeepSkin semantic output is ordered as background, body/tissue, wound. The
    combined mask deliberately excludes the background channel and keeps tissue
    plus wound; this is required when WoundAmbit is used for wound-only masks.
    """
    wound_segmentation = _load_deepskin_segmentation()
    os.makedirs(wound_only_folder, exist_ok=True)
    os.makedirs(wound_tissue_folder, exist_ok=True)

    image_names = sorted(
        name for name in os.listdir(input_folder) if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    for image_name in image_names:
        image_path = os.path.join(input_folder, image_name)
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Failed to read image: {image_name}")
            continue

        semantic_mask = wound_segmentation(img=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if semantic_mask.ndim != 3 or semantic_mask.shape[2] < 3:
            raise ValueError(f"DeepSkin returned an invalid semantic mask for: {image_name}")

        tissue_channel = semantic_mask[:, :, 1]
        wound_channel = semantic_mask[:, :, 0]
        wound_tissue_mask = np.bitwise_or(wound_channel, tissue_channel)

        # Keep the historical '<image filename>.png' naming convention, e.g. 00001.jpg.png.
        cv2.imwrite(os.path.join(wound_tissue_folder, image_name + ".png"), wound_tissue_mask)

def process_folder(input_folder, output_folder, num_workers=4, Use_SAM2=False, Use_DeepSkin=False, Use_WoundAmbit=False,
                   use_mask_adjustment=False, extrude_masks=False, min_mask_ratio=0.1, max_mask_ratio=0.8,
                   neighbors_count=3, sam2_checkpoint=None, sam2_model_config=None,
                   woundambit_executable=None, woundambit_working_directory=None):
    """
    Process a folder of images to generate segmentation masks.
    
    Args:
        input_folder (str): Path to folder containing input images
        output_folder (str): Path to folder for saving masks (wound+tissue combined)
        num_workers (int): Number of parallel workers (for non-DeepSkin methods)
        Use_SAM2 (bool): Use SAM2 for segmentation
        Use_DeepSkin (bool): Use DeepSkin for wound segmentation (default: True)
        use_mask_adjustment (bool): Apply mask adjustment workflow after segmentation
        min_mask_ratio (float): Minimum acceptable mask area ratio (default: 0.1)
        max_mask_ratio (float): Maximum acceptable mask area ratio (default: 0.8)
        neighbors_count (int): Number of neighbors for mask adjustment (default: 3)
    
    When Use_DeepSkin=True, creates:
        - {output_folder}: wound+tissue combined masks
        - {output_folder}/../masks_wound_only: wound-only masks
    
    Example:
        process_folder('images/', 'masks/', Use_DeepSkin=True, use_mask_adjustment=True)
    """

    mask_outputs_ready = False

    if Use_SAM2:
        if sam2_checkpoint is None or sam2_model_config is None:
            raise ValueError(
                "SAM2 masking requires 'sam2_checkpoint' and 'sam2_model_config' in the pipeline configuration."
            )
        checkpoint = os.fspath(sam2_checkpoint)
        model_cfg = os.fspath(sam2_model_config)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")
        if not os.path.isfile(model_cfg):
            raise FileNotFoundError(f"SAM2 model configuration not found: {model_cfg}")
        result = process_video_with_sam2(input_folder, checkpoint, model_cfg)
        save_masks(result, output_folder)
        return
    
    elif Use_DeepSkin:
        print("DeepSkin module found.")
        wound_only_folder = os.path.join(os.path.dirname(output_folder), 'masks_wound_only')
        _create_deepskin_masks(input_folder, wound_only_folder, output_folder)
        mask_outputs_ready = True
    
    elif Use_WoundAmbit:
        import subprocess
        import shutil
        import stat
        import tempfile
        import uuid

        if woundambit_executable not in (None, "wsl"):
            raise ValueError(
                "The legacy WoundAmbit adapter currently supports only 'wsl' as woundambit_executable. "
                "Configure masking.woundambit_executable: wsl."
            )
        default_woundambit_wrapper_directory = Path(__file__).resolve().parent / "utils"
        woundambit_working_directory = (
            Path(woundambit_working_directory)
            if woundambit_working_directory is not None
            else default_woundambit_wrapper_directory
        )
        woundambit_working_directory = os.fspath(woundambit_working_directory.resolve())
        if not os.path.isfile(os.path.join(woundambit_working_directory, "WoundAmbitWrapper.py")):
            raise FileNotFoundError(
                "WoundAmbitWrapper.py was not found in the configured/default WoundAmbit wrapper directory: "
                f"{woundambit_working_directory}"
            )
        
        wound_only_folder = os.path.join(os.path.dirname(output_folder), 'masks_wound_only')
        wound_tissue_folder = output_folder  # wound + tissue combined
        os.makedirs(wound_only_folder, exist_ok=True)
        os.makedirs(wound_tissue_folder, exist_ok=True)

        print("Generating DeepSkin wound+tissue masks for WoundAmbit background removal...")
        _create_deepskin_masks(input_folder, wound_only_folder, wound_tissue_folder)
        mask_outputs_ready = True
        
        # Convert Windows paths to WSL paths for the command.
        # Use wslpath first because mapped/network drives are not always represented by /mnt/<drive>/...
        def win_to_wsl_path(win_path):
            normalized = os.path.normpath(win_path)
            result = subprocess.run(
                ["wsl", "wslpath", "-a", normalized],
                capture_output=True,
                text=True,
                shell=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()

            drive_letter = normalized[0].lower()
            rest = normalized[3:].replace('\\', '/')
            return f"/mnt/{drive_letter}/{rest}"

        def wsl_dir_exists(wsl_path):
            escaped = wsl_path.replace("'", "'\\''")
            check = subprocess.run(
                ["wsl", "bash", "-lc", f"test -d '{escaped}'"],
                capture_output=True,
                text=True,
                shell=False,
            )
            return check.returncode == 0

        def write_woundambit_masks(source_folder, target_folder):
            for old_name in os.listdir(source_folder):
                old_path = os.path.join(source_folder, old_name)
                if not os.path.isfile(old_path):
                    continue
                print("renaming masks...")
                base_name = os.path.splitext(old_name)[0]
                new_img_name = base_name + ".jpg.png"
                new_path = os.path.join(target_folder, new_img_name)
                image = cv2.imread(old_path)
                if image is None:
                    continue
                cv2.imwrite(new_path, image)
                if os.path.abspath(source_folder) == os.path.abspath(target_folder):
                    os.remove(old_path)

        output_folder_full = wound_only_folder
        input_folder_wsl = win_to_wsl_path(input_folder)
        output_folder_wsl = win_to_wsl_path(output_folder_full)

        stage_dir = None
        stage_input_win = None
        stage_output_win = None
        run_input_wsl = input_folder_wsl
        run_output_wsl = output_folder_wsl

        if not wsl_dir_exists(input_folder_wsl):
            print(f"WSL cannot access input path directly: {input_folder_wsl}")
            print("Staging images to a C-drive temp folder for WoundAmbit.")
            stage_dir = os.path.join(tempfile.gettempdir(), f"woundambit_staging_{uuid.uuid4().hex}")
            stage_input_win = os.path.join(stage_dir, "input")
            stage_output_win = os.path.join(stage_dir, "output")
            os.makedirs(stage_input_win, exist_ok=True)
            os.makedirs(stage_output_win, exist_ok=True)

            for entry in os.listdir(input_folder):
                source = os.path.join(input_folder, entry)
                if os.path.isfile(source):
                    shutil.copy2(source, os.path.join(stage_input_win, entry))

            run_input_wsl = win_to_wsl_path(stage_input_win)
            run_output_wsl = win_to_wsl_path(stage_output_win)
        
        woundambit_available = False
        
        # Check if WoundAmbit conda environment exists
        env_check_cmd = "source ~/anaconda3/etc/profile.d/conda.sh && conda env list | grep woundambit"
        env_check_result = subprocess.run(
            ["wsl", "bash", "-lc", env_check_cmd],
            capture_output=True,
            text=True,
            shell=False
        )
        
        woundambit_available = ("woundambit" in env_check_result.stdout or "woundambit" in env_check_result.stderr)
        
        if not woundambit_available:
            print("WARNING: WoundAmbit conda environment 'woundambit' not found.")
        else:
            print(f"WoundAmbit environment found. Running WoundAmbit inference...")
            print(f"  Input folder (WSL):  {run_input_wsl}")
            print(f"  Output folder (WSL): {run_output_wsl}")
            
            # Escape single quotes in paths for the shell command
            input_escaped = run_input_wsl.replace("'", "'\\''")
            output_escaped = run_output_wsl.replace("'", "'\\''")
            
            cmd = f"""
            set -e
            cd '{win_to_wsl_path(woundambit_working_directory)}'
            source ~/anaconda3/etc/profile.d/conda.sh
            conda activate woundambit
            python WoundAmbitWrapper.py --input_folder '{input_escaped}' --output_folder '{output_escaped}'
            """

            result = subprocess.run(
                ["wsl", "bash", "-lc", cmd],
                capture_output=True,
                text=True,
            )

            print("Return code:", result.returncode)
            print("--- STDOUT ---")
            print(result.stdout)
            print("--- STDERR ---")
            print(result.stderr)

            if result.returncode == 0:
                if stage_output_win is not None:
                    write_woundambit_masks(stage_output_win, wound_only_folder)
                    shutil.rmtree(stage_dir, ignore_errors=True)
                else:
                    write_woundambit_masks(output_folder_full, wound_only_folder)
            else:
                print(f"WARNING: WoundAmbit inference failed with return code {result.returncode}.")
                woundambit_available = False
                if stage_dir is not None:
                    shutil.rmtree(stage_dir, ignore_errors=True)

        if not woundambit_available:
            print("WoundAmbit was unavailable; retaining DeepSkin wound-only masks as the fallback.")
            Use_WoundAmbit = False
    # Apply mask adjustment if requested
    
    # If WoundAmbit failed or was unavailable, use the conservative (LAB-based) fallback
    if not mask_outputs_ready and not Use_WoundAmbit and not Use_SAM2 and not Use_DeepSkin:
        print("\n" + "="*60)
        print("Running Conservative (LAB-based) Segmentation")
        print("="*60)
        os.makedirs(output_folder, exist_ok=True)
        image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            executor.map(process_image, image_paths, [output_folder] * len(image_paths))
        print(f"Conservative segmentation complete. Masks saved to: {output_folder}")
    
    if use_mask_adjustment:
        print("\n" + "="*60)
        print("Applying Mask Adjustment Workflow")
        print("="*60)
        
        from wound3d.legacy.modules.masking.MaskAdjustment import MaskAdjustment
        
        # Apply to wound+tissue masks
        print("\nAdjusting wound+tissue masks...")
        mask_adjuster = MaskAdjustment(
            input_folder,
            output_folder,
            output_folder,  # Output to the configured wound+tissue mask folder
            min_mask_ratio=min_mask_ratio,
            max_mask_ratio=max_mask_ratio,
            neighbors_count=neighbors_count
        )
        
        # Preprocess masks
        mask_adjuster.perform_preprocessing(min_solidity=0.7)
        
        # Adjust small masks
        mask_adjuster.simple_small_mask_adjustment()
        
        # Evaluate and adjust invalid masks
        mask_adjuster.evaluate_masks(visualize=False)
        mask_adjuster.adjust_masks(save_adjusted=False)
        
        # Save adjusted masks
        mask_adjuster._save_all_masks()
           
    if extrude_masks:
        from wound3d.legacy.modules.masking.ExtendWoundMask import create_extended_wound_masks
        print("Extruding masks...")
        extruded_wound_only_folder = os.path.join(os.path.dirname(output_folder), 'masks_extruded_wound_only')
        os.makedirs(extruded_wound_only_folder, exist_ok=True)
        stats = create_extended_wound_masks(
            os.path.join(os.path.dirname(output_folder), 'masks_wound_only'),
            output_folder,
            extruded_wound_only_folder,
            extension_percentage=1.0,  # 70% extension
            visualize_samples=0  # Visualize first 3 samples
        )

            
        print("Mask adjustment complete!")



def process_image_with_sam2(image_path, checkpoint, model_cfg,  multimask_output=True):
    if not SAM2_AVAILABLE:
        print("Error: SAM2 is not available. Cannot process image with SAM2.")
        return None
        
    # Assume model_cfg has the same base name as checkpoint, but with .yaml extension
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
        
    if not os.path.exists(model_cfg):
        print(f"Config file not found for checkpoint {checkpoint}: {model_cfg}")
        return None
        
    print(f"Using checkpoint: {checkpoint}")
    print(f"Using config: {model_cfg}")
    
    try:
        predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        
        points = [[img.shape[1] // 2, img.shape[0] // 2]]
        input_point = np.array(points, dtype=int)
        print(input_point)
        input_label = np.ones(len(input_point), dtype=int)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(img)
            masks, _, _ = predictor.predict(point_coords=input_point,
                                          point_labels=input_label, multimask_output=multimask_output)
        return masks
    except Exception as e:
        print(f"Error processing image with SAM2: {e}")
        return None

def process_video_with_sam2(image_folder_path, checkpoint=None, model_cfg=None, input_points=None, interactive=False):
    if not SAM2_AVAILABLE:
        print("Error: SAM2 is not available. Cannot process video with SAM2.")
        return {}
        
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError:
        print("Error: Could not import build_sam2_video_predictor from SAM2.")
        return {}
        
    # Set default paths relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if checkpoint is None:
        checkpoint = os.path.join(script_dir, '..', 'external', 'sam2', 'checkpoints', 'sam2.1_hiera_base_plus.pt')
    if model_cfg is None:
        model_cfg = os.path.join(script_dir, '..', 'external', 'sam2', 'sam2', 'configs', 'sam2.1', 'sam2.1_hiera_b+.yaml')
    checkpoint = os.path.abspath(checkpoint)
    model_cfg = os.path.abspath(model_cfg)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")

    predictor = build_sam2_video_predictor(model_cfg, checkpoint)

    frame_names = [
        p for p in os.listdir(image_folder_path)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = predictor.init_state(video_path=image_folder_path)
    predictor.reset_state(inference_state)
    
    if input_points is not None:
        points = input_points
        labels = np.ones(len(points), dtype=np.int32)
    elif interactive:
        if len(frame_names) > 0:
            frame_idx = min(780, len(frame_names) - 1)  # Use 780 or the last available frame
            plt.imshow(cv2.cvtColor(cv2.imread(os.path.join(image_folder_path, frame_names[frame_idx])), cv2.COLOR_BGR2RGB))
            plt.title("Click to select points")
            plt.axis("off")
            points = plt.ginput(n=0, timeout=0)
            plt.close()
            points = np.array(points, dtype=np.int32)
            labels = np.ones(len(points), dtype=np.int32)
        else:
            print("Warning: No frames available in the folder.")
            return {}
    else:
        img_path = os.path.join(image_folder_path, frame_names[9])
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        radius = int(0.4 * min(h, w) / 2)
        
        # Generate 10 random points within the circle
        points = []
        while len(points) < 5:
            x = np.random.randint(center_x - radius, center_x + radius)
            y = np.random.randint(center_y - radius, center_y + radius)
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                points.append([x, y])
        
        labels = np.ones(len(points), dtype=np.int32)

        
        
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=points,
        labels=labels,
    )

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    return video_segments
 
def save_masks(result, output_path):
    os.makedirs(output_path, exist_ok=True)
    for frame_idx, obj_masks in result.items():
        for obj_id, mask in obj_masks.items():
            mask_img = (mask.squeeze() * 255).astype(np.uint8)
            out_filename = f"{frame_idx + 1:05d}.png"
            out_filepath = os.path.join(output_path, out_filename)
            cv2.imwrite(out_filepath, mask_img)
        
def crop_masks(input_folder):

    for filename in os.listdir(input_folder):
        combined_mask = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)   
        combined_mask_original = combined_mask.copy()     
        # Create a distance map from the center of the image
        w, h = combined_mask.shape
        kernel_size = int(min(w, h) * 0.03)  # 3% of image dimension
        kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)  # Ensure odd number
        filled_combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, 
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ((kernel_size, kernel_size))))
        
        # Apply erosion only to the outer part of the mask
        kernel_size = int(min(w, h) * 0.06)  # 6% of image dimension
        kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)  # Ensure odd number
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        

        # Erode outer region
        eroded_outer = cv2.erode(filled_combined_mask, kernel, iterations=1)
        
       
        
        # Optional: Perform a closing operation to smooth the mask borders
        combined_mask_cropped = cv2.morphologyEx(eroded_outer, cv2.MORPH_CLOSE, 
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        
       
        cv2.imwrite(os.path.join(input_folder, filename), combined_mask_cropped)
