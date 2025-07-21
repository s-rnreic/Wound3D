import cv2
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'external', 'sam2'))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
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

def process_image(image_path, output_folder):
    mask = generate_mask(image_path)
    if mask is not None:
        output_path = os.path.join(output_folder, os.path.basename(image_path) +'.png')
        cv2.imwrite(output_path, mask)

def process_folder(input_folder, output_folder, num_workers=4):
    os.makedirs(output_folder, exist_ok=True)
    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_image, image_paths, [output_folder] * len(image_paths))

def process_image_with_sam2(image_path, checkpoint, model_cfg,  multimask_output=True):
     # Assume model_cfg has the same base name as checkpoint, but with .yaml extension
    img= cv2.imread(image_path)
    model_cfg = model_cfg
    if not os.path.exists(model_cfg):
        print(f"Config file not found for checkpoint {checkpoint}: {model_cfg}")
    print(f"Using checkpoint: {checkpoint}")
    print(f"Using config: {model_cfg}")
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

def process_video_with_sam2(image_folder_path, checkpoint=None, model_cfg=None, input_points=None, interactive=False):
    from sam2.build_sam import build_sam2_video_predictor
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
        img_path = os.path.join(image_folder_path, frame_names[0])
        img = cv2.imread(img_path)
        points = [[img.shape[1] // 2, img.shape[0] // 2]]
        labels = np.array([1], np.int32)
        
        
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
            
if __name__ == '__main__':
    # Your multiprocessing code here
    # Example usage
    
    checkpoint_dir = r"C:\Users\Reich\LocalCoding\SfM_Pipeline_alternative\wundpro_vid2cloud\external\sam2\checkpoints\sam2.1_hiera_base_plus.pt"
    config_dir = r"C:\Users\Reich\LocalCoding\SfM_Pipeline_alternative\wundpro_vid2cloud\external\sam2\sam2\configs\sam2.1\sam2.1_hiera_b+.yaml"
    img_folder_path = r"G:\Reco_Motorad\sharp_images"
    output_path = r"G:\Reco_Motorad\masks"


    
    result = process_video_with_sam2(img_folder_path, checkpoint_dir, config_dir, interactive=True)
    save_masks(result, output_path)
    # for frame_idx, obj_masks in result.items():
    #     img_path_jpg = os.path.join(img_folder_path, f"{frame_idx:05d}.jpg")
    #     img_path_png = os.path.join(img_folder_path, f"{frame_idx:05d}.png")
    #     print(img_path_jpg)
    #     if os.path.exists(img_path_jpg):
    #         img_path = img_path_jpg
    #     elif os.path.exists(img_path_png):
    #         img_path = img_path_png
    #     else:
    #         print(f"Warning: No image found for frame {frame_idx} (tried .jpg and .png)")
    #         continue
    #     img = cv2.imread(img_path)
    #     if img is None:
    #         print(f"Warning: Failed to load image at {img_path}")
    #         continue
    #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     for obj_id, mask in obj_masks.items():
    #         mask_bool = mask.squeeze().astype(bool)
    #         masked_img = np.zeros_like(img_rgb)
    #         masked_img[mask_bool] = img_rgb[mask_bool]
    #         plt.imshow(masked_img)
    #         plt.title(f"Frame {frame_idx}, Object {obj_id}")
    #         plt.axis("off")
    #         plt.show()
    # Save all masks in result to output_path
