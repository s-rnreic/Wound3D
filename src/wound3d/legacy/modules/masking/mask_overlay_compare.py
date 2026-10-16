"""
Compare two mask folders by overlaying each mask on the corresponding image.

Usage:
  python mask_overlay_compare.py \
    --images_dir C:\path\to\images \
    --mask_dir1 C:\path\to\masks1 \
    --mask_dir2 C:\path\to\masks2

Masks are matched by exact filename.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_images(images_dir: Path) -> Iterable[Path]:
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def load_mask(mask_path: Path) -> np.ndarray | None:
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return mask


def resize_mask_to_image(mask: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    if mask.shape[0] == h and mask.shape[1] == w:
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    if mask is None:
        return image_rgb.copy()
    overlay = image_rgb.copy()
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return overlay
    color_arr = np.array(color, dtype=np.uint8)
    overlay[mask_bool] = (1.0 - alpha) * overlay[mask_bool] + alpha * color_arr
    return overlay


def compare_overlays(images_dir: Path, mask_dir1: Path, mask_dir2: Path,
                     alpha: float, max_images: int | None) -> None:
    images = list(iter_images(images_dir))
    if max_images is not None:
        images = images[:max_images]

    if not images:
        print(f"No images found in: {images_dir}")
        return

    for img_path in images:
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        mask1_path = mask_dir1 / img_path.name
        mask2_path = mask_dir2 / img_path.name

        mask1 = load_mask(mask1_path)
        mask2 = load_mask(mask2_path)

        if mask1 is None or mask2 is None:
            print(f"Skipping {img_path.name}: missing mask in one folder")
            continue

        mask1 = resize_mask_to_image(mask1, (h, w))
        mask2 = resize_mask_to_image(mask2, (h, w))

        overlay1 = overlay_mask(image_rgb, mask1, color=(255, 0, 0), alpha=alpha)
        overlay2 = overlay_mask(image_rgb, mask2, color=(0, 255, 0), alpha=alpha)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(overlay1)
        axes[0].set_title(f"Mask 1: {img_path.name}")
        axes[0].axis("off")

        axes[1].imshow(overlay2)
        axes[1].set_title(f"Mask 2: {img_path.name}")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()


def main() -> None:
    images_dir = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\test\sharp_images")
    mask_dir1 = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\test\mask_wound_only_undistorted\_colmap_mask_input")
    mask_dir2 = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\test\mask_wound_only_undistorted\images")
    alpha = 0.5
    max_images = None
    compare_overlays(
        images_dir=images_dir,
        mask_dir1=mask_dir1,
        mask_dir2=mask_dir2,
        alpha=alpha,
        max_images=max_images,
    )


if __name__ == "__main__":
    main()
