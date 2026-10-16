"""
Undistort mask images using COLMAP sparse model and image_undistorter.

Masks are named like the original images with an extra .png extension
(e.g., image.jpg.png). This script runs COLMAP's image_undistorter
directly on the mask folder.
"""
from __future__ import annotations
import os

import argparse
import shutil
import subprocess
import wound3d.legacy.run_command
from pathlib import Path


def undistort_masks_with_colmap(
	mask_dir: Path,
	sparse_model_dir: Path,
	output_dir: Path,
	colmap_exe: str = "colmap",
	output_type: str = "COLMAP",
	max_image_size: int | None = None,
) -> None:
	"""Run COLMAP image_undistorter on a folder of masks.

	Args:
		mask_dir: Folder containing mask images (as image_name.jpg.png).
		sparse_model_dir: COLMAP sparse model folder (e.g., sparse/0).
		output_dir: Output folder for undistorted masks (will contain image_name.jpg.png).
		colmap_exe: COLMAP executable name or full path.
		output_type: COLMAP output type (e.g., COLMAP).
		max_image_size: Optional max image size (int) passed to COLMAP.
	"""
	mask_dir = Path(mask_dir)
	sparse_model_dir = Path(sparse_model_dir)
	output_dir = Path(output_dir)

	if not mask_dir.exists():
		raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
	if not sparse_model_dir.exists():
		raise FileNotFoundError(f"Sparse model directory not found: {sparse_model_dir}")

	if shutil.which(colmap_exe) is None:
		raise FileNotFoundError(
			f"COLMAP executable not found: {colmap_exe}. "
			"Add COLMAP to PATH or pass --colmap-exe."
		)

	output_dir.mkdir(parents=True, exist_ok=True)
	
	# Create temporary folder with masks as image_name.jpg (without .png extension)
	temp_input_dir = output_dir / "_temp_colmap_input"
	temp_output_dir = output_dir / "_temp_colmap_output"
	
	try:
		_prepare_masks_for_undistortion(mask_dir, temp_input_dir)

		cmd = (
			f"\"{colmap_exe}\" image_undistorter "
			f"--image_path \"{temp_input_dir}\" "
			f"--input_path \"{sparse_model_dir}\" "
			f"--output_path \"{temp_output_dir}\" "
			f"--output_type {output_type}"
		)

		if max_image_size is not None:
			cmd += f" --max_image_size {max_image_size}"

		print("Running:", cmd)
		wound3d.legacy.run_command.run_command(cmd)
		
		# Move undistorted masks from temp output to final output and add .png extension
		_finalize_undistorted_masks(temp_output_dir, output_dir)
		
	finally:
		# Clean up temporary directories
		if temp_input_dir.exists():
			shutil.rmtree(temp_input_dir)
		if temp_output_dir.exists():
			shutil.rmtree(temp_output_dir)


def _prepare_masks_for_undistortion(mask_dir: Path, temp_input_dir: Path) -> None:
	"""
	Copy masks from mask_dir to temp_input_dir, removing the .png extension.
	
	Converts: image_name.jpg.png -> image_name.jpg
	"""
	temp_input_dir.mkdir(parents=True, exist_ok=True)

	mask_files = list(mask_dir.glob("*.png"))
	if not mask_files:
		raise FileNotFoundError(f"No .png masks found in: {mask_dir}")

	for mask_path in mask_files:
		# image.jpg.png -> image.jpg
		output_name = mask_path.stem  # Remove .png extension
		output_path = temp_input_dir / output_name

		if output_path.exists():
			continue

		shutil.copy2(mask_path, output_path)
		

def _finalize_undistorted_masks(temp_output_dir: Path, final_output_dir: Path) -> None:
	"""
	Move undistorted masks from temp COLMAP output to final folder,
	adding back the .png extension.
	
	Converts: image_name.jpg -> image_name.jpg.png
	"""
	final_output_dir.mkdir(parents=True, exist_ok=True)
	
	# COLMAP outputs to temp_output_dir/images/
	images_dir = temp_output_dir / "images"
	if not images_dir.exists():
		raise FileNotFoundError(f"COLMAP output images directory not found: {images_dir}")
	
	mask_files = list(images_dir.glob("*"))
	if not mask_files:
		raise FileNotFoundError(f"No undistorted masks found in: {images_dir}")
	
	for mask_path in mask_files:
		if not mask_path.is_file():
			continue
		
		# image.jpg -> image.jpg.png
		output_name = mask_path.name + ".png"
		output_path = final_output_dir / output_name
		
		if output_path.exists():
			continue
		
		shutil.copy2(mask_path, output_path)


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Undistort mask images using COLMAP image_undistorter."
	)
	parser.add_argument(
		"--mask_dir",
		required=True,
		type=Path,
		help="Path to mask folder (e.g., images_woundmask).",
	)
	parser.add_argument(
		"--sparse_model_dir",
		required=True,
		type=Path,
		help="Path to COLMAP sparse model (e.g., sparse/0).",
	)
	parser.add_argument(
		"--output_dir",
		required=True,
		type=Path,
		help="Output folder for undistorted masks.",
	)
	parser.add_argument(
		"--colmap_exe",
		default="colmap",
		help="COLMAP executable name or full path.",
	)
	parser.add_argument(
		"--output_type",
		default="COLMAP",
		help="COLMAP output type (default: COLMAP).",
	)
	parser.add_argument(
		"--max_image_size",
		type=int,
		default=None,
		help="Optional max image size passed to COLMAP.",
	)
	return parser.parse_args()


def main() -> None:
	#args = _parse_args()
    mask_dir =          r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\test\masks_wound_only"
    sparse_model_dir =  r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\test\sparse\0"
    output_dir =        r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\test\mask_wound_only_undistorted"
    undistort_masks_with_colmap(
		mask_dir=mask_dir,
		sparse_model_dir=sparse_model_dir,
		output_dir=output_dir,
		colmap_exe="colmap",
		output_type="COLMAP",
		max_image_size=None,
	)


if __name__ == "__main__":
	main()
