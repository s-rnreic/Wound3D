"""
Undistort mask images using COLMAP sparse model and image_undistorter.

Masks are named like the original images with an extra .png extension
(e.g., image.jpg.png). This script runs COLMAP's image_undistorter
directly on the mask folder.
"""
from __future__ import annotations
import os
import sys


_current_dir = os.path.dirname(os.path.abspath(__file__))
_pipe_code_dir = os.path.abspath(os.path.join(_current_dir, '..', '..'))
_workspace_root = os.path.abspath(os.path.join(_current_dir, '..', '..', '..'))

if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)
if _pipe_code_dir not in sys.path:
    sys.path.insert(0, _pipe_code_dir)


import argparse
import shutil
import subprocess
import Pipe_Code.run_command
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
		mask_dir: Folder containing mask images.
		sparse_model_dir: COLMAP sparse model folder (e.g., sparse/0).
		output_dir: Output folder for undistorted masks.
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
	compat_mask_dir = output_dir / "_colmap_mask_input"
	_ensure_colmap_mask_names(mask_dir, compat_mask_dir)

	cmd = (
		f"\"{colmap_exe}\" image_undistorter "
		f"--image_path \"{compat_mask_dir}\" "
		f"--input_path \"{sparse_model_dir}\" "
		f"--output_path \"{output_dir}\" "
		f"--output_type {output_type}"
	)

	if max_image_size is not None:
		cmd += f" --max_image_size {max_image_size}"

	print("Running:", cmd)
	Pipe_Code.run_command.run_command(cmd)


def _ensure_colmap_mask_names(mask_dir: Path, compat_mask_dir: Path) -> None:
	"""Create a COLMAP-compatible mask folder by removing the trailing .png."""
	compat_mask_dir.mkdir(parents=True, exist_ok=True)

	mask_files = list(mask_dir.glob("*.png"))
	if not mask_files:
		raise FileNotFoundError(f"No .png masks found in: {mask_dir}")

	for mask_path in mask_files:
		# image.jpg.png -> image.jpg, image.png.png -> image.png
		output_name = mask_path.stem
		output_path = compat_mask_dir / output_name

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
