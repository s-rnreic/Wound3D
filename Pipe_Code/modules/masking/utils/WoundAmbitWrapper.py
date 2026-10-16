"""
WoundAmbit Wrapper - Accepts input/output folders via command-line arguments.

Usage:
    python WoundAmbitWrapper.py --input_folder /path/to/images --output_folder /path/to/masks

The wrapper overrides the default IMG_PATH and save path in the WoundAmbit inference helper
to point to the specified input/output folders.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


try:
    from medseg.util.path_builder import PathBuilder
except ModuleNotFoundError as error:
    raise RuntimeError(
        "WoundAmbit is not installed in this environment. Initialize the external/WoundAmbit submodule and "
        "install it in the selected environment before running the wrapper."
    ) from error


class FolderOverridePathBuilder(PathBuilder):
    """
    A PathBuilder subclass that returns a fixed folder path instead of the default WoundAmbit path.
    This replaces the default IMG_PATH PathBuilder so WoundAmbit reads from our input folder.
    """
    def __init__(self, folder_path: str):
        self._folder_path = folder_path

    def build(self, make_dirs=True):
        if make_dirs:
            os.makedirs(self._folder_path, exist_ok=True)
        return self._folder_path

    def clone(self):
        return FolderOverridePathBuilder(self._folder_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WoundAmbit inference with custom input/output folders.")
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Path to folder containing input images")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to folder where output masks will be saved")
    args = parser.parse_args()

    input_folder = os.path.abspath(args.input_folder)
    output_folder = os.path.abspath(args.output_folder)

    # Verify input folder exists
    if not os.path.isdir(input_folder):
        print(f"ERROR: Input folder does not exist: {input_folder}")
        sys.exit(1)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    from medseg.tools.inference.inference_helper_optimized import (  # noqa: WPS433
        MODEL_MAPPING,
        ImageInferenceHelper,
    )

    model_name = "TransNeXt"
    print(f"Running WoundAmbit inference for model: {model_name}")
    print(f"  Input folder:  {input_folder}")
    print(f"  Output folder: {output_folder}")

    # Override IMG_PATH to point to our input folder
    img_path_override = FolderOverridePathBuilder(input_folder)

    # Override the _save_mask method on the instance after creation to use our output folder
    original_init = ImageInferenceHelper.__init__

    def patched_init(self, model_names, model_ckpts_paths, img_path, dataset_name,
                     is_kfold=True, save_contours=False, batch_size=1, num_workers=16,
                     ensemble_name="ensemble"):
        # Call original init with our overridden img_path
        original_init(self, model_names, model_ckpts_paths, img_path, dataset_name,
                      is_kfold=is_kfold, save_contours=save_contours,
                      batch_size=batch_size, num_workers=num_workers,
                      ensemble_name=ensemble_name)

        # Override _save_mask to save to our output folder
        original_save_mask = self._save_mask

        def patched_save_mask(image_name, mask_img, annotated_image=None):
            """Save mask to the specified output folder instead of the default path."""
            save_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            mask_img.save(save_path)
            print(f"Saved mask for {image_name} at {save_path}")

        self._save_mask = patched_save_mask

    ImageInferenceHelper.__init__ = patched_init

    inference = ImageInferenceHelper(
        model_names=[model_name],
        model_ckpts_paths={
            model_name: PathBuilder().root().out().add("k_fold_models").add(MODEL_MAPPING[model_name][1])
        },
        img_path=img_path_override,
        dataset_name="pipeline_inference",
        is_kfold=True,
        save_contours=False,
        batch_size=1,
        num_workers=4,
    )
    inference.run_inference()

    print(f"\nWoundAmbit inference complete! Masks saved to: {output_folder}")


if __name__ == "__main__":
    main()