"""Explicit-state orchestration for the wound3d reconstruction workflow."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from wound3d.config import PipelineConfig
from wound3d.workspace import WorkspacePaths


def _log_time(log_file: Path, step: str, elapsed_time: float | None) -> None:
    message = f"{step}: {elapsed_time:.2f} seconds\n" if elapsed_time is not None else f"{step}: elapsed time is None\n"
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(message)


def _total_elapsed_time(log_file: Path) -> float:
    if not log_file.is_file():
        return 0.0
    total = 0.0
    for line in log_file.read_text(encoding="utf-8").splitlines():
        token = line.rsplit(":", maxsplit=1)[-1].strip().split()[0]
        if token != "elapsed":
            total += float(token)
    return total


def _runtime_modules():
    """Load legacy scientific implementation modules only for a pipeline run."""
    from wound3d.legacy import MVS_Meshing as mvs
    from wound3d.legacy import create_temp_glomap_ini as glomap_ini
    from wound3d.legacy import create_temp_project_ini as project_ini
    from wound3d.legacy import prepare_vid_for_colmap_v5 as video
    from wound3d.legacy import rerun_viewer
    from wound3d.legacy.modules.dense_postprocessing.PointcloudWorkflow import execute_dense_postprocessing
    from wound3d.legacy.modules.masking import build_masks
    from wound3d.legacy.modules.WoundEdgeDetection.Wound_Detection_Mesh import main as execute_segmentation
    from wound3d.legacy.run_command import run_command
    from wound3d.legacy.utils.ArUcoTracking import main as execute_apriltag_scaling
    from wound3d.legacy.utils.read_write_model import write_model

    return {
        "mvs": mvs,
        "glomap_ini": glomap_ini,
        "project_ini": project_ini,
        "video": video,
        "rerun_viewer": rerun_viewer,
        "execute_dense_postprocessing": execute_dense_postprocessing,
        "build_masks": build_masks,
        "execute_segmentation": execute_segmentation,
        "run_command": run_command,
        "execute_apriltag_scaling": execute_apriltag_scaling,
        "write_model": write_model,
    }


def run_pipeline(config: PipelineConfig) -> tuple[dict, float]:
    modules = _runtime_modules()
    paths = WorkspacePaths.create(config.workspace)
    image_options = {"reduce_size": config.resize_factor}

    start = time.time()
    modules["video"].extract_images_from_video(str(config.input_video), str(paths.raw_images))
    sharpness = modules["video"].get_image_sharpness(str(paths.raw_images), mask_folder_path=None, image_options=image_options)
    selected = modules["video"].select_sharp_images(
        sharpness, config.segments, absolut_min=config.sharpness_percentile, input_path=str(paths.root)
    )
    modules["video"].copy_selected_images(selected, str(paths.raw_images), str(paths.sharp_images), image_options)
    if config.masking.enabled:
        modules["build_masks"].process_folder(
            str(paths.sharp_images),
            str(paths.masks),
            Use_SAM2=config.masking.backend == "sam2",
            Use_DeepSkin=config.masking.backend == "deepskin",
            Use_WoundAmbit=config.masking.backend == "woundambit",
            use_mask_adjustment=config.masking.use_adjustment,
            extrude_masks=config.masking.extrude_masks,
            min_mask_ratio=config.masking.min_mask_ratio,
            max_mask_ratio=config.masking.max_mask_ratio,
            neighbors_count=config.masking.neighbors_count,
            sam2_checkpoint=config.masking.sam2_checkpoint,
            sam2_model_config=config.masking.sam2_model_config,
            woundambit_executable=config.masking.woundambit_executable,
            woundambit_working_directory=config.masking.woundambit_working_directory,
        )
    if config.enhance_option:
        modules["video"].enhance_img(str(paths.sharp_images), str(paths.sharp_images), config.enhance_option)
    _log_time(paths.log_file, "Video_Pipeline", time.time() - start)

    start = time.time()
    ini_path, database_path = modules["project_ini"].standard_ini_pipline(
        str(paths.root), str(paths.sharp_images), str(paths.masks), Masking=config.masking.enabled
    )
    feature_ini = modules["project_ini"].create_temp_project_ini(ini_path, "HEADER", "ImageReader", "SiftExtraction")
    modules["run_command"](f'colmap feature_extractor --project_path "{feature_ini}"')
    modules["project_ini"].append_to_total_project_ini(feature_ini, str(paths.root / "total_project.ini"))
    match_ini = modules["project_ini"].create_temp_project_ini(ini_path, "HEADER", "SiftMatching", "TwoViewGeometry", "ExhaustiveMatching")
    modules["project_ini"].delete_keywords_from_project_ini(match_ini, "HEADER", "image_path")
    modules["run_command"](f'colmap exhaustive_matcher --project_path "{match_ini}"')
    modules["project_ini"].append_to_total_project_ini(match_ini, str(paths.root / "total_project.ini"))
    _log_time(paths.log_file, "COLMAP_Pipeline", time.time() - start)

    paths.sparse.mkdir(parents=True, exist_ok=True)
    modules["glomap_ini"].create_glomap_ini(str(paths.glomap_ini))
    start = time.time()
    modules["glomap_ini"].execute_glomap(str(paths.glomap_ini), database_path, str(paths.sparse), str(paths.sharp_images))
    modules["project_ini"].append_to_total_project_ini(str(paths.glomap_ini), str(paths.root / "total_project.ini"))
    _log_time(paths.log_file, "GLOMAP_Pipeline", time.time() - start)

    for model_directory in (entry for entry in paths.sparse.iterdir() if entry.is_dir()):
        result = subprocess.run(
            f'colmap model_analyzer --path "{model_directory}"',
            shell=True,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

    if config.run_rerun_filter:
        start = time.time()
        cameras, images, filtered_points = modules["rerun_viewer"].Rerun_Colmap_Pipeline(str(paths.root))
        modules["write_model"](cameras, images, filtered_points, paths.sparse / "0")
        _log_time(paths.log_file, "COLMAP_RERUN", time.time() - start)

    if config.run_apriltag_scaling:
        start = time.time()
        modules["execute_apriltag_scaling"](str(paths.root), config.tag_size_m)
        _log_time(paths.log_file, "AprilTagScaling", time.time() - start)

    elapsed = modules["mvs"].execute_dense_reconstruction(str(paths.root))
    _log_time(paths.log_file, "MVS", elapsed)

    processed_cloud = paths.dense / "processed_pointcloud.ply"
    _, elapsed = modules["execute_dense_postprocessing"](
        dense_model_path=paths.dense,
        output_path=processed_cloud,
        radial_percentile=0.99,
        target_points=100000,
        SOR_std=1.5,
        SOR_n=10,
        use_pointcleannet=config.use_pointcleannet,
        pointcleannet_denoise=False,
        pointcleannet_outliers=False,
        denoise_iterations=1,
    )
    _log_time(paths.log_file, "DensePostProcessing", elapsed)

    elapsed = modules["mvs"].execute_poisson_meshing(str(paths.root), input_ply=str(processed_cloud))
    _log_time(paths.log_file, "Meshing", elapsed)

    start = time.time()
    results = modules["execute_segmentation"](str(paths.root))
    _log_time(paths.log_file, "3DSegmentation", time.time() - start)
    return results, _total_elapsed_time(paths.log_file)