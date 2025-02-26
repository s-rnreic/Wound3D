from __future__ import annotations

import io
import os
import re
import zipfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Final
import rerun.blueprint as rrb
import cv2
import numpy as np
import numpy.typing as npt
import requests
import rerun as rr  # pip install rerun-sdk
from tqdm import tqdm
from read_write_model import Camera, read_model  # type: ignore[attr-defined]

FILTER_MIN_VISIBLE: Final = 500
DESCRIPTION = """
# Sparse reconstruction by COLMAP
This example was generated from the output of a sparse reconstruction done with COLMAP.

[COLMAP](https://colmap.github.io/index.html) is a general-purpose Structure-from-Motion (SfM) and Multi-View Stereo
(MVS) pipeline with a graphical and command-line interface.

In this example a short video clip has been processed offline by the COLMAP pipeline, and we use Rerun to visualize the
individual camera frames, estimated camera poses, and resulting point clouds over time.

The full source code for this example is available
[on GitHub](https://github.com/rerun-io/rerun/blob/latest/examples/python/structure_from_motion).
""".strip()

def scale_camera(camera: Camera, resize: tuple[int, int]) -> tuple[Camera, npt.NDArray[np.float_]]:
    """Scale the camera intrinsics to match the resized image."""
    #assert camera.model == "PINHOLE"
    new_width = resize[0]
    new_height = resize[1]
    scale_factor = np.array([new_width / camera.width, new_height / camera.height])

    # For PINHOLE camera model, params are: [focal_length_x, focal_length_y, principal_point_x, principal_point_y]
    new_params = np.append(camera.params[:2] * scale_factor, camera.params[2:] * scale_factor)

    return (Camera(camera.id, camera.model, new_width, new_height, new_params), scale_factor)

def map_error_to_color(errors: list[float]) -> list[list[int]]:
    """Map reprojection errors to colors based on thresholds."""
    mean_error = np.mean(errors)
    colors = []
    for error in errors:
        if error > 3:
            colors.append([255, 0, 0])  # Red
        elif error < 0.5 * mean_error:
            colors.append([0, 255, 0])  # Green
        else:
            normalized_error = (error - 0.5 * mean_error) / (3 - 0.5 * mean_error)
            red = int(255 * normalized_error)
            green = int(255 * (1 - normalized_error))
            colors.append([red, green, 0])
    return colors

def read_and_log_sparse_reconstruction(dataset_path: Path, filter_output: bool, resize: tuple[int, int] | None) -> None:
    #print("Reading sparse COLMAP reconstruction")
    sparse_path = dataset_path / "sparse//0"
    cameras, images, points3D = read_model(sparse_path, ext=".bin")
    #print("Building visualization by logging to Rerun")

    if filter_output:
        # Filter out noisy points
        points3D = {id: point for id, point in points3D.items()}

    
    #rr.log("description", rr.TextDocument(DESCRIPTION, media_type=rr.MediaType.MARKDOWN), static=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    
    all_points = [point.xyz for point in points3D.values()]
    all_point_colors = [point.rgb for point in points3D.values()]
    all_point_errors = [point.error for point in points3D.values()]
    rr.log("all_points", rr.Points3D(all_points, colors=all_point_colors), rr.AnyValues(error=all_point_errors), static=True)

    # Log points with colors based on reprojection error
    error_colors = map_error_to_color(all_point_errors)
    rr.log("color_map_mean_repo_error", rr.Points3D(all_points, colors=error_colors), static=True)

    # Log points with colors based on track length
    track_lengths = [len(point.image_ids) for point in points3D.values()]
    mean_track_length = np.mean(track_lengths)
    track_colors = [
        [int(255 * (1 - (length / mean_track_length))), 
         int(255 * (length / mean_track_length)), 
         0]
        for length in track_lengths
    ]
    rr.log("track_length", rr.Points3D(all_points, colors=track_colors), static=True)
    rr.log("track_length", rr.Scalar(track_lengths), static=True)
    #colormap of reprojection error
    
    
    # Iterate through images (video frames) logging data related to each frame.
    image_list = sorted(images.values(), key=lambda im: im.name)
    for image in image_list:  # type: ignore[no-any-return]
        image_file = dataset_path / "sharp_images" / image.name
        mask_file = dataset_path / "masks" / (image.name + '.jpg')
        if not os.path.exists(image_file):
            continue

        # COLMAP sets image ids that don't match the original video frame
        idx_match = re.search(r"(\d+)(?=\.\w+$)", image.name)
        assert idx_match is not None
        frame_idx = int(idx_match.group(0))

        quat_xyzw = image.qvec[[1, 2, 3, 0]]  # COLMAP uses wxyz quaternions
        camera = cameras[image.camera_id]
        if resize:
            camera, scale_factor = scale_camera(camera, resize)
        else:
            scale_factor = np.array([1.0, 1.0])

        visible = [id != -1 and points3D.get(id) is not None for id in image.point3D_ids]
        visible_ids = image.point3D_ids[visible]

        if filter_output and len(visible_ids) < FILTER_MIN_VISIBLE:
            continue

        visible_xyzs = [points3D[id] for id in visible_ids]
        visible_xys = image.xys[visible]
        if resize:
            visible_xys *= scale_factor

        #rr.set_time_sequence("frames", frame_idx)

        points = [point.xyz for point in visible_xyzs]
        point_colors = [point.rgb for point in visible_xyzs]
        point_errors = [point.error for point in visible_xyzs]

        if point_errors:
            rr.log("plot/avg_reproj_err", rr.Scalar(np.mean(point_errors)))

        rr.log("points", rr.Points3D(points, colors=point_colors), rr.AnyValues(error=point_errors))

        # COLMAP's camera transform is "camera from world"
        rr.log(
            "camera", rr.Transform3D(translation=image.tvec, rotation=rr.Quaternion(xyzw=quat_xyzw), from_parent=True)
        )
        rr.log("camera", rr.ViewCoordinates.RDF, static=True)  # X=Right, Y=Down, Z=Forward

        # Log camera intrinsics
        #assert camera.model == "PINHOLE"
        rr.log(
            "camera/image",
            rr.Pinhole(
                resolution=[camera.width, camera.height],
                focal_length=camera.params[:2],
                principal_point=camera.params[2:],
            ),
        )

        if resize:
            bgr = cv2.imread(str(image_file))
            bgr = cv2.resize(bgr, resize)
            rr.log("camera/image", rr.Image(bgr, color_model="BGR").compress(jpeg_quality=75))
        else:
            rr.log("camera/image", rr.EncodedImage(path=dataset_path / "sharp_images" / image.name))

        rr.log("camera/image/keypoints", rr.Points2D(visible_xys, colors=[34, 138, 167]))
        

def Rerun_Colmap_Pipeline(workspace_path:str)->None:
    parser = ArgumentParser(description="Visualize the output of COLMAP's sparse reconstruction on a video.")
    parser.add_argument("--unfiltered", action="store_true", help="If set, we don't filter away any noisy data.")
    parser.add_argument(
        "--dataset",
        action="store",
        default="colmap_rusty_car",
        choices=["colmap_rusty_car", "colmap_fiat"],
        help="Which dataset to download",
    )
    parser.add_argument("--resize", action="store", help="Target resolution to resize images")
    rr.script_add_args(parser)
    args = parser.parse_args()
    if args.resize:
        args.resize = tuple(int(x) for x in args.resize.split("x"))

    rr.script_setup(args, "COLMAP_PIPELINE_VISUIZATION",recording_id=workspace_path)
    dataset_path=Path(workspace_path)
    print(f"Using dataset at {dataset_path}")
    read_and_log_sparse_reconstruction(dataset_path, filter_output=not args.unfiltered, resize=args.resize)
    rr.script_teardown(args)

def main():
    parser = ArgumentParser(description="Visualize the output of COLMAP's sparse reconstruction on a video.")
    parser.add_argument("--unfiltered", action="store_true", help="If set, we don't filter away any noisy data.")
    parser.add_argument(
        "--dataset",
        action="store",
        default="colmap_rusty_car",
        choices=["colmap_rusty_car", "colmap_fiat"],
        help="Which dataset to download",
    )
    parser.add_argument("--resize", action="store", help="Target resolution to resize images")
    rr.script_add_args(parser)
    args = parser.parse_args()
    print(args)
    if args.resize:
        args.resize = tuple(int(x) for x in args.resize.split("x"))

    rr.script_setup(args, "COLMAP_PIPELINE_VISUZATN")
    dataset_path=Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\test")
    print(f"Using dataset at {dataset_path}")
    read_and_log_sparse_reconstruction(dataset_path, filter_output=False, resize=args.resize) #not args.unfiltered
    rr.script_teardown(args)
    
    
if __name__ == "__main__":
    #main()
    Rerun_Colmap_Pipeline(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\benchmark_new\06_circ_mask_more_features")
