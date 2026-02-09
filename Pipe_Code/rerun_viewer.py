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
import sys
sys.path.append('C:\\Users\\Reich\\LocalCoding\\COLMAP_Pipline\\external')  # Add this line
from Pipe_Code.utils.read_write_model import Camera, read_model  # type: ignore[attr-defined]
import matplotlib.pyplot as plt

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


def read_and_log_sparse_reconstruction(dataset_path: Path, filter_output: bool=False, resize: tuple[int, int]=None):
    #print("Reading sparse COLMAP reconstruction")
    sparse_path = dataset_path / 'sparse/0'
    cameras, images, points3D = read_model(sparse_path, ext=".bin")

    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    
    all_points_xyz = [point.xyz for point in points3D.values()]
    # Calculate the center of the point cloud
    center_point = np.mean(all_points_xyz, axis=0)
    # Calculate the position of each point relative to the center
    all_points_xyz = [point - center_point for point in all_points_xyz]
    all_point_colors = [point.rgb for point in points3D.values()]
    all_point_errors = [point.error for point in points3D.values()]

    error_histogram, bin_edges = np.histogram(all_point_errors, bins='auto')

    rr.log("all_points", rr.Points3D(all_points_xyz, colors=all_point_colors), rr.AnyValues(error=all_point_errors), static=True)
    rr.log("Bar/avg_reproj_err", rr.BarChart(error_histogram), static=True)
    
    
    # Log points with colors based on reprojection error
    error_colors = map_error_to_color(all_point_errors)
    rr.log("color_map_mean_repo_error", rr.Points3D(all_points_xyz, colors=error_colors), static=True)
    rr.log("color_map_mean_repo_error", rr.Scalars(all_point_errors), static=True)

    # Log points with colors based on track length
    track_lengths = [len(point.image_ids) for point in points3D.values()]
    track_lengths_histo, bin_edges= np.histogram(track_lengths, bins='auto')
    mean_track_length = np.mean(track_lengths)
    track_colors = [
        [int(255 * (1 - (length / mean_track_length))), 
         int(255 * (length / mean_track_length)), 
         0]
        for length in track_lengths
    ]
    rr.log("track_length", rr.Points3D(all_points_xyz, colors=track_colors), static=True)
    rr.log("track_length", rr.Scalars(track_lengths), static=True)
    rr.log("Bar/track_length", rr.BarChart(track_lengths_histo), static=True)
    
    
    # Iterate through images (video frames) logging data related to each frame.
    image_list = sorted(images.values(), key=lambda im: im.name)
    for image in image_list:  # type: ignore[no-any-return]
        image_file = dataset_path / 'sharp_images' / image.name
        mask_file = dataset_path / 'masks' / f"{image.name}.png"
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
            rr.log("/plot/avg_reproj_err", rr.Scalars(np.mean(point_errors)))

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
            image= cv2.imread(str(image_file), cv2.IMREAD_COLOR_RGB)
            if os.path.exists(mask_file):
                masked_image = cv2.bitwise_and(image, image, mask=cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE))
                combined = cv2.addWeighted(image, 0.7, cv2.cvtColor(cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR), 0.3, 0)

                rr.log("camera/image", rr.Image(combined))
            rr.log("camera/image", rr.Image(image))
            
        rr.log("camera/image/keypoints", rr.Points2D(visible_xys, colors=[34, 138, 167]))

    filter_output = True
    if filter_output:
        # Filter out noisy points
        filtered_points3D = {id: point for id, point in points3D.items() if point.rgb.any() and point.error < 1.4}
    
    center_point = np.mean([point.xyz for point in filtered_points3D.values()], axis=0)
    rr.log("/filtered_points", rr.Points3D([point.xyz - center_point for point in filtered_points3D.values()], colors=[point.rgb for point in filtered_points3D.values()]), rr.AnyValues(error=[point.error for point in filtered_points3D.values()]), static=True)
    #distance array of all points
    distance_arr = [np.linalg.norm(point) for point in [point - center_point for point in [point.xyz for point in filtered_points3D.values()]]]
    
    # Calculate the 95th percentile distance
    distance_95th_percentile = np.percentile(distance_arr, 95)
    print(f"95th percentile distance after filtering: {distance_95th_percentile}")
    distance_filtered_points=[point for point in filtered_points3D.values() if np.linalg.norm(point.xyz -center_point) < distance_95th_percentile]
    rr.log("/filtered_points3D_95_distance", rr.Points3D([point.xyz for point in distance_filtered_points], colors=[point.rgb for point in distance_filtered_points]), rr.AnyValues(error=[point.error for point in distance_filtered_points]), static=True)

    
    filtered_points3D = {id: point for id, point in points3D.items() if point.rgb.any() and point.error < 1.4 and np.linalg.norm(point.xyz -center_point) < distance_95th_percentile} 
    
    return cameras, images, filtered_points3D
    

def Rerun_Colmap_Pipeline(workspace_path:str)->None:

    rr.init('COLMAP_Visualization', recording_id=workspace_path, spawn=True)

    # Define a blueprint (layout) to organize how data appears
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.Spatial2DView(origin="/camera/image"),
                        rrb.Spatial3DView(origin="/", contents=["+ $origin/**", "- /all_points", "- /color_map_mean_repo_error/**", "- /track_length/**", "- /filtered_points/**", "- /filtered_points3D_95_distance/**"]),
                
                    ),
                    rrb.Tabs(
                        rrb.Horizontal(rrb.Tabs(
                            rrb.Spatial3DView(origin="all_points"),
                            rrb.Spatial3DView(origin="color_map_mean_repo_error"),
                            rrb.Spatial3DView(origin="track_length"),
                            ),
                            rrb.Tabs(rrb.TimeSeriesView(origin="/plot/avg_reproj_err"),rrb.BarChartView(origin="/Bar"))),
                        rrb.Horizontal(
                            rrb.Tabs(rrb.Spatial3DView(origin="filtered_points"),
                            rrb.Spatial3DView(origin="filtered_points3D_95_distance"),
                            )
                            ),
                    )
                )
            )
        )
    )

    dataset_path = Path(workspace_path)
    print(f"Using dataset at {dataset_path}")
    cameras, images, filtered_points3D=read_and_log_sparse_reconstruction(dataset_path)
    return cameras, images, filtered_points3D

if __name__ == "__main__":
    #main()
    Rerun_Colmap_Pipeline(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\1_reso_0.1_sharp_circ_08_mask")
    # base_path = r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments"
    # for folder in os.listdir(base_path):
    #     full_path = os.path.join(base_path, folder)
    #     Rerun_Colmap_Pipeline(full_path)
