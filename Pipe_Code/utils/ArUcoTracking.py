from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from pupil_apriltags import Detector

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Pipe_Code.utils.read_write_model import Camera, Image, Point3D, qvec2rotmat, read_model, write_model


def create_detector() -> Detector:
    return Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=True,
    )


def camera_matrix_and_distortion(camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    model = camera.model.upper()
    params = np.asarray(camera.params, dtype=np.float64)

    if model == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1:3]
        distortion = np.zeros((4, 1), dtype=np.float64)
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
        distortion = np.zeros((4, 1), dtype=np.float64)
    elif model == "SIMPLE_RADIAL":
        fx = fy = params[0]
        cx, cy = params[1:3]
        distortion = np.array([params[3], 0.0, 0.0, 0.0], dtype=np.float64).reshape(-1, 1)
    elif model == "RADIAL":
        fx = fy = params[0]
        cx, cy = params[1:3]
        distortion = np.array([params[3], params[4], 0.0, 0.0], dtype=np.float64).reshape(-1, 1)
    elif model == "OPENCV":
        fx, fy, cx, cy = params[:4]
        distortion = np.array(params[4:8], dtype=np.float64).reshape(-1, 1)
    elif model == "FULL_OPENCV":
        fx, fy, cx, cy = params[:4]
        distortion = np.array(params[4:12], dtype=np.float64).reshape(-1, 1)
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {camera.model}")

    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return camera_matrix, distortion


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def detect_apriltags(image_bgr: np.ndarray, detector: Detector) -> list:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return detector.detect(gray)


def annotate_detections(image_bgr: np.ndarray, detections: list) -> np.ndarray:
    annotated = image_bgr.copy()
    for detection in detections:
        corners = detection.corners.astype(int)
        for index in range(4):
            p1 = tuple(corners[index])
            p2 = tuple(corners[(index + 1) % 4])
            cv2.line(annotated, p1, p2, (0, 255, 0), 2)

        center = tuple(detection.center.astype(int))
        cv2.circle(annotated, center, 5, (0, 0, 255), -1)
        cv2.putText(
            annotated,
            str(detection.tag_id),
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )
    return annotated


def estimate_tag_center_in_camera(
    corners_px: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    tag_size_m: float,
) -> np.ndarray | None:
    half = tag_size_m / 2.0
    object_points = np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=np.float64,
    )

    success, _, tvec = cv2.solvePnP(
        object_points,
        corners_px.astype(np.float64),
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    return tvec.reshape(3)


def camera_center_from_image(image: Image) -> np.ndarray:
    rotation = qvec2rotmat(image.qvec)
    return -rotation.T @ np.asarray(image.tvec, dtype=np.float64)


def world_from_camera_rotation(image: Image) -> np.ndarray:
    return qvec2rotmat(image.qvec).T


def estimate_scene_scale_from_tags(
    observations: list[tuple[Image, int, np.ndarray]],
) -> tuple[float, float, dict[int, np.ndarray]]:
    tag_to_rows: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for image, tag_id, tag_center_camera in observations:
        camera_center = camera_center_from_image(image)
        world_from_camera = world_from_camera_rotation(image)
        transformed = world_from_camera @ tag_center_camera
        tag_to_rows.setdefault(tag_id, []).append((camera_center, transformed))

    inv_scales: list[float] = []
    tag_positions: dict[int, np.ndarray] = {}

    for tag_id, rows in tag_to_rows.items():
        if len(rows) < 2:
            continue

        matrix_rows = []
        rhs_rows = []
        for camera_center, transformed in rows:
            matrix_rows.append(np.hstack([np.eye(3), -transformed.reshape(3, 1)]))
            rhs_rows.append(camera_center.reshape(3, 1))

        matrix = np.vstack(matrix_rows)
        rhs = np.vstack(rhs_rows).reshape(-1)
        solution, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
        tag_position = solution[:3]
        inv_scale = solution[3]

        if np.isfinite(inv_scale) and inv_scale > 0:
            inv_scales.append(float(inv_scale))
            tag_positions[tag_id] = tag_position

    if not inv_scales:
        raise ValueError("Need at least one AprilTag seen in at least two images to estimate scale.")

    # Compute scale factor as median of inverse scales
    median_inv_scale = float(np.median(inv_scales))
    scale_factor = 1.0 / median_inv_scale

    # Compute standard deviation in scale-space for confidence indication
    scales = [1.0 / inv for inv in inv_scales]
    scale_std = float(np.std(scales, ddof=1)) if len(scales) > 1 else 0.0

    return scale_factor, scale_std, tag_positions


def scale_sparse_model(
    cameras: dict[int, Camera],
    images: dict[int, Image],
    points3d: dict[int, Point3D],
    scale_factor: float,
    center: np.ndarray | None = None,
) -> tuple[dict[int, Camera], dict[int, Image], dict[int, Point3D]]:
    if center is None:
        if points3d:
            center = np.mean([np.asarray(point.xyz, dtype=np.float64) for point in points3d.values()], axis=0)
        else:
            center = np.zeros(3, dtype=np.float64)

    scaled_points3d: dict[int, Point3D] = {}
    for point_id, point in points3d.items():
        scaled_xyz = (np.asarray(point.xyz, dtype=np.float64) - center) * scale_factor + center
        scaled_points3d[point_id] = Point3D(
            id=point.id,
            xyz=scaled_xyz,
            rgb=np.asarray(point.rgb, dtype=np.uint8),
            error=float(point.error),
            image_ids=np.asarray(point.image_ids, dtype=np.int32),
            point2D_idxs=np.asarray(point.point2D_idxs, dtype=np.int32),
        )

    scaled_images: dict[int, Image] = {}
    for image_id, image in images.items():
        rotation = qvec2rotmat(image.qvec)
        camera_center = -rotation.T @ np.asarray(image.tvec, dtype=np.float64)
        scaled_center = (camera_center - center) * scale_factor + center
        scaled_tvec = -rotation @ scaled_center
        scaled_images[image_id] = Image(
            id=image.id,
            qvec=np.asarray(image.qvec, dtype=np.float64),
            tvec=scaled_tvec,
            camera_id=image.camera_id,
            name=image.name,
            xys=np.asarray(image.xys, dtype=np.float64),
            point3D_ids=np.asarray(image.point3D_ids, dtype=np.int64),
        )

    return cameras, scaled_images, scaled_points3d

def scale_dense_model() -> None:
    # Placeholder for scaling the dense model (e.g., depth maps, normal maps) if needed.
    # This would involve reading the dense model data, applying the same scale factor, and writing it back.
    pass 

def gather_tag_observations(
    image_dir: Path,
    cameras: dict[int, Camera],
    images: dict[int, Image],
    tag_size_m: float,
) -> tuple[list[tuple[Image, int, np.ndarray]], dict[str, np.ndarray]]:
    detector = create_detector()
    observations: list[tuple[Image, int, np.ndarray]] = []
    previews: dict[str, np.ndarray] = {}

    image_by_name = {image.name: image for image in images.values()}
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue

        image = load_image(image_path)
        detections = detect_apriltags(image, detector)
        if not detections:
            continue

        previews[image_path.name] = annotate_detections(image, detections)
        colmap_image = image_by_name.get(image_path.name)
        if colmap_image is None:
            continue

        camera = cameras[colmap_image.camera_id]
        camera_matrix, distortion = camera_matrix_and_distortion(camera)
        for detection in detections:
            tag_center_camera = estimate_tag_center_in_camera(
                detection.corners,
                camera_matrix,
                distortion,
                tag_size_m,
            )
            if tag_center_camera is None:
                continue
            observations.append((colmap_image, int(detection.tag_id), tag_center_camera))

    return observations, previews


def save_previews(previews: dict[str, np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, preview in previews.items():
        cv2.imwrite(str(output_dir / name), preview)


def main(workspace_path: str, Tag_size: float) -> None:
    
    # workspace_path: Path to the workspace directory. Should contain 'sharp_images' and 'sparse' from COLMAP reconstruction
    # Tag_size: Physical size of the AprilTags in meters (e.g., 0.04 for 4cm tags)
    
    workspace_path = Path(workspace_path)
    
    manuel_scale = None
    save_previews = None
    show = False

    image_dir = workspace_path/"sharp_images"
    
    # Determine sparse model directory: try workspace_path/sparse first, then workspace_path/sparse/0
    sparse_model_dir = None
    sparse_base = workspace_path / "sparse"
    candidates_to_check = [sparse_base, sparse_base / "0"]
    
    for candidate in candidates_to_check:
        if candidate.exists() and candidate.is_dir():
            # Check if any .bin files exist in this directory
            bin_files = list(candidate.glob("*.bin"))
            if bin_files:
                sparse_model_dir = candidate
                break
    
    if sparse_model_dir is None:
        # If not found in direct candidates, search all subdirectories of sparse_base
        if sparse_base.exists() and sparse_base.is_dir():
            for sub_dir in sorted(sparse_base.iterdir()):
                if sub_dir.is_dir():
                    bin_files = list(sub_dir.glob("*.bin"))
                    if bin_files:
                        sparse_model_dir = sub_dir
                        break
    
    if sparse_model_dir is None:
        print("no sparse reconstruction done or cant be found")
        return
    
    output_model_dir = sparse_model_dir
    
    cameras, images, points3d = read_model(sparse_model_dir, ext=".bin")
    observations, previews = gather_tag_observations(image_dir, cameras, images, Tag_size)

    if not observations:
        raise RuntimeError("No AprilTag observations matched the COLMAP images.")

    if manuel_scale is not None:
        scale_factor = float(manuel_scale)
        print(f"Using manual scale factor: {scale_factor:.6f}")
    else:
        scale_factor, scale_std, tag_positions = estimate_scene_scale_from_tags(observations)
        print(f"Estimated metric scale factor: {scale_factor:.6f} ± {scale_std:.6f} m / COLMAP unit")
        if scale_std > 0:
            print(f"Scale confidence (relative std): {scale_std / scale_factor * 100:.2f}%")
        for tag_id, position in sorted(tag_positions.items()):
            print(f"Tag {tag_id} position in COLMAP units: {position}")

    if output_model_dir is not None:
        scaled_cameras, scaled_images, scaled_points3d = scale_sparse_model(
            cameras,
            images,
            points3d,
            scale_factor,
        )
        output_model_dir.mkdir(parents=True, exist_ok=True)
        write_model(scaled_cameras, scaled_images, scaled_points3d, output_model_dir, ext=".bin")
        print(f"Wrote scaled model to: {output_model_dir}")
        
        scale_dense_model()

    if save_previews is not None:
        save_previews(previews, save_previews)
        print(f"Saved annotated images to: {save_previews}")

    if show and previews:
        first_preview = next(iter(previews.values()))
        cv2.imshow("AprilTags", first_preview)
        cv2.waitKey(0)
        
    print("AprilTag-based scaling completed successfully. Scale factor: {:.6f} m / COLMAP unit".format(scale_factor))
    return scale_factor

if __name__ == "__main__":
    main(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Ferse_ArUco2", 0.0123)
    # folder_path = r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\AprilTags_Scaling\Ferse\images"
    # for img in os.listdir(folder_path):
    #     img_path = os.path.join(folder_path, img)
    #     img = cv2.imread(img_path)
    #     img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    #     detector = create_detector()
    #     detections = detect_apriltags(img, detector)
    #     annotated = annotate_detections(img, detections)
    #     cv2.imshow("AprilTags", annotated)
    #     cv2.waitKey(0)
    