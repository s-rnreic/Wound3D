"""
PointCleanNet Wrapper Module
Provides easy-to-use interface for denoising and outlier removal on point clouds.
Handles PLY to XYZ conversion automatically.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import torch
from pathlib import Path

# Add PointCleanNet to path
POINTCLEANNET_ROOT = Path(__file__).parent.parent.parent.parent / "external" / "PointCleanNet"


def read_ply(ply_path):
    """
    Read a PLY file and extract XYZ coordinates.
    
    Args:
        ply_path: Path to the PLY file
        
    Returns:
        numpy array of shape (N, 3) with XYZ coordinates
    """
    try:
        from plyfile import PlyData
        ply_data = PlyData.read(ply_path)
        vertex = ply_data['vertex']
        points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
        return points
    except ImportError:
        # Fallback to simple parsing if plyfile is not installed
        print("Warning: plyfile not installed, using simple parser")
        return read_ply_simple(ply_path)


def read_ply_simple(ply_path):
    """
    Simple PLY reader for ASCII format (fallback).
    
    Args:
        ply_path: Path to the PLY file
        
    Returns:
        numpy array of shape (N, 3) with XYZ coordinates
    """
    with open(ply_path, 'r') as f:
        lines = f.readlines()
    
    # Find header end
    vertex_count = 0
    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith('element vertex'):
            vertex_count = int(line.split()[-1])
        if line.startswith('end_header'):
            header_end = i + 1
            break
    
    # Read vertex data
    points = []
    for i in range(header_end, header_end + vertex_count):
        parts = lines[i].strip().split()
        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    
    return np.array(points)


def write_ply(points, output_path, colors=None):
    """
    Write points to a PLY file.
    
    Args:
        points: numpy array of shape (N, 3) with XYZ coordinates
        output_path: Path to save the PLY file
        colors: Optional numpy array of shape (N, 3) with RGB colors (0-255)
    """
    try:
        from plyfile import PlyData, PlyElement
        
        if colors is not None:
            vertices = np.zeros(points.shape[0], 
                              dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                     ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
            vertices['x'] = points[:, 0]
            vertices['y'] = points[:, 1]
            vertices['z'] = points[:, 2]
            vertices['red'] = colors[:, 0]
            vertices['green'] = colors[:, 1]
            vertices['blue'] = colors[:, 2]
        else:
            vertices = np.zeros(points.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            vertices['x'] = points[:, 0]
            vertices['y'] = points[:, 1]
            vertices['z'] = points[:, 2]
        
        el = PlyElement.describe(vertices, 'vertex')
        PlyData([el]).write(output_path)
    except ImportError:
        # Fallback to simple writing
        write_ply_simple(points, output_path, colors)


def write_ply_simple(points, output_path, colors=None):
    """
    Simple PLY writer (fallback).
    
    Args:
        points: numpy array of shape (N, 3) with XYZ coordinates
        output_path: Path to save the PLY file
        colors: Optional numpy array of shape (N, 3) with RGB colors (0-255)
    """
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(len(points)):
            if colors is not None:
                f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]} "
                       f"{int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}\n")
            else:
                f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]}\n")


def denoise_pointcloud(input_ply, output_ply, num_iterations=3, model_dir=None, n_neighbours=100):
    """
    Denoise a point cloud using PointCleanNet.
    
    Args:
        input_ply: Path to input PLY file
        output_ply: Path to output PLY file
        num_iterations: Number of denoising iterations (default: 3)
        model_dir: Path to model directory (default: external/PointCleanNet/models/denoisingModel)
        n_neighbours: Number of neighbors for smoothing (default: 100)
        
    Returns:
        Path to output file
    """
    if model_dir is None:
        model_dir = str(POINTCLEANNET_ROOT / "models" / "denoisingModel")
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}\n"
                              "Please download the model first using:\n"
                              "cd external/PointCleanNet/models && python download_models.py --task denoising")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Read PLY and convert to XYZ
        print(f"Reading point cloud from {input_ply}...")
        points = read_ply(input_ply)
        
        # Save as XYZ (initial iteration)
        temp_xyz = temp_dir / "cloud_0.xyz"
        np.savetxt(temp_xyz, points)
        print(f"Saved temporary XYZ with {len(points)} points")
        
        # Create dataset list file
        dataset_file = temp_dir / "dataset.txt"
        with open(dataset_file, 'w') as f:
            f.write("cloud_0.xyz\n")
        
        # Import PointCleanNet denoising module
        noise_removal_path = str(POINTCLEANNET_ROOT / "noise_removal")
        if noise_removal_path in sys.path:
            sys.path.remove(noise_removal_path)
        sys.path.insert(0, noise_removal_path)
        import eval_pcpnet as denoise_module
        eval_pcpnet_denoise = denoise_module.eval_pcpnet
        sys.path.remove(noise_removal_path)
        
        # Run denoising iterations
        for i in range(num_iterations):
            print(f"\n--- Iteration {i+1}/{num_iterations} ---")
            
            # Create arguments object
            class Args:
                pass
            
            opt = Args()
            opt.indir = str(temp_dir)
            opt.outdir = str(temp_dir)
            opt.dataset = 'dataset.txt'
            opt.modeldir = model_dir
            opt.model = 'PointCleanNet'
            opt.modelpostfix = '_model.pth'
            opt.parmpostfix = '_params.pth'
            opt.n_neighbours = n_neighbours
            opt.sparse_patches = False
            opt.sampling = 'full'
            opt.patches_per_shape = 20000
            opt.seed = 40938661
            opt.batchSize = 512
            opt.workers = 0
            opt.cache_capacity = 10
            opt.nrun = i + 1
            opt.shapename = 'cloud_{i}'  # Will be formatted by PointCleanNet as cloud_0, cloud_1, etc.
            
            try:
                eval_pcpnet_denoise(opt)
            except Exception as e:
                print(f"Error during denoising: {e}")
                raise
        
        # Read final denoised result
        final_xyz = temp_dir / f"cloud_{num_iterations}.xyz"
        if not final_xyz.exists():
            raise FileNotFoundError(f"Output file not found: {final_xyz}")
        
        denoised_points = np.loadtxt(final_xyz)
        print(f"\nDenoising complete. Output has {len(denoised_points)} points")
        
        # Save as PLY
        print(f"Saving result to {output_ply}...")
        write_ply(denoised_points, output_ply)
    
    return output_ply


def remove_outliers(input_ply, output_ply, threshold=0.5, model_dir=None):
    """
    Remove outliers from a point cloud using PointCleanNet.
    
    Args:
        input_ply: Path to input PLY file
        output_ply: Path to output PLY file (inliers only)
        threshold: Threshold for outlier classification (default: 0.5)
        model_dir: Path to model directory (default: external/PointCleanNet/models/outliersRemovalModel)
        
    Returns:
        Tuple of (output_path, num_inliers, num_outliers)
    """
    if model_dir is None:
        model_dir = str(POINTCLEANNET_ROOT / "models" / "outliersRemovalModel")
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}\n"
                              "Please download the model first using:\n"
                              "cd external/PointCleanNet/models && python download_models.py --task outliers_removal")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Read PLY and convert to XYZ
        print(f"Reading point cloud from {input_ply}...")
        points = read_ply(input_ply)
        
        # Save as XYZ
        temp_xyz = temp_dir / "cloud.xyz"
        np.savetxt(temp_xyz, points)
        print(f"Saved temporary XYZ with {len(points)} points")
        
        # Create shape list file
        shapes_file = temp_dir / "shapes.txt"
        with open(shapes_file, 'w') as f:
            f.write("cloud\n")
        
        # Import PointCleanNet outlier removal module
        outliers_removal_path = str(POINTCLEANNET_ROOT / "outliers_removal")
        if outliers_removal_path in sys.path:
            sys.path.remove(outliers_removal_path)
        sys.path.insert(0, outliers_removal_path)
        import eval_pcpnet as outliers_module
        eval_pcpnet_outliers = outliers_module.eval_pcpnet
        sys.path.remove(outliers_removal_path)
        
        # Create arguments object
        class Args:
            pass
        
        opt = Args()
        opt.indir = str(temp_dir)
        opt.outdir = str(temp_dir)
        opt.dataset = 'shapes.txt'
        opt.modeldir = model_dir
        opt.model = 'PointCleanNetOutliers'
        opt.modelpostfix = '_model.pth'
        opt.parmpostfix = '_params.pth'
        opt.sparse_patches = False
        opt.sampling = 'full'
        opt.patches_per_shape = 20000
        opt.seed = 40938661
        opt.batchSize = 512
        opt.workers = 0
        opt.cache_capacity = 10
        opt.shapename = 'cloud'  # Will be formatted by PointCleanNet as cloud.xyz
        
        print("\nRunning outlier detection...")
        try:
            eval_pcpnet_outliers(opt)
        except Exception as e:
            print(f"Error during outlier removal: {e}")
            raise
        
        # Read outlier scores
        outlier_info_file = temp_dir / opt.model / "outliers_value_cloud.info"
        if not outlier_info_file.exists():
            raise FileNotFoundError(f"Outlier info file not found: {outlier_info_file}")
        
        outlier_data = np.loadtxt(outlier_info_file)
        # First 3 columns are XYZ, 4th column is outlier score
        outlier_scores = outlier_data[:, 3]
        
        # Filter inliers
        inlier_mask = outlier_scores <= threshold
        inliers = points[inlier_mask]
        
        num_inliers = np.sum(inlier_mask)
        num_outliers = len(points) - num_inliers
        
        print(f"\nOutlier removal complete:")
        print(f"  Inliers: {num_inliers}")
        print(f"  Outliers: {num_outliers}")
        print(f"  Outlier ratio: {num_outliers/len(points)*100:.2f}%")
        
        # Save as PLY
        print(f"Saving result to {output_ply}...")
        write_ply(inliers, output_ply)
    
    return output_ply, num_inliers, num_outliers


def clean_pointcloud(input_ply, output_ply, denoise=True, remove_outliers_flag=True,
                     num_iterations=3, outlier_threshold=0.5):
    """
    Complete point cloud cleaning: outlier removal followed by denoising.
    
    Args:
        input_ply: Path to input PLY file
        output_ply: Path to output PLY file
        denoise: Whether to apply denoising (default: True)
        remove_outliers_flag: Whether to remove outliers (default: True)
        num_iterations: Number of denoising iterations (default: 3)
        outlier_threshold: Threshold for outlier classification (default: 0.5)
        
    Returns:
        Path to output file
    """
    temp_file = None
    
    try:
        if remove_outliers_flag and denoise:
            # Both operations: use temporary file
            with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp:
                temp_file = tmp.name
            
            print("=== Step 1: Outlier Removal ===")
            remove_outliers(input_ply, temp_file, threshold=outlier_threshold)
            
            print("\n=== Step 2: Denoising ===")
            denoise_pointcloud(temp_file, output_ply, num_iterations=num_iterations)
            
        elif remove_outliers_flag:
            # Only outlier removal
            remove_outliers(input_ply, output_ply, threshold=outlier_threshold)
            
        elif denoise:
            # Only denoising
            denoise_pointcloud(input_ply, output_ply, num_iterations=num_iterations)
            
        else:
            # Nothing to do, just copy
            shutil.copy(input_ply, output_ply)
            
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
    
    return output_ply


if __name__ == "__main__":
    """
    Example usage of PointCleanNet wrapper
    """
    # Example paths - adjust to your actual data
    workspace_root = Path(__file__).parent.parent.parent.parent
    
    # Example 1: Denoise only
    print("=" * 60)
    print("Example 1: Denoising a point cloud")
    print("=" * 60)
    
    input_file = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_WoundDetection\Ferse1\sor_cleaned_pointcloud.ply")
    output_file = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_WoundDetection\Ferse1\denoised_pointcloud.ply")
    
    # Check if input exists
    if input_file.exists():
        denoise_pointcloud(
            input_ply=str(input_file),
            output_ply=str(output_file),
            num_iterations=1,
            n_neighbours=100
        )
        print(f"\n✓ Denoised point cloud saved to: {output_file}")
    else:
        print(f"⚠ Input file not found: {input_file}")
        print("\nTo use this example, provide a valid PLY file path:")
        print("  input_file = 'path/to/your/pointcloud.ply'")
    
    #Example 2: Remove outliers only
    print("\n" + "=" * 60)
    print("Example 2: Removing outliers")
    print("=" * 60)
    
    output_file_outliers = output_file = Path(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Experiments_WoundDetection\Ferse1\outlier_pointcloud.ply")
    
    if input_file.exists():
        remove_outliers(
            input_ply=str(input_file),
            output_ply=str(output_file_outliers),
            threshold=0.5
        )
        print(f"\n✓ Cleaned point cloud saved to: {output_file_outliers}")
    else:
        print(f"⚠ Input file not found: {input_file}")
    
    # Example 3: Complete cleaning (outliers + denoising)
    print("\n" + "=" * 60)
    print("Example 3: Complete cleaning (outliers + denoising)")
    print("=" * 60)
    
    output_file_clean = workspace_root / "Experiments" / "test_Knete_20ml_4K" / "dense" / "dense_cleaned.ply"
    
    if input_file.exists():
        clean_pointcloud(
            input_ply=str(input_file),
            output_ply=str(output_file_clean),
            denoise=True,
            remove_outliers_flag=True,
            num_iterations=3,
            outlier_threshold=0.5
        )
        print(f"\n✓ Fully cleaned point cloud saved to: {output_file_clean}")
    else:
        print(f"⚠ Input file not found: {input_file}")
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
