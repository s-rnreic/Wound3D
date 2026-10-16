"""
Example usage of the sparse point filtering and bundle adjustment workflow.

This file demonstrates various configurations and thresholds for the sparse
post-processing pipeline.
"""

import os
import sys

# Add workspace root to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_workspace_root = os.path.abspath(os.path.join(_current_dir, '../../..'))
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

from wound3d.legacy.modules.sparse_postprocessing import (
    apply_sparse_filtering,
    print_point_statistics
)
from wound3d.legacy.utils.read_write_model import read_model
from pathlib import Path


def example_1_conservative_filtering():
    """
    Conservative filtering: keep most points, only remove clear outliers.
    
    Good for: Highly textured scenes, high quality input images
    """
    print("=" * 70)
    print("EXAMPLE 1: Conservative Filtering")
    print("=" * 70)
    
    workspace_path = r"C:\path\to\workspace"  # Update this
    sparse_path = os.path.join(workspace_path, 'sparse')
    ini_path = os.path.join(workspace_path, 'project.ini')
    database_path = os.path.join(workspace_path, 'database.db')
    log_file = os.path.join(workspace_path, 'pipeline_times.log')
    
    cameras, images, points3D, elapsed_time = apply_sparse_filtering(
        workspace_path,
        sparse_path,
        ini_path,
        database_path,
        max_reprojection_error=2.0,      # More lenient error threshold
        min_track_length=2,              # Fewer observations required
        run_ba=True,
        ba_camera_model="PINHOLE"
    )
    
    print(f"\nTotal elapsed time: {elapsed_time:.2f} seconds")


def example_2_balanced_filtering():
    """
    Balanced filtering: good overall quality with reasonable point count.
    
    Good for: General purpose reconstruction, average quality input
    """
    print("=" * 70)
    print("EXAMPLE 2: Balanced Filtering (Default)")
    print("=" * 70)
    
    workspace_path = r"C:\path\to\workspace"  # Update this
    sparse_path = os.path.join(workspace_path, 'sparse')
    ini_path = os.path.join(workspace_path, 'project.ini')
    database_path = os.path.join(workspace_path, 'database.db')
    log_file = os.path.join(workspace_path, 'pipeline_times.log')
    
    cameras, images, points3D, elapsed_time = apply_sparse_filtering(
        workspace_path,
        sparse_path,
        ini_path,
        database_path,
        max_reprojection_error=1.0,      # Moderate error threshold
        min_track_length=3,              # Standard minimum observations
        run_ba=True,
        ba_camera_model="PINHOLE"
    )
    
    print(f"\nTotal elapsed time: {elapsed_time:.2f} seconds")


def example_3_aggressive_filtering():
    """
    Aggressive filtering: high quality points only, fewer outliers.
    
    Good for: Precision-critical applications, clean input images
    """
    print("=" * 70)
    print("EXAMPLE 3: Aggressive Filtering")
    print("=" * 70)
    
    workspace_path = r"C:\path\to\workspace"  # Update this
    sparse_path = os.path.join(workspace_path, 'sparse')
    ini_path = os.path.join(workspace_path, 'project.ini')
    database_path = os.path.join(workspace_path, 'database.db')
    log_file = os.path.join(workspace_path, 'pipeline_times.log')
    
    cameras, images, points3D, elapsed_time = apply_sparse_filtering(
        workspace_path,
        sparse_path,
        ini_path,
        database_path,
        max_reprojection_error=0.5,      # Strict error threshold
        min_track_length=5,              # More observations required
        run_ba=True,
        ba_camera_model="PINHOLE"
    )
    
    print(f"\nTotal elapsed time: {elapsed_time:.2f} seconds")


def example_4_filtering_without_ba():
    """
    Filtering without bundle adjustment.
    
    Use when: You want to see filtering results without refining cameras,
    or when cameras are already well calibrated.
    """
    print("=" * 70)
    print("EXAMPLE 4: Filtering Without Bundle Adjustment")
    print("=" * 70)
    
    workspace_path = r"C:\path\to\workspace"  # Update this
    sparse_path = os.path.join(workspace_path, 'sparse')
    ini_path = os.path.join(workspace_path, 'project.ini')
    database_path = os.path.join(workspace_path, 'database.db')
    log_file = os.path.join(workspace_path, 'pipeline_times.log')
    
    cameras, images, points3D, elapsed_time = apply_sparse_filtering(
        workspace_path,
        sparse_path,
        ini_path,
        database_path,
        max_reprojection_error=1.0,
        min_track_length=3,
        run_ba=False,                   # Skip bundle adjustment
        ba_camera_model="PINHOLE"
    )
    
    print(f"\nTotal elapsed time: {elapsed_time:.2f} seconds")


def example_5_custom_analysis():
    """
    Analyze point statistics before deciding on filtering thresholds.
    
    Use this to understand the distribution of errors and track lengths
    in your specific reconstruction.
    """
    print("=" * 70)
    print("EXAMPLE 5: Custom Analysis")
    print("=" * 70)
    
    workspace_path = r"C:\path\to\workspace"  # Update this
    sparse_path = os.path.join(workspace_path, 'sparse')
    model_path = Path(sparse_path) / "0"
    
    # Read the current model
    cameras, images, points3D = read_model(model_path, ext=".bin")
    
    # Print statistics
    print_point_statistics(points3D, "Pre-Filtering Statistics")
    
    # Analyze distributions
    import numpy as np
    errors = np.array([point.error for point in points3D.values()])
    track_lengths = np.array([len(point.image_ids) for point in points3D.values()])
    
    print("\nError Distribution:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(errors, p)
        print(f"  {p}th percentile: {value:.4f}")
    
    print("\nTrack Length Distribution:")
    for p in percentiles:
        value = np.percentile(track_lengths, p)
        print(f"  {p}th percentile: {int(value)}")
    
    # Suggest thresholds based on data
    print("\nSuggested Thresholds Based on Distribution:")
    print(f"  - Conservative: max_error={np.percentile(errors, 90):.2f}, min_track={int(np.percentile(track_lengths, 10))}")
    print(f"  - Balanced:     max_error={np.percentile(errors, 75):.2f}, min_track={int(np.percentile(track_lengths, 25))}")
    print(f"  - Aggressive:   max_error={np.percentile(errors, 50):.2f}, min_track={int(np.percentile(track_lengths, 50))}")


def example_6_iterate_thresholds():
    """
    Iterate through different threshold combinations to find optimal settings.
    
    Use when: You want to experiment with multiple configurations and compare results.
    """
    print("=" * 70)
    print("EXAMPLE 6: Iterate Through Threshold Combinations")
    print("=" * 70)
    
    workspace_path = r"C:\path\to\workspace"  # Update this
    sparse_path = os.path.join(workspace_path, 'sparse')
    ini_path = os.path.join(workspace_path, 'project.ini')
    database_path = os.path.join(workspace_path, 'database.db')
    log_file = os.path.join(workspace_path, 'pipeline_times.log')
    
    # Define threshold combinations to try
    error_thresholds = [0.5, 1.0, 2.0]
    track_length_thresholds = [2, 3, 5]
    
    results = []
    
    for max_error in error_thresholds:
        for min_track in track_length_thresholds:
            print(f"\nTesting: max_error={max_error}, min_track={min_track}")
            
            try:
                cameras, images, points3D, elapsed_time = apply_sparse_filtering(
                    workspace_path,
                    sparse_path,
                    ini_path,
                    database_path,
                    max_reprojection_error=max_error,
                    min_track_length=min_track,
                    run_ba=False,  # Don't run BA for quick iteration
                    ba_camera_model="PINHOLE"
                )
                
                results.append({
                    'max_error': max_error,
                    'min_track': min_track,
                    'point_count': len(points3D),
                    'elapsed_time': elapsed_time
                })
                
                print(f"  Points remaining: {len(points3D)}")
                
            except Exception as e:
                print(f"  Failed: {e}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Max Error':<12} {'Min Track':<12} {'Points':<12} {'Time (s)':<12}")
    print("-" * 70)
    for result in results:
        print(f"{result['max_error']:<12.2f} {result['min_track']:<12} "
              f"{result['point_count']:<12} {result['elapsed_time']:<12.2f}")


if __name__ == "__main__":
    # Choose which example to run:
    
    # example_1_conservative_filtering()
    # example_2_balanced_filtering()
    # example_3_aggressive_filtering()
    # example_4_filtering_without_ba()
    example_5_custom_analysis()
    # example_6_iterate_thresholds()
