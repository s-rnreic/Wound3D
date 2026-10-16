# Sparse Point Cloud Post-Processing Module

This module provides a complete workflow for filtering sparse 3D point clouds after COLMAP/GLOMAP reconstruction and optionally running bundle adjustment to refine camera parameters.

## Features

- **Point Filtering**: Remove points based on:
  - Reprojection error (pixels)
  - Track length (number of image observations)
  
- **Automatic Image Updates**: Updates image references to maintain 2D-3D correspondences after point removal

- **Bundle Adjustment**: Optionally refine camera intrinsics after filtering

- **Comprehensive Statistics**: Detailed statistics before, after filtering, and after bundle adjustment

- **Visualization**: Integration with Rerun for before/after comparison visualization

## Workflow

The complete workflow integrates into the main pipeline:

```
GLOMAP Pipeline
    ↓
ANALYSE_COLMAP_OUTPUT
    ↓
Sparse_Filtering_and_BA
    ├─ Filter points based on criteria
    ├─ Update image references
    ├─ Run bundle adjustment (optional)
    └─ Update camera intrinsics
    ↓
COLMAP_RERUN (Visualization)
    ↓
MVS (Dense Reconstruction)
    ↓
DensePostProcessing
    ↓
Meshing
```

## Configuration

Edit the `max_reprojection_error` and `min_track_length` parameters in `main.py`:

```python
Sparse_Filtering_and_BA(
    workspace_path,
    sparse_path,
    ini_path,
    database_path,
    log_file,
    max_reprojection_error=1.0,      # Points with higher error are removed
    min_track_length=3,               # Points seen in fewer images are removed
    run_bundle_adjustment=True        # Run BA after filtering
)
```

## API Reference

### `filter_points3D()`
Filter 3D points based on reprojection error and track length.

**Parameters:**
- `points3D`: Dictionary of Point3D objects
- `images`: Dictionary of Image objects
- `max_reprojection_error`: Maximum allowed error in pixels (default: 1.0)
- `min_track_length`: Minimum observations required (default: 3)

**Returns:**
- Tuple of (filtered_points3D, updated_images)

### `apply_sparse_filtering()`
Complete workflow: filter points and optionally run bundle adjustment.

**Parameters:**
- `workspace_path`: Path to workspace
- `sparse_path`: Path to sparse reconstruction
- `ini_path`: Path to project.ini
- `database_path`: Path to database.db
- `max_reprojection_error`: Max error threshold
- `min_track_length`: Min track length threshold
- `run_ba`: Whether to run bundle adjustment (default: True)
- `ba_camera_model`: Camera model for BA (default: "PINHOLE")

**Returns:**
- Tuple of (cameras, images, points3D, elapsed_time)

### `print_point_statistics()`
Print formatted statistics about the point cloud.

**Parameters:**
- `points3D`: Dictionary of Point3D objects
- `label`: Label for output (default: "Point Cloud")

### `run_bundle_adjustment()`
Run COLMAP bundle adjustment on the sparse model.

**Parameters:**
- `workspace_path`: Path to workspace
- `sparse_path`: Path to sparse reconstruction
- `ini_path`: Path to project.ini
- `database_path`: Path to database.db
- `camera_model`: Camera model to use (default: "PINHOLE")

**Returns:**
- Elapsed time for the operation

## Output

The process creates:

1. **Filtered sparse model**: Updated in `sparse/0/` with filtered points
2. **Point statistics**: Logged to `pipeline_times.log`
3. **Camera parameters**: Updated intrinsics stored in camera.bin
4. **Rerun visualization**: 
   - `/all_points`: Original sparse point cloud
   - `/color_map_mean_repo_error`: Points colored by reprojection error
   - `/track_length`: Points colored by track length
   - `/sparse_filtered`: Post-filtering point cloud results

## Statistics Provided

For each stage (before filtering, after filtering, after BA):

- **Point count**
- **Reprojection error**: mean, median, max, min
- **Track length**: mean, median, max, min

Example output:
```
Point Cloud Statistics:
  Count: 15234
  Reprojection Error - Mean: 0.3456, Median: 0.2123, Max: 2.4531, Min: 0.0012
  Track Length - Mean: 4.52, Median: 4.0, Max: 28, Min: 3
```

## Integration with Main Pipeline

The sparse filtering is automatically called after GLOMAP reconstruction:

```python
def run_colmap_process(...):
    # ... GLOMAP_Pipeline execution ...
    
    # Apply sparse filtering and bundle adjustment
    Sparse_Filtering_and_BA(
        workspace_path,
        sparse_path,
        ini_path,
        database_path,
        log_file,
        max_reprojection_error=1.0,
        min_track_length=3,
        run_bundle_adjustment=True
    )
    
    # Visualize with rerun
    cameras, images, filtered_points3D = COLMAP_RERUN(workspace_path, log_file)
```

## Recommended Thresholds

### Conservative (Keep more points)
- `max_reprojection_error`: 2.0
- `min_track_length`: 2

### Balanced (Default)
- `max_reprojection_error`: 1.0
- `min_track_length`: 3

### Aggressive (Keep fewer, higher quality points)
- `max_reprojection_error`: 0.5
- `min_track_length`: 5

## Bundle Adjustment Parameters

The bundle adjustment currently uses:
- Refine principal point: enabled
- Refine focal length: enabled
- Refine extra parameters: enabled
- Multi-threading: enabled for >100 residuals

To modify these parameters, edit `run_bundle_adjustment()` in `point_filtering.py`.

## Troubleshooting

### Bundle Adjustment Fails
- Ensure COLMAP is installed and available in PATH
- Check that the sparse model is valid
- Verify database.db and project.ini paths are correct

### Points Not Filtering
- Check that thresholds are appropriate for your data
- Review statistics to understand error/track distributions
- Try less aggressive thresholds first

### Camera Intrinsics Not Updated
- Bundle adjustment must be run (`run_bundle_adjustment=True`)
- Verify bundle adjustment completes successfully
- Check rerun visualization for camera parameter changes
