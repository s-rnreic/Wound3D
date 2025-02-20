## Conservative Glomap Workflow for Wound Segmentation and 3D Reconstruction

### Pipeline Components

1. **Video Analysis/Processing**
    - `video_edit.py`: Decomposes video into images and selects images based on sharpness and equidistance.
    - `mask_edit.py`: Automatically creates masks using thresholding.
        - Optional: Use Retinex or other color homogenization techniques.

2. **COLMAP/SFM Pipeline**
    - Parameters can be specified through a project file.
        - Automatic creation of a project file where parameters and settings are specified.
        - The change of parameters is presented to the user in a text editor
            - List of parameters:
                - segments, resize_factor (main.py)
                - min_sharpness_percentile (prepare_vid_for_colmap_v5.py)
                - List of Parameters given in project.ini files
        - Alternatively, execute the COLMAP pipeline with predefined standard values 
    - Save results in the standard COLMAP folder structure as sparse.
        - Output workspace structure:

        --  Workspace
            |   database.db
            |   glomap_config.ini
            |   pipeline_times.log
            |   project.ini
            |   TempProject.ini
            |   TotalProject.ini
            +---raw_images    
            +---sharp_images
            \---sparse
                \---0
                        cameras.bin
                        images.bin
                        points3D.bin
                        

    - For each execution the main.py: saving relative Parameters in sqlite3 Database "Trial.db"

3. **Post-Processing**
    - Possibly filter backgrounds, outstanding edges, etc.

### Features of the Glomap Pipeline in `main.py`

- **User Input**: Allows user to input video path and workspace path.
- **Workspace Creation**: Creates a structured workspace directory with subfolders for raw images, sharp images, masks, sparse, and dense data.
- **Video Pipeline**: 
    - Extracts images from video.
    - Selects sharp and equidistant images.
    - Optionally processes images to create masks.
- **COLMAP Pipeline**:
    - Executes COLMAP commands for feature extraction, feature matching, and mapping using temporary project files.
    - Appends the data from temporary project files to a total project file (`total_project.ini`).
- **Analysis**: Analyzes the output of the COLMAP pipeline.
- **Rerun Streaming**: Reruns the COLMAP pipeline for visualization.
- **Timing Analysis**: Logs the elapsed time for each step of the pipeline and calculates the total elapsed time.
- **Database Integration**: Loads existing experiments into a database with details such as experiment description, project file, cameras, images, 3D points, observations, mean reprojection error, mean track length, mean observations per image, resize factor, fraction of images, and elapsed time.

### Visualization

The visualization of the trial is achieved via the Rerun viewer.

<video width="640" height="360" controls>
  <source src="video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
