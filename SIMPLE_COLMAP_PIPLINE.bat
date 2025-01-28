@echo off
SETLOCAL

REM Define input parameters
SET OUTPUT_DIR=%1
SET IMAGE_PATH=%2
SET IMAGE_MASKS_PATH=%3

REM Check if the correct number of arguments are provided

IF "%OUTPUT_DIR%"=="" (
    echo "Output directory is required."
    EXIT /B 1
)


@REM REM Create the output directory if it doesn't exist
@REM IF NOT EXIST "%OUTPUT_DIR%" (
@REM     mkdir "%OUTPUT_DIR%"
@REM )

REM Step 1: Feature Extraction
IF %IMAGE_PATH%=="" (
    call colmap feature_extractor --database_path "%OUTPUT_DIR%/database.db" --image_path "%OUTPUT_DIR%/images" --ImageReader.mask_path "%OUTPUT_DIR%/mask_path"
)
ELSE (
    call colmap feature_extractor --database_path "%OUTPUT_DIR%/database.db" --image_path "%IMAGE_PATH%" --ImageReader.mask_path "%IMAGE_MASKS_PATH%"
)
IF ERRORLEVEL 1 (
    echo "Feature extraction failed."
    EXIT /B 1
)


REM Step 2: Exhaustive Matching
call colmap exhaustive_matcher --database_path "%OUTPUT_DIR%/database.db"
IF ERRORLEVEL 1 (
    echo "Exhaustive matching failed."
    EXIT /B 1
)

mkdir "%OUTPUT_DIR%/sparse"

REM Step 3: Sparse Reconstruction
IF %IMAGE_PATH%=="" (
    call colmap mapper --database_path "%OUTPUT_DIR%/database.db" --image_path "%OUTPUT_DIR%/images" --output_path "%OUTPUT_DIR%/sparse"
)
ELSE (
    call colmap mapper --database_path "%OUTPUT_DIR%/database.db" --image_path "%IMAGE_PATH%" --output_path "%OUTPUT_DIR%/sparse"
)
if ERRORLEVEL 1 (
    echo "Sparse reconstruction failed."
    EXIT /B 1
)

mkdir "%OUTPUT_DIR%/dense"

@REM REM Step 4: Image Undistortion (for dense reconstruction)
IF %IMAGE_PATH%=="" (
    call colmap image_undistorter --image_path "%OUTPUT_DIR%/images" --input_path "%OUTPUT_DIR%/sparse/0" --output_path "%OUTPUT_DIR%/dense" --output_type COLMAP
)
ELSE (
    call colmap image_undistorter --image_path "%IMAGE_PATH%" --input_path "%OUTPUT_DIR%/sparse/0" --output_path "%OUTPUT_DIR%/dense" --output_type COLMAP
)
if ERRORLEVEL 1 (
    echo "Image undistortion failed."
    EXIT /B 1
)

@REM REM Step 5: Dense Reconstruction
call colmap patch_match_stereo --workspace_path "%OUTPUT_DIR%/dense" --workspace_format COLMAP --PatchMatchStereo.geom_consistency true
if ERRORLEVEL 1 (
    echo "Dense reconstruction failed."
    EXIT /B 1
)


@REM REM Step 6: Dense Point Cloud Fusion
call colmap stereo_fusion --workspace_path "%OUTPUT_DIR%/dense" --workspace_format COLMAP --input_type geometric --output_path "%OUTPUT_DIR%/dense/fused.ply"
if ERRORLEVEL 1 (
    echo "Dense point cloud fusion failed."
    EXIT /B 1
)

@REM REM Step 7: (Optional) Create a Mesh from the Dense Point Cloud
call colmap poisson_mesher --input_path "%OUTPUT_DIR%/dense/fused.ply" --output_path "%OUTPUT_DIR%/dense/meshed.ply"
if ERRORLEVEL 1 (
    echo "Mesh creation failed."
    EXIT /B 1
)

ENDLOCAL