![Demo](git_images\wound3d.gif)



# wound3d

`wound3d` reconstructs a 3D wound model from a video. The pipeline extracts frames, selects sharp images, creates optional
masks, performs sparse and dense reconstruction, generates a mesh, and runs mesh-based wound segmentation.

![Pipeline overview](git_images/Pipeline_Roadmap-Page-2.svg)

## Features

- Video frame extraction and sharpness-based frame selection
- Optional conservative, SAM2, DeepSkin, and WoundAmbit masking backends
- COLMAP feature extraction, matching, dense reconstruction, and Poisson meshing
- GLOMAP sparse reconstruction
- Optional AprilTag metric scaling and Rerun visualization/filtering
- Dense point-cloud post-processing and optional PointCleanNet integration
- Mesh-based wound-edge segmentation

## Pipeline workflow

Given an input video, the pipeline performs these stages:

1. Extract video frames.
2. Measure frame sharpness and select sharp images.
3. Create optional image masks.
4. Run COLMAP feature extraction and matching.
5. Run GLOMAP sparse reconstruction.
6. Optionally filter/visualize the sparse model and apply AprilTag scaling.
7. Run COLMAP dense reconstruction.
8. Post-process the dense point cloud.
9. Generate a Poisson mesh.
10. Run three-dimensional wound-edge segmentation.

## Code layout

The application package is located in:

```text
src/wound3d/
```

The main user-facing modules are:

```text
src/wound3d/cli.py         Command-line execution
src/wound3d/config.py      YAML configuration loading
src/wound3d/pipeline.py    Pipeline orchestration
src/wound3d/workspace.py   Generated workspace paths
```

Existing reconstruction, masking, post-processing, and segmentation implementation modules are currently located in:

```text
src/wound3d/legacy/
```

## Installation

### Requirements

Install the following external programs and make sure they are available on `PATH`:

```text
COLMAP
GLOMAP
FFmpeg
Git
```

Python 3.10 or newer is required. A CUDA-capable GPU and a compatible PyTorch/CUDA installation may be required by the
selected masking backend.

### Install the package

Clone the repository together with its external submodules:

```powershell
git clone --recurse-submodules <repository-url>
cd COLMAP_Pipline
```

For an existing clone, initialize the submodules with:

```powershell
git submodule update --init --recursive
```

Create and activate a Python environment, then install the pipeline:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional masking and post-processing backends must be installed/configured separately when selected in the YAML
configuration.

## Configure a run

Create a local configuration file from the example:

```powershell
Copy-Item .\configs\local.example.yaml .\configs\local.yaml
```

Set the input video and output workspace in `configs/local.yaml`:

```yaml
input_video: C:/path/to/input_video.mp4
workspace: C:/path/to/wound3d_workspace

processing:
  segments: 10
  resize_factor: 1.0
  sharpness_percentile: 25.0

masking:
  enabled: true
  backend: conservative

reconstruction:
  tag_size_m: 0.0123
  use_pointcleannet: false
```

Available masking backends are:

```text
conservative
sam2
deepskin
woundambit
```

For Windows paths in YAML, use forward slashes such as `C:/path/to/file.mp4`, or use single-quoted backslash paths.

## Execute the pipeline

Run the pipeline with:

```powershell
python -m wound3d --config .\configs\local.yaml
```

After installing the package, the equivalent console command is:

```powershell
wound3d-pipeline --config .\configs\local.yaml
```

## Generated workspace structure

The configured `workspace` directory contains all files generated for one run:

```text
workspace/
├── raw_images/                     Extracted video frames
├── sharp_images/                   Frames selected for reconstruction
├── masks/                          Wound + tissue masks used for reconstruction
├── masks_wound_only/               Wound-only masks
├── masks_extruded_wound_only/      Extended wound masks when mask extrusion is enabled
├── sparse/
│   └── 0/                          Sparse COLMAP/GLOMAP model
├── dense/                          Dense reconstruction, point cloud, and mesh outputs
├── database.db                     COLMAP database
├── project.ini                     COLMAP project configuration
├── TempProject.ini                 Temporary COLMAP command configuration
├── total_project.ini               Combined COLMAP/GLOMAP configuration record
├── glomap_config.ini               GLOMAP configuration
└── pipeline_times.log              Per-stage execution timing
```

Important dense outputs typically include:

```text
dense/fused.ply
dense/processed_pointcloud.ply
dense/poisson.ply
```

> **Research use only.** This software is not a medical device and is not intended for clinical diagnosis or treatment.

## License and citation

A project license and citation metadata must be selected before a public release. See
[docs/release-metadata.md](docs/release-metadata.md). Third-party components retain their own licenses and terms; see
[docs/third-party-software.md](docs/third-party-software.md).