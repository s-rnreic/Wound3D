# wound3d

Video-to-3D wound reconstruction research software built around COLMAP and GLOMAP. The pipeline extracts sharp video
frames, optionally creates masks, reconstructs sparse and dense geometry, performs point-cloud and mesh processing, and
runs mesh-based wound segmentation.

> **Research-use only.** This project is not a medical device and is not validated for diagnosis, treatment, or clinical
> decision-making. Read the [medical and research-use disclaimer](docs/medical-use-disclaimer.md) before using it with
> wound imagery.

![Pipeline overview](git_images/Pipeline_Roadmap-Page-2.svg)

## Status

`wound3d` is under active development. All project Python code is contained in the standard `src/wound3d` package.
The `src/wound3d/legacy` namespace contains implementation modules retained during the staged refactor; new code should
target the top-level `wound3d` modules and avoid adding new dependencies on legacy internals.

## Features

- Video frame extraction and sharpness-based frame selection
- Optional conservative, SAM2, DeepSkin, and WoundAmbit masking backends
- COLMAP feature extraction, matching, dense reconstruction, and Poisson meshing
- GLOMAP sparse reconstruction
- Optional AprilTag metric scaling and Rerun visualization/filtering
- Dense point-cloud post-processing and optional PointCleanNet integration
- Mesh-based wound-edge segmentation

## Requirements

### System tools

Install these separately and ensure they are available on `PATH`:

- [COLMAP](https://colmap.github.io/)
- GLOMAP
- FFmpeg
- Git

A CUDA-capable GPU and a PyTorch/CUDA combination compatible with the selected optional backend may be required.

### Python

- Python 3.10 or newer
- Windows is the primary development environment; WoundAmbit currently requires its separately configured runtime
  environment, commonly WSL

## Installation

Clone with optional backend source repositories:

```powershell
git clone --recurse-submodules <repository-url>
cd COLMAP_Pipline
```

For an existing clone:

```powershell
git submodule update --init --recursive
```

Create and activate an environment, then install the package:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

For tests and linting:

```powershell
python -m pip install -e .[dev]
```

## Configuration

Copy the public template; the local configuration is ignored by Git:

```powershell
Copy-Item .\configs\local.example.yaml .\configs\local.yaml
```

Set at least these fields in `configs/local.yaml`:

```yaml
input_video: C:/path/to/input_video.mp4
workspace: C:/path/to/wound3d_workspace

masking:
  backend: conservative
```

Use forward slashes in Windows YAML paths, or use single-quoted paths with backslashes. Do not use double-quoted
backslash paths because YAML treats backslashes as escape characters.

See `configs/default.yaml` for all supported options.

## Run

```powershell
python -m wound3d --config .\configs\local.yaml
```

Or use the PowerShell helper:

```powershell
.\scripts\run_pipeline.ps1 -Config .\configs\local.yaml
```

After editable installation, this console command is also available:

```powershell
wound3d-pipeline --config .\configs\local.yaml
```

## Optional backends

Optional backends are not downloaded, cloned, or installed during a reconstruction run. Initialize submodules and install
or configure the selected backend before enabling it in YAML.

| Backend | YAML selection | Setup summary |
| --- | --- | --- |
| Conservative masks | `masking.backend: conservative` | Included; no external model required. |
| SAM2 | `masking.backend: sam2` | Install `external/sam2` into the active environment and provide checkpoint/config paths. |
| DeepSkin | `masking.backend: deepskin` | Install `external/Deepskin` into the active environment. |
| WoundAmbit | `masking.backend: woundambit` | Configure its dedicated environment and `woundambit_executable` when required. The repository's `WoundAmbitWrapper.py` location is detected automatically. DeepSkin must also be installed: it generates the wound+tissue mask used for background removal, while WoundAmbit supplies the wound-only mask. |
| PointCleanNet | `reconstruction.use_pointcleannet: true` | Configure `external/PointCleanNet` as an importable package. |

Read [third-party software and optional backends](docs/third-party-software.md) before redistribution or model use.

## Output workspace

A run writes generated artifacts below the configured workspace, for example:

```text
workspace/
├── raw_images/
├── sharp_images/
├── masks/
├── masks_wound_only/
├── sparse/
│   └── 0/
├── dense/
├── database.db
├── project.ini
├── glomap_config.ini
└── pipeline_times.log
```

These files are intentionally ignored by Git. Do not commit reconstruction outputs, patient images/videos, masks, model
weights, or local configuration.

## Development

```powershell
python -m pytest
ruff check src tests
python scripts/check_release_ready.py
```

The release check detects tracked bytecode, model weights, generated geometry, local configuration, output folders, and
submodule problems. See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/release-checklist.md](docs/release-checklist.md).

## Privacy and responsible use

Do not upload or commit protected health information, identifiable patient data, or derived artifacts without the required
permissions. Users are responsible for consent, ethics approval, institutional review, and compliance with applicable
data-protection requirements.

## License and citation

A project license and citation metadata must be selected before a public release. See
[docs/release-metadata.md](docs/release-metadata.md). Third-party components retain their own licenses and terms; see
[docs/third-party-software.md](docs/third-party-software.md).
