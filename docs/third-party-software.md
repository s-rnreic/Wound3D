# Third-Party Software and Optional Backends

This repository orchestrates external tools and optional repositories. They remain subject to their own licenses, model
terms, installation requirements, and hardware requirements.

## External executables

- COLMAP
- GLOMAP
- FFmpeg

These executables are not installed by `pip` and must be available on `PATH` before a reconstruction run.

## Git submodules

Initialize optional source dependencies with:

```powershell
git submodule update --init --recursive
```

| Submodule | Purpose | Runtime policy |
| --- | --- | --- |
| `external/sam2` | Optional SAM2 image/video masking | Install it in the active environment before selecting `masking.backend: sam2`. Model checkpoints are not stored in Git. |
| `external/Deepskin` | Optional DeepSkin masking | Install it in the active environment before selecting `masking.backend: deepskin`. |
| `external/PointCleanNet` | Optional dense point-cloud cleaning | Configure it as an importable package before enabling `reconstruction.use_pointcleannet`. |
| `external/WoundAmbit` | Optional wound-only segmentation backend | Install/configure it in its dedicated environment before selecting `masking.backend: woundambit`. DeepSkin is also required in this mode because it produces the combined wound+tissue mask used to remove background. |

No external repository is cloned, modified, or installed during a pipeline run. Pinning is handled through Git submodule
commits in the parent repository. Review every upstream repository's license and model/data terms before redistribution.