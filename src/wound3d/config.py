"""Configuration loading and validation for the supported wound3d CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_MASK_BACKENDS = {"conservative", "sam2", "woundambit", "deepskin"}


@dataclass(frozen=True)
class MaskingConfig:
    enabled: bool = True
    backend: str = "conservative"
    use_adjustment: bool = False
    extrude_masks: bool = True
    min_mask_ratio: float = 0.05
    max_mask_ratio: float = 0.5
    neighbors_count: int = 3
    sam2_checkpoint: Path | None = None
    sam2_model_config: Path | None = None
    woundambit_executable: str | None = None
    woundambit_working_directory: Path | None = None


@dataclass(frozen=True)
class PipelineConfig:
    input_video: Path
    workspace: Path
    segments: int = 10
    resize_factor: float = 1.0
    sharpness_percentile: float = 25.0
    enhance_option: str | None = None
    tag_size_m: float = 0.0123
    masking: MaskingConfig = MaskingConfig()
    use_pointcleannet: bool = False
    run_rerun_filter: bool = True
    run_apriltag_scaling: bool = True


def _resolve_path(value: str | Path | None, base_directory: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_directory / path).resolve()


def _require(mapping: dict[str, Any], key: str) -> Any:
    value = mapping.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required configuration key: '{key}'.")
    return value


def load_pipeline_config(config_path: str | Path) -> PipelineConfig:
    """Load a YAML configuration, resolving relative paths from its directory."""
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise RuntimeError("PyYAML is required. Install project dependencies with 'python -m pip install -e .'.") from error

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except yaml.YAMLError as error:
        raise ValueError(
            f"Invalid YAML in {path}. For Windows paths, use C:/path/to/file or single-quoted backslash paths."
        ) from error

    if not isinstance(raw, dict):
        raise ValueError("Pipeline configuration must be a YAML mapping.")

    processing = raw.get("processing", {})
    masking_raw = raw.get("masking", {})
    reconstruction = raw.get("reconstruction", {})
    if not all(isinstance(item, dict) for item in (processing, masking_raw, reconstruction)):
        raise ValueError("processing, masking, and reconstruction must be YAML mappings.")

    backend = str(masking_raw.get("backend", "conservative")).lower()
    if backend not in VALID_MASK_BACKENDS:
        raise ValueError(f"Unsupported masking.backend '{backend}'. Choose: {', '.join(sorted(VALID_MASK_BACKENDS))}.")

    min_ratio = float(masking_raw.get("min_mask_ratio", 0.05))
    max_ratio = float(masking_raw.get("max_mask_ratio", 0.5))
    if not 0.0 <= min_ratio <= max_ratio <= 1.0:
        raise ValueError("masking ratios must satisfy 0 <= min_mask_ratio <= max_mask_ratio <= 1.")

    config_directory = path.parent
    return PipelineConfig(
        input_video=_resolve_path(_require(raw, "input_video"), config_directory),
        workspace=_resolve_path(_require(raw, "workspace"), config_directory),
        segments=int(processing.get("segments", 10)),
        resize_factor=float(processing.get("resize_factor", 1.0)),
        sharpness_percentile=float(processing.get("sharpness_percentile", 25.0)),
        enhance_option=processing.get("enhance_option"),
        tag_size_m=float(reconstruction.get("tag_size_m", 0.0123)),
        masking=MaskingConfig(
            enabled=bool(masking_raw.get("enabled", True)),
            backend=backend,
            use_adjustment=bool(masking_raw.get("use_adjustment", False)),
            extrude_masks=bool(masking_raw.get("extrude_masks", True)),
            min_mask_ratio=min_ratio,
            max_mask_ratio=max_ratio,
            neighbors_count=int(masking_raw.get("neighbors_count", 3)),
            sam2_checkpoint=_resolve_path(masking_raw.get("sam2_checkpoint"), config_directory),
            sam2_model_config=_resolve_path(masking_raw.get("sam2_model_config"), config_directory),
            woundambit_executable=masking_raw.get("woundambit_executable"),
            woundambit_working_directory=_resolve_path(
                masking_raw.get("woundambit_working_directory"), config_directory
            ),
        ),
        use_pointcleannet=bool(reconstruction.get("use_pointcleannet", False)),
        run_rerun_filter=bool(reconstruction.get("run_rerun_filter", True)),
        run_apriltag_scaling=bool(reconstruction.get("run_apriltag_scaling", True)),
    )