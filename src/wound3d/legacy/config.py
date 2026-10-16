"""Loading and validation of user-editable pipeline configuration files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VALID_MASK_BACKENDS = {"conservative", "sam2", "woundambit"}


@dataclass(frozen=True)
class MaskingConfig:
    enabled: bool
    backend: str
    use_adjustment: bool
    extrude_masks: bool
    min_mask_ratio: float
    max_mask_ratio: float
    neighbors_count: int
    sam2_checkpoint: Path | None
    sam2_model_config: Path | None


@dataclass(frozen=True)
class PipelineConfig:
    input_video: Path
    workspace: Path
    segments: int
    resize_factor: float
    sharpness_percentile: float
    enhance_option: str | None
    tag_size_m: float
    masking: MaskingConfig
    use_pointcleannet: bool


def _resolve_path(value: str | None) -> Path | None:
    if value in (None, ""):
        return None

    path = Path(value).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping or mapping[key] in (None, ""):
        raise ValueError(f"Missing required configuration key: '{key}'.")
    return mapping[key]


def load_pipeline_config(config_path: str | Path) -> PipelineConfig:
    """Load a YAML pipeline configuration and validate its supported values."""
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "YAML configuration support requires PyYAML. Install project dependencies with "
            "'python -m pip install -e .'."
        ) from error

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file) or {}
    except yaml.YAMLError as error:
        raise ValueError(
            f"Invalid YAML in pipeline configuration: {path}. "
            "For Windows paths, use forward slashes (C:/Users/...) or single quotes around backslash paths."
        ) from error

    if not isinstance(raw_config, dict):
        raise ValueError("Pipeline configuration must contain a YAML mapping.")

    processing = raw_config.get("processing", {})
    masking_raw = raw_config.get("masking", {})
    reconstruction = raw_config.get("reconstruction", {})
    if not all(isinstance(section, dict) for section in (processing, masking_raw, reconstruction)):
        raise ValueError("The processing, masking, and reconstruction sections must be mappings.")

    backend = str(masking_raw.get("backend", "conservative")).lower()
    if backend not in VALID_MASK_BACKENDS:
        valid_backends = ", ".join(sorted(VALID_MASK_BACKENDS))
        raise ValueError(f"Unsupported masking.backend '{backend}'. Choose one of: {valid_backends}.")

    min_mask_ratio = float(masking_raw.get("min_mask_ratio", 0.05))
    max_mask_ratio = float(masking_raw.get("max_mask_ratio", 0.5))
    if not 0.0 <= min_mask_ratio <= max_mask_ratio <= 1.0:
        raise ValueError("masking min_mask_ratio and max_mask_ratio must satisfy 0 <= min <= max <= 1.")

    return PipelineConfig(
        input_video=_resolve_path(str(_required(raw_config, "input_video"))),
        workspace=_resolve_path(str(_required(raw_config, "workspace"))),
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
            min_mask_ratio=min_mask_ratio,
            max_mask_ratio=max_mask_ratio,
            neighbors_count=int(masking_raw.get("neighbors_count", 3)),
            sam2_checkpoint=_resolve_path(masking_raw.get("sam2_checkpoint")),
            sam2_model_config=_resolve_path(masking_raw.get("sam2_model_config")),
        ),
        use_pointcleannet=bool(reconstruction.get("use_pointcleannet", False)),
    )