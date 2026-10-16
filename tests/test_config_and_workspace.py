from pathlib import Path

import cv2
import numpy as np
import pytest

from wound3d.config import load_pipeline_config
from wound3d.workspace import WorkspacePaths


def test_load_pipeline_config_resolves_relative_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        """\
input_video: input.mp4
workspace: workspace
masking:
  backend: sam2
  sam2_checkpoint: models/model.pt
  sam2_model_config: models/config.yaml
reconstruction:
  run_rerun_filter: false
  run_apriltag_scaling: false
""",
        encoding="utf-8",
    )

    config = load_pipeline_config(config_path)

    assert config.input_video == tmp_path / "input.mp4"
    assert config.workspace == tmp_path / "workspace"
    assert config.masking.sam2_checkpoint == tmp_path / "models" / "model.pt"
    assert config.run_rerun_filter is False
    assert config.run_apriltag_scaling is False


def test_invalid_mask_backend_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("input_video: video.mp4\nworkspace: run\nmasking:\n  backend: unknown\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported masking.backend"):
        load_pipeline_config(config_path)


def test_workspace_paths_are_created_explicitly(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path / "run")

    assert paths.root.is_dir()
    assert paths.raw_images.is_dir()
    assert paths.sharp_images.is_dir()
    assert paths.masks.is_dir()
    assert paths.database == paths.root / "database.db"


def test_deepskin_masks_use_wound_channel_two_and_tissue_channel_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from Pipe_Code.modules.masking import build_masks

    input_dir = tmp_path / "images"
    wound_only_dir = tmp_path / "masks_wound_only"
    wound_tissue_dir = tmp_path / "masks"
    input_dir.mkdir()
    cv2.imwrite(str(input_dir / "00001.jpg"), np.zeros((2, 2, 3), dtype=np.uint8))

    semantic_mask = np.zeros((2, 2, 3), dtype=np.uint8)
    semantic_mask[0, 0, 0] = 255  # Background: must be excluded.
    semantic_mask[0, 1, 1] = 255  # Tissue.
    semantic_mask[1, 0, 2] = 255  # Wound.
    monkeypatch.setattr(build_masks, "_load_deepskin_segmentation", lambda: lambda **_kwargs: semantic_mask)

    build_masks._create_deepskin_masks(str(input_dir), str(wound_only_dir), str(wound_tissue_dir))

    wound_only = cv2.imread(str(wound_only_dir / "00001.jpg.png"), cv2.IMREAD_GRAYSCALE)
    wound_tissue = cv2.imread(str(wound_tissue_dir / "00001.jpg.png"), cv2.IMREAD_GRAYSCALE)
    assert wound_only[1, 0] == 255
    assert wound_only[0, 0] == 0
    assert wound_tissue[0, 1] == 255
    assert wound_tissue[1, 0] == 255
    assert wound_tissue[0, 0] == 0