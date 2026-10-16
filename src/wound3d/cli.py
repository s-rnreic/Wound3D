"""Command-line interface for the supported wound3d package."""

from __future__ import annotations

import argparse
from pathlib import Path

from wound3d.config import load_pipeline_config
from wound3d.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the wound3d reconstruction pipeline.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML pipeline configuration.")
    args = parser.parse_args(argv)

    config = load_pipeline_config(args.config)
    if not config.input_video.is_file():
        raise FileNotFoundError(f"Configured input video does not exist: {config.input_video}")

    results, elapsed_seconds = run_pipeline(config)
    print(f"Pipeline completed in {elapsed_seconds:.2f} seconds.")
    print(f"Segmentation results: {results}")