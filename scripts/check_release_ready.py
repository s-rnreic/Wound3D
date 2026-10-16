"""Fail fast on common repository artifacts that must not be published."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".pyc", ".mp4", ".mov", ".avi", ".ply", ".bin", ".db", ".pt", ".pth", ".ckpt", ".onnx"}
FORBIDDEN_PARTS = {"__pycache__", "outputs", "workspaces", "experiments", "models", "checkpoints"}


def git_lines(*arguments: str) -> list[str]:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return [line for line in result.stdout.splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check for artifacts that must not be published.")
    parser.add_argument(
        "--skip-submodules",
        action="store_true",
        help="Skip submodule initialization/pinning checks for source-only CI jobs.",
    )
    args = parser.parse_args(argv)
    failures: list[str] = []
    for name in git_lines("ls-files"):
        path = Path(name)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or any(part.lower() in FORBIDDEN_PARTS for part in path.parts):
            failures.append(f"Forbidden tracked artifact: {name}")

    if "configs/local.yaml" in git_lines("ls-files"):
        failures.append("Local configuration is tracked: configs/local.yaml")

    if not args.skip_submodules:
        for status_line in git_lines("submodule", "status", "--recursive"):
            if status_line.startswith(("+", "-", "U")):
                failures.append(f"Submodule is not pinned/initialized cleanly: {status_line}")

        submodule_paths = git_lines("config", "--file", ".gitmodules", "--get-regexp", r"^submodule\..*\.path$")
        for line in submodule_paths:
            _, submodule_path = line.split(maxsplit=1)
            if not (ROOT / submodule_path).is_dir():
                continue
            result = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT / submodule_path, capture_output=True, text=True)
            if result.returncode != 0:
                failures.append(f"Unable to inspect submodule: {submodule_path}")
            elif result.stdout.strip():
                failures.append(f"Submodule has uncommitted changes: {submodule_path}")

    if failures:
        print("Release readiness check failed:", file=sys.stderr)
        print("\n".join(f"- {failure}" for failure in failures), file=sys.stderr)
        return 1

    print("Release readiness check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())