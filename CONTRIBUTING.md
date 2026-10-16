# Contributing to wound3d

## Before opening a pull request

1. Do not add patient data, identifiable images, videos, masks, reconstructed geometry, model checkpoints, credentials,
   or local workstation configuration.
2. Keep new pipeline behavior configurable through YAML or CLI arguments; do not add hard-coded paths.
3. Keep optional backends optional. A pipeline run must never clone repositories or install packages dynamically.
4. Add or update tests for behavior changes.

## Local checks

```powershell
python -m pip install -e .[dev]
python -m pytest
ruff check src tests
python scripts/check_release_ready.py
```

## Code structure

- Put supported application code in `src/wound3d`.
- Put transitional implementation modules in `src/wound3d/legacy`; do not add new top-level packages.
- Put user-editable configuration templates in `configs/`.
- Put executable maintenance helpers in `scripts/`.
- Put tests in `tests/`.