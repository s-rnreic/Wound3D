# Release Checklist

## Repository hygiene

- [ ] `git status --short` is clean.
- [ ] No `__pycache__`, `.pyc`, video, patient image, mesh, database, model weight, or runtime output is tracked.
- [ ] `configs/local.yaml` is absent from Git history and public examples contain no personal paths.
- [ ] Every submodule is initialized, clean, and pinned to the intended commit.
- [ ] `python scripts/check_release_ready.py` succeeds (without `--skip-submodules`).

## Quality

- [ ] `python -m pip install -e .[dev]` succeeds in a fresh environment.
- [ ] `python -m pytest` succeeds.
- [ ] `ruff check src tests` succeeds.
- [ ] `python -m wound3d --help` succeeds.
- [ ] A documented smoke test has been run with non-sensitive test data.

## Documentation and legal review

- [ ] `README.md` reflects the current CLI and configuration format.
- [ ] `LICENSE` has been selected and reviewed.
- [ ] `CITATION.cff` has correct authors, repository URL, and release version/date.
- [ ] The medical/research-use disclaimer and third-party notice have been reviewed.
- [ ] Supported Python, CUDA, PyTorch, COLMAP, GLOMAP, and OS versions are documented.

## Release

- [ ] Version in `pyproject.toml` is updated.
- [ ] A changelog/release note is prepared.
- [ ] A signed or annotated tag is created, for example `v0.1.0`.
- [ ] A GitHub/GitLab release is created.
- [ ] If applicable, the release is archived with Zenodo for a DOI.