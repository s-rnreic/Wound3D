# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- Installable `src/wound3d` package and `wound3d-pipeline` command-line entry point.
- YAML-based runtime configuration and explicit workspace path management.
- Public configuration templates, release checks, CI workflow, tests, contributor guidance, security guidance, and
  research-use documentation.

### Changed

- Optional SAM2, DeepSkin, PointCleanNet, and WoundAmbit backends are explicit preconfigured dependencies.
- Pipeline runs no longer clone external repositories or install packages dynamically.
- Legacy `Pipe_Code.main` is now a compatibility launcher for the `wound3d` CLI.

### Removed

- Hard-coded workstation paths and standalone personal-path examples from supported pipeline modules.
- Tracked Python bytecode artifacts.

## [0.1.0] - Planned

First public research release. Add the final release date, license, citation metadata, and tag information when this
release is created.