"""Deprecated compatibility entry point for the wound3d pipeline.

Use ``python -m wound3d --config configs/local.yaml`` for all new workflows.
This module remains only so older commands using ``python -m Pipe_Code.main``
continue to invoke the same supported CLI.
"""

from __future__ import annotations

from wound3d.cli import main


if __name__ == "__main__":
    main()
