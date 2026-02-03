"""
Convenience entrypoint for running the collector from repo root.

This avoids requiring `PYTHONPATH=src` or an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parent
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> None:
    _ensure_src_on_path()
    from gaia_react.collect_traces import run

    run()


if __name__ == "__main__":
    main()

