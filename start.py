#!/usr/bin/env python3
"""Small launcher for the Thermal Bat Blob Detector GUI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _preferred_python(root: Path) -> str:
    candidates = [
        root / ".venv" / "bin" / "python",
        root / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def build_command(root: Path) -> list[str]:
    return [_preferred_python(root), "-m", "gui"]


def build_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src = str(root / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not existing else src + os.pathsep + existing
    return env


def main() -> int:
    root = _repo_root()
    cmd = build_command(root)
    if "--print-command" in sys.argv[1:]:
        print(" ".join(cmd))
        return 0
    return subprocess.call(cmd, cwd=root, env=build_env(root))


if __name__ == "__main__":
    raise SystemExit(main())
