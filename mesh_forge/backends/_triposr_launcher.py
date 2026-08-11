"""Launch TripoSR run.py with mesh_forge vendor shims on sys.path."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

_VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor"
sys.path.insert(0, str(_VENDOR_DIR))

if __name__ == "__main__":
    triposr_root = Path(sys.argv[1])
    run_py = Path(sys.argv[2])
    os.chdir(triposr_root)
    sys.path.insert(0, str(triposr_root))
    sys.argv = [str(run_py), *sys.argv[3:]]
    runpy.run_path(str(run_py), run_name="__main__")
