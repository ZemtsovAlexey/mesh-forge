from __future__ import annotations

import shutil
from pathlib import Path

_SHIM_SRC = Path(__file__).resolve().parent.parent / "vendor" / "torchmcubes"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TRIPOSR_VENV = _PROJECT_ROOT / "venv-triposr"


def _shim_target() -> Path:
    if (_TRIPOSR_VENV / "Scripts" / "python.exe").is_file():
        return _TRIPOSR_VENV / "Lib" / "site-packages" / "torchmcubes"
    import sys

    return Path(sys.prefix) / "Lib" / "site-packages" / "torchmcubes"


def ensure_torchmcubes_shim() -> None:
    """Install CPU torchmcubes shim into the TripoSR venv if needed."""
    if not _SHIM_SRC.is_dir():
        raise FileNotFoundError(f"torchmcubes shim not found: {_SHIM_SRC}")

    target = _shim_target()
    if (target / "__init__.py").is_file():
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(_SHIM_SRC, target)
