from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from mesh_forge.config import load_config


def triposr_path() -> Path:
    cfg = load_config()
    if cfg.paths.triposr:
        return Path(cfg.paths.triposr)
    raise FileNotFoundError("TripoSR path not set in config.yaml")


def run_triposr(image_path: Path, output_dir: Path, *, remove_bg: bool = True) -> Path:
    triposr = triposr_path()
    run_py = triposr / "run.py"
    if not run_py.is_file():
        raise FileNotFoundError(f"TripoSR run.py not found: {run_py}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(run_py),
        str(image_path),
        "--output-dir", str(output_dir),
        "--model-save-format", "obj",
    ]
    if not remove_bg:
        cmd.append("--no-remove-bg")
    if load_config().gpu.vram_gb <= 6:
        cmd.extend(["--chunk-size", "200", "--mc-resolution", "128"])

    result = subprocess.run(cmd, cwd=str(triposr), capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        raise RuntimeError(f"TripoSR failed:\n{result.stderr[-3000:]}")

    objs = list(output_dir.glob("**/*.obj"))
    if not objs:
        raise RuntimeError("TripoSR produced no OBJ output")
    return objs[0]


def triposr_available() -> bool:
    try:
        return (triposr_path() / "run.py").is_file()
    except FileNotFoundError:
        return False
