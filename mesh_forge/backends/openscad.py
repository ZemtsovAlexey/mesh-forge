from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from mesh_forge.config import load_config


def openscad_path() -> Path:
    cfg = load_config()
    if cfg.paths.openscad:
        return Path(cfg.paths.openscad)
    raise FileNotFoundError("OpenSCAD path not set in config.yaml")


def _clean_scad(code: str) -> str:
    code = re.sub(r"^```(?:openscad|scad)?\s*", "", code.strip(), flags=re.I)
    code = re.sub(r"```\s*$", "", code.strip())
    return code


def render_scad_to_stl(scad_code: str, out_stl: Path) -> Path:
    scad = _clean_scad(scad_code)
    openscad = openscad_path()
    with tempfile.NamedTemporaryFile("w", suffix=".scad", delete=False, encoding="utf-8") as f:
        f.write(scad)
        scad_path = Path(f.name)
    out_stl.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(openscad), "-o", str(out_stl), str(scad_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    scad_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"OpenSCAD failed:\n{result.stderr}")
    return out_stl


def openscad_available() -> bool:
    try:
        return openscad_path().is_file()
    except FileNotFoundError:
        return False
