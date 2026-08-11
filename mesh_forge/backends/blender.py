from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from pathlib import Path

from mesh_forge.config import load_config

logger = logging.getLogger("mesh_forge.blender")

BLENDER_REPAIR_SCRIPT = """
import bpy
import sys
argv = sys.argv
argv = argv[argv.index("--") + 1:]
inp, outp, thickness = argv[0], argv[1], float(argv[2])
bpy.ops.wm.read_factory_settings(use_empty=True)
if inp.lower().endswith(".stl"):
    bpy.ops.wm.stl_import(filepath=inp)
else:
    bpy.ops.wm.obj_import(filepath=inp)
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.remove_doubles(threshold=0.001)
bpy.ops.object.mode_set(mode="OBJECT")
mod = obj.modifiers.new(name="Solidify", type="SOLIDIFY")
mod.thickness = thickness
bpy.ops.object.modifier_apply(modifier=mod.name)
bpy.ops.wm.stl_export(filepath=outp, export_selected_objects=True)
"""


def _blender_path() -> Path:
    cfg = load_config()
    if cfg.paths.blender:
        return Path(cfg.paths.blender)
    raise FileNotFoundError("Blender path not set in config.yaml")


def run_blender_script(script: str, extra_args: list[str] | None = None) -> None:
    blender = _blender_path()
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(script)
        script_path = f.name
    cmd = [str(blender), "--background", "--python", script_path]
    if extra_args:
        cmd.extend(["--", *extra_args])
    logger.info("Blender run args=%s", extra_args)
    started = time.perf_counter()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600, encoding="utf-8", errors="replace",
    )
    elapsed = time.perf_counter() - started
    Path(script_path).unlink(missing_ok=True)
    if result.returncode != 0:
        logger.error(
            "Blender failed rc=%s after %.1fs\nstderr:\n%s\nstdout:\n%s",
            result.returncode,
            elapsed,
            result.stderr[-3000:],
            result.stdout[-1500:],
        )
        raise RuntimeError(f"Blender failed:\n{result.stderr[-2000:]}\n{result.stdout[-1000:]}")
    logger.info("Blender done in %.1fs", elapsed)
    if result.stderr.strip():
        logger.debug("Blender stderr tail:\n%s", result.stderr[-1000:])


def repair_and_export(inp: Path, out: Path, solidify_mm: float = 0.0) -> Path:
    if not inp.is_file():
        raise FileNotFoundError(f"Blender input mesh not found: {inp}")
    out.parent.mkdir(parents=True, exist_ok=True)
    thickness = solidify_mm if solidify_mm > 0 else 0.001
    logger.info("Blender repair %s -> %s solidify=%.3f", inp, out, thickness)
    run_blender_script(BLENDER_REPAIR_SCRIPT, [str(inp), str(out), str(thickness)])
    if not out.is_file():
        raise RuntimeError(f"Blender did not create output: {out}")
    logger.info("Blender output %s (%d bytes)", out, out.stat().st_size)
    return out


def blender_available() -> bool:
    try:
        return _blender_path().is_file()
    except FileNotFoundError:
        return False
