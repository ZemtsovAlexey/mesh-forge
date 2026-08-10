from __future__ import annotations

from pathlib import Path

from mesh_forge.backends.blender import blender_available, repair_and_export
from mesh_forge.backends.triposr import run_triposr, triposr_available
from mesh_forge.ops.geometry import load_mesh, save_mesh


def create_from_photo(image_path: Path, work_dir: Path, *, remove_bg: bool = True, solidify_mm: float = 0.0) -> Path:
    if not triposr_available():
        raise RuntimeError("TripoSR not configured. Set paths.triposr in config.yaml")
    triposr_out = work_dir / "triposr"
    obj_path = run_triposr(image_path, triposr_out, remove_bg=remove_bg)
    stl_path = work_dir / "photo_raw.stl"
    mesh = load_mesh(obj_path)
    save_mesh(mesh, stl_path)
    if blender_available() and solidify_mm > 0:
        final = work_dir / "photo_final.stl"
        repair_and_export(stl_path, final, solidify_mm=solidify_mm)
        return final
    return stl_path
