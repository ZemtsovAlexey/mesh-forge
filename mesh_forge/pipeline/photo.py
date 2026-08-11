from __future__ import annotations

import logging
from pathlib import Path

from mesh_forge.backends.blender import blender_available, repair_and_export
from mesh_forge.backends.triposr import run_triposr, triposr_available
from mesh_forge.ops.geometry import load_mesh, orient_upright, save_mesh, smooth_mesh
from mesh_forge import progress as prog

logger = logging.getLogger("mesh_forge.pipeline.photo")


def create_from_photo(
    image_path: Path,
    work_dir: Path,
    *,
    remove_bg: bool = True,
    solidify_mm: float = 0.0,
    project_id: str | None = None,
) -> Path:
    if not triposr_available():
        raise RuntimeError(
            "TripoSR not available. Build Docker image (docker/triposr/build.ps1) "
            "or set docker.enabled: false and paths.triposr in config.yaml"
        )
    triposr_out = work_dir / "triposr"
    logger.info("photo pipeline: triposr image=%s remove_bg=%s", image_path, remove_bg)
    if project_id:
        prog.update(project_id, 8, "Запуск TripoSR…")
    obj_path = run_triposr(image_path, triposr_out, remove_bg=remove_bg, project_id=project_id)
    if project_id:
        prog.update(project_id, 88, "Ориентация и конвертация…")
    stl_path = work_dir / "photo_raw.stl"
    logger.info("photo pipeline: convert %s -> %s", obj_path, stl_path)
    mesh = load_mesh(obj_path)
    mesh = orient_upright(mesh)
    # Light smooth reduces marching-cubes stair-steps without killing detail
    try:
        mesh = smooth_mesh(mesh, iterations=1)
    except Exception as exc:
        logger.warning("smooth skipped: %s", exc)
    save_mesh(mesh, stl_path)
    if blender_available() and solidify_mm > 0:
        final = work_dir / "photo_final.stl"
        if project_id:
            prog.update(project_id, 94, "Solidify в Blender…")
        logger.info("photo pipeline: solidify %.2f mm", solidify_mm)
        repair_and_export(stl_path, final, solidify_mm=solidify_mm)
        return final
    return stl_path
