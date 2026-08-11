from __future__ import annotations

import logging
from pathlib import Path

from mesh_forge.backends.blender import blender_available, repair_and_export
from mesh_forge.backends.hunyuan3d import hunyuan3d_available, run_hunyuan3d
from mesh_forge.backends.triposr import run_triposr, triposr_available
from mesh_forge.config import load_config
from mesh_forge.ops.geometry import (
    load_mesh,
    normalize_height_mm,
    orient_upright,
    save_mesh,
    smooth_mesh,
    try_make_watertight,
)
from mesh_forge import progress as prog

logger = logging.getLogger("mesh_forge.pipeline.photo")


def resolve_photo_backend(requested: str | None = None) -> str:
    cfg = load_config()
    backend = (requested or cfg.photo.backend or "hunyuan3d").strip().lower()
    if backend in {"hunyuan", "hunyuan3d", "hunyuan3d-2mini", "hy3d"}:
        return "hunyuan3d"
    if backend in {"triposr", "tripo"}:
        return "triposr"
    raise ValueError(f"Unknown photo backend: {backend}. Use hunyuan3d or triposr.")


def create_from_photo(
    image_path: Path,
    work_dir: Path,
    *,
    remove_bg: bool = True,
    solidify_mm: float = 0.0,
    backend: str | None = None,
    project_id: str | None = None,
) -> Path:
    cfg = load_config()
    chosen = resolve_photo_backend(backend)
    if chosen == "hunyuan3d":
        if not hunyuan3d_available():
            raise RuntimeError(
                "Hunyuan3D not available. Build Docker image "
                "(docker/hunyuan3d/build.ps1) or switch photo.backend to triposr."
            )
        out_dir = work_dir / "hunyuan"
        label = "Hunyuan3D-2mini"
        if project_id:
            prog.update(project_id, 8, f"Запуск {label}…")
        logger.info("photo pipeline: %s image=%s remove_bg=%s", label, image_path, remove_bg)
        obj_path = run_hunyuan3d(
            image_path, out_dir, remove_bg=remove_bg, project_id=project_id
        )
    else:
        if not triposr_available():
            raise RuntimeError(
                "TripoSR not available. Build Docker image (docker/triposr/build.ps1) "
                "or set docker.enabled: false and paths.triposr in config.yaml"
            )
        out_dir = work_dir / "triposr"
        label = "TripoSR"
        if project_id:
            prog.update(project_id, 8, f"Запуск {label}…")
        logger.info("photo pipeline: %s image=%s remove_bg=%s", label, image_path, remove_bg)
        obj_path = run_triposr(
            image_path, out_dir, remove_bg=remove_bg, project_id=project_id
        )

    if project_id:
        prog.update(project_id, 88, "Ориентация, масштаб и ремонт…")
    stl_path = work_dir / "photo_raw.stl"
    logger.info("photo pipeline: convert %s -> %s", obj_path, stl_path)
    mesh = load_mesh(obj_path)
    mesh = orient_upright(mesh)
    target_h = float(cfg.photo.target_height_mm or 160.0)
    before = float(mesh.extents.max())
    mesh = normalize_height_mm(mesh, target_h)
    logger.info(
        "photo pipeline: scale %.4f -> %.1f mm (target_height)",
        before,
        float(mesh.extents.max()),
    )
    mesh = try_make_watertight(mesh)
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
