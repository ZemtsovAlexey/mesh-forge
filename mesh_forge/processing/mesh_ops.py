from __future__ import annotations

import logging
import shutil
from pathlib import Path

from mesh_forge.config import load_config
from mesh_forge.ops.geometry import (
    apply_operations,
    load_mesh,
    normalize_height_mm,
    orient_upright,
    repair_reconstruction_mesh,
    save_mesh,
)
from mesh_forge.ops.repair import clean_scan

logger = logging.getLogger("mesh_forge.mesh_ops")


def _warn_solidify_skipped(solidify_mm: float) -> None:
    logger.warning(
        "solidify_mm=%.3f requested, but Blender dependency was removed; "
        "returning STL without wall-thickness pass",
        solidify_mm,
    )


class MeshProcessingService:
    def __init__(self) -> None:
        self.config = load_config()

    def finalize_reconstruction(
        self,
        mesh_path: Path,
        work_dir: Path,
        *,
        solidify_mm: float = 0.0,
    ) -> Path:
        work_dir.mkdir(parents=True, exist_ok=True)
        stl_path = work_dir / "mesh_from_reconstruction.stl"
        photo = self.config.photo
        mesh = load_mesh(mesh_path)
        mesh = orient_upright(mesh)
        mesh = normalize_height_mm(mesh, float(photo.target_height_mm or 160.0))
        mesh = repair_reconstruction_mesh(
            mesh,
            target_faces=int(getattr(photo, "finalize_target_faces", 120_000) or 120_000),
            smooth_iters=int(getattr(photo, "finalize_smooth_iters", 3) or 0),
            min_edge_mm=float(getattr(photo, "finalize_min_edge_mm", 0.08) or 0.08),
            close_holes=bool(getattr(photo, "finalize_close_holes", True)),
            voxel_mm=float(getattr(photo, "finalize_voxel_mm", 1.0) or 0.0),
        )
        save_mesh(mesh, stl_path)
        if solidify_mm > 0:
            _warn_solidify_skipped(solidify_mm)
        return stl_path

    def cleanup_mesh(
        self,
        mesh_path: Path,
        work_dir: Path,
        *,
        mode: str = "light",
        smooth_iters: int = 1,
        solidify_mm: float = 0.0,
    ) -> Path:
        work_dir.mkdir(parents=True, exist_ok=True)
        cleaned = work_dir / "mesh_cleaned.stl"
        clean_scan(mesh_path, cleaned, mode=mode, smooth_iters=smooth_iters)
        if solidify_mm > 0:
            _warn_solidify_skipped(solidify_mm)
        return cleaned

    def apply_edit_operations(
        self,
        mesh_path: Path,
        operations: list[dict],
        work_dir: Path,
        *,
        solidify_mm: float = 0.0,
    ) -> Path:
        work_dir.mkdir(parents=True, exist_ok=True)
        edited = work_dir / "mesh_edited.stl"
        if operations:
            apply_operations(mesh_path, operations, edited)
        else:
            shutil.copy2(mesh_path, edited)
        if solidify_mm > 0:
            _warn_solidify_skipped(solidify_mm)
        return edited
