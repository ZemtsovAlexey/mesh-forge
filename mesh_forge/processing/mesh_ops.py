from __future__ import annotations

import logging
import shutil
from pathlib import Path

from mesh_forge.config import load_config
from mesh_forge.ops.geometry import (
    apply_operations,
    keep_largest_component,
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
        """Post-process ComfyUI mesh → STL (or raw export if mesh_postprocess=False)."""
        self.config = load_config()
        work_dir.mkdir(parents=True, exist_ok=True)
        stl_path = work_dir / "mesh_from_reconstruction.stl"
        photo = self.config.photo
        mesh = load_mesh(mesh_path)

        if not bool(getattr(photo, "mesh_postprocess", True)):
            save_mesh(mesh, stl_path)
            if solidify_mm > 0:
                _warn_solidify_skipped(solidify_mm)
            logger.info(
                "finalize (postprocess off): %s -> %s (%d faces)",
                mesh_path.name,
                stl_path.name,
                len(mesh.faces),
            )
            return stl_path

        try:
            mesh = keep_largest_component(mesh, single=True)
        except Exception as exc:
            logger.warning("keep_largest_component failed: %s", exc)
        mesh = repair_reconstruction_mesh(
            mesh,
            target_faces=int(photo.finalize_target_faces or 0),
            smooth_iters=int(photo.finalize_smooth_iters or 0),
            min_edge_mm=float(photo.finalize_min_edge_mm or 0.08),
            close_holes=bool(photo.finalize_close_holes),
            voxel_mm=float(getattr(photo, "finalize_voxel_mm", 0.0) or 0.0),
        )
        mesh = orient_upright(mesh)
        mesh = normalize_height_mm(mesh, float(photo.target_height_mm or 160.0))
        save_mesh(mesh, stl_path)
        if solidify_mm > 0:
            _warn_solidify_skipped(solidify_mm)
        logger.info(
            "finalize: %s -> %s (%d faces, wt=%s)",
            mesh_path.name,
            stl_path.name,
            len(mesh.faces),
            bool(getattr(mesh, "is_watertight", False)),
        )
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
