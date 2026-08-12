from __future__ import annotations

import shutil
from pathlib import Path

from mesh_forge.backends.blender import blender_available, repair_and_export
from mesh_forge.config import load_config
from mesh_forge.ops.geometry import (
    apply_operations,
    load_mesh,
    normalize_height_mm,
    orient_upright,
    save_mesh,
    smooth_mesh,
    try_make_watertight,
)
from mesh_forge.ops.repair import clean_scan


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
        mesh = load_mesh(mesh_path)
        mesh = orient_upright(mesh)
        mesh = normalize_height_mm(mesh, float(self.config.photo.target_height_mm or 160.0))
        mesh = try_make_watertight(mesh)
        mesh = smooth_mesh(mesh, iterations=1)
        save_mesh(mesh, stl_path)
        if blender_available() and solidify_mm > 0:
            final_path = work_dir / "mesh_from_reconstruction_solid.stl"
            return repair_and_export(stl_path, final_path, solidify_mm=solidify_mm)
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
        if blender_available() and solidify_mm > 0:
            final_path = work_dir / "mesh_cleaned_solid.stl"
            return repair_and_export(cleaned, final_path, solidify_mm=solidify_mm)
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
        if blender_available() and solidify_mm > 0:
            final_path = work_dir / "mesh_edited_solid.stl"
            return repair_and_export(edited, final_path, solidify_mm=solidify_mm)
        return edited
