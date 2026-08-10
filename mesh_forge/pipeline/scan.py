from __future__ import annotations

from pathlib import Path

from mesh_forge.backends.blender import blender_available, repair_and_export
from mesh_forge.mesh_qc import analyze_mesh
from mesh_forge.ops.repair import clean_scan


def create_from_scan(scan_path: Path, work_dir: Path, *, mode: str = "light", smooth_iters: int = 1, solidify_mm: float = 0.0) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    cleaned = work_dir / "scan_cleaned.stl"
    clean_scan(scan_path, cleaned, mode=mode, smooth_iters=smooth_iters)

    stats = analyze_mesh(cleaned)
    if stats.issues and (stats.triangle_count == 0 or stats.vertex_count == 0):
        raise ValueError(
            "Scan cleanup produced an invalid mesh. "
            + "; ".join(stats.issues)
            + ". Try 'light' mode or check the input STL."
        )

    if blender_available() and solidify_mm > 0:
        final = work_dir / "scan_final.stl"
        repair_and_export(cleaned, final, solidify_mm=solidify_mm)
        return final
    return cleaned
