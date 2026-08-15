from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from trimesh import Scene

from mesh_forge.ops.geometry import read_trimesh


@dataclass
class MeshStats:
    triangle_count: int = 0
    vertex_count: int = 0
    watertight: bool = False
    winding_consistent: bool = False
    bbox_mm: list[float] | None = None
    volume_mm3: float | None = None
    surface_area_mm2: float | None = None
    min_edge_mm: float | None = None
    issues: list[str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"Triangles: {self.triangle_count:,}",
            f"Vertices: {self.vertex_count:,}",
            f"Watertight: {'yes' if self.watertight else 'NO'}",
            f"Winding OK: {'yes' if self.winding_consistent else 'no'}",
        ]
        if self.bbox_mm:
            lines.append(f"BBox (mm): {self.bbox_mm[0]:.1f} x {self.bbox_mm[1]:.1f} x {self.bbox_mm[2]:.1f}")
        if self.volume_mm3 is not None:
            lines.append(f"Volume: {self.volume_mm3:.1f} mm³")
        if self.issues:
            lines.append("Issues:")
            lines.extend(f"  - {i}" for i in self.issues)
        return "\n".join(lines)


def analyze_mesh(mesh_path: Path) -> MeshStats:
    issues: list[str] = []

    if not mesh_path.is_file():
        issues.append("Mesh file not found")
        return MeshStats(issues=issues)
    if mesh_path.stat().st_size == 0:
        issues.append("Mesh file is empty")
        return MeshStats(issues=issues)

    loaded = read_trimesh(mesh_path)
    if loaded is None or isinstance(loaded, Scene):
        issues.append("Scene contains no geometry")
        return MeshStats(issues=issues)
    mesh = loaded

    if mesh is None or len(getattr(mesh, "vertices", [])) == 0:
        issues.append("Mesh has no vertices")
        return MeshStats(issues=issues)
    if len(getattr(mesh, "faces", [])) == 0:
        issues.append("Mesh has no faces")
        return MeshStats(
            triangle_count=0,
            vertex_count=len(mesh.vertices),
            issues=issues,
        )

    # STL reloads as disconnected triangle soup unless vertices are merged.
    try:
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    if not mesh.is_watertight:
        issues.append("Mesh is not watertight (holes or open edges)")
    if not mesh.is_winding_consistent:
        issues.append("Inconsistent face winding (normals may be flipped)")

    edges = mesh.edges_unique_length
    min_edge = float(np.min(edges)) if len(edges) else None
    if min_edge is not None and min_edge < 0.1:
        issues.append(f"Very small edges detected (min {min_edge:.3f} mm)")

    extents = None
    bbox = mesh.bounds
    if bbox is not None and len(bbox) == 2:
        extents = (bbox[1] - bbox[0]).tolist()
    else:
        issues.append("Could not compute bounding box")

    volume = None
    if mesh.is_watertight:
        try:
            volume = float(mesh.volume)
        except Exception:
            issues.append("Could not compute volume")

    surface_area = None
    try:
        surface_area = float(mesh.area)
    except Exception:
        issues.append("Could not compute surface area")

    return MeshStats(
        triangle_count=len(mesh.faces),
        vertex_count=len(mesh.vertices),
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        bbox_mm=extents,
        volume_mm3=volume,
        surface_area_mm2=surface_area,
        min_edge_mm=min_edge,
        issues=issues or None,
    )


def is_print_ready(stats: MeshStats, min_wall_hint_mm: float = 0.4) -> bool:
    if not stats.watertight:
        return False
    if stats.min_edge_mm is not None and stats.min_edge_mm < min_wall_hint_mm:
        return False
    return True


def mesh_is_usable(path: Path, *, min_vertices: int = 8, min_faces: int = 4) -> tuple[bool, str]:
    stats = analyze_mesh(path)
    if stats.vertex_count < min_vertices or stats.triangle_count < min_faces:
        return False, stats.summary()
    return True, stats.summary()
