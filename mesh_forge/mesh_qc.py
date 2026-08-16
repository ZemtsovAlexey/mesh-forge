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
            f"Треугольники: {self.triangle_count}",
            f"Вершины: {self.vertex_count}",
            f"Замкнут: {'да' if self.watertight else 'нет'}",
            f"Нормали: {'в порядке' if self.winding_consistent else 'перепутаны'}",
        ]
        if self.bbox_mm:
            lines.append(
                f"Габарит (мм): {self.bbox_mm[0]:.1f} × {self.bbox_mm[1]:.1f} × {self.bbox_mm[2]:.1f}"
            )
        if self.volume_mm3 is not None:
            lines.append(f"Объём: {self.volume_mm3:.1f} мм³")
        if self.issues:
            lines.append("Проблемы:")
            lines.extend(f"  - {i}" for i in self.issues)
        return "\n".join(lines)


def analyze_mesh(mesh_path: Path) -> MeshStats:
    issues: list[str] = []

    if not mesh_path.is_file():
        issues.append("Файл меша не найден")
        return MeshStats(issues=issues)
    if mesh_path.stat().st_size == 0:
        issues.append("Файл меша пустой")
        return MeshStats(issues=issues)

    loaded = read_trimesh(mesh_path)
    if loaded is None or isinstance(loaded, Scene):
        issues.append("В сцене нет геометрии")
        return MeshStats(issues=issues)
    mesh = loaded

    if mesh is None or len(getattr(mesh, "vertices", [])) == 0:
        issues.append("Нет вершин")
        return MeshStats(issues=issues)
    if len(getattr(mesh, "faces", [])) == 0:
        issues.append("Нет граней")
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

    # Open surface / mixed winding / tiny edges are typical for Hunyuan STL.
    # Status fields (watertight, winding_consistent, min_edge_mm) still record them.

    edges = mesh.edges_unique_length
    min_edge = float(np.min(edges)) if len(edges) else None

    extents = None
    bbox = mesh.bounds
    if bbox is not None and len(bbox) == 2:
        extents = (bbox[1] - bbox[0]).tolist()
    else:
        issues.append("Не удалось посчитать габарит")

    volume = None
    if mesh.is_watertight:
        try:
            volume = float(mesh.volume)
        except Exception:
            issues.append("Не удалось посчитать объём")

    surface_area = None
    try:
        surface_area = float(mesh.area)
    except Exception:
        issues.append("Не удалось посчитать площадь")

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
