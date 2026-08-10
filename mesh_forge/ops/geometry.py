from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if hasattr(g, "vertices") and len(g.vertices) > 0]
        if not geoms:
            raise ValueError(f"No geometry in mesh file: {path}")
        loaded = trimesh.util.concatenate(geoms)
    if len(loaded.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {path}")
    return loaded


def save_mesh(mesh: trimesh.Trimesh, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return path


def scale_axis(mesh: trimesh.Trimesh, axis: str, value_mm: float) -> trimesh.Trimesh:
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    extents = mesh.extents.copy()
    if extents[axis_idx] < 1e-6:
        return mesh
    factor = value_mm / extents[axis_idx]
    scale = np.ones(3)
    scale[axis_idx] = factor
    mesh = mesh.copy()
    mesh.apply_scale(scale)
    return mesh


def scale_uniform(mesh: trimesh.Trimesh, factor: float) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.apply_scale(factor)
    return mesh


def smooth_mesh(mesh: trimesh.Trimesh, iterations: int = 2) -> trimesh.Trimesh:
    return trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=iterations)


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= target_faces:
        return mesh
    return mesh.simplify_quadric_decimation(target_faces)


def remesh_voxel(mesh: trimesh.Trimesh, voxel_mm: float) -> trimesh.Trimesh:
    return mesh.voxelized(voxel_mm).marching_cubes


def apply_operations(mesh_path: Path, operations: list[dict[str, Any]], out_path: Path) -> Path:
    mesh = load_mesh(mesh_path)
    for op in operations:
        name = op.get("op", "")
        if name == "scale_axis":
            mesh = scale_axis(mesh, op["axis"], float(op["value_mm"]))
        elif name == "scale_uniform":
            mesh = scale_uniform(mesh, float(op["factor"]))
        elif name == "smooth":
            mesh = smooth_mesh(mesh, int(op.get("iterations", 2)))
        elif name == "decimate":
            mesh = decimate(mesh, int(op["target_faces"]))
        elif name == "remesh_voxel":
            mesh = remesh_voxel(mesh, float(op.get("voxel_mm", 1.0)))
        elif name == "fill_holes":
            trimesh.repair.fill_holes(mesh)
    return save_mesh(mesh, out_path)
