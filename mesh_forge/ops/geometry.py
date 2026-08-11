from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

logger = logging.getLogger("mesh_forge.geometry")


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


def orient_upright(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Rotate so the longest bbox axis becomes +Y (viewer up), then sit on ground."""
    mesh = mesh.copy()
    extents = np.asarray(mesh.extents, dtype=float)
    long_axis = int(np.argmax(extents))
    if long_axis != 1:
        if long_axis == 0:  # X -> Y
            matrix = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
        else:  # Z -> Y
            matrix = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        mesh.apply_transform(matrix)

    # Prefer taller half above centroid (avoid head-down)
    if float(np.mean(mesh.vertices[:, 1] - mesh.centroid[1])) < 0:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))

    bounds = mesh.bounds
    # Center XZ, put min Y on ground
    shift = np.array([
        -(bounds[0][0] + bounds[1][0]) / 2,
        -bounds[0][1],
        -(bounds[0][2] + bounds[1][2]) / 2,
    ])
    mesh.apply_translation(shift)
    return mesh


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


def normalize_height_mm(mesh: trimesh.Trimesh, target_height_mm: float = 160.0) -> trimesh.Trimesh:
    """Uniform-scale so the longest axis matches target height (photo nets are ~unit cube)."""
    mesh = mesh.copy()
    extents = np.asarray(mesh.extents, dtype=float)
    longest = float(np.max(extents))
    if longest < 1e-9 or target_height_mm <= 0:
        return mesh
    mesh.apply_scale(target_height_mm / longest)
    # Re-seat on ground after scale
    bounds = mesh.bounds
    mesh.apply_translation([
        -(bounds[0][0] + bounds[1][0]) / 2,
        -bounds[0][1],
        -(bounds[0][2] + bounds[1][2]) / 2,
    ])
    return mesh


def try_make_watertight(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Light repair only — aggressive fill_holes can explode face count on dense nets."""
    mesh = mesh.copy()
    try:
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    # Soft hole fill: only if mesh is small enough / few holes
    try:
        if (not mesh.is_watertight) and len(mesh.faces) < 80_000:
            trimesh.repair.fill_holes(mesh)
    except Exception:
        pass
    return mesh


def smooth_mesh(mesh: trimesh.Trimesh, iterations: int = 2) -> trimesh.Trimesh:
    iterations = max(0, min(int(iterations), 5))
    if iterations <= 0:
        return mesh
    # volume_constraint blows up on open / non-manifold scans
    try:
        return trimesh.smoothing.filter_laplacian(
            mesh, lamb=0.5, iterations=iterations, volume_constraint=False
        )
    except TypeError:
        return trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=iterations)
    except Exception:
        return mesh


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Reduce face count. Uses fast-simplification / Open3D; never passes face count as percent."""
    n = int(len(mesh.faces))
    target = int(target_faces)
    if target <= 0 or n <= target:
        return mesh
    # Keep at least a tiny mesh
    target = max(4, min(target, n - 1))

    try:
        from fast_simplification import simplify

        vertices, faces = simplify(
            points=np.asarray(mesh.vertices, dtype=np.float64),
            triangles=np.asarray(mesh.faces, dtype=np.int32),
            target_count=target,
        )
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    except Exception as exc:
        logger.warning("fast_simplification failed (%s); trying Open3D", exc)

    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "Decimate requires fast-simplification or open3d"
        ) from exc

    o3 = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    simplified = o3.simplify_quadric_decimation(target_number_of_triangles=target)
    return trimesh.Trimesh(
        vertices=np.asarray(simplified.vertices),
        faces=np.asarray(simplified.triangles),
        process=False,
    )


def remesh_voxel(mesh: trimesh.Trimesh, voxel_mm: float) -> trimesh.Trimesh:
    """Voxel remesh with safety clamp — tiny voxels on meter-scale scans destroy the mesh."""
    extents = np.asarray(mesh.extents, dtype=float)
    longest = float(np.max(extents)) if extents.size else 0.0
    pitch = float(voxel_mm) if voxel_mm and voxel_mm > 0 else 1.0
    if longest > 0:
        # Cap resolution: never more than ~200 voxels along the longest axis
        min_pitch = max(longest / 200.0, 0.5)
        if pitch < min_pitch:
            logger.warning(
                "remesh_voxel: clamping pitch %.4f -> %.4f mm (longest %.1f mm)",
                pitch,
                min_pitch,
                longest,
            )
            pitch = min_pitch
        dims = np.maximum(extents / pitch, 1.0)
        if float(np.prod(dims)) > 8_000_000:
            pitch = max(pitch, float(np.prod(extents) ** (1.0 / 3.0) / 200.0))
            logger.warning("remesh_voxel: further clamp to %.4f mm for memory safety", pitch)
    try:
        return mesh.voxelized(pitch).marching_cubes
    except Exception as exc:
        logger.warning("remesh_voxel failed (%s); leaving mesh unchanged", exc)
        return mesh


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
            target = op.get("target_faces", op.get("face_count", op.get("faces")))
            if target is None:
                # LLM sometimes sends percent/reduction instead
                pct = op.get("percent", op.get("target_reduction"))
                if pct is not None:
                    target = int(len(mesh.faces) * (1.0 - float(pct)))
                else:
                    target = max(1000, len(mesh.faces) // 2)
            mesh = decimate(mesh, int(target))
        elif name == "remesh_voxel":
            mesh = remesh_voxel(mesh, float(op.get("voxel_mm", 1.0)))
        elif name == "fill_holes":
            try:
                if len(mesh.faces) < 200_000:
                    trimesh.repair.fill_holes(mesh)
                else:
                    logger.warning("fill_holes skipped on large mesh (%s faces)", len(mesh.faces))
            except Exception as exc:
                logger.warning("fill_holes failed: %s", exc)
    return save_mesh(mesh, out_path)
