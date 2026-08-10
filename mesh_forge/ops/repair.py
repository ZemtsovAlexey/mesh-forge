from __future__ import annotations

import tempfile
from pathlib import Path

import trimesh

import numpy as np

from mesh_forge.ops.geometry import load_mesh, save_mesh, smooth_mesh


def _pymeshlab_available() -> bool:
    try:
        import pymeshlab  # noqa: F401
        return True
    except ImportError:
        return False


def _pml_method(ms, name: str, **kwargs) -> None:
    fn = getattr(ms, name, None)
    if not callable(fn):
        raise AttributeError(f"PyMeshLab missing method: {name}")
    fn(**kwargs)


def clean_scan_pymeshlab(inp: Path, out: Path, *, smooth_iters: int = 1) -> Path:
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(inp))
    _pml_method(ms, "meshing_remove_duplicate_vertices")
    _pml_method(ms, "meshing_remove_unreferenced_vertices")
    _pml_method(ms, "meshing_remove_duplicate_faces")
    _pml_method(ms, "meshing_remove_null_faces")
    _pml_method(ms, "meshing_repair_non_manifold_edges")
    try:
        _pml_method(ms, "meshing_close_holes", maxholesize=100)
    except Exception:
        pass
    if smooth_iters > 0:
        _pml_method(ms, "apply_coord_laplacian_smoothing", stepsmoothnum=smooth_iters)
    out.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(out))
    return out


def clean_scan_trimesh(inp: Path, out: Path, *, smooth_iters: int = 0) -> Path:
    mesh = load_mesh(inp)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)

    unique_faces = getattr(mesh, "unique_faces", None)
    nondegenerate_faces = getattr(mesh, "nondegenerate_faces", None)
    if callable(unique_faces) and callable(nondegenerate_faces):
        mesh.update_faces(unique_faces() & nondegenerate_faces())
    elif callable(nondegenerate_faces):
        mesh.update_faces(nondegenerate_faces())

    remove_unref = getattr(mesh, "remove_unreferenced_vertices", None)
    if callable(remove_unref):
        remove_unref()

    trimesh.repair.fill_holes(mesh)
    if smooth_iters > 0:
        mesh = smooth_mesh(mesh, smooth_iters)
    return save_mesh(mesh, out)


def clean_scan(inp: Path, out: Path, *, mode: str = "light", smooth_iters: int = 1) -> Path:
    if mode == "rebuild":
        try:
            return rebuild_poisson(inp, out)
        except Exception:
            pass
    return clean_scan_trimesh(inp, out, smooth_iters=smooth_iters)


def rebuild_poisson(inp: Path, out: Path, depth: int = 9) -> Path:
    try:
        import open3d as o3d
    except ImportError:
        mesh = load_mesh(inp)
        mesh = smooth_mesh(mesh, 2)
        return save_mesh(mesh, out)

    mesh = load_mesh(inp)
    if len(mesh.vertices) == 0:
        raise ValueError("Input mesh has no vertices")

    sample_count = min(200_000, max(50_000, len(mesh.vertices)))
    pcd = mesh.sample(sample_count)
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    o3d_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
    )
    o3d_pcd, _ = o3d_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        o3d_pcd, depth=depth
    )
    if len(densities) > 0:
        keep = np.asarray(densities) > np.quantile(np.asarray(densities), 0.02)
        poisson = poisson.select_by_index(np.where(keep)[0])

    poisson.remove_degenerate_triangles()
    poisson.remove_duplicated_triangles()
    poisson.remove_duplicated_vertices()
    poisson.remove_non_manifold_edges()

    bbox = o3d_pcd.get_axis_aligned_bounding_box()
    poisson = poisson.crop(bbox)

    if len(poisson.vertices) == 0 or len(poisson.triangles) == 0:
        raise ValueError("Poisson rebuild produced empty mesh — try 'light' cleanup mode")

    out.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(out), poisson):
        raise ValueError("Failed to write rebuilt mesh")
    return out
