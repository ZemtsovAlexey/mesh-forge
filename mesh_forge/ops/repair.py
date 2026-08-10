from __future__ import annotations

import tempfile
from pathlib import Path

import trimesh

from mesh_forge.ops.geometry import load_mesh, save_mesh, smooth_mesh


def _pymeshlab_available() -> bool:
    try:
        import pymeshlab  # noqa: F401
        return True
    except ImportError:
        return False


def clean_scan_pymeshlab(inp: Path, out: Path, *, smooth_iters: int = 1) -> Path:
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(inp))
    ms.apply_filter("remove_duplicate_vertices")
    ms.apply_filter("remove_unreferenced_vertices")
    ms.apply_filter("remove_non_manifold_edges")
    try:
        ms.apply_filter("close_holes", maxholesize=100)
    except Exception:
        pass
    if smooth_iters > 0:
        ms.apply_filter("laplacian_smooth", stepsmoothnum=smooth_iters)
    out.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(out))
    return out


def clean_scan_trimesh(inp: Path, out: Path) -> Path:
    mesh = load_mesh(inp)
    mesh.merge_vertices()
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fill_holes(mesh)
    return save_mesh(mesh, out)


def clean_scan(inp: Path, out: Path, *, mode: str = "light", smooth_iters: int = 1) -> Path:
    if mode == "rebuild":
        return rebuild_poisson(inp, out)
    if _pymeshlab_available():
        return clean_scan_pymeshlab(inp, out, smooth_iters=smooth_iters)
    return clean_scan_trimesh(inp, out)


def rebuild_poisson(inp: Path, out: Path, depth: int = 9) -> Path:
    try:
        import open3d as o3d
    except ImportError:
        mesh = load_mesh(inp)
        mesh = smooth_mesh(mesh, 2)
        return save_mesh(mesh, out)

    mesh = load_mesh(inp)
    pcd = mesh.sample(100_000)
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    o3d_pcd.estimate_normals()
    o3d_pcd, _ = o3d_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    poisson, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(o3d_pcd, depth=depth)
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out), poisson)
    return out
