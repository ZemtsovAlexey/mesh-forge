from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def render_mesh_preview(mesh_path: Path, out_path: Path, size: int = 512) -> Path:
    mesh = _load_render_mesh(mesh_path)
    fig = plt.figure(figsize=(4, 4), dpi=max(64, size // 4))
    ax = fig.add_subplot(111, projection="3d")
    _draw_mesh(ax, mesh, color=(0.78, 0.8, 0.86))
    ax.set_axis_off()
    ax.view_init(elev=25, azim=45)
    _equal_aspect(ax, mesh)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, facecolor="#1a1a2e")
    plt.close(fig)
    return out_path


def render_mesh_front_clay(mesh_path: Path, out_path: Path, size: int = 768) -> Path:
    """Orthographic-ish front bake for guided-edit anchors (studio clay look)."""
    mesh = _load_render_mesh(mesh_path)
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    _draw_mesh(ax, mesh, color=(0.93, 0.93, 0.91))
    ax.set_axis_off()
    # Near-front elevation for reconstruction-style silhouette.
    ax.view_init(elev=8, azim=-90)
    _equal_aspect(ax, mesh)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, facecolor="#9a9a9a")
    plt.close(fig)
    return out_path


def _load_render_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    # Cap face count so matplotlib stays responsive on photo reconstructions.
    if len(mesh.faces) > 80_000:
        try:
            mesh = mesh.simplify_quadric_decimation(80_000)
        except Exception:
            step = max(1, len(mesh.faces) // 80_000)
            mesh = mesh.submesh([np.arange(0, len(mesh.faces), step)], append=True)
    return mesh


def _draw_mesh(ax, mesh: trimesh.Trimesh, *, color: tuple[float, float, float]) -> None:
    verts = mesh.vertices
    faces = mesh.faces
    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=faces,
        color=color,
        edgecolor="none",
        alpha=1.0,
        shade=True,
    )


def _equal_aspect(ax, mesh: trimesh.Trimesh) -> None:
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) * 0.5
    extent = float(np.max(bounds[1] - bounds[0]) or 1.0) * 0.55
    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] - extent, center[1] + extent)
    ax.set_zlim(center[2] - extent, center[2] + extent)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
