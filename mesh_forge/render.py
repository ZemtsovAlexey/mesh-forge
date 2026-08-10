from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh


def render_mesh_preview(mesh_path: Path, out_path: Path, size: int = 512) -> Path:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    fig = plt.figure(figsize=(4, 4), dpi=size // 4)
    ax = fig.add_subplot(111, projection="3d")
    verts = mesh.vertices
    faces = mesh.faces
    ax.plot_trisurf(
        verts[:, 0], verts[:, 1], verts[:, 2],
        triangles=faces, color=(0.7, 0.75, 0.85), edgecolor="none", alpha=0.95,
    )
    ax.set_axis_off()
    ax.view_init(elev=25, azim=45)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, facecolor="#1a1a2e")
    plt.close(fig)
    return out_path
