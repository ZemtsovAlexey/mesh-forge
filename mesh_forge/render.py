from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mesh_forge.ops.geometry import read_trimesh

_PREVIEW_FACES = 8_000
_CLAY = (0.77, 0.65, 0.45)
_STUDIO = (0.93, 0.93, 0.91)
_VIEWER_BG = "#100f0d"


def render_mesh_preview(mesh_path: Path, out_path: Path, size: int = 512) -> Path:
    """3/4 view matching the chat MeshViewer: Y-up, object sitting on XZ ground."""
    mesh = _load_render_mesh(mesh_path)
    return _render(
        mesh,
        out_path,
        size=size,
        color=_CLAY,
        background=_VIEWER_BG,
        elev=25,
        azim=45,
        pad=0.04,
        ground=True,
    )


def render_mesh_front_clay(mesh_path: Path, out_path: Path, size: int = 768) -> Path:
    """Orthographic-ish front bake for guided-edit anchors (studio clay look)."""
    mesh = _load_render_mesh(mesh_path)
    return _render(
        mesh,
        out_path,
        size=size,
        color=_STUDIO,
        background="#9a9a9a",
        elev=8,
        azim=0,
        pad=0.02,
    )


def _render(
    mesh: trimesh.Trimesh,
    out_path: Path,
    *,
    size: int,
    color: tuple[float, float, float],
    background: str,
    elev: float,
    azim: float,
    pad: float,
    ground: bool = False,
) -> Path:
    dpi = 100
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    fig.patch.set_facecolor(background)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(background)
    _hide_panes(ax, background)
    if ground:
        _draw_ground(ax, mesh)
    _draw_mesh(ax, mesh, color=color)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim, vertical_axis="y")
    _equal_aspect(ax, mesh)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        bbox_inches="tight",
        pad_inches=pad,
        facecolor=background,
        edgecolor=background,
        transparent=False,
    )
    plt.close(fig)
    return out_path


def _load_render_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = read_trimesh(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if getattr(g, "faces", None) is not None]
        if not geoms:
            raise ValueError(f"No geometry to preview: {mesh_path}")
        mesh = trimesh.util.concatenate(geoms)
    return _simplify(mesh, _PREVIEW_FACES)


def _simplify(mesh: trimesh.Trimesh, face_count: int) -> trimesh.Trimesh:
    if len(getattr(mesh, "faces", [])) == 0:
        return mesh
    try:
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    n = len(mesh.faces)
    if n <= face_count:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=int(face_count))
    except Exception:
        pass
    try:
        percent = min(1.0, max(0.01, float(face_count) / float(n)))
        return mesh.simplify_quadric_decimation(percent=percent)
    except Exception:
        pass
    step = max(1, int(np.ceil(n / face_count)))
    reduced = mesh.submesh([np.arange(0, n, step)], append=True)
    return reduced if reduced is not None else mesh


def _draw_mesh(ax, mesh: trimesh.Trimesh, *, color: tuple[float, float, float]) -> None:
    if len(getattr(mesh, "faces", [])) == 0 or len(getattr(mesh, "vertices", [])) == 0:
        return
    triangles = mesh.vertices[mesh.faces]
    light = np.array([0.4, 0.85, 0.45], dtype=float)
    light /= float(np.linalg.norm(light) or 1.0)
    normals = np.asarray(mesh.face_normals, dtype=float)
    shade = 0.22 + 0.78 * np.clip(normals @ light, 0.0, 1.0)
    rgb = np.clip(np.outer(shade, np.asarray(color, dtype=float)), 0.0, 1.0)
    coll = Poly3DCollection(triangles, linewidths=0, antialiased=False)
    coll.set_facecolor(rgb)
    coll.set_edgecolor("none")
    ax.add_collection3d(coll)


def _hide_panes(ax, background: str) -> None:
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor(background)
            axis.pane.set_edgecolor(background)
            axis.pane.fill = False
            axis.line.set_color(background)
        except Exception:
            pass


def _draw_ground(ax, mesh: trimesh.Trimesh) -> None:
    bounds = np.asarray(mesh.bounds, dtype=float)
    y0 = float(bounds[0][1])
    center = (bounds[0] + bounds[1]) * 0.5
    span = float(np.max(bounds[1] - bounds[0]) or 1.0) * 0.7
    xs = np.linspace(center[0] - span, center[0] + span, 9)
    zs = np.linspace(center[2] - span, center[2] + span, 9)
    color = (0.23, 0.20, 0.17, 0.85)
    for x in xs:
        ax.plot([x, x], [y0, y0], [zs[0], zs[-1]], color=color, linewidth=0.5)
    for z in zs:
        ax.plot([xs[0], xs[-1]], [y0, y0], [z, z], color=color, linewidth=0.5)


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
