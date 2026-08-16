from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from mesh_forge.ops.geometry import load_mesh

logger = logging.getLogger("mesh_forge.render")

_CPU_FACES = 120_000
_CLAY = (0.769, 0.647, 0.455)
_STUDIO = (0.93, 0.93, 0.91)
_VIEWER_BG = "#100f0d"
_FOV_DEG = 45.0
_GL_TO_O3D = np.diag([1.0, -1.0, -1.0, 1.0])
_CAMERAS = {
    "viewer": (0.85, 0.55, 0.85),
    "front": (0.0, 0.18, 1.0),
    "back": (0.0, 0.18, -1.0),
    "left": (-1.0, 0.18, 0.0),
    "right": (1.0, 0.18, 0.0),
    "top": (0.18, 1.0, 0.18),
}
# Look-at as a fraction of seated extent (origin = XZ center, y=0 ground).
_REGIONS = {
    "": (0.0, 0.45, 0.0),
    "center": (0.0, 0.45, 0.0),
    "top": (0.0, 0.82, 0.0),
    "bottom": (0.0, 0.12, 0.0),
    "legs": (0.0, 0.10, 0.0),
    "seat": (0.0, 0.42, 0.0),
    "left": (-0.32, 0.45, 0.0),
    "right": (0.32, 0.45, 0.0),
    "front": (0.0, 0.45, 0.32),
    "back": (0.0, 0.52, -0.32),
    "backrest": (0.0, 0.78, -0.22),
}


def load_render_mesh(mesh_path: Path) -> trimesh.Trimesh:
    """Welded mesh, full resolution — look must match the chat MeshViewer."""
    return load_mesh(mesh_path)


def render_mesh_preview(
    mesh_path: Path,
    out_path: Path,
    size: int = 512,
    *,
    camera: str = "viewer",
    zoom: float = 1.0,
    region: str = "",
    mesh: trimesh.Trimesh | None = None,
) -> Path:
    """Y-up preview. camera: viewer|front|left|right|back|top. zoom>1 closer. region crops look-at."""
    loaded = mesh if mesh is not None else _load_render_mesh(mesh_path)
    return _render(
        loaded,
        out_path,
        size=size,
        color=_CLAY,
        background=_VIEWER_BG,
        camera=camera,
        pad=1.0,
        ground=True,
        zoom=zoom,
        region=region,
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
        camera="front",
        pad=1.04,
        ground=False,
        zoom=1.0,
        region="",
    )


def _render(
    mesh: trimesh.Trimesh,
    out_path: Path,
    *,
    size: int,
    color: tuple[float, float, float],
    background: str,
    camera: str,
    pad: float,
    ground: bool = False,
    zoom: float = 1.0,
    region: str = "",
) -> Path:
    verts, faces, extent = _seat_for_viewer(mesh)
    raster = max(int(size) * 2, 2)
    if len(verts) == 0 or len(faces) == 0:
        image = np.tile(_hex_rgb(background), (raster, raster, 1))
        return _save_png(image, out_path, size=size)
    eye, target = _camera_eye_target(extent, camera, pad=pad, zoom=zoom, region=region)
    try:
        _render_open3d(
            verts,
            faces,
            eye,
            target,
            out_path,
            size=raster,
            color=color,
            background=background,
            ground=ground,
        )
        if raster != size:
            img = Image.open(out_path).convert("RGB")
            img.resize((size, size), Image.Resampling.LANCZOS).save(out_path)
        return out_path
    except Exception as exc:
        logger.warning("Open3D preview failed (%s); using CPU rasterizer", exc)
    verts, faces = _limit_cpu_faces(verts, faces)
    rgb, depth = _project_points(verts, eye, target, raster)
    vert_rgb = _vertex_colors(verts, faces, color)
    image, zbuf = _rasterize(rgb, depth, faces, vert_rgb, raster, _hex_rgb(background))
    if ground:
        _draw_ground(image, zbuf, verts, eye, target, raster)
    return _save_png(image, out_path, size=size)


def _load_render_mesh(mesh_path: Path) -> trimesh.Trimesh:
    return load_mesh(mesh_path)


def _limit_cpu_faces(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Only if the CPU path must run — keep the surface, never stride-subsample."""
    n = len(faces)
    if n <= _CPU_FACES:
        return verts, faces
    try:
        from mesh_forge.ops.geometry import decimate

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        reduced = decimate(mesh, _CPU_FACES)
        if reduced is not None and 4 <= len(reduced.faces) < n:
            return np.asarray(reduced.vertices, dtype=np.float64), np.asarray(reduced.faces, dtype=np.int64)
    except Exception as exc:
        logger.warning("CPU decimate skipped: %s", exc)
    return verts, faces


def _seat_for_viewer(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center on XZ and sit on y=0, matching MeshViewer.fit()."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(verts) == 0:
        return verts, faces, np.ones(3)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center = (bounds[0] + bounds[1]) * 0.5
    extent = np.maximum(bounds[1] - bounds[0], 1e-9)
    seated = verts - center
    seated[:, 1] += extent[1] * 0.5
    return seated, faces, extent


def _camera_eye_target(
    extent: np.ndarray,
    camera: str,
    *,
    pad: float,
    zoom: float = 1.0,
    region: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    max_dim = float(np.max(extent) or 1.0)
    zoom = max(1.0, min(float(zoom or 1.0), 4.0))
    dist = max_dim * 2.2 * float(pad) / zoom
    key = (region or "").strip().lower()
    frac = _REGIONS.get(key, _REGIONS[""])
    target = np.array(
        [frac[0] * extent[0], frac[1] * extent[1], frac[2] * extent[2]],
        dtype=np.float64,
    )
    direction = np.array(_CAMERAS.get((camera or "viewer").strip().lower(), _CAMERAS["viewer"]), dtype=np.float64)
    direction /= float(np.linalg.norm(direction) or 1.0)
    eye = target + direction * dist
    return eye, target


def _render_open3d(
    verts: np.ndarray,
    faces: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    out_path: Path,
    *,
    size: int,
    color: tuple[float, float, float],
    background: str,
    ground: bool,
) -> Path:
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(faces),
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(list(color))
    vis = o3d.visualization.Visualizer()
    if not vis.create_window(width=int(size), height=int(size), visible=False):
        raise RuntimeError("Open3D create_window failed")
    try:
        vis.add_geometry(mesh)
        if ground:
            vis.add_geometry(_ground_lineset(verts))
        opt = vis.get_render_option()
        opt.background_color = _hex_rgb(background)[:3]
        opt.mesh_show_back_face = True
        opt.light_on = True
        vis.poll_events()
        vis.update_renderer()
        ctr = vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        fov = np.deg2rad(_FOV_DEG)
        fx = float(size) / (2.0 * np.tan(fov * 0.5))
        cx = float(size) * 0.5 - 0.5
        params.intrinsic.set_intrinsics(int(size), int(size), fx, fx, cx, cx)
        params.extrinsic = _GL_TO_O3D @ _look_at(eye, target, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        if not ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True):
            raise RuntimeError("Open3D camera rejected")
        vis.poll_events()
        vis.update_renderer()
        buf = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        if buf.size == 0 or float(np.max(buf)) < 0.02:
            raise RuntimeError("Open3D capture empty")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.clip(buf * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(pixels, mode="RGB").save(out_path)
    finally:
        vis.destroy_window()
    if not out_path.is_file() or out_path.stat().st_size < 64:
        raise RuntimeError("Open3D wrote an empty preview")
    return out_path


def _ground_lineset(verts: np.ndarray):
    import open3d as o3d

    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    center = (lo + hi) * 0.5
    span = float(np.max(hi - lo) or 1.0) * 0.7
    xs = np.linspace(center[0] - span, center[0] + span, 9)
    zs = np.linspace(center[2] - span, center[2] + span, 9)
    points: list[list[float]] = []
    lines: list[list[int]] = []
    for x in xs:
        i = len(points)
        points.extend([[x, 0.0, zs[0]], [x, 0.0, zs[-1]]])
        lines.append([i, i + 1])
    for z in zs:
        i = len(points)
        points.extend([[xs[0], 0.0, z], [xs[-1], 0.0, z]])
        lines.append([i, i + 1])
    geom = o3d.geometry.LineSet()
    geom.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    geom.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    geom.paint_uniform_color([0.23, 0.20, 0.17])
    return geom


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    zaxis = eye - target
    zaxis /= float(np.linalg.norm(zaxis) or 1.0)
    xaxis = np.cross(up, zaxis)
    xnorm = float(np.linalg.norm(xaxis))
    if xnorm < 1e-12:
        xaxis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        xaxis /= xnorm
    yaxis = np.cross(zaxis, xaxis)
    view = np.eye(4, dtype=np.float64)
    view[0, :3] = xaxis
    view[1, :3] = yaxis
    view[2, :3] = zaxis
    view[:3, 3] = -np.array([xaxis, yaxis, zaxis], dtype=np.float64) @ eye
    return view


def _project_points(
    verts: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    view = _look_at(eye, target, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    hom = np.concatenate([verts, np.ones((len(verts), 1), dtype=np.float64)], axis=1)
    cam = (view @ hom.T).T
    depth = np.maximum(-cam[:, 2], 1e-6)
    f = 1.0 / np.tan(np.deg2rad(_FOV_DEG) * 0.5)
    ndc_x = (f * cam[:, 0]) / depth
    ndc_y = (f * cam[:, 1]) / depth
    px = (ndc_x * 0.5 + 0.5) * (size - 1)
    py = (1.0 - (ndc_y * 0.5 + 0.5)) * (size - 1)
    return np.stack([px, py], axis=1), depth


def _vertex_colors(
    verts: np.ndarray,
    faces: np.ndarray,
    color: tuple[float, float, float],
) -> np.ndarray:
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(verts)
    np.add.at(vn, faces[:, 0], fn)
    np.add.at(vn, faces[:, 1], fn)
    np.add.at(vn, faces[:, 2], fn)
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = np.divide(vn, np.maximum(norms, 1e-12))
    key = np.array([4.0, 6.0, 3.0], dtype=np.float64)
    key /= float(np.linalg.norm(key) or 1.0)
    fill = np.array([-3.0, 2.0, -2.0], dtype=np.float64)
    fill /= float(np.linalg.norm(fill) or 1.0)
    shade = 0.20 + 0.70 * np.clip(np.abs(vn @ key), 0.0, 1.0) + 0.16 * np.clip(np.abs(vn @ fill), 0.0, 1.0)
    return np.clip(np.outer(shade, np.asarray(color, dtype=np.float64)), 0.0, 1.0)


def _rasterize(
    xy: np.ndarray,
    depth: np.ndarray,
    faces: np.ndarray,
    vert_rgb: np.ndarray,
    size: int,
    background: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    image = np.broadcast_to(background, (size, size, 3)).copy()
    zbuf = np.full((size, size), np.inf, dtype=np.float32)
    pts = xy[faces]
    z = depth[faces]
    colors = vert_rgb[faces]
    behind = np.all(z <= 1e-5, axis=1)
    for i in range(len(faces)):
        if behind[i]:
            continue
        _fill_triangle(image, zbuf, pts[i], z[i], colors[i], size)
    return image, zbuf


def _fill_triangle(
    image: np.ndarray,
    zbuf: np.ndarray,
    pts: np.ndarray,
    z: np.ndarray,
    rgb: np.ndarray,
    size: int,
) -> None:
    minx = int(np.floor(pts[:, 0].min()))
    maxx = int(np.ceil(pts[:, 0].max()))
    miny = int(np.floor(pts[:, 1].min()))
    maxy = int(np.ceil(pts[:, 1].max()))
    if maxx < 0 or maxy < 0 or minx >= size or miny >= size:
        return
    minx = max(minx, 0)
    miny = max(miny, 0)
    maxx = min(maxx, size - 1)
    maxy = min(maxy, size - 1)
    if minx > maxx or miny > maxy:
        return
    a, b, c = pts
    area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
    if abs(area) < 1e-8:
        return
    ys, xs = np.mgrid[miny : maxy + 1, minx : maxx + 1]
    px = xs.astype(np.float64) + 0.5
    py = ys.astype(np.float64) + 0.5
    w0 = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
    w1 = (c[0] - b[0]) * (py - b[1]) - (c[1] - b[1]) * (px - b[0])
    w2 = (a[0] - c[0]) * (py - c[1]) - (a[1] - c[1]) * (px - c[0])
    if area < 0:
        mask = (w0 <= 0) & (w1 <= 0) & (w2 <= 0)
    else:
        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    if not np.any(mask):
        return
    b0 = ((c[0] - b[0]) * (py - b[1]) - (c[1] - b[1]) * (px - b[0])) / area
    b1 = ((a[0] - c[0]) * (py - c[1]) - (a[1] - c[1]) * (px - c[0])) / area
    b2 = 1.0 - b0 - b1
    inv_z = b0 / np.maximum(z[0], 1e-6) + b1 / np.maximum(z[1], 1e-6) + b2 / np.maximum(z[2], 1e-6)
    pix_z = 1.0 / np.maximum(inv_z, 1e-9)
    closer = mask & (pix_z < zbuf[miny : maxy + 1, minx : maxx + 1])
    if not np.any(closer):
        return
    zbuf[miny : maxy + 1, minx : maxx + 1][closer] = pix_z[closer]
    if rgb.ndim == 1:
        pix = rgb
    else:
        pix = b0[..., None] * rgb[0] + b1[..., None] * rgb[1] + b2[..., None] * rgb[2]
        pix = pix[closer]
    image[miny : maxy + 1, minx : maxx + 1][closer] = pix


def _draw_ground(
    image: np.ndarray,
    zbuf: np.ndarray,
    verts: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    size: int,
) -> None:
    if len(verts) == 0:
        return
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    center = (lo + hi) * 0.5
    span = float(np.max(hi - lo) or 1.0) * 0.7
    xs = np.linspace(center[0] - span, center[0] + span, 9)
    zs = np.linspace(center[2] - span, center[2] + span, 9)
    y0 = 0.0
    color = np.array([0.23, 0.20, 0.17], dtype=np.float64)
    lines: list[np.ndarray] = []
    for x in xs:
        lines.append(np.array([[x, y0, zs[0]], [x, y0, zs[-1]]], dtype=np.float64))
    for z in zs:
        lines.append(np.array([[xs[0], y0, z], [xs[-1], y0, z]], dtype=np.float64))
    pts = np.concatenate(lines, axis=0)
    xy, depth = _project_points(pts, eye, target, size)
    for i in range(0, len(pts), 2):
        _draw_line(image, zbuf, xy[i], xy[i + 1], depth[i], depth[i + 1], color, size)


def _draw_line(
    image: np.ndarray,
    zbuf: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    z0: float,
    z1: float,
    color: np.ndarray,
    size: int,
) -> None:
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]), 1.0)) + 1
    ts = np.linspace(0.0, 1.0, n)
    xs = p0[0] + (p1[0] - p0[0]) * ts
    ys = p0[1] + (p1[1] - p0[1]) * ts
    zs = z0 + (z1 - z0) * ts
    ix = np.rint(xs).astype(np.int32)
    iy = np.rint(ys).astype(np.int32)
    on = (ix >= 0) & (ix < size) & (iy >= 0) & (iy < size) & (zs > 1e-5)
    if not np.any(on):
        return
    ix, iy, zs = ix[on], iy[on], zs[on]
    closer = zs < zbuf[iy, ix]
    if not np.any(closer):
        return
    ix, iy = ix[closer], iy[closer]
    zbuf[iy, ix] = zs[closer]
    image[iy, ix] = color


def _hex_rgb(value: str) -> np.ndarray:
    raw = value.lstrip("#")
    if len(raw) != 6:
        return np.array([0.06, 0.06, 0.05], dtype=np.float64)
    r = int(raw[0:2], 16) / 255.0
    g = int(raw[2:4], 16) / 255.0
    b = int(raw[4:6], 16) / 255.0
    return np.array([r, g, b], dtype=np.float64)


def _save_png(image: np.ndarray, out_path: Path, *, size: int | None = None) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels, mode="RGB")
    if size is not None and img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    img.save(out_path)
    return out_path
