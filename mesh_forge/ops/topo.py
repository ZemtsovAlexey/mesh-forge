from __future__ import annotations

from typing import Any, Literal

import numpy as np
import trimesh

from mesh_forge.ops.region import pick_world_point, _connected_in_mask

ElemKind = Literal["vertex", "edge", "face"]


class TopoError(ValueError):
    """Bad vertex / edge / face target."""


def empty_topo() -> dict[str, Any]:
    return {"kind": "face", "vertex": -1, "face": -1, "edge": [], "mesh": "", "hops": 12}


def topo_valid(topo: dict[str, Any] | None) -> bool:
    if not topo:
        return False
    return int(topo.get("face", -1)) >= 0 or int(topo.get("vertex", -1)) >= 0


def parse_edge(edge: str | list[int] | tuple[int, ...] | None) -> list[int]:
    if edge is None or edge == "" or edge == []:
        return []
    if isinstance(edge, (list, tuple)) and len(edge) >= 2:
        return [int(edge[0]), int(edge[1])]
    raw = str(edge).strip().replace(",", "-").replace(" ", "")
    if "-" not in raw:
        raise TopoError("edge must be two vertex ids like 12-34.")
    a, b = raw.split("-", 1)
    return [int(a), int(b)]


def parse_kind(kind: str | None) -> ElemKind:
    raw = (kind or "face").strip().lower()
    aliases = {
        "face": "face",
        "polygon": "face",
        "poly": "face",
        "грань": "face",
        "полигон": "face",
        "vertex": "vertex",
        "vert": "vertex",
        "point": "vertex",
        "точка": "vertex",
        "вершина": "vertex",
        "edge": "edge",
        "ребро": "edge",
    }
    mapped = aliases.get(raw, raw)
    if mapped not in {"vertex", "edge", "face"}:
        raise TopoError("elem must be vertex, edge, or face.")
    return mapped  # type: ignore[return-value]


def _normalize_point(mesh: trimesh.Trimesh, point: np.ndarray) -> list[float]:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    span = np.maximum(bounds[1] - bounds[0], 1e-9)
    n = (np.asarray(point, dtype=np.float64) - bounds[0]) / span
    return [float(np.clip(n[0], 0, 1)), float(np.clip(n[1], 0, 1)), float(np.clip(n[2], 0, 1))]


def _element_point(mesh: trimesh.Trimesh, topo: dict[str, Any]) -> np.ndarray:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    kind = parse_kind(str(topo.get("kind") or "face"))
    if kind == "vertex":
        i = int(topo["vertex"])
        return verts[i]
    if kind == "edge":
        a, b = int(topo["edge"][0]), int(topo["edge"][1])
        return 0.5 * (verts[a] + verts[b])
    f = int(topo["face"])
    return verts[faces[f]].mean(axis=0)


def hit_topology(
    mesh: trimesh.Trimesh,
    nx: float,
    ny: float,
    nz: float,
    *,
    kind: str | None = None,
) -> dict[str, Any]:
    """Nearest face / vertex / edge on the mesh to a normalized AABB click."""
    if mesh is None or int(len(mesh.faces)) == 0:
        raise TopoError("Mesh has no faces.")
    kind_name = parse_kind(kind)
    point = pick_world_point(mesh, nx, ny, nz)
    face = _closest_face(mesh, point)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tri = faces[face]
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    corners = verts[tri]
    vertex = int(tri[int(np.argmin(np.linalg.norm(corners - point, axis=1)))])
    edge = _closest_edge(tri, corners, point)
    topo = {
        "kind": kind_name,
        "vertex": vertex,
        "face": int(face),
        "edge": edge,
    }
    snapped = _normalize_point(mesh, _element_point(mesh, topo))
    topo["nx"], topo["ny"], topo["nz"] = snapped
    return topo


def topology_from_ids(
    mesh: trimesh.Trimesh,
    *,
    kind: str | None = None,
    vertex: int | None = None,
    face: int | None = None,
    edge: str | list[int] | None = None,
) -> dict[str, Any]:
    kind_name = parse_kind(kind)
    n_v = int(len(mesh.vertices))
    n_f = int(len(mesh.faces))
    faces = np.asarray(mesh.faces, dtype=np.int64)
    topo = empty_topo()
    topo["kind"] = kind_name
    if face is not None and int(face) >= 0:
        f = int(face)
        if f >= n_f:
            raise TopoError(f"face {f} is out of range (0..{n_f - 1}).")
        topo["face"] = f
        tri = [int(x) for x in faces[f]]
        topo["vertex"] = tri[0]
        topo["edge"] = [tri[0], tri[1]]
    if vertex is not None and int(vertex) >= 0:
        v = int(vertex)
        if v >= n_v:
            raise TopoError(f"vertex {v} is out of range (0..{n_v - 1}).")
        topo["vertex"] = v
        if topo["face"] < 0:
            hits = np.where(np.any(faces == v, axis=1))[0]
            if len(hits) == 0:
                raise TopoError(f"vertex {v} is not on any face.")
            topo["face"] = int(hits[0])
            others = [int(x) for x in faces[topo["face"]] if int(x) != v]
            topo["edge"] = [v, others[0]] if others else [v, v]
    pair = parse_edge(edge)
    if pair:
        a, b = pair
        if a >= n_v or b >= n_v or a < 0 or b < 0:
            raise TopoError(f"edge {a}-{b} is out of range.")
        topo["edge"] = [a, b]
        topo["vertex"] = a
        if topo["face"] < 0:
            hits = np.where(np.any(faces == a, axis=1) & np.any(faces == b, axis=1))[0]
            if len(hits) == 0:
                raise TopoError(f"edge {a}-{b} is not on any face.")
            topo["face"] = int(hits[0])
    if not topo_valid(topo):
        raise TopoError("Need face, vertex, or edge.")
    snapped = _normalize_point(mesh, _element_point(mesh, topo))
    topo["nx"], topo["ny"], topo["nz"] = snapped
    return topo


def face_mask_for_topo(mesh: trimesh.Trimesh, topo: dict[str, Any]) -> np.ndarray:
    n = int(len(mesh.faces))
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    kind = parse_kind(str(topo.get("kind") or "face"))
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if kind == "face":
        f = int(topo.get("face", -1))
        if 0 <= f < n:
            mask[f] = True
    elif kind == "vertex":
        v = int(topo.get("vertex", -1))
        if v >= 0:
            mask = np.any(faces == v, axis=1)
    else:
        pair = list(topo.get("edge") or [])
        if len(pair) >= 2:
            a, b = int(pair[0]), int(pair[1])
            mask = np.any(faces == a, axis=1) & np.any(faces == b, axis=1)
    hops = int(topo.get("hops", 12) or 0)
    if hops <= 0 or not np.any(mask):
        return mask
    return grow_face_patch(mesh, mask, hops=hops)


def grow_face_patch(
    mesh: trimesh.Trimesh,
    seed: np.ndarray,
    *,
    hops: int = 12,
    max_frac: float = 0.025,
    max_faces: int = 800,
) -> np.ndarray:
    """Expand seed faces by adjacency, but never a large slab of the model."""
    n = int(len(mesh.faces))
    out = np.asarray(seed, dtype=bool).copy()
    if n == 0 or not np.any(out):
        return out
    allowed = np.ones(n, dtype=bool)
    limit = max(1, min(int(hops), 24))
    cap = max(16, min(int(max_faces), int(max_frac * n)))
    grown = np.zeros(n, dtype=bool)
    for seed_i in np.where(out)[0][:12]:
        grown |= _connected_in_mask(mesh, int(seed_i), allowed, max_hops=limit)
        if int(grown.sum()) >= cap:
            break
    if int(grown.sum()) > cap:
        hops_try = limit
        while hops_try > 1 and int(grown.sum()) > cap:
            hops_try -= 1
            grown = np.zeros(n, dtype=bool)
            for seed_i in np.where(out)[0][:4]:
                grown |= _connected_in_mask(mesh, int(seed_i), allowed, max_hops=hops_try)
        if int(grown.sum()) > cap:
            return out
    return grown if np.any(grown) else out


def grow_visible_lump(
    mesh: trimesh.Trimesh,
    seed_face: int,
    *,
    seated_verts: np.ndarray,
    seated_faces: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Whole island or protrusion under the aim, stopping at the neck to the body."""
    n = int(len(mesh.faces))
    out = np.zeros(n, dtype=bool)
    seed = int(seed_face)
    if n == 0 or seed < 0 or seed >= n:
        return out
    faces = np.asarray(seated_faces, dtype=np.int64)
    verts = np.asarray(seated_verts, dtype=np.float64)
    if len(faces) != n:
        return out
    nbrs = _face_neighbors(mesh, n, verts=verts, faces=faces)
    if not nbrs or not any(nbrs):
        out[seed] = True
        return out
    component = _flood_faces(nbrs, seed)
    n_comp = int(component.sum())
    # A real disconnected island (the extra bit) — keep it, even if remesh
    # made the flap only a handful of triangles. One isolated face is a miss.
    if 6 <= n_comp < n and n_comp <= _extra_bit_cap(n):
        return component
    centers = verts[faces].mean(axis=1)
    body = verts.mean(axis=0)
    axis = _outward_axis(verts, faces, seed)
    inward = float(np.percentile((verts - body) @ axis, 55.0))
    ear = grow_ear_from_seed(verts, faces, nbrs, seed, centers, body)
    candidates: list[np.ndarray] = []
    for cand in (
        ear,
        _protrusion_from_seed(verts, faces, nbrs, seed, eye=eye, target=target),
        _visible_bfs_from_seed(verts, faces, nbrs, seed, eye, target, n),
        _face_flood_above(nbrs, seed, centers, body, axis, inward, n),
        _view_shell_lump(mesh, verts, faces, nbrs, seed, eye, target, band_frac=0.06),
    ):
        if np.any(cand) and cand[seed]:
            candidates.append(cand)
    if not candidates:
        out[seed] = True
        return out
    return _pick_visible_lump(mesh, candidates, centers, body, n, seed)


def _pick_visible_lump(
    mesh: trimesh.Trimesh,
    candidates: list[np.ndarray],
    centers: np.ndarray,
    body: np.ndarray,
    n: int,
    seed: int,
) -> np.ndarray:
    scored: list[tuple[float, np.ndarray]] = []
    for cand in candidates:
        score = _lump_outward_score(cand, centers, body, n, seed)
        if score >= 0.0:
            scored.append((score, cand))
    if scored:
        return max(scored, key=lambda item: item[0])[1]
    cap = _extra_bit_cap(n)
    min_k = _extra_bit_min(n)
    medium = [c for c in candidates if min_k <= int(c.sum()) <= cap]
    if medium:
        return min(medium, key=lambda mask: int(mask.sum()))
    local = grow_face_patch(
        mesh,
        np.array([seed], dtype=bool),
        hops=16,
        max_frac=0.02,
        max_faces=cap,
    )
    if int(local.sum()) >= min_k:
        return local
    rest = [c for c in candidates if int(c.sum()) < n]
    if rest:
        return min(rest, key=lambda mask: int(mask.sum()))
    out = np.zeros(n, dtype=bool)
    if 0 <= seed < n:
        out[seed] = True
    return out


def _extra_bit_cap(n: int) -> int:
    """Hard ceiling for an extra bit — a skirt panel is not a petal."""
    return int(min(8000, max(48, int(0.045 * max(n, 1)))))


def _extra_bit_min(n: int) -> int:
    """Small remeshed flaps are a few triangles; 24 was skipping the whole petal."""
    return int(min(6, max(3, int(0.0003 * max(n, 1)))))


def mask_is_tiny(mask: np.ndarray, verts: np.ndarray, faces: np.ndarray) -> bool:
    """True when red would be invisible: a crumb, not a small remeshed flap."""
    mask = np.asarray(mask, dtype=bool)
    k = int(mask.sum())
    if k <= 0:
        return True
    if k >= 8:
        return False
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) != len(mask):
        return k < 6
    tri = verts[faces]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cr, axis=1)
    mesh_area = float(np.sum(areas)) or 1.0
    frac = float(np.sum(areas[mask])) / mesh_area
    return frac < 0.0015


def grow_ear_from_seed(
    verts: np.ndarray,
    faces: np.ndarray,
    nbrs: list[list[int]],
    seed: int,
    centers: np.ndarray,
    body: np.ndarray,
) -> np.ndarray:
    """Grow a thin sticking-out ear and stop at the neck into the body/skirt."""
    n = int(len(faces))
    out = np.zeros(n, dtype=bool)
    if n == 0 or seed < 0 or seed >= n:
        return out
    axis = _outward_axis(verts, faces, seed)
    along = (centers - body) @ axis
    tip = float(along[seed])
    spread = float(np.max(along) - np.min(along)) or 1.0
    perp = np.linalg.norm(centers - body - along[:, None] * axis, axis=1)
    mesh_w = float(np.max(np.max(verts, axis=0) - np.min(verts, axis=0))) or 1.0
    cap = _extra_bit_cap(n)
    min_k = _extra_bit_min(n)
    prev: np.ndarray | None = None
    prev_w = 0.0
    ts = np.linspace(tip, tip - 0.24 * spread, 20)
    for t in ts:
        mask = _face_flood_above(nbrs, seed, centers, body, axis, float(t), n)
        k = int(mask.sum())
        if k < min_k:
            continue
        w = float(np.percentile(perp[mask], 88))
        if prev is not None and int(prev.sum()) >= min_k:
            if w > 1.75 * max(prev_w, 1e-9) or k > 2.8 * max(int(prev.sum()), 1):
                return prev
        if k > cap:
            return prev if prev is not None and int(prev.sum()) >= min_k else mask
        if prev is None and w > 0.20 * mesh_w:
            return out
        prev = mask
        prev_w = w
    return prev if prev is not None else out


def _lump_is_slab(mask: np.ndarray, centers: np.ndarray, body: np.ndarray, seed: int, n: int) -> bool:
    k = int(mask.sum())
    if k < _extra_bit_min(n):
        return False
    if k > _extra_bit_cap(n):
        return True
    axis = centers[seed] - body
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return True
    axis = axis / norm
    pts = centers[mask]
    along = (pts - body) @ axis
    perp = np.linalg.norm(pts - body - along[:, None] * axis, axis=1)
    length = float(np.max(along) - np.min(along))
    width = float(np.percentile(perp, 88))
    return width > 1.45 * max(length, 1e-9)


def erode_face_mask(mesh: trimesh.Trimesh, mask: np.ndarray, hops: int = 1) -> np.ndarray:
    """Peel the mask boundary so a skirt panel shrinks toward the extra bit."""
    n = int(len(mesh.faces))
    out = np.asarray(mask, dtype=bool).copy()
    if out.shape[0] != n or not np.any(out):
        return out
    nbrs = _face_neighbors(mesh, n)
    min_k = _extra_bit_min(n)
    steps = max(1, min(int(hops), 8))
    for _ in range(steps):
        nxt = out.copy()
        for i in np.flatnonzero(out):
            row = nbrs[int(i)]
            if not row or any(not out[int(j)] for j in row):
                nxt[int(i)] = False
        if int(nxt.sum()) < min_k:
            break
        out = nxt
    return out


def dilate_face_mask(
    mesh: trimesh.Trimesh,
    mask: np.ndarray,
    hops: int = 1,
    *,
    seated_verts: np.ndarray | None = None,
    seated_faces: np.ndarray | None = None,
) -> np.ndarray:
    """Grow the mask by face adjacency, staying under the extra-bit cap."""
    n = int(len(mesh.faces))
    out = np.asarray(mask, dtype=bool).copy()
    if out.shape[0] != n or not np.any(out):
        return out
    verts = seated_verts if seated_verts is not None else np.asarray(mesh.vertices, dtype=np.float64)
    faces = seated_faces if seated_faces is not None else np.asarray(mesh.faces, dtype=np.int64)
    nbrs = _face_neighbors(mesh, n, verts=verts, faces=faces)
    cap = _extra_bit_cap(n)
    steps = max(1, min(int(hops), 8))
    for _ in range(steps):
        extra: list[int] = []
        for i in np.flatnonzero(out):
            extra.extend(int(j) for j in nbrs[int(i)] if not out[int(j)])
        if not extra:
            break
        nxt = out.copy()
        nxt[np.asarray(extra, dtype=np.int64)] = True
        if int(nxt.sum()) > cap:
            break
        out = nxt
    return out


def keep_outward_blob(
    mesh: trimesh.Trimesh,
    mask: np.ndarray,
    *,
    seated_verts: np.ndarray,
    seated_faces: np.ndarray,
) -> np.ndarray:
    """If red split into several islands, keep the one farthest from the body."""
    n = int(len(mesh.faces))
    out = np.asarray(mask, dtype=bool).copy()
    if out.shape[0] != n or not np.any(out):
        return out
    verts = np.asarray(seated_verts, dtype=np.float64)
    faces = np.asarray(seated_faces, dtype=np.int64)
    nbrs = _face_neighbors(mesh, n, verts=verts, faces=faces)
    centers = verts[faces].mean(axis=1)
    body = verts.mean(axis=0)
    seen = np.zeros(n, dtype=bool)
    best: np.ndarray | None = None
    best_score = -1.0
    for seed in np.flatnonzero(out):
        if seen[int(seed)]:
            continue
        blob = _keep_reachable(out, nbrs, np.array([int(seed)], dtype=np.int64))
        seen[blob] = True
        k = int(blob.sum())
        if k < _extra_bit_min(n):
            continue
        score = float(np.linalg.norm(centers[blob].mean(axis=0) - body))
        if score > best_score:
            best_score = score
            best = blob
    return best if best is not None else out


def silhouette_spike_face(
    verts: np.ndarray,
    faces: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    *,
    side: str = "right",
    y_lo: float = 0.0,
    y_hi: float = 100.0,
) -> int:
    """Tip of a sharp silhouette ear, or -1 if the outline is a smooth/boxy body."""
    from mesh_forge.render import _project_points

    faces = np.asarray(faces, dtype=np.int64)
    verts = np.asarray(verts, dtype=np.float64)
    n = int(len(faces))
    if n == 0:
        return -1
    centers = verts[faces].mean(axis=1)
    size = 1000
    xy, depth = _project_points(centers, eye, target, size)
    in_frame = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < size)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < size)
        & (depth > 1e-4)
    )
    if not np.any(in_frame):
        return -1
    occ = xy[in_frame, 1]
    y0 = float(np.percentile(occ, max(0.0, min(100.0, y_lo))))
    y1 = float(np.percentile(occ, max(0.0, min(100.0, y_hi))))
    band = in_frame & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    if not np.any(band):
        band = in_frame
    rows = 64
    row = np.clip(np.floor(xy[:, 1] / float(size) * rows).astype(np.int32), 0, rows - 1)
    side = (side or "right").strip().lower()
    rightward = side not in {"left", "лево", "слева"}
    row_ext = np.full(rows, -np.inf if rightward else np.inf, dtype=np.float64)
    if rightward:
        np.maximum.at(row_ext, row[band], xy[band, 0])
    else:
        np.minimum.at(row_ext, row[band], xy[band, 0])
    occ_rows = np.where(np.isfinite(row_ext))[0]
    if len(occ_rows) < 6:
        return -1
    smooth = np.full(rows, np.nan, dtype=np.float64)
    for r in occ_rows:
        window = row_ext[max(0, r - 8) : r + 9]
        window = window[np.isfinite(window)]
        if len(window):
            smooth[r] = float(np.median(window))
    spike = np.zeros(rows, dtype=np.float64)
    valid = np.isfinite(row_ext) & np.isfinite(smooth)
    if rightward:
        spike[valid] = row_ext[valid] - smooth[valid]
    else:
        spike[valid] = smooth[valid] - row_ext[valid]
    lo = occ_rows[int(0.14 * len(occ_rows))]
    hi = occ_rows[int(0.86 * len(occ_rows))]
    if y_lo >= 35.0:
        lo = occ_rows[int(0.06 * len(occ_rows))]
        hi = occ_rows[-1]
    spike[:lo] = 0.0
    spike[hi + 1 :] = 0.0
    peak = float(np.max(spike)) if spike.size else 0.0
    thresh = 0.022 * float(size)
    if peak < thresh:
        return -1
    flags = spike >= max(thresh, 0.45 * peak)
    run = 0
    best_run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best_run = max(best_run, run)
    if best_run < 3:
        return -1
    on_spike = band & flags[row]
    score = np.where(on_spike, xy[:, 0] if rightward else -xy[:, 0], -np.inf)
    if not np.any(np.isfinite(score)):
        return -1
    return int(np.argmax(score))


def silhouette_lobe_face(
    verts: np.ndarray,
    faces: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    *,
    side: str = "right",
    y_lo: float = 0.0,
    y_hi: float = 100.0,
) -> int:
    """Wide hanging flap on the outline — not a thin ear, still not the whole skirt."""
    from mesh_forge.render import _project_points

    faces = np.asarray(faces, dtype=np.int64)
    verts = np.asarray(verts, dtype=np.float64)
    n = int(len(faces))
    if n == 0:
        return -1
    centers = verts[faces].mean(axis=1)
    size = 1000
    xy, depth = _project_points(centers, eye, target, size)
    in_frame = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < size)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < size)
        & (depth > 1e-4)
    )
    if not np.any(in_frame):
        return -1
    occ = xy[in_frame, 1]
    y0 = float(np.percentile(occ, max(0.0, min(100.0, y_lo))))
    y1 = float(np.percentile(occ, max(0.0, min(100.0, y_hi))))
    band = in_frame & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    if not np.any(band):
        band = in_frame
    rows = 64
    row = np.clip(np.floor(xy[:, 1] / float(size) * rows).astype(np.int32), 0, rows - 1)
    side = (side or "right").strip().lower()
    rightward = side not in {"left", "лево", "слева"}
    row_ext = np.full(rows, -np.inf if rightward else np.inf, dtype=np.float64)
    if rightward:
        np.maximum.at(row_ext, row[band], xy[band, 0])
    else:
        np.minimum.at(row_ext, row[band], xy[band, 0])
    occ_rows = np.where(np.isfinite(row_ext))[0]
    if len(occ_rows) < 6:
        return -1
    hull = np.full(rows, np.nan, dtype=np.float64)
    for r in occ_rows:
        window = row_ext[max(0, r - 20) : r + 21]
        window = window[np.isfinite(window)]
        if len(window) >= 5:
            hull[r] = float(np.percentile(window, 30.0 if rightward else 70.0))
    lobe = np.zeros(rows, dtype=np.float64)
    valid = np.isfinite(row_ext) & np.isfinite(hull)
    if rightward:
        lobe[valid] = row_ext[valid] - hull[valid]
    else:
        lobe[valid] = hull[valid] - row_ext[valid]
    lo = occ_rows[int(0.12 * len(occ_rows))]
    hi = occ_rows[int(0.88 * len(occ_rows))]
    if y_lo >= 35.0:
        lo = occ_rows[int(0.06 * len(occ_rows))]
        hi = occ_rows[-1]
    lobe[:lo] = 0.0
    lobe[hi + 1 :] = 0.0
    peak = float(np.max(lobe)) if lobe.size else 0.0
    thresh = 0.016 * float(size)
    if peak < thresh:
        return -1
    flags = lobe >= max(thresh, 0.40 * peak)
    run = 0
    best_run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best_run = max(best_run, run)
    if best_run < 3:
        return -1
    on_lobe = band & flags[row]
    score = np.where(on_lobe, xy[:, 0] if rightward else -xy[:, 0], -np.inf)
    if not np.any(np.isfinite(score)):
        return -1
    return int(np.argmax(score))


def silhouette_extra_face(
    verts: np.ndarray,
    faces: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    *,
    side: str = "right",
    y_lo: float = 0.0,
    y_hi: float = 100.0,
) -> int:
    """Thin ear or a wide hanging flap on the silhouette, else -1."""
    seed = silhouette_spike_face(
        verts, faces, eye, target, side=side, y_lo=y_lo, y_hi=y_hi
    )
    if seed >= 0:
        return seed
    return silhouette_lobe_face(
        verts, faces, eye, target, side=side, y_lo=y_lo, y_hi=y_hi
    )


def mask_silhouette_cameras(describe: str, current: str = "") -> list[str]:
    """Cameras where a left/right extra bit sticks out of the outline, not face-on."""
    side = mask_aim_side(describe, None)
    cam = (current or "").split(",")[0].strip().lower()
    if side in {"left", "right"}:
        ordered = ["front", "viewer"]
        if cam and cam not in ordered and cam in {"front", "left", "right", "back", "top", "viewer"}:
            ordered.append(cam)
        return ordered
    return [cam] if cam in {"front", "left", "right", "back", "top", "viewer"} else ["front"]


def silhouette_extreme_face(
    verts: np.ndarray,
    faces: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    *,
    side: str = "right",
    y_lo: float = 0.0,
    y_hi: float = 100.0,
) -> int:
    """Face on the look-frame silhouette in a screen direction (not nearest-to-empty-pixel)."""
    from mesh_forge.render import _project_points

    faces = np.asarray(faces, dtype=np.int64)
    verts = np.asarray(verts, dtype=np.float64)
    n = int(len(faces))
    if n == 0:
        return -1
    centers = verts[faces].mean(axis=1)
    size = 1000
    xy, depth = _project_points(centers, eye, target, size)
    in_frame = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < size)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < size)
        & (depth > 1e-4)
    )
    if not np.any(in_frame):
        return int(np.argmax(centers[:, 0]))
    occ = xy[in_frame, 1]
    y0 = float(np.percentile(occ, max(0.0, min(100.0, y_lo))))
    y1 = float(np.percentile(occ, max(0.0, min(100.0, y_hi))))
    band = in_frame & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    if not np.any(band):
        band = in_frame
    rows = 48
    row = np.clip(np.floor(xy[:, 1] / float(size) * rows).astype(np.int32), 0, rows - 1)
    margin = 0.045 * float(size)
    side = (side or "right").strip().lower()
    if side in {"left", "лево", "слева"}:
        row_ext = np.full(rows, np.inf, dtype=np.float64)
        np.minimum.at(row_ext, row[band], xy[band, 0])
        on_edge = band & (xy[:, 0] <= row_ext[row] + margin)
        score = np.where(on_edge, -xy[:, 0], -np.inf)
    elif side in {"top", "верх", "сверху"}:
        col = np.clip(np.floor(xy[:, 0] / float(size) * rows).astype(np.int32), 0, rows - 1)
        col_ext = np.full(rows, np.inf, dtype=np.float64)
        np.minimum.at(col_ext, col[band], xy[band, 1])
        on_edge = band & (xy[:, 1] <= col_ext[col] + margin)
        score = np.where(on_edge, -xy[:, 1], -np.inf)
    elif side in {"bottom", "низ", "снизу"}:
        col = np.clip(np.floor(xy[:, 0] / float(size) * rows).astype(np.int32), 0, rows - 1)
        col_ext = np.full(rows, -np.inf, dtype=np.float64)
        np.maximum.at(col_ext, col[band], xy[band, 1])
        on_edge = band & (xy[:, 1] >= col_ext[col] - margin)
        score = np.where(on_edge, xy[:, 1], -np.inf)
    else:
        row_ext = np.full(rows, -np.inf, dtype=np.float64)
        np.maximum.at(row_ext, row[band], xy[band, 0])
        on_edge = band & (xy[:, 0] >= row_ext[row] - margin)
        score = np.where(on_edge, xy[:, 0], -np.inf)
    if not np.any(np.isfinite(score)):
        score = np.where(band, xy[:, 0], -np.inf)
    extreme = int(np.argmax(score))
    spike = silhouette_spike_face(
        verts, faces, eye, target, side=side, y_lo=y_lo, y_hi=y_hi
    )
    return spike if spike >= 0 else extreme


def mask_aim_side(describe: str, x: float | None = None) -> str:
    """Screen side of the extra bit, from the user's words or the aim x."""
    text = (describe or "").lower()
    if any(word in text for word in ("справа", "правой", "правый", "правая", "right")):
        return "right"
    if any(word in text for word in ("слева", "левой", "левый", "левая", "left")):
        return "left"
    if any(word in text for word in ("сверху", "верх", "голов", "шляп", "head", "hat")):
        if not any(word in text for word in ("юбк", "низ", "ног", "skirt", "лепест", "petal")):
            return "top"
    if x is not None:
        if float(x) > 0.62:
            return "right"
        if float(x) < 0.38:
            return "left"
    return "right"


def mask_aim_y_band(describe: str) -> tuple[float, float]:
    """Occupied-screen Y percentiles for the extra bit (0 = top of frame)."""
    text = (describe or "").lower()
    low = any(word in text for word in ("юбк", "низ", "ног", "ступн", "skirt", "leg", "foot", "лепест", "petal"))
    high = any(word in text for word in ("голов", "шляп", "верх", "head", "hat"))
    if low and not high:
        return 40.0, 100.0
    if high and not low:
        return 0.0, 55.0
    return 0.0, 100.0


def _lump_outward_score(
    mask: np.ndarray,
    centers: np.ndarray,
    body: np.ndarray,
    n: int,
    seed: int,
) -> float:
    """Prefer a compact ear: not one face, not a skirt panel, as far out as the seed."""
    k = int(mask.sum())
    if k < _extra_bit_min(n) or not mask[seed]:
        return -1.0
    if k > _extra_bit_cap(n):
        return -1.0
    if _lump_is_slab(mask, centers, body, seed, n):
        return -1.0
    rad = np.linalg.norm(centers[mask] - body, axis=1)
    seed_r = float(np.linalg.norm(centers[seed] - body))
    if float(np.percentile(rad, 80)) < seed_r * 0.72:
        return -1.0
    return float(np.percentile(rad, 90)) / float(np.log1p(k))


def _view_shell_lump(
    mesh: trimesh.Trimesh,
    verts: np.ndarray,
    faces: np.ndarray,
    face_nbrs: list[list[int]],
    seed: int,
    eye: np.ndarray,
    target: np.ndarray,
    *,
    band_frac: float = 0.06,
) -> np.ndarray:
    """Visible outer shell from the camera through the seed — follows protrusions sideways."""
    from collections import deque

    n = int(len(faces))
    view = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    view /= float(np.linalg.norm(view) or 1.0)
    centers = verts[faces].mean(axis=1)
    depths = (centers - np.asarray(eye, dtype=np.float64)) @ view
    try:
        normals = np.asarray(mesh.face_normals, dtype=np.float64)
    except Exception:
        normals = np.zeros((n, 3), dtype=np.float64)
    seed_d = float(depths[seed])
    span = float(np.percentile(depths, 95) - np.percentile(depths, 5)) or 1.0
    band = max(float(band_frac) * span, 1e-4)
    seed_r = float(np.linalg.norm(centers[seed] - centers.mean(axis=0)))
    out = np.zeros(n, dtype=bool)
    out[seed] = True
    q: deque[int] = deque([seed])
    while q:
        i = q.popleft()
        for j in face_nbrs[i]:
            if out[j]:
                continue
            if float(depths[j]) > seed_d + band:
                continue
            if float(np.dot(normals[j], view)) < -0.12:
                continue
            if float(np.linalg.norm(centers[j] - centers.mean(axis=0))) + 1e-9 < seed_r * 0.72:
                continue
            out[j] = True
            q.append(j)
    return out


def grow_screen_lump(
    mesh: trimesh.Trimesh,
    seed_face: int,
    *,
    seated_verts: np.ndarray,
    seated_faces: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    x: float,
    y: float,
    size: int = 512,
) -> np.ndarray:
    """Visible island under a look-frame click, from the depth buffer.

    Flood walks neighbor depth (occlusion edges). Faces in the pixel blob are
    kept even if they do not share a mesh edge — on a dense Hunyuan mesh the
    z-buffer only samples a subset of triangles, and a single-seed adjacency
    walk collapses that subset to a dozen faces. Shoot-through is blocked by
    matching centroid depth to the z-buffer, then extras must reach the blob.
    """
    from collections import deque

    from mesh_forge.render import _project_points, rasterize_face_ids

    n = int(len(mesh.faces))
    out = np.zeros(n, dtype=bool)
    seed = int(seed_face)
    if n == 0 or seed < 0 or seed >= n:
        return out
    faces = np.asarray(seated_faces, dtype=np.int64)
    verts = np.asarray(seated_verts, dtype=np.float64)
    if len(faces) != n:
        return out
    size = int(size)
    face_buf, zbuf = rasterize_face_ids(verts, faces, eye, target, size)
    px = int(round(float(x) * (size - 1)))
    py = int(round(float(y) * (size - 1)))
    px = max(0, min(size - 1, px))
    py = max(0, min(size - 1, py))
    hit = int(face_buf[py, px])
    if hit < 0:
        best: tuple[int, int, int] | None = None
        best_d = 1e18
        radius = max(12, int(0.07 * size))
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                cy, cx = py + dy, px + dx
                if cy < 0 or cx < 0 or cy >= size or cx >= size:
                    continue
                fid = int(face_buf[cy, cx])
                if fid < 0:
                    continue
                d = float(dx * dx + dy * dy)
                if d < best_d:
                    best_d = d
                    best = (cx, cy, fid)
        if best:
            px, py, hit = best
    if hit < 0:
        hit = seed
        out[seed] = True
        return out
    seed_z = float(zbuf[py, px])
    if not np.isfinite(seed_z):
        out[hit] = True
        return out
    z_valid = zbuf[np.isfinite(zbuf) & (zbuf > 0)]
    span = float(np.percentile(z_valid, 92) - np.percentile(z_valid, 8)) if len(z_valid) else 1.0
    step_tol = max(0.035 * span, 1e-4)
    near = seed_z + 0.22 * span
    blob = np.zeros((size, size), dtype=bool)
    q: deque[tuple[int, int]] = deque([(px, py)])
    while q:
        cx, cy = q.popleft()
        if cx < 0 or cy < 0 or cx >= size or cy >= size or blob[cy, cx]:
            continue
        fid = int(face_buf[cy, cx])
        zpix = float(zbuf[cy, cx])
        if fid < 0 or not np.isfinite(zpix) or zpix > near:
            continue
        blob[cy, cx] = True
        for dx, dy in (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ):
            nx, ny = cx + dx, cy + dy
            if nx < 0 or ny < 0 or nx >= size or ny >= size or blob[ny, nx]:
                continue
            nz = float(zbuf[ny, nx])
            if not np.isfinite(nz) or nz > near:
                continue
            if abs(nz - zpix) > step_tol:
                continue
            q.append((nx, ny))
    ids = face_buf[blob]
    ids = ids[ids >= 0]
    if len(ids) == 0:
        out[hit] = True
        return out
    out[ids] = True
    dilated = blob.copy()
    dilated[1:, :] |= blob[:-1, :]
    dilated[:-1, :] |= blob[1:, :]
    dilated[:, 1:] |= blob[:, :-1]
    dilated[:, :-1] |= blob[:, 1:]
    xy, depth = _project_points(verts[faces].mean(axis=1), eye, target, size)
    sx = np.clip(np.rint(xy[:, 0]).astype(np.int32), 0, size - 1)
    sy = np.clip(np.rint(xy[:, 1]).astype(np.int32), 0, size - 1)
    in_frame = (xy[:, 0] >= 0) & (xy[:, 0] < size) & (xy[:, 1] >= 0) & (xy[:, 1] < size)
    vis_z = zbuf[sy, sx]
    visible = in_frame & np.isfinite(depth) & np.isfinite(vis_z) & (np.abs(depth - vis_z) <= step_tol)
    extra = visible & dilated[sy, sx] & ~out
    if np.any(extra):
        candidate = out.copy()
        candidate[extra] = True
        out = _keep_reachable(
            candidate,
            _face_neighbors(mesh, n, verts=verts, faces=faces),
            np.flatnonzero(out),
        )
    return out


def paint_screen_region(
    mesh: trimesh.Trimesh,
    verts: np.ndarray,
    faces: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    side: str = "right",
    size: int = 512,
) -> np.ndarray:
    """Faces that appear in a look-frame box — no mesh-walk. What you see is what you paint."""
    from mesh_forge.render import _project_points, rasterize_face_ids

    n = int(len(faces))
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    faces = np.asarray(faces, dtype=np.int64)
    verts = np.asarray(verts, dtype=np.float64)
    x0, x1 = sorted((max(0.0, min(1.0, float(x0))), max(0.0, min(1.0, float(x1)))))
    y0, y1 = sorted((max(0.0, min(1.0, float(y0))), max(0.0, min(1.0, float(y1)))))
    if x1 - x0 < 0.04:
        mid = 0.5 * (x0 + x1)
        x0, x1 = max(0.0, mid - 0.05), min(1.0, mid + 0.05)
    if y1 - y0 < 0.04:
        mid = 0.5 * (y0 + y1)
        y0, y1 = max(0.0, mid - 0.06), min(1.0, mid + 0.06)
    size = int(max(64, size))
    centers = verts[faces].mean(axis=1)
    xy, depth = _project_points(centers, eye, target, size)
    in_frame = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < size)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < size)
        & (depth > 1e-4)
    )
    px0, px1 = x0 * float(size - 1), x1 * float(size - 1)
    py0, py1 = y0 * float(size - 1), y1 * float(size - 1)
    in_box = (
        in_frame
        & (xy[:, 0] >= px0)
        & (xy[:, 0] <= px1)
        & (xy[:, 1] >= py0)
        & (xy[:, 1] <= py1)
    )
    if n < 300000:
        face_buf, _zbuf = rasterize_face_ids(verts, faces, eye, target, size)
        ix0 = max(0, min(size - 1, int(np.floor(px0))))
        ix1 = max(0, min(size - 1, int(np.ceil(px1))))
        iy0 = max(0, min(size - 1, int(np.floor(py0))))
        iy1 = max(0, min(size - 1, int(np.ceil(py1))))
        patch = face_buf[iy0 : iy1 + 1, ix0 : ix1 + 1]
        ids = np.unique(patch[patch >= 0])
        if len(ids):
            out[ids] = True
    if not np.any(out):
        out |= in_box
    return out


def screen_face_at(
    verts: np.ndarray,
    faces: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    x: float,
    y: float,
    *,
    size: int = 512,
) -> int:
    """Front-most face under a look-frame pixel (0–1 coords)."""
    from mesh_forge.render import rasterize_face_ids

    n = int(len(faces))
    if n <= 0:
        return -1
    size = int(max(64, size))
    face_buf, _zbuf = rasterize_face_ids(
        np.asarray(verts, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        eye,
        target,
        size,
    )
    px = int(round(float(x) * (size - 1)))
    py = int(round(float(y) * (size - 1)))
    px = max(0, min(size - 1, px))
    py = max(0, min(size - 1, py))
    hit = int(face_buf[py, px])
    if hit >= 0:
        return hit
    best: tuple[int, int, int] | None = None
    best_d = 1e18
    radius = max(10, int(0.06 * size))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            cy, cx = py + dy, px + dx
            if cy < 0 or cx < 0 or cy >= size or cx >= size:
                continue
            fid = int(face_buf[cy, cx])
            if fid < 0:
                continue
            d = float(dx * dx + dy * dy)
            if d < best_d:
                best_d = d
                best = (cx, cy, fid)
    return best[2] if best else -1


def complete_visual_mask(
    mesh: trimesh.Trimesh,
    verts: np.ndarray,
    faces: np.ndarray,
    visual: np.ndarray,
    *,
    eye: np.ndarray,
    target: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    seed: int = -1,
    aim_x: float | None = None,
    aim_y: float | None = None,
    hops: int = 2,
    size: int = 512,
) -> np.ndarray:
    """Fill raster holes by face adjacency, staying inside the look-frame box.

    Visual set is the source of truth. Topology only adds neighbors that still
    project into the box, then keeps the component under the click.
    """
    from mesh_forge.render import _project_points

    n = int(len(faces))
    out = np.asarray(visual, dtype=bool).copy()
    if out.shape[0] != n or not np.any(out):
        return np.zeros(n, dtype=bool)
    faces = np.asarray(faces, dtype=np.int64)
    verts = np.asarray(verts, dtype=np.float64)
    size = int(max(64, size))
    x0, x1 = sorted((max(0.0, min(1.0, float(x0))), max(0.0, min(1.0, float(x1)))))
    y0, y1 = sorted((max(0.0, min(1.0, float(y0))), max(0.0, min(1.0, float(y1)))))
    centers = verts[faces].mean(axis=1)
    xy, depth = _project_points(centers, eye, target, size)
    px0, px1 = x0 * float(size - 1), x1 * float(size - 1)
    py0, py1 = y0 * float(size - 1), y1 * float(size - 1)
    in_box = (
        (xy[:, 0] >= px0)
        & (xy[:, 0] <= px1)
        & (xy[:, 1] >= py0)
        & (xy[:, 1] <= py1)
        & (depth > 1e-4)
    )
    nbrs = _face_neighbors(mesh, n, verts=verts, faces=faces)
    steps = max(0, min(int(hops), 6))
    for _ in range(steps):
        extra: list[int] = []
        for i in np.flatnonzero(out):
            extra.extend(int(j) for j in nbrs[int(i)] if not out[int(j)] and bool(in_box[int(j)]))
        if not extra:
            break
        out[np.asarray(extra, dtype=np.int64)] = True
    click = int(seed)
    if aim_x is not None and aim_y is not None:
        px = float(aim_x) * float(size - 1)
        py = float(aim_y) * float(size - 1)
    elif 0 <= click < n:
        px = float(xy[click, 0]) if np.isfinite(xy[click, 0]) else 0.5 * (px0 + px1)
        py = float(xy[click, 1]) if np.isfinite(xy[click, 1]) else 0.5 * (py0 + py1)
    else:
        px = 0.5 * (px0 + px1)
        py = 0.5 * (py0 + py1)
    if 0 <= click < n and out[click]:
        return _keep_reachable(out, nbrs, np.array([click], dtype=np.int64))
    sel = np.flatnonzero(out)
    if sel.size == 0:
        return out
    dist = (xy[sel, 0] - px) ** 2 + (xy[sel, 1] - py) ** 2
    nearest = int(sel[int(np.argmin(dist))])
    return _keep_reachable(out, nbrs, np.array([nearest], dtype=np.int64))


def visible_face_ids(
    verts: np.ndarray,
    faces: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    *,
    size: int = 512,
) -> np.ndarray:
    from mesh_forge.render import rasterize_face_ids

    face_buf, _zbuf = rasterize_face_ids(
        np.asarray(verts, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        eye,
        target,
        int(max(64, size)),
    )
    return np.unique(face_buf[face_buf >= 0])


def box_face_ids(
    verts: np.ndarray,
    faces: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    size: int = 512,
) -> np.ndarray:
    from mesh_forge.render import rasterize_face_ids

    size = int(max(64, size))
    x0, x1 = sorted((max(0.0, min(1.0, float(x0))), max(0.0, min(1.0, float(x1)))))
    y0, y1 = sorted((max(0.0, min(1.0, float(y0))), max(0.0, min(1.0, float(y1)))))
    face_buf, _zbuf = rasterize_face_ids(
        np.asarray(verts, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        eye,
        target,
        size,
    )
    px0, px1 = x0 * float(size - 1), x1 * float(size - 1)
    py0, py1 = y0 * float(size - 1), y1 * float(size - 1)
    ix0 = max(0, min(size - 1, int(np.floor(px0))))
    ix1 = max(0, min(size - 1, int(np.ceil(px1))))
    iy0 = max(0, min(size - 1, int(np.floor(py0))))
    iy1 = max(0, min(size - 1, int(np.ceil(py1))))
    patch = face_buf[iy0 : iy1 + 1, ix0 : ix1 + 1]
    return np.unique(patch[patch >= 0])


def mask_from_view_observations(
    mesh: trimesh.Trimesh,
    verts: np.ndarray,
    faces: np.ndarray,
    observations: list[dict[str, Any]],
    *,
    size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a 3D candidate mask from several 2D detections across cameras."""
    n = int(len(faces))
    if n <= 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)
    scores = np.zeros(n, dtype=np.float64)
    support = np.zeros(n, dtype=np.int32)
    hit_sets: list[tuple[dict[str, Any], np.ndarray]] = []
    size = int(max(64, size))
    for obs in observations:
        try:
            eye = np.asarray(obs["eye"], dtype=np.float64).reshape(3)
            target = np.asarray(obs["target"], dtype=np.float64).reshape(3)
        except Exception:
            continue
        conf = max(0.0, min(1.0, float(obs.get("confidence") or 0.0)))
        visible = visible_face_ids(verts, faces, eye, target, size=size)
        if visible.size == 0:
            continue
        if not bool(obs.get("visible")):
            scores[visible] -= 0.10 * max(conf, 0.25)
            continue
        try:
            hits = box_face_ids(
                verts,
                faces,
                eye,
                target,
                x0=float(obs["x0"]),
                y0=float(obs["y0"]),
                x1=float(obs["x1"]),
                y1=float(obs["y1"]),
                size=size,
            )
        except Exception:
            continue
        if hits.size == 0:
            continue
        hit_sets.append((dict(obs), np.asarray(hits, dtype=np.int64)))
        scores[visible] -= 0.18 * max(conf, 0.25)
        scores[hits] += 1.05 * max(conf, 0.25)
        support[hits] += 1
        kind = str(obs.get("kind") or "").strip().lower()
        if kind == "protrusion":
            scores[hits] += 0.10 * max(conf, 0.25)
    scores += 0.22 * np.maximum(support - 1, 0)
    scored = _mask_from_scores(mesh, scores, verts=verts, faces=faces)
    seeded = _mask_from_seed_support(
        mesh,
        scores,
        support,
        hit_sets,
        verts=verts,
        faces=faces,
    )
    if np.any(seeded) and np.any(scored):
        mask = seeded
        if _mask_support_score(scored, scores, verts=verts, faces=faces) > _mask_support_score(
            seeded,
            scores,
            verts=verts,
            faces=faces,
        ):
            mask = scored
    else:
        mask = seeded if np.any(seeded) else scored
    return mask, scores


def _mask_from_seed_support(
    mesh: trimesh.Trimesh,
    scores: np.ndarray,
    support: np.ndarray,
    hit_sets: list[tuple[dict[str, Any], np.ndarray]],
    *,
    verts: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    n = int(len(scores))
    if n <= 0 or not hit_sets:
        return np.zeros(max(n, 0), dtype=bool)
    best = np.zeros(n, dtype=bool)
    best_score = -1e18
    global_seed = _best_seed_from_support(scores, support, verts=verts, faces=faces)
    for obs, hits in hit_sets:
        seeds: list[int] = []
        local_seed = _best_seed_in_hits(
            obs,
            hits,
            scores,
            support,
            verts=verts,
            faces=faces,
        )
        if local_seed >= 0:
            seeds.append(int(local_seed))
        silhouette_seed = _best_silhouette_seed(obs, verts=verts, faces=faces)
        if silhouette_seed >= 0 and int(silhouette_seed) not in seeds:
            seeds.append(int(silhouette_seed))
        if global_seed >= 0 and int(global_seed) not in seeds and np.any(np.asarray(hits, dtype=np.int64) == int(global_seed)):
            seeds.append(int(global_seed))
        if not seeds:
            continue
        try:
            eye = np.asarray(obs["eye"], dtype=np.float64).reshape(3)
            target = np.asarray(obs["target"], dtype=np.float64).reshape(3)
        except Exception:
            continue
        for seed in seeds:
            grown = grow_visible_lump(
                mesh,
                int(seed),
                seated_verts=verts,
                seated_faces=faces,
                eye=eye,
                target=target,
            )
            grown = np.asarray(grown, dtype=bool)
            score = _mask_support_score(grown, scores, verts=verts, faces=faces)
            if score > best_score:
                best_score = score
                best = grown
    return best


def _best_seed_from_support(
    scores: np.ndarray,
    support: np.ndarray,
    *,
    verts: np.ndarray,
    faces: np.ndarray,
) -> int:
    n = int(len(scores))
    if n <= 0:
        return -1
    base = np.asarray(scores, dtype=np.float64).reshape(-1)
    sup = np.asarray(support, dtype=np.int32).reshape(-1)
    positive = np.flatnonzero(base > 0.10)
    if positive.size == 0:
        positive = np.flatnonzero(base > 0.0)
    if positive.size == 0:
        return -1
    centers = verts[faces].mean(axis=1)
    body = verts.mean(axis=0)
    radii = np.linalg.norm(centers - body, axis=1)
    scale = float(np.percentile(radii, 95)) if radii.size else 1.0
    scale = scale if scale > 1e-9 else 1.0
    seed_score = (
        np.maximum(base[positive], 0.0)
        + 0.32 * np.maximum(sup[positive] - 1, 0)
        + 0.18 * (radii[positive] / scale)
    )
    return int(positive[int(np.argmax(seed_score))])


def _best_seed_in_hits(
    obs: dict[str, Any],
    hits: np.ndarray,
    scores: np.ndarray,
    support: np.ndarray,
    *,
    verts: np.ndarray,
    faces: np.ndarray,
) -> int:
    idx = np.unique(np.asarray(hits, dtype=np.int64))
    idx = idx[(idx >= 0) & (idx < len(scores))]
    if idx.size == 0:
        return -1
    centers = verts[faces].mean(axis=1)
    body = verts.mean(axis=0)
    radii = np.linalg.norm(centers - body, axis=1)
    scale = float(np.percentile(radii, 95)) if radii.size else 1.0
    scale = scale if scale > 1e-9 else 1.0
    base = np.maximum(np.asarray(scores, dtype=np.float64).reshape(-1)[idx], 0.0)
    sup = np.asarray(support, dtype=np.int32).reshape(-1)[idx]
    seed_score = base + 0.32 * np.maximum(sup - 1, 0) + 0.18 * (radii[idx] / scale)
    try:
        from mesh_forge.render import _project_points

        eye = np.asarray(obs["eye"], dtype=np.float64).reshape(3)
        target = np.asarray(obs["target"], dtype=np.float64).reshape(3)
        xy, _depth = _project_points(centers[idx], eye, target, 512)
        cx = 0.5 * (float(obs.get("x0", 0.5)) + float(obs.get("x1", 0.5)))
        cy = 0.5 * (float(obs.get("y0", 0.5)) + float(obs.get("y1", 0.5)))
        px = xy[:, 0] / 511.0
        py = xy[:, 1] / 511.0
        if cx >= 0.55:
            seed_score += 0.16 * np.maximum(px - cx, 0.0)
        elif cx <= 0.45:
            seed_score += 0.16 * np.maximum(cx - px, 0.0)
        if cy >= 0.55:
            seed_score += 0.10 * np.maximum(py - cy, 0.0)
        elif cy <= 0.45:
            seed_score += 0.10 * np.maximum(cy - py, 0.0)
    except Exception:
        pass
    return int(idx[int(np.argmax(seed_score))])


def _best_silhouette_seed(
    obs: dict[str, Any],
    *,
    verts: np.ndarray,
    faces: np.ndarray,
) -> int:
    try:
        eye = np.asarray(obs["eye"], dtype=np.float64).reshape(3)
        target = np.asarray(obs["target"], dtype=np.float64).reshape(3)
    except Exception:
        return -1
    side, y_lo, y_hi = _obs_seed_hint(obs)
    return int(
        silhouette_extra_face(
            verts,
            faces,
            eye,
            target,
            side=side,
            y_lo=y_lo,
            y_hi=y_hi,
        )
    )


def _obs_seed_hint(obs: dict[str, Any]) -> tuple[str, float, float]:
    x0 = float(obs.get("x0", 0.0))
    x1 = float(obs.get("x1", 1.0))
    y0 = float(obs.get("y0", 0.0))
    y1 = float(obs.get("y1", 1.0))
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    if cx <= 0.45:
        side = "left"
    elif cx >= 0.55:
        side = "right"
    elif cy <= 0.45:
        side = "top"
    else:
        side = "right"
    if cy >= 0.60:
        return side, 40.0, 100.0
    if cy <= 0.40:
        return side, 0.0, 55.0
    return side, 0.0, 100.0


def _mask_support_score(
    mask: np.ndarray,
    scores: np.ndarray,
    *,
    verts: np.ndarray,
    faces: np.ndarray,
) -> float:
    current = np.asarray(mask, dtype=bool).reshape(-1)
    if current.shape[0] != len(scores) or not np.any(current):
        return -1e18
    base = np.asarray(scores, dtype=np.float64).reshape(-1)
    idx = np.flatnonzero(current)
    pos = np.maximum(base[idx], 0.0)
    coverage = float(np.mean(base[idx] > 0.02))
    centers = verts[faces].mean(axis=1)
    body = verts.mean(axis=0)
    seed = int(idx[int(np.argmax(base[idx]))])
    shape = _lump_outward_score(current, centers, body, len(faces), seed)
    score = float(pos.sum()) * (0.45 + 0.55 * coverage)
    if shape < 0.0:
        score -= 3.0
    else:
        score += 0.35 * float(shape)
    if mask_is_tiny(current, verts, faces):
        score -= 4.0
    return score


def _mask_from_scores(
    mesh: trimesh.Trimesh,
    scores: np.ndarray,
    *,
    verts: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    n = int(len(scores))
    if n <= 0:
        return np.zeros(0, dtype=bool)
    support = np.asarray(scores, dtype=np.float64).reshape(-1)
    centers = verts[faces].mean(axis=1)
    body = verts.mean(axis=0)
    pos = support > 0.42
    if not np.any(pos):
        pos = support > 0.10
    if not np.any(pos):
        return np.zeros(n, dtype=bool)
    nbrs = _face_neighbors(mesh, n, verts=verts, faces=faces)
    best = np.zeros(n, dtype=bool)
    seen = np.zeros(n, dtype=bool)
    best_score = -1e18
    for seed in np.flatnonzero(pos):
        i = int(seed)
        if seen[i]:
            continue
        comp = _keep_reachable(pos, nbrs, np.array([i], dtype=np.int64))
        seen |= comp
        idx = np.flatnonzero(comp)
        if idx.size == 0:
            continue
        comp_score = float(np.sum(np.maximum(support[idx], 0.0)))
        seed = int(idx[int(np.argmax(support[idx]))])
        shape_score = _lump_outward_score(comp, centers, body, n, seed)
        if shape_score < 0.0:
            comp_score -= 2.5
        else:
            comp_score += 0.35 * shape_score
        if comp_score > best_score:
            best_score = comp_score
            best = comp
    if not np.any(best):
        return best
    best = keep_outward_blob(mesh, best, seated_verts=verts, seated_faces=faces)
    fringe = best.copy()
    for i in np.flatnonzero(best):
        for j in nbrs[int(i)]:
            if support[int(j)] > 0.02:
                fringe[int(j)] = True
    return _keep_reachable(fringe, nbrs, np.flatnonzero(best)[:1])


def mask_geometry_metrics(
    mesh: trimesh.Trimesh,
    mask: np.ndarray,
    *,
    verts: np.ndarray,
    faces: np.ndarray,
) -> dict[str, float | int | bool]:
    n = int(len(faces))
    current = np.asarray(mask, dtype=bool).reshape(-1)
    if n <= 0 or current.shape[0] != n or not np.any(current):
        return {
            "faces": 0,
            "components": 0,
            "largest_component_faces": 0,
            "area_frac": 0.0,
            "outward_score": -1.0,
            "is_slab": False,
        }
    nbrs = _face_neighbors(mesh, n, verts=verts, faces=faces)
    centers = verts[faces].mean(axis=1)
    body = verts.mean(axis=0)
    seen = np.zeros(n, dtype=bool)
    components = 0
    largest = 0
    best_shape = -1.0
    best_slab = False
    for seed in np.flatnonzero(current):
        i = int(seed)
        if seen[i]:
            continue
        comp = _keep_reachable(current, nbrs, np.array([i], dtype=np.int64))
        seen |= comp
        k = int(comp.sum())
        if k <= 0:
            continue
        components += 1
        if k > largest:
            largest = k
        radii = np.linalg.norm(centers[comp] - body, axis=1)
        local_seed = int(np.flatnonzero(comp)[int(np.argmax(radii))])
        shape = _lump_outward_score(comp, centers, body, n, local_seed)
        if shape > best_shape:
            best_shape = shape
            best_slab = _lump_is_slab(comp, centers, body, local_seed, n)
    tri = verts[faces]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cr, axis=1)
    mesh_area = float(np.sum(areas)) or 1.0
    area_frac = float(np.sum(areas[current])) / mesh_area
    return {
        "faces": int(current.sum()),
        "components": int(components),
        "largest_component_faces": int(largest),
        "area_frac": float(area_frac),
        "outward_score": float(best_shape),
        "is_slab": bool(best_slab),
    }


def _keep_reachable(mask: np.ndarray, nbrs: list[list[int]], seeds: np.ndarray) -> np.ndarray:
    from collections import deque

    n = int(len(mask))
    out = np.zeros(n, dtype=bool)
    q: deque[int] = deque()
    for seed in np.asarray(seeds, dtype=np.int64).reshape(-1):
        s = int(seed)
        if 0 <= s < n and mask[s] and not out[s]:
            out[s] = True
            q.append(s)
    if not q:
        return mask
    while q:
        i = q.popleft()
        for j in nbrs[i]:
            if out[j] or not mask[j]:
                continue
            out[j] = True
            q.append(j)
    return out


def _face_neighbors(
    mesh: trimesh.Trimesh,
    n: int,
    *,
    verts: np.ndarray | None = None,
    faces: np.ndarray | None = None,
) -> list[list[int]]:
    nbrs: list[list[int]] = [[] for _ in range(n)]
    try:
        for a, b in np.asarray(mesh.face_adjacency, dtype=np.int64):
            ia, ib = int(a), int(b)
            if 0 <= ia < n and 0 <= ib < n:
                nbrs[ia].append(ib)
                nbrs[ib].append(ia)
    except Exception:
        nbrs = [[] for _ in range(n)]
    isolated = sum(1 for row in nbrs if not row)
    if isolated > max(8, int(0.02 * n)) and verts is not None and faces is not None:
        spatial = _centroid_neighbors(verts, faces, n)
        for i, extra in enumerate(spatial):
            if not nbrs[i]:
                nbrs[i] = extra
    return nbrs


def _centroid_neighbors(verts: np.ndarray, faces: np.ndarray, n: int, k: int = 8) -> list[list[int]]:
    """Connect faces that almost share an edge (Hunyuan / STL soup gaps)."""
    faces = np.asarray(faces, dtype=np.int64)
    verts = np.asarray(verts, dtype=np.float64)
    if n == 0 or len(faces) != n:
        return [[] for _ in range(n)]
    centers = verts[faces].mean(axis=1)
    edge = np.linalg.norm(verts[faces[:, 0]] - verts[faces[:, 1]], axis=1)
    radius = 2.8 * float(np.median(edge) or 1e-4)
    kk = int(min(max(k, 1) + 1, n))
    dist, idx = _knn_centers(centers, kk)
    out: list[list[int]] = [[] for _ in range(n)]
    rad = float(radius)
    for i in range(n):
        di = np.atleast_1d(dist[i])
        ji = np.atleast_1d(idx[i])
        row: list[int] = []
        for d, j in zip(di, ji):
            jj = int(j)
            if jj == i or jj < 0 or not np.isfinite(d) or float(d) > rad:
                continue
            row.append(jj)
        out[i] = row
    return out


def _knn_centers(centers: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(centers, dtype=np.float64)
    n = int(len(pts))
    k = int(min(max(k, 1), n))
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(pts)
        try:
            return tree.query(pts, k=k, workers=-1)
        except TypeError:
            return tree.query(pts, k=k)
    except Exception:
        pass
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        tree = o3d.geometry.KDTreeFlann(pcd)
        dist = np.full((n, k), np.inf, dtype=np.float64)
        idx = np.full((n, k), -1, dtype=np.int64)
        for i in range(n):
            found, ii, dd = tree.search_knn_vector_3d(pts[i], k)
            take = min(int(found), k)
            if take:
                idx[i, :take] = np.asarray(ii[:take], dtype=np.int64)
                dist[i, :take] = np.sqrt(np.asarray(dd[:take], dtype=np.float64))
        return dist, idx
    except Exception:
        return _knn_centers_grid(pts, k)


def _knn_centers_grid(pts: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(len(pts))
    dist = np.full((n, k), np.inf, dtype=np.float64)
    idx = np.full((n, k), -1, dtype=np.int64)
    if n == 0:
        return dist, idx
    span = np.maximum(pts.max(axis=0) - pts.min(axis=0), 1e-9)
    spacing = float(np.prod(span) / max(n, 1)) ** (1.0 / 3.0)
    cell = max(spacing, 1e-6)
    keys = np.floor((pts - pts.min(axis=0)) / cell).astype(np.int32)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for i, (a, b, c) in enumerate(keys):
        buckets.setdefault((int(a), int(b), int(c)), []).append(i)
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for i, (a, b, c) in enumerate(keys):
        cand: list[int] = []
        for dx, dy, dz in offsets:
            cand.extend(buckets.get((int(a) + dx, int(b) + dy, int(c) + dz), ()))
        if not cand:
            continue
        ids = np.asarray(cand, dtype=np.int64)
        d = np.linalg.norm(pts[ids] - pts[i], axis=1)
        order = np.argsort(d)[:k]
        take = int(len(order))
        dist[i, :take] = d[order]
        idx[i, :take] = ids[order]
    return dist, idx


def _flood_faces(nbrs: list[list[int]], seed: int) -> np.ndarray:
    from collections import deque

    n = len(nbrs)
    out = np.zeros(n, dtype=bool)
    out[seed] = True
    q = deque([seed])
    while q:
        i = q.popleft()
        for j in nbrs[i]:
            if out[j]:
                continue
            out[j] = True
            q.append(j)
    return out


def _outward_axis(verts: np.ndarray, faces: np.ndarray, seed: int) -> np.ndarray:
    centers = verts[faces].mean(axis=1)
    body = verts.mean(axis=0)
    axis = centers[seed] - body
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return axis / norm


def _visible_bfs_from_seed(
    verts: np.ndarray,
    faces: np.ndarray,
    face_nbrs: list[list[int]],
    seed: int,
    eye: np.ndarray,
    target: np.ndarray,
    n: int,
) -> np.ndarray:
    from collections import deque

    centers = verts[faces].mean(axis=1)
    view = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    view /= float(np.linalg.norm(view) or 1.0)
    depths = (centers - eye) @ view
    seed_d = float(depths[seed])
    span = float(np.percentile(depths, 92) - np.percentile(depths, 8)) or 1.0
    tau = max(0.12 * span, 1e-4)
    out = np.zeros(n, dtype=bool)
    out[seed] = True
    q: deque[int] = deque([seed])
    while q:
        i = q.popleft()
        for j in face_nbrs[i]:
            if out[j]:
                continue
            if abs(float(depths[j]) - seed_d) > tau:
                continue
            out[j] = True
            q.append(j)
    return out


def _protrusion_from_seed(
    verts: np.ndarray,
    faces: np.ndarray,
    face_nbrs: list[list[int]],
    seed: int,
    *,
    eye: np.ndarray | None = None,
    target: np.ndarray | None = None,
) -> np.ndarray:
    """Faces on the seed's side of the thinnest neck toward the body centroid."""
    from collections import deque

    n = int(len(faces))
    out = np.zeros(n, dtype=bool)
    out[seed] = True
    centers = verts[faces].mean(axis=1)
    body = verts.mean(axis=0)
    axis = centers[seed] - body
    norm = float(np.linalg.norm(axis))
    if eye is not None and target is not None:
        view = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
        view /= float(np.linalg.norm(view) or 1.0)
        if norm > 1e-9:
            axis = axis / norm
            axis = axis + 0.65 * view
            axis /= float(np.linalg.norm(axis) or 1.0)
        else:
            axis = view
    elif norm < 1e-9:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = axis / norm
    scores = (verts - body) @ axis
    seed_vs = np.unique(np.asarray(faces[seed], dtype=np.int64).reshape(-1))
    tip = float(np.max(scores[seed_vs]))
    spread = float(np.max(scores) - np.min(scores)) or 1.0
    lo = float(np.percentile(scores, 58.0))
    ordered = np.sort(scores)
    window = ordered[(ordered >= lo) & (ordered <= tip + 1e-12)]
    shell = lo
    if len(window) > 8:
        gaps = np.diff(window)
        j = int(np.argmax(gaps))
        if float(gaps[j]) > 0.025 * spread:
            shell = float(window[j])
    inward = shell + 0.015 * max(tip - shell, 1e-9)
    v_n = int(len(verts))
    v_nbrs: list[list[int]] = [[] for _ in range(v_n)]
    for a, b, c in faces:
        ia, ib, ic = int(a), int(b), int(c)
        v_nbrs[ia].extend((ib, ic))
        v_nbrs[ib].extend((ia, ic))
        v_nbrs[ic].extend((ia, ib))
    lump = np.zeros(v_n, dtype=bool)
    q: deque[int] = deque()
    for v in seed_vs:
        vi = int(v)
        if 0 <= vi < v_n:
            lump[vi] = True
            q.append(vi)
    while q:
        i = q.popleft()
        for j in v_nbrs[i]:
            if lump[j] or float(scores[j]) < inward:
                continue
            lump[j] = True
            q.append(j)
    counts = lump[faces].sum(axis=1)
    out = counts >= 2
    if int(out.sum()) < 12:
        out = counts >= 1
    if int(out.sum()) < 8:
        out = _face_flood_above(face_nbrs, seed, centers, body, axis, inward, n)
    return out


def _face_flood_above(
    face_nbrs: list[list[int]],
    seed: int,
    centers: np.ndarray,
    body: np.ndarray,
    axis: np.ndarray,
    inward: float,
    n: int,
) -> np.ndarray:
    from collections import deque

    score_f = (centers - body) @ axis
    out = np.zeros(n, dtype=bool)
    out[seed] = True
    q: deque[int] = deque([seed])
    while q:
        i = q.popleft()
        for j in face_nbrs[i]:
            if out[j] or float(score_f[j]) < inward:
                continue
            out[j] = True
            q.append(j)
    return out


def vertex_mask_for_topo(mesh: trimesh.Trimesh, topo: dict[str, Any]) -> np.ndarray:
    n = int(len(mesh.vertices))
    mask = np.zeros(n, dtype=bool)
    kind = parse_kind(str(topo.get("kind") or "face"))
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if kind == "vertex":
        v = int(topo.get("vertex", -1))
        if 0 <= v < n:
            mask[v] = True
            hits = np.where(np.any(faces == v, axis=1))[0]
            for f in hits:
                mask[faces[f]] = True
        return mask
    drop = face_mask_for_topo(mesh, topo)
    if np.any(drop):
        mask[faces[drop].ravel()] = True
    return mask


def format_topo(topo: dict[str, Any] | None) -> str:
    if not topo_valid(topo):
        return ""
    assert topo is not None
    kind = parse_kind(str(topo.get("kind") or "face"))
    edge = topo.get("edge") or []
    edge_s = f"{int(edge[0])}-{int(edge[1])}" if len(edge) >= 2 else "?"
    return (
        f"{kind} face={int(topo.get('face', -1))} "
        f"vertex={int(topo.get('vertex', -1))} edge={edge_s}"
    )


def _closest_face(mesh: trimesh.Trimesh, point: np.ndarray) -> int:
    try:
        _, _, fid = trimesh.proximity.closest_point(mesh, point.reshape(1, 3))
        face = int(np.asarray(fid).reshape(-1)[0])
        if 0 <= face < int(len(mesh.faces)):
            return face
    except Exception:
        pass
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    return int(np.argmin(np.linalg.norm(centers - point, axis=1)))


def _closest_edge(tri: np.ndarray, corners: np.ndarray, point: np.ndarray) -> list[int]:
    best = 1e18
    pair = [int(tri[0]), int(tri[1])]
    for i in range(3):
        a = corners[i]
        b = corners[(i + 1) % 3]
        ab = b - a
        denom = float(np.dot(ab, ab)) or 1e-12
        t = max(0.0, min(1.0, float(np.dot(point - a, ab) / denom)))
        q = a + t * ab
        d = float(np.linalg.norm(point - q))
        if d < best:
            best = d
            pair = [int(tri[i]), int(tri[(i + 1) % 3])]
    return pair


def _ray_first_face(mesh: trimesh.Trimesh, origin: np.ndarray, direction: np.ndarray) -> int:
    """Möller–Trumbore, no rtree. Closest hit along the ray."""
    tris = np.asarray(mesh.triangles, dtype=np.float64)
    if len(tris) == 0:
        return -1
    v0 = tris[:, 0]
    e1 = tris[:, 1] - v0
    e2 = tris[:, 2] - v0
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    pvec = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    hit = np.abs(det) > 1e-10
    inv = np.zeros(len(tris), dtype=np.float64)
    inv[hit] = 1.0 / det[hit]
    tvec = o - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv
    hit &= (u >= 0.0) & (u <= 1.0)
    qvec = np.cross(tvec, e1)
    v = np.einsum("ij,j->i", qvec, d) * inv
    hit &= (v >= 0.0) & (u + v <= 1.0)
    t = np.einsum("ij,ij->i", e2, qvec) * inv
    hit &= t > 1e-6
    if not np.any(hit):
        return -1
    dist = np.where(hit, t, np.inf)
    return int(np.argmin(dist))


def _face_at_preview_xy(
    seated: trimesh.Trimesh,
    eye: np.ndarray,
    target: np.ndarray,
    x: float,
    y: float,
) -> tuple[int, float]:
    """Nearest face centroid in the look preview. Returns (face, pixel distance in 0–1)."""
    from mesh_forge.render import _project_points

    centers = np.asarray(seated.triangles_center, dtype=np.float64)
    if len(centers) == 0:
        return -1, 1.0
    size = 1000
    xy, depth = _project_points(centers, eye, target, size)
    px = float(x) * (size - 1)
    py = float(y) * (size - 1)
    valid = depth > 1e-4
    dist = np.full(len(centers), np.inf, dtype=np.float64)
    if np.any(valid):
        dx = xy[valid, 0] - px
        dy = xy[valid, 1] - py
        dist[valid] = np.sqrt(dx * dx + dy * dy)
    radius = max(18.0, 0.035 * float(size - 1))
    near = dist <= radius
    if not np.any(near):
        face = int(np.argmin(dist))
        best = float(dist[face])
        if not np.isfinite(best):
            return -1, 1.0
        return face, best / float(size - 1)
    cand = np.where(near & valid)[0]
    face = int(cand[np.argmin(depth[cand])])
    best = float(dist[face])
    if not np.isfinite(best):
        return -1, 1.0
    return face, best / float(size - 1)


def viewport_hit(
    mesh: trimesh.Trimesh,
    *,
    camera: str = "right",
    views: str | None = None,
    yaw: float | None = None,
    pitch: float | None = None,
    x: float = 0.5,
    y: float = 0.5,
    zoom: float = 1.0,
    kind: str | None = None,
    hops: int = 12,
) -> dict[str, Any]:
    """Pick a face in a look-style frame. x,y are 0–1 in the preview (0,0 = top-left)."""
    from mesh_forge.render import _FOV_DEG, _camera_eye_target, _seat_for_viewer
    from mesh_forge.tools.look import parse_look_shots

    if int(len(mesh.faces)) == 0:
        raise TopoError("Mesh has no faces.")
    shot = parse_look_shots(
        str(views or camera or "right"),
        zoom=float(zoom or 1.0),
        yaw=yaw,
        pitch=pitch,
    )[0]
    verts, faces, extent = _seat_for_viewer(mesh)
    seated = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    eye, target = _camera_eye_target(
        extent,
        shot.camera,
        pad=1.0,
        zoom=float(shot.zoom or 1.0),
        yaw=shot.yaw,
        pitch=shot.pitch,
        shift=shot.shift,
    )
    forward = target - eye
    forward = forward / float(np.linalg.norm(forward) or 1.0)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if float(np.linalg.norm(right)) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right = right / float(np.linalg.norm(right))
    up = np.cross(right, forward)
    up = up / float(np.linalg.norm(up) or 1.0)
    ndc_x = float(x) * 2.0 - 1.0
    ndc_y = 1.0 - float(y) * 2.0
    tan = float(np.tan(np.deg2rad(_FOV_DEG) * 0.5))
    direction = forward + ndc_x * tan * right + ndc_y * tan * up
    direction = direction / float(np.linalg.norm(direction) or 1.0)
    face = _ray_first_face(seated, eye, direction)
    snapped = 0.0
    face_scr, snap_scr = _face_at_preview_xy(seated, eye, target, float(x), float(y))
    if face < 0:
        face = face_scr
        snapped = float(snap_scr)
    elif face_scr >= 0 and float(snap_scr) <= 0.04:
        centers = np.asarray(seated.triangles_center, dtype=np.float64)
        view = forward
        depths = (centers - eye) @ view
        if float(depths[face_scr]) + 1e-6 < float(depths[face]):
            face = face_scr
            snapped = float(snap_scr)
    if face < 0:
        raise TopoError("Viewport aim missed the mesh. Change views/x/y.")
    topo = topology_from_ids(mesh, kind=kind or "face", face=face)
    topo["hops"] = 0
    topo["faces"] = 1
    topo["aim_x"] = float(x)
    topo["aim_y"] = float(y)
    topo["aim_snap"] = float(snapped)
    topo["camera"] = shot.camera
    topo["yaw"] = shot.yaw
    topo["pitch"] = shot.pitch
    topo["zoom"] = float(shot.zoom or 1.0)
    return topo
