from __future__ import annotations

from typing import Literal

import numpy as np
import trimesh

RegionName = Literal["legs", "seat", "back", "left", "right", "top", "bottom", "front"]

REGION_NAMES: tuple[str, ...] = (
    "legs",
    "seat",
    "back",
    "left",
    "right",
    "top",
    "bottom",
    "front",
)

# Normalized AABB (left, right, bottom, top, back, front). Always two axes
# clipped so a call cannot become a full left/right slab.
REGION_BOXES: dict[str, tuple[float, float, float, float, float, float]] = {
    "legs": (0.04, 0.96, 0.00, 0.34, 0.12, 0.96),
    "seat": (0.08, 0.92, 0.28, 0.58, 0.22, 0.98),
    "back": (0.10, 0.90, 0.42, 1.00, 0.00, 0.42),
    "left": (0.00, 0.22, 0.42, 1.00, 0.00, 0.55),
    "right": (0.78, 1.00, 0.42, 1.00, 0.00, 0.55),
    "top": (0.10, 0.90, 0.78, 1.00, 0.05, 0.95),
    "bottom": (0.04, 0.96, 0.00, 0.18, 0.05, 0.95),
    "front": (0.10, 0.90, 0.18, 0.88, 0.78, 1.00),
}

_ALIASES = {
    "legs": "legs",
    "ножки": "legs",
    "ноги": "legs",
    "seat": "seat",
    "сиденье": "seat",
    "back": "back",
    "backrest": "back",
    "спинка": "back",
    "left": "left",
    "слева": "left",
    "right": "right",
    "справа": "right",
    "top": "top",
    "верх": "top",
    "bottom": "bottom",
    "низ": "bottom",
    "front": "front",
    "спереди": "front",
}


_KNIFE_SIDES: dict[str, tuple[int, int]] = {
    "right": (0, 1),
    "left": (0, -1),
    "top": (1, 1),
    "bottom": (1, -1),
    "front": (2, 1),
    "back": (2, -1),
}

_KNIFE_BANDS: dict[str, tuple[int, float, float]] = {
    "bottom": (1, 0.0, 0.42),
    "top": (1, 0.58, 1.0),
    "left": (0, 0.0, 0.42),
    "right": (0, 0.58, 1.0),
    "back": (2, 0.0, 0.42),
    "front": (2, 0.58, 1.0),
}


class RegionError(ValueError):
    """Unknown or missing semantic region."""


def parse_region(region: str | None) -> str:
    raw = (region or "").strip().lower()
    if not raw:
        raise RegionError("Need region: legs|seat|back|left|right|top|bottom|front.")
    mapped = _ALIASES.get(raw, raw)
    if mapped not in REGION_BOXES:
        raise RegionError(
            f"Unknown region {region!r}. Use legs|seat|back|left|right|top|bottom|front."
        )
    return mapped


def region_box(region: str | None) -> tuple[float, float, float, float, float, float]:
    return REGION_BOXES[parse_region(region)]


def faces_in_box(
    mesh: trimesh.Trimesh,
    box: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    """Boolean mask of faces whose centroids lie in a normalized AABB."""
    n = int(len(mesh.faces))
    if n == 0:
        return np.zeros(0, dtype=bool)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    span = np.maximum(hi - lo, 1e-9)
    left, right, bottom, top, back, front = box
    box_lo = lo + span * np.array([left, bottom, back], dtype=np.float64)
    box_hi = lo + span * np.array([right, top, front], dtype=np.float64)
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    return np.all((centers >= box_lo) & (centers <= box_hi), axis=1)


def faces_in_region(mesh: trimesh.Trimesh, region: str | None) -> np.ndarray:
    return faces_in_box(mesh, region_box(region))


DEFAULT_PICK_RADIUS = 0.022


def pick_box(
    nx: float,
    ny: float,
    nz: float,
    radius: float = DEFAULT_PICK_RADIUS,
) -> tuple[float, float, float, float, float, float]:
    """Local AABB around a normalized click. Prefer faces_near_pick for remove."""
    r = max(0.015, min(float(radius), 0.05))

    def _span(value: float) -> tuple[float, float]:
        lo = max(0.0, min(1.0, float(value) - r))
        hi = max(0.0, min(1.0, float(value) + r))
        if hi - lo < 0.03:
            mid = float(value)
            lo = max(0.0, mid - 0.015)
            hi = min(1.0, mid + 0.015)
        return lo, hi

    left, right = _span(nx)
    bottom, top = _span(ny)
    back, front = _span(nz)
    return left, right, bottom, top, back, front


def pick_world_point(mesh: trimesh.Trimesh, nx: float, ny: float, nz: float) -> np.ndarray:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    span = np.maximum(bounds[1] - bounds[0], 1e-9)
    return bounds[0] + span * np.array(
        [float(nx), float(ny), float(nz)],
        dtype=np.float64,
    )


def faces_near_pick(
    mesh: trimesh.Trimesh,
    nx: float,
    ny: float,
    nz: float,
    radius: float = DEFAULT_PICK_RADIUS,
    *,
    max_frac: float = 0.025,
    max_hops: int = 8,
) -> np.ndarray:
    """Faces around a click: small sphere + hop-limited patch."""
    n = int(len(mesh.faces))
    empty = np.zeros(n, dtype=bool)
    if n == 0:
        return empty
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    span = np.maximum(np.asarray(mesh.extents, dtype=np.float64), 1e-9)
    point = pick_world_point(mesh, nx, ny, nz)
    dist = np.linalg.norm(centers - point, axis=1)
    seed = int(np.argmin(dist))
    r_frac = max(0.012, min(float(radius), 0.04))
    for _ in range(8):
        radius_w = r_frac * float(np.max(span))
        nearby = dist <= radius_w
        if not bool(nearby[seed]):
            nearby[seed] = True
        patch = _connected_in_mask(mesh, seed, nearby, max_hops=max_hops)
        if int(patch.sum()) <= max(8, int(max_frac * n)):
            return patch
        r_frac *= 0.7
        max_hops = max(3, max_hops - 1)
    patch = np.zeros(n, dtype=bool)
    patch[seed] = True
    return patch


def _connected_in_mask(
    mesh: trimesh.Trimesh,
    seed: int,
    allowed: np.ndarray,
    *,
    max_hops: int = 8,
) -> np.ndarray:
    n = int(len(mesh.faces))
    out = np.zeros(n, dtype=bool)
    if seed < 0 or seed >= n or not bool(allowed[seed]):
        return out
    try:
        raw = getattr(mesh, "face_adjacency", None)
        if raw is None or len(raw) == 0:
            out[seed] = True
            return out
        neighbors: list[list[int]] = [[] for _ in range(n)]
        for a, b in np.asarray(raw, dtype=np.int64):
            neighbors[int(a)].append(int(b))
            neighbors[int(b)].append(int(a))
    except Exception:
        out[seed] = True
        return out
    hops = np.full(n, -1, dtype=np.int32)
    hops[seed] = 0
    stack = [seed]
    out[seed] = True
    limit = max(1, int(max_hops))
    while stack:
        i = stack.pop()
        if hops[i] >= limit:
            continue
        for j in neighbors[i]:
            if out[j] or not bool(allowed[j]):
                continue
            out[j] = True
            hops[j] = hops[i] + 1
            stack.append(j)
    return out


def parse_knife_side(side: str | None) -> str:
    raw = (side or "").strip().lower()
    mapped = _ALIASES.get(raw, raw)
    if mapped not in _KNIFE_SIDES:
        raise RegionError(
            f"Unknown knife {side!r}. Use left|right|top|bottom|front|back."
        )
    return mapped


def parse_knife_along(along: str | None) -> str:
    raw = (along or "").strip().lower()
    if not raw:
        return ""
    mapped = _ALIASES.get(raw, raw)
    if mapped not in _KNIFE_BANDS:
        raise RegionError(
            f"Unknown along {along!r}. Use top|bottom|left|right|front|back."
        )
    return mapped


def knife_plane(
    mesh: trimesh.Trimesh,
    side: str | None = None,
    *,
    at: float | None = None,
    pick: list[float] | tuple[float, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Axis-aligned bisect on the mesh AABB (not the camera).

    Named side: scrap is toward that bbox face (+X right, -X left, +Y top, -Y bottom,
    +Z front, -Z back). at: 0–1 on that mesh axis. Click sets at on the same axis.
    """
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    span = np.maximum(hi - lo, 1e-9)
    center = 0.5 * (lo + hi)
    side_name = parse_knife_side(side)
    axis, sign = _KNIFE_SIDES[side_name]
    t = float(at) if at is not None else None
    if t is None and pick is not None and len(pick) >= 3:
        t = float(pick[axis])
    if t is None:
        t = 0.82 if sign > 0 else 0.18
    t = max(0.02, min(0.98, t))
    origin = center.copy()
    origin[axis] = lo[axis] + t * span[axis]
    keep = np.zeros(3, dtype=np.float64)
    keep[axis] = -float(sign)
    return origin, keep, f"knife:{side_name} at={t:.2f}"


def knife_pick(
    mesh: trimesh.Trimesh,
    side: str,
    along: str = "",
) -> list[float]:
    """Normalized tip of the lump sticking out ``side``, optionally in an ``along`` band."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(verts) == 0:
        raise RegionError("Mesh has no vertices for knife.")
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    span = np.maximum(hi - lo, 1e-9)
    idx = knife_tip_index(mesh, side, along=along)
    p = (verts[idx] - lo) / span
    return [float(p[0]), float(p[1]), float(p[2])]


def knife_tip_index(mesh: trimesh.Trimesh, side: str, along: str = "") -> int:
    side_name = parse_knife_side(side)
    along_name = parse_knife_along(along)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(verts) == 0:
        raise RegionError("Mesh has no vertices for knife.")
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    span = np.maximum(hi - lo, 1e-9)
    norm = (verts - lo) / span
    mask = np.ones(len(verts), dtype=bool)
    axis, sign = _KNIFE_SIDES[side_name]
    if sign > 0:
        mask &= norm[:, axis] >= 0.52
    else:
        mask &= norm[:, axis] <= 0.48
    if along_name:
        b_axis, lo_n, hi_n = _KNIFE_BANDS[along_name]
        mask &= (norm[:, b_axis] >= lo_n) & (norm[:, b_axis] <= hi_n)
    if not np.any(mask):
        mask = np.ones(len(verts), dtype=bool)
    scores = verts[mask, axis] * float(sign)
    local = int(np.argmax(scores))
    return int(np.where(mask)[0][local])


def knife_lump_faces(
    mesh: trimesh.Trimesh,
    side: str,
    *,
    along: str = "",
    at: float | None = None,
    hops: int = 18,
) -> np.ndarray:
    """Faces of a connected protrusion on the mesh, not an AABB plane."""
    del at
    n = int(len(mesh.faces))
    out = np.zeros(n, dtype=bool)
    if n == 0 or int(len(mesh.vertices)) == 0:
        return out
    side_name = parse_knife_side(side)
    axis, sign = _KNIFE_SIDES[side_name]
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    scores = verts[:, axis] * float(sign)
    tip = knife_tip_index(mesh, side, along=along)
    tip_score = float(scores[tip])
    ordered = np.sort(scores)
    spread = float(ordered[-1] - ordered[0]) or 1.0
    k = max(1, int(0.45 * len(ordered)))
    shell = float(np.percentile(scores, 88.0))
    if k < len(ordered) - 2:
        window = ordered[k:]
        gaps = np.diff(window)
        j = int(np.argmax(gaps))
        if float(gaps[j]) > 0.06 * spread:
            shell = float(window[j])
    if tip_score <= shell + 1e-12:
        inward_lim = float(np.percentile(scores, 80.0))
    else:
        inward_lim = shell + 0.04 * (tip_score - shell)
    nbrs: list[set[int]] = [set() for _ in range(len(verts))]
    for a, b, c in faces:
        ia, ib, ic = int(a), int(b), int(c)
        nbrs[ia].update((ib, ic))
        nbrs[ib].update((ia, ic))
        nbrs[ic].update((ia, ib))
    lump = np.zeros(len(verts), dtype=bool)
    lump[tip] = True
    current = [tip]
    limit = max(4, min(int(hops), 28))
    for _ in range(limit):
        nxt: list[int] = []
        for v in current:
            for n_i in nbrs[v]:
                if lump[n_i]:
                    continue
                if float(scores[n_i]) < inward_lim:
                    continue
                nxt.append(n_i)
        if not nxt:
            break
        uniq = list(dict.fromkeys(nxt))
        if int(lump.sum()) >= 12:
            ring_score = float(np.median(scores[np.asarray(uniq, dtype=np.int64)]))
            entered_body = ring_score <= shell + 0.04 * max(tip_score - shell, 1e-9)
            exploded = len(uniq) >= max(24, int(3.5 * len(current)))
            if entered_body and exploded:
                break
        for n_i in uniq:
            lump[n_i] = True
        current = uniq
    counts = lump[faces].sum(axis=1)
    out = counts >= 2
    if not np.any(out):
        out = np.any(lump[faces], axis=1)
    return out


def infer_region(nx: float, ny: float, nz: float) -> str:
    """Best semantic name for a click, for the LLM brief — edits still use the pick box."""
    hits: list[tuple[float, str]] = []
    for name, box in REGION_BOXES.items():
        left, right, bottom, top, back, front = box
        if left <= nx <= right and bottom <= ny <= top and back <= nz <= front:
            vol = (right - left) * (top - bottom) * (front - back)
            hits.append((vol, name))
    if hits:
        hits.sort()
        return hits[0][1]
    point = np.array([nx, ny, nz], dtype=np.float64)
    best = "seat"
    best_d = 1e9
    for name, box in REGION_BOXES.items():
        left, right, bottom, top, back, front = box
        center = np.array(
            [(left + right) * 0.5, (bottom + top) * 0.5, (back + front) * 0.5],
            dtype=np.float64,
        )
        dist = float(np.linalg.norm(center - point))
        if dist < best_d:
            best_d = dist
            best = name
    return best


def vertex_mask_from_faces(mesh: trimesh.Trimesh, face_mask: np.ndarray) -> np.ndarray:
    verts = np.zeros(len(mesh.vertices), dtype=bool)
    if not np.any(face_mask):
        return verts
    idx = np.unique(np.asarray(mesh.faces, dtype=np.int64)[face_mask].ravel())
    verts[idx] = True
    return verts
