from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import trimesh

from mesh_forge.ops.geometry import carve_faces, carve_region, decimate, smooth_mesh, try_make_watertight
from mesh_forge.ops.region import (
    DEFAULT_PICK_RADIUS,
    faces_in_box,
    faces_near_pick,
    knife_lump_faces,
    knife_pick,
    parse_region,
    region_box,
    vertex_mask_from_faces,
)
from mesh_forge.ops.topo import face_mask_for_topo, format_topo, vertex_mask_for_topo

logger = logging.getLogger("mesh_forge.edit")

MatchHow = Literal["mirror", "height", "flat"]


class EditError(ValueError):
    """User-facing local-edit failure."""


def _label_box(
    region: str,
    box: tuple[float, float, float, float, float, float] | None,
) -> tuple[str, tuple[float, float, float, float, float, float]]:
    if box is not None:
        return region or "pick", box
    name = parse_region(region)
    return name, region_box(name)


def remove_in_region(
    mesh: trimesh.Trimesh,
    region: str,
    *,
    close: bool = False,
    box: tuple[float, float, float, float, float, float] | None = None,
    protect_sides: bool = True,
    pick: list[float] | tuple[float, ...] | None = None,
    knife: str | None = None,
    along: str = "",
    at: float | None = None,
    hops: int | None = None,
    topo: dict[str, Any] | None = None,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    if topo:
        out, stats = remove_topo(mesh, topo)
        if close:
            out = try_make_watertight(out)
            stats["closed"] = True
        return out, stats
    if knife:
        return knife_bisect(
            mesh,
            knife,
            at=at,
            close=close,
            along=along,
            hops=hops,
        )
    if pick is not None and len(pick) >= 3:
        out, stats = remove_near_pick(mesh, pick)
        if close:
            out = try_make_watertight(out)
            stats["closed"] = True
        stats["region"] = region or "pick"
        return out, stats
    label, target = _label_box(region, box)
    out, stats = carve_region(mesh, target, action="remove", protect_sides=protect_sides)
    if close:
        out = try_make_watertight(out)
        stats["closed"] = True
    stats["region"] = label
    return out, stats


def knife_bisect(
    mesh: trimesh.Trimesh,
    side: str | None,
    *,
    at: float | None = None,
    close: bool = False,
    along: str = "",
    hops: int | None = None,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Delete a connected protrusion on the mesh surface (not an AABB plane)."""
    if not side:
        raise EditError("Need knife=left|right|top|bottom|front|back.")
    drop = knife_lump_faces(mesh, side, along=along or "", at=at, hops=int(hops or 18))
    if not np.any(drop):
        raise EditError("Knife found no protruding faces. Change side or along.")
    try:
        out, stats = carve_faces(mesh, drop, min_keep_ratio=0.50, min_keep_faces=8)
    except Exception as exc:
        raise EditError(str(exc)) from exc
    if close:
        try:
            trimesh.repair.fill_holes(out)
        except Exception:
            logger.warning("knife cap fill_holes failed", exc_info=True)
        out = try_make_watertight(out)
        stats["closed"] = True
    stats["region"] = f"knife:{side}" + (f" along={along}" if along else "")
    return out, stats


def remove_topo(
    mesh: trimesh.Trimesh,
    topo: dict[str, Any],
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    drop = face_mask_for_topo(mesh, topo)
    if not np.any(drop):
        raise EditError("That vertex/edge/face is gone. Click again.")
    out, stats = carve_faces(mesh, drop, min_keep_ratio=0.50, min_keep_faces=8)
    stats["region"] = format_topo(topo) or "topo"
    return out, stats


def remove_near_pick(
    mesh: trimesh.Trimesh,
    pick: list[float] | tuple[float, ...],
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    nx, ny, nz = float(pick[0]), float(pick[1]), float(pick[2])
    radius = float(pick[3]) if len(pick) > 3 else DEFAULT_PICK_RADIUS
    drop = faces_near_pick(mesh, nx, ny, nz, radius)
    out, stats = carve_faces(mesh, drop, min_keep_ratio=0.92, min_keep_faces=80)
    stats["region"] = "pick"
    return out, stats


def fill_in_region(
    mesh: trimesh.Trimesh,
    region: str,
    *,
    box: tuple[float, float, float, float, float, float] | None = None,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    mesh = mesh.copy()
    label, target = _label_box(region, box)
    mask = faces_in_box(mesh, target)
    if not np.any(mask):
        raise EditError(f"Nothing in region {region!r}. look, then pick another region.")
    inside_idx = np.where(mask)[0]
    outside_idx = np.where(~mask)[0]
    patch = mesh.submesh([inside_idx], append=True, repair=False)
    if patch is None or len(patch.faces) == 0:
        raise EditError(f"Could not extract region {region!r}.")
    before = int(len(patch.faces))
    try:
        trimesh.repair.fill_holes(patch)
    except Exception as exc:
        logger.warning("fill_holes failed: %s", exc)
    after = int(len(patch.faces))
    if len(outside_idx) == 0:
        out = patch
    else:
        rest = mesh.submesh([outside_idx], append=True, repair=False)
        out = trimesh.util.concatenate([rest, patch])
        try:
            out.merge_vertices()
            out.remove_unreferenced_vertices()
        except Exception:
            pass
    return out, {
        "region": label,
        "faces_before": int(len(mesh.faces)),
        "faces_after": int(len(out.faces)),
        "faces_added": max(0, after - before),
    }


def split_in_region(
    mesh: trimesh.Trimesh,
    region: str,
    *,
    box: tuple[float, float, float, float, float, float] | None = None,
    protect_sides: bool = True,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Delete a thin slice through the region so two pieces stay in one mesh."""
    label, target = _label_box(region, box)
    left, right, bottom, top, back, front = target
    wx, wy, wz = right - left, top - bottom, front - back
    if wx >= wz:
        mid = 0.5 * (left + right)
        pad = max(0.08, min(0.16, 0.22 * wx))
        slice_box = (mid - pad, mid + pad, bottom, top, back, front)
    else:
        mid = 0.5 * (back + front)
        pad = max(0.08, min(0.16, 0.22 * wz))
        slice_box = (left, right, bottom, top, mid - pad, mid + pad)
    out, stats = carve_region(
        mesh,
        slice_box,
        action="remove",
        min_keep_ratio=0.08,
        min_keep_faces=80,
        drop_crumbs=False,
        protect_sides=protect_sides,
    )
    stats["region"] = label
    return out, stats


def join_in_region(
    mesh: trimesh.Trimesh,
    region: str,
    *,
    box: tuple[float, float, float, float, float, float] | None = None,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Weld close vertices in the region and fill small gaps."""
    mesh = mesh.copy()
    label, target = _label_box(region, box)
    mask = faces_in_box(mesh, target)
    if not np.any(mask):
        raise EditError(f"Nothing in region {region!r}. look, then pick another region.")
    vmask = vertex_mask_from_faces(mesh, mask)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    idx = np.where(vmask)[0]
    if len(idx) < 6:
        raise EditError(f"Too few vertices in region {region!r} to join.")
    span = float(np.linalg.norm(np.ptp(verts[idx], axis=0)))
    merge_d = max(span * 0.03, 1e-4)
    _weld_vertices(mesh, idx, merge_d)
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    return mesh, {
        "region": label,
        "merge_mm": merge_d,
        "faces_after": int(len(mesh.faces)),
    }


def match_in_region(
    mesh: trimesh.Trimesh,
    region: str,
    how: MatchHow,
    *,
    box: tuple[float, float, float, float, float, float] | None = None,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    how = (how or "mirror").strip().lower()  # type: ignore[assignment]
    if how not in {"mirror", "height", "flat"}:
        raise EditError("how must be mirror, height, or flat.")
    if how == "mirror":
        return _match_mirror(mesh, region, box=box)
    if how == "height":
        return _match_height(mesh, region, box=box)
    return _match_flat(mesh, region, box=box)


def smooth_in_region(
    mesh: trimesh.Trimesh,
    region: str | None,
    iterations: int,
    *,
    box: tuple[float, float, float, float, float, float] | None = None,
    topo: dict[str, Any] | None = None,
) -> trimesh.Trimesh:
    if topo:
        mesh = mesh.copy()
        vmask = vertex_mask_for_topo(mesh, topo)
        if not np.any(vmask):
            raise EditError("That vertex/edge/face is gone. Click again.")
        steps = max(1, min(int(iterations), 5))
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        neighbors = _vertex_neighbors(len(verts), faces)
        movable = np.where(vmask)[0]
        for _ in range(steps):
            nxt = verts.copy()
            for i in movable:
                nbr = neighbors[i]
                if not nbr:
                    continue
                nxt[i] = verts[np.asarray(nbr, dtype=np.int64)].mean(axis=0)
            verts = nxt
        mesh.vertices = verts
        return mesh
    if box is None and not (region or "").strip():
        return smooth_mesh(mesh, iterations)
    mesh = mesh.copy()
    _, target = _label_box(region or "pick", box)
    mask = faces_in_box(mesh, target)
    if not np.any(mask):
        raise EditError(f"Nothing in region {region!r}. look, then pick another region.")
    vmask = vertex_mask_from_faces(mesh, mask)
    vmask = _interior_vertices(mesh, vmask)
    if not np.any(vmask):
        return mesh
    steps = max(1, min(int(iterations), 5))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    neighbors = _vertex_neighbors(len(verts), faces)
    movable = np.where(vmask)[0]
    for _ in range(steps):
        nxt = verts.copy()
        for i in movable:
            nbr = neighbors[i]
            if not nbr:
                continue
            nxt[i] = verts[np.asarray(nbr, dtype=np.int64)].mean(axis=0)
        verts = nxt
    mesh.vertices = verts
    return mesh


def remesh_mesh(mesh: trimesh.Trimesh, target_faces: int | None = None) -> trimesh.Trimesh:
    n = int(len(mesh.faces))
    target = int(target_faces) if target_faces else max(1000, n // 2)
    return decimate(mesh, target)


def extract_in_region(
    mesh: trimesh.Trimesh,
    region: str,
    *,
    box: tuple[float, float, float, float, float, float] | None = None,
    pick: list[float] | tuple[float, ...] | None = None,
    knife: str | None = None,
    along: str = "",
    topo: dict[str, Any] | None = None,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, dict[str, Any]]:
    if topo:
        mask = face_mask_for_topo(mesh, topo)
    else:
        mask = _edit_face_mask(mesh, region, box=box, pick=pick, knife=knife, along=along)
    n = int(len(mesh.faces))
    if pick is not None or knife:
        mask = _extract_components(mesh, mask)
    dropped = int(mask.sum())
    if dropped < 1:
        raise EditError("Nothing to extract. Click the piece or pass knife/region.")
    if dropped > int(0.85 * n):
        raise EditError("Extract would take almost the whole mesh. Use a click or knife.")
    piece_idx = np.where(mask)[0]
    rest_idx = np.where(~mask)[0]
    piece = mesh.submesh([piece_idx], append=True, repair=False)
    rest = mesh.submesh([rest_idx], append=True, repair=False)
    if piece is None or rest is None or len(piece.faces) == 0 or len(rest.faces) == 0:
        raise EditError("Extract failed to split the mesh.")
    if topo:
        label = format_topo(topo)
    else:
        label = f"knife:{knife}" if knife else (region or "pick")
    return rest, piece, {
        "region": label,
        "faces_extracted": int(len(piece.faces)),
        "faces_left": int(len(rest.faces)),
    }


def offset_in_region(
    mesh: trimesh.Trimesh,
    region: str,
    amount: float,
    *,
    box: tuple[float, float, float, float, float, float] | None = None,
    pick: list[float] | tuple[float, ...] | None = None,
    knife: str | None = None,
    along: str = "",
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    delta = max(-0.25, min(0.25, float(amount)))
    if abs(delta) < 1e-6:
        raise EditError("offset amount is 0. Use +inflate or -deflate, e.g. 0.04.")
    mesh = mesh.copy()
    mask = _edit_face_mask(mesh, region, box=box, pick=pick, knife=knife, along=along)
    if not np.any(mask):
        raise EditError("Nothing in this region to offset.")
    vmask = _interior_vertices(mesh, vertex_mask_from_faces(mesh, mask))
    if not np.any(vmask):
        vmask = vertex_mask_from_faces(mesh, mask)
    try:
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    except Exception as exc:
        raise EditError(f"No vertex normals for offset: {exc}") from exc
    scale = delta * float(np.max(np.asarray(mesh.extents, dtype=np.float64)))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    verts[vmask] = verts[vmask] + normals[vmask] * scale
    mesh.vertices = verts
    label = f"knife:{knife}" if knife else (region or "pick")
    return mesh, {"region": label, "amount": delta, "mm": scale}


def add_primitive_in_region(
    mesh: trimesh.Trimesh,
    region: str,
    shape: str,
    *,
    box: tuple[float, float, float, float, float, float] | None = None,
    pick: list[float] | tuple[float, ...] | None = None,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    kind = (shape or "cylinder").strip().lower()
    if kind not in {"cylinder", "box", "sphere"}:
        raise EditError("shape must be cylinder, box, or sphere.")
    label, target = _label_box(region or "pick", box)
    wlo, whi = _world_box(mesh, target)
    if pick is not None and len(pick) >= 3:
        from mesh_forge.ops.region import pick_world_point

        center = pick_world_point(mesh, float(pick[0]), float(pick[1]), float(pick[2]))
        span = float(np.max(np.asarray(mesh.extents, dtype=np.float64)))
        half = np.array([0.04, 0.08, 0.04], dtype=np.float64) * span
        wlo, whi = center - half, center + half
        label = "pick"
    size = np.maximum(whi - wlo, 1e-4)
    center = 0.5 * (wlo + whi)
    if kind == "box":
        prim = trimesh.creation.box(extents=size)
    elif kind == "sphere":
        prim = trimesh.creation.icosphere(subdivisions=2, radius=0.5 * float(np.min(size)))
    else:
        prim = trimesh.creation.cylinder(
            radius=0.5 * float(min(size[0], size[2])),
            height=float(size[1]),
            sections=24,
        )
    prim.apply_translation(center)
    out = trimesh.util.concatenate([mesh, prim])
    try:
        out.merge_vertices()
        out.remove_unreferenced_vertices()
    except Exception:
        pass
    return out, {
        "region": label,
        "shape": kind,
        "faces_added": int(len(prim.faces)),
        "faces_after": int(len(out.faces)),
    }


def restore_patch_in_region(
    mesh: trimesh.Trimesh,
    region: str,
    *,
    box: tuple[float, float, float, float, float, float] | None = None,
    pick: list[float] | tuple[float, ...] | None = None,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Rebuild missing surface in the region: fill holes, else mirror the intact side."""
    try:
        filled, stats = fill_in_region(mesh, region, box=box)
        if int(stats.get("faces_added") or 0) > 0:
            stats["how"] = "fill"
            return filled, stats
    except EditError:
        filled = None
        stats = {}
    try:
        mirrored, mstats = _match_mirror(mesh, region, box=box)
        mstats["how"] = "mirror"
        return mirrored, mstats
    except EditError as exc:
        if filled is not None:
            stats["how"] = "fill"
            return filled, stats
        raise EditError(
            f"Nothing to restore in region {region!r}. {exc}"
        ) from exc


def _world_box(
    mesh: trimesh.Trimesh,
    box: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    span = np.maximum(hi - lo, 1e-9)
    left, right, bottom, top, back, front = box
    wlo = lo + span * np.array([left, bottom, back], dtype=np.float64)
    whi = lo + span * np.array([right, top, front], dtype=np.float64)
    return wlo, whi


def _extract_components(mesh: trimesh.Trimesh, seed: np.ndarray) -> np.ndarray:
    """If the click sits on a small separate body, take that whole body."""
    n = int(len(mesh.faces))
    if n == 0 or not np.any(seed):
        return seed
    limit = max(12, int(0.20 * n))
    out = np.zeros(n, dtype=bool)
    used_small = False
    for group in _face_components(mesh, np.ones(n, dtype=bool)):
        if not np.any(seed[group]):
            continue
        if len(group) <= limit:
            out[group] = True
            used_small = True
        else:
            out[group] = seed[group]
    return out if used_small or np.any(out) else seed


def _edit_face_mask(
    mesh: trimesh.Trimesh,
    region: str,
    *,
    box: tuple[float, float, float, float, float, float] | None = None,
    pick: list[float] | tuple[float, ...] | None = None,
    knife: str | None = None,
    along: str = "",
) -> np.ndarray:
    if pick is not None and len(pick) >= 3:
        radius = float(pick[3]) if len(pick) > 3 else DEFAULT_PICK_RADIUS
        return faces_near_pick(mesh, float(pick[0]), float(pick[1]), float(pick[2]), radius)
    if knife:
        tip = knife_pick(mesh, knife, along)
        return faces_near_pick(mesh, tip[0], tip[1], tip[2])
    _, target = _label_box(region, box)
    return faces_in_box(mesh, target)


def _match_mirror(
    mesh: trimesh.Trimesh,
    region: str,
    *,
    box: tuple[float, float, float, float, float, float] | None = None,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    mesh = mesh.copy()
    label, target = _label_box(region, box)
    mask = faces_in_box(mesh, target)
    if int(mask.sum()) < 8:
        raise EditError(f"Not enough faces in region {region!r} to mirror.")
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    mid_x = float(np.mean(mesh.bounds[:, 0]))
    left = mask & (centers[:, 0] < mid_x)
    right = mask & (centers[:, 0] >= mid_x)
    left_a = float(areas[left].sum())
    right_a = float(areas[right].sum())
    if left_a < 1e-12 or right_a < 1e-12:
        raise EditError("Need geometry on both left and right in this region for mirror.")
    donor = left if left_a >= right_a else right
    drop = right if left_a >= right_a else left
    keep = ~drop
    donor_idx = np.where(donor)[0]
    rest = mesh.submesh([np.where(keep)[0]], append=True, repair=False)
    piece = mesh.submesh([donor_idx], append=True, repair=False)
    piece = piece.copy()
    piece.vertices[:, 0] = 2.0 * mid_x - piece.vertices[:, 0]
    out = trimesh.util.concatenate([rest, piece])
    try:
        out.merge_vertices()
        out.remove_unreferenced_vertices()
    except Exception:
        pass
    return out, {
        "region": label,
        "how": "mirror",
        "donor": "left" if left_a >= right_a else "right",
        "faces_after": int(len(out.faces)),
    }


def _match_height(
    mesh: trimesh.Trimesh,
    region: str,
    *,
    box: tuple[float, float, float, float, float, float] | None = None,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    mesh = mesh.copy()
    label, target = _label_box(region, box)
    mask = faces_in_box(mesh, target)
    if not np.any(mask):
        raise EditError(f"Nothing in region {region!r}.")
    groups = _face_components(mesh, mask)
    if len(groups) < 2:
        raise EditError("Need at least two separate parts in this region to equalize height.")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    spans: list[tuple[np.ndarray, float, float, float]] = []
    for faces_idx in groups:
        vidx = np.unique(faces[faces_idx].ravel())
        ys = verts[vidx, 1]
        y0, y1 = float(ys.min()), float(ys.max())
        spans.append((vidx, y0, y1, y1 - y0))
    target_span = max(s[3] for s in spans)
    target_ymin = min(s[1] for s in spans)
    if target_span < 1e-9:
        raise EditError("Region parts have no height to match.")
    for vidx, y0, y1, span in spans:
        if span < 1e-9:
            verts[vidx, 1] += target_ymin - y0
            continue
        scale = target_span / span
        verts[vidx, 1] = y1 - (y1 - verts[vidx, 1]) * scale
        new_min = float(verts[vidx, 1].min())
        verts[vidx, 1] += target_ymin - new_min
    mesh.vertices = verts
    return mesh, {
        "region": label,
        "how": "height",
        "parts": len(groups),
        "faces_after": int(len(mesh.faces)),
    }


def _match_flat(
    mesh: trimesh.Trimesh,
    region: str,
    *,
    box: tuple[float, float, float, float, float, float] | None = None,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    mesh = mesh.copy()
    label, target = _label_box(region, box)
    mask = faces_in_box(mesh, target)
    vmask = vertex_mask_from_faces(mesh, mask)
    interior = _interior_vertices(mesh, vmask)
    targets = interior if np.any(interior) else vmask
    if not np.any(targets):
        raise EditError(f"Nothing to flatten in region {region!r}.")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    plane_y = float(np.median(verts[targets, 1]))
    verts[targets, 1] = plane_y
    mesh.vertices = verts
    return mesh, {
        "region": label,
        "how": "flat",
        "plane_y": plane_y,
        "faces_after": int(len(mesh.faces)),
    }


def _face_components(mesh: trimesh.Trimesh, face_mask: np.ndarray) -> list[np.ndarray]:
    idx = np.where(face_mask)[0]
    if len(idx) == 0:
        return []
    try:
        sub = mesh.submesh([idx], append=True, repair=False)
        if sub is None or len(getattr(sub, "face_adjacency", [])) == 0:
            return [idx]
        comps = list(
            trimesh.graph.connected_components(
                sub.face_adjacency,
                min_len=1,
                nodes=np.arange(len(sub.faces)),
            )
        )
        return [idx[np.asarray(c, dtype=np.int64)] for c in comps]
    except Exception:
        return [idx]


def _vertex_neighbors(n_verts: int, faces: np.ndarray) -> list[list[int]]:
    nbrs: list[set[int]] = [set() for _ in range(n_verts)]
    for a, b, c in faces:
        nbrs[int(a)].update((int(b), int(c)))
        nbrs[int(b)].update((int(a), int(c)))
        nbrs[int(c)].update((int(a), int(b)))
    return [list(s) for s in nbrs]


def _slice_keep_plane(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    normal: np.ndarray,
) -> trimesh.Trimesh:
    """Keep the half-space (v-origin)·normal >= 0. Splits triangles that cross the plane."""
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    normal = np.asarray(normal, dtype=np.float64).reshape(3)
    normal = normal / float(np.linalg.norm(normal) or 1.0)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    dist = (verts - origin) @ normal
    out_verts = verts.tolist()
    cache: dict[tuple[int, int], int] = {}

    def split_edge(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        found = cache.get(key)
        if found is not None:
            return found
        di, dj = float(dist[i]), float(dist[j])
        t = di / (di - dj) if abs(di - dj) > 1e-18 else 0.5
        t = max(0.0, min(1.0, t))
        point = verts[i] + t * (verts[j] - verts[i])
        idx = len(out_verts)
        out_verts.append(point.tolist())
        cache[key] = idx
        return idx

    new_faces: list[list[int]] = []
    eps = 1e-12
    for a, b, c in faces:
        tri = (int(a), int(b), int(c))
        keep = [float(dist[i]) >= -eps for i in tri]
        nkeep = int(sum(keep))
        if nkeep == 3:
            new_faces.append([tri[0], tri[1], tri[2]])
            continue
        if nkeep == 0:
            continue
        if nkeep == 2:
            out_k = next(k for k in range(3) if not keep[k])
            i0, i1 = (out_k + 1) % 3, (out_k + 2) % 3
            p0, p1, o = tri[i0], tri[i1], tri[out_k]
            x0 = split_edge(p0, o)
            x1 = split_edge(p1, o)
            new_faces.append([p0, p1, x1])
            new_faces.append([p0, x1, x0])
        else:
            in_k = next(k for k in range(3) if keep[k])
            p = tri[in_k]
            o0, o1 = tri[(in_k + 1) % 3], tri[(in_k + 2) % 3]
            x0 = split_edge(p, o0)
            x1 = split_edge(p, o1)
            new_faces.append([p, x0, x1])
    if not new_faces:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))
    out = trimesh.Trimesh(
        vertices=np.asarray(out_verts, dtype=np.float64),
        faces=np.asarray(new_faces, dtype=np.int64),
        process=False,
    )
    try:
        out.remove_unreferenced_vertices()
    except Exception:
        pass
    return out


def _interior_vertices(mesh: trimesh.Trimesh, vmask: np.ndarray) -> np.ndarray:
    """Drop vertices that also belong to faces outside the mask (the seam)."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    outside = ~vmask
    if not np.any(outside):
        return vmask
    seam = np.zeros(len(vmask), dtype=bool)
    for tri in faces:
        inside = vmask[tri]
        if bool(inside.any()) and not bool(inside.all()):
            seam[tri] = True
    out = vmask.copy()
    out[seam] = False
    return out


def _weld_vertices(mesh: trimesh.Trimesh, indices: np.ndarray, distance: float) -> None:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    pts = verts[indices]
    used = np.zeros(len(pts), dtype=bool)
    remap = {int(i): int(i) for i in indices}
    for a, src in enumerate(indices):
        if used[a]:
            continue
        d = np.linalg.norm(pts - pts[a], axis=1)
        cluster = np.where((~used) & (d <= distance))[0]
        used[cluster] = True
        mean = pts[cluster].mean(axis=0)
        root = int(indices[int(cluster[0])])
        verts[root] = mean
        for c in cluster:
            remap[int(indices[int(c)])] = root
    faces = np.asarray(mesh.faces, dtype=np.int64)
    mapped = np.fromiter(
        (remap.get(int(i), int(i)) for i in faces.ravel()),
        dtype=np.int64,
        count=faces.size,
    )
    mesh.vertices = verts
    mesh.faces = mapped.reshape(faces.shape)
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
    except Exception:
        pass
