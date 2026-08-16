from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

logger = logging.getLogger("mesh_forge.geometry")


def read_trimesh(path: Path):
    """Load a mesh file, including GLB bytes that were saved with a .stl suffix."""
    data = path.read_bytes()
    if data[:4] == b"glTF":
        loaded = trimesh.load(io.BytesIO(data), file_type="glb", force="mesh", process=False)
    else:
        loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if hasattr(g, "vertices") and len(g.vertices) > 0]
        if not geoms:
            return loaded
        loaded = trimesh.util.concatenate(geoms)
    return loaded


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = read_trimesh(path)
    if isinstance(loaded, trimesh.Scene):
        raise ValueError(f"No geometry in mesh file: {path}")
    if len(getattr(loaded, "vertices", [])) == 0:
        raise ValueError(f"Mesh has no vertices: {path}")
    # STL is a triangle soup (3 unique verts per face). Topology ops need welds.
    return _weld_mesh_vertices(loaded)


def save_mesh(mesh: trimesh.Trimesh, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return path


def orient_upright(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Seat the mesh with the most stable base on -Y ground (viewer up = +Y).

    Tries the 6 axis-aligned "which axis is up" poses and keeps the one with the
    largest bottom footprint and lowest center of mass. Longest-axis→up was wrong
    for lying animals (tipped them onto nose/tail).
    """
    mesh = mesh.copy()
    if len(mesh.vertices) == 0:
        return mesh

    best = None
    best_score = -1.0
    for matrix in _axis_up_rotations():
        cand = mesh.copy()
        if matrix is not None:
            cand.apply_transform(matrix)
        cand = _seat_on_ground(cand)
        score = _base_stability_score(cand)
        if score > best_score:
            best_score = score
            best = cand
    return best if best is not None else _seat_on_ground(mesh)


def _axis_up_rotations() -> list:
    """Identity + rotations that map ±X / ±Z onto +Y."""
    R = trimesh.transformations.rotation_matrix
    return [
        None,  # +Y already up
        R(np.pi, [1, 0, 0]),  # -Y -> +Y
        R(np.pi / 2, [0, 0, 1]),  # +X -> +Y
        R(-np.pi / 2, [0, 0, 1]),  # -X -> +Y
        R(-np.pi / 2, [1, 0, 0]),  # +Z -> +Y
        R(np.pi / 2, [1, 0, 0]),  # -Z -> +Y
    ]


def _seat_on_ground(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    bounds = mesh.bounds
    mesh.apply_translation([
        -(bounds[0][0] + bounds[1][0]) / 2,
        -bounds[0][1],
        -(bounds[0][2] + bounds[1][2]) / 2,
    ])
    return mesh


def _footprint_score(mesh: trimesh.Trimesh, *, end: str) -> float:
    """XZ extent of vertices near the low or high end of the Y axis."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(verts) == 0:
        return 0.0
    ys = verts[:, 1]
    y_min = float(ys.min())
    y_max = float(ys.max())
    span = max(y_max - y_min, 1e-9)
    band = 0.12 * span
    if end == "low":
        mask = ys <= (y_min + band)
    else:
        mask = ys >= (y_max - band)
    slice_pts = verts[mask]
    if len(slice_pts) < 8:
        return float(len(slice_pts))
    return float(np.ptp(slice_pts[:, 0]) * np.ptp(slice_pts[:, 2])) * (1.0 + 0.01 * len(slice_pts))


def _base_stability_score(mesh: trimesh.Trimesh) -> float:
    """Higher = flatter, wider base and lower center of mass."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(verts) < 8:
        return 0.0
    ys = verts[:, 1]
    y_min = float(ys.min())
    y_max = float(ys.max())
    span = max(y_max - y_min, 1e-9)
    footprint = _footprint_score(mesh, end="low")
    com_y = float(np.mean(ys))
    # 0 at ground, 1 at top
    com_rel = (com_y - y_min) / span
    return float(footprint) / (0.25 + com_rel)


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


def keep_largest_component(
    mesh: trimesh.Trimesh,
    *,
    single: bool = True,
    min_ratio: float = 0.02,
) -> trimesh.Trimesh:
    """Drop floaters / duplicate Hunyuan bodies; keep the primary subject.

    ``single=True`` (default): keep the largest component **plus** nearby
    fragments that belong to the same body (thin parts often disconnect). Far
    twin shells of similar size are dropped.

    ``single=False``: keep every component above ``min_ratio`` of the largest.
    """
    mesh = mesh.copy()
    try:
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    if len(mesh.faces) == 0:
        return mesh

    components: list[np.ndarray] = []
    try:
        if len(getattr(mesh, "face_adjacency", [])) == 0:
            try:
                trimesh.grouping.merge_vertices(mesh, digits_vertex=5)
                mesh.remove_unreferenced_vertices()
                mesh._cache.clear()
            except Exception:
                pass
        if len(getattr(mesh, "face_adjacency", [])) > 0:
            components = list(
                trimesh.graph.connected_components(
                    mesh.face_adjacency,
                    min_len=1,
                    nodes=np.arange(len(mesh.faces)),
                )
            )
    except Exception as exc:
        logger.debug("face adjacency components failed: %s", exc)

    if len(components) <= 1:
        if len(getattr(mesh, "face_adjacency", [])) == 0:
            logger.debug("keep_largest_component: no face adjacency; skip")
        return mesh

    sizes = [len(c) for c in components]
    largest_i = int(np.argmax(sizes))
    largest = sizes[largest_i]
    n_faces = len(mesh.faces)
    # Cracked reconstructions: thousands of tiny patches, no real "primary shell".
    if largest <= 4 and len(components) > 32:
        logger.warning(
            "keep_largest_component: fragmented adjacency (%d parts, largest=%d); skip",
            len(components),
            largest,
        )
        return mesh
    if largest < max(32, int(0.08 * n_faces)):
        logger.warning(
            "keep_largest_component: largest part is only %d/%d faces; skip",
            largest,
            n_faces,
        )
        return mesh

    before = mesh
    if single:
        keep_idx = _select_primary_body_faces(mesh, components, largest_i)
    else:
        threshold = max(50, int(min_ratio * largest))
        keep_idx = np.concatenate([c for c in components if len(c) >= threshold])

    if len(keep_idx) >= n_faces:
        return mesh
    mask = np.zeros(n_faces, dtype=bool)
    mask[np.asarray(keep_idx, dtype=np.int64)] = True
    dropped = int((~mask).sum())
    kept_n = 0
    dropped_parts = 0
    for c in components:
        idx = np.asarray(c, dtype=np.int64)
        if not idx.size:
            continue
        if bool(mask[idx].all()):
            kept_n += 1
        else:
            dropped_parts += 1
    # Cracked seats/cushions: thousands of interior patches, not one far twin.
    if dropped > int(0.12 * n_faces) and dropped_parts > 32:
        logger.warning(
            "keep_largest_component: would drop %d/%d faces across %d parts; skip",
            dropped,
            n_faces,
            dropped_parts,
        )
        return before
    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()
    logger.info(
        "keep_largest_component: %d parts -> kept %d (%d faces, dropped %d, single=%s)",
        len(components),
        kept_n,
        int(mask.sum()),
        dropped,
        single,
    )
    return mesh


def _select_primary_body_faces(
    mesh: trimesh.Trimesh,
    components: list[np.ndarray],
    largest_i: int,
) -> np.ndarray:
    """Largest shell + every fragment that still sits inside the body.

    Hunyuan chairs/sofas crack into thousands of patches. Size and
    paper-thin filters used to delete the seat/back and leave a skeleton.
    Only drop spatial outliers and far duplicate bodies.
    """
    seed = np.asarray(components[largest_i], dtype=np.int64)
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    seed_pts = centers[seed]
    seed_centroid = seed_pts.mean(axis=0)
    seed_lo = seed_pts.min(axis=0)
    seed_hi = seed_pts.max(axis=0)
    diag = float(np.linalg.norm(seed_hi - seed_lo))
    pad = max(0.55 * diag, 1e-6)
    lo, hi = seed_lo - pad, seed_hi + pad
    seed_faces = int(seed.size)

    kept: list[np.ndarray] = [seed]
    for i, faces in enumerate(components):
        if i == largest_i:
            continue
        idx = np.asarray(faces, dtype=np.int64)
        n = int(idx.size)
        centroid = centers[idx].mean(axis=0)
        # Far twin of similar size (classic Hunyuan double body)
        if n >= 0.45 * seed_faces:
            dist = float(np.linalg.norm(centroid - seed_centroid))
            if dist > 0.45 * diag:
                continue
        if not (np.all(centroid >= lo) and np.all(centroid <= hi)):
            continue
        kept.append(idx)
    return np.concatenate(kept)


# Normalized AABB: X left→right, Y bottom→top, Z back→front (same as look cameras).
_CARVE_SIDES = {
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
    "back": "back",
    "сзади": "back",
}


class CarveError(ValueError):
    """User-facing carve failure (empty cut, too aggressive, missing region)."""


def normalize_carve_side(side: str | None) -> str:
    raw = (side or "").strip().lower()
    if not raw:
        return ""
    mapped = _CARVE_SIDES.get(raw, raw)
    if mapped not in {"left", "right", "front", "back", "top", "bottom"}:
        raise CarveError(f"Unknown side {side!r}. Use left/right/front/back/top/bottom.")
    return mapped


def carve_box_from_side(side: str, amount: float) -> tuple[float, float, float, float, float, float]:
    """Slab from one AABB side. Returns (left, right, bottom, top, back, front) in 0–1."""
    amount = max(0.02, min(float(amount), 0.28))
    left, right, bottom, top, back, front = 0.0, 1.0, 0.0, 1.0, 0.0, 1.0
    if side == "left":
        right = amount
    elif side == "right":
        left = 1.0 - amount
    elif side == "bottom":
        top = amount
    elif side == "top":
        bottom = 1.0 - amount
    elif side == "back":
        front = amount
    elif side == "front":
        back = 1.0 - amount
    return left, right, bottom, top, back, front


def resolve_carve_box(
    *,
    side: str | None = None,
    amount: float = 0.10,
    left: float | None = None,
    right: float | None = None,
    bottom: float | None = None,
    top: float | None = None,
    back: float | None = None,
    front: float | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Build a 0–1 AABB. ``side``+``amount`` sets a slab; explicit bounds override that axis."""
    mapped = normalize_carve_side(side)
    explicit = any(v is not None for v in (left, right, bottom, top, back, front))
    if not mapped and not explicit:
        raise CarveError(
            "Specify side (left/right/front/back/top/bottom) or a box "
            "(left/right/bottom/top/back/front as 0–1 of bbox)."
        )
    box = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    if mapped:
        box = list(carve_box_from_side(mapped, amount))
    for i, value in enumerate((left, right, bottom, top, back, front)):
        if value is not None:
            box[i] = float(value)
    left_n, right_n = sorted((max(0.0, min(box[0], 1.0)), max(0.0, min(box[1], 1.0))))
    bottom_n, top_n = sorted((max(0.0, min(box[2], 1.0)), max(0.0, min(box[3], 1.0))))
    back_n, front_n = sorted((max(0.0, min(box[4], 1.0)), max(0.0, min(box[5], 1.0))))
    if right_n - left_n < 0.01 or top_n - bottom_n < 0.01 or front_n - back_n < 0.01:
        raise CarveError("Carve box is too thin. Widen left/right, bottom/top, or back/front.")
    return left_n, right_n, bottom_n, top_n, back_n, front_n


def carve_constrained_axes(box: tuple[float, float, float, float, float, float]) -> tuple[bool, bool, bool]:
    """True when that axis is not a full 0–1 span."""
    left, right, bottom, top, back, front = box
    return (right - left) < 0.98, (top - bottom) < 0.98, (front - back) < 0.98


def carve_region(
    mesh: trimesh.Trimesh,
    box: tuple[float, float, float, float, float, float],
    *,
    action: str = "remove",
    min_keep_ratio: float = 0.12,
    min_keep_faces: int = 200,
    drop_crumbs: bool = True,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Keep or delete faces whose centroids fall in a normalized AABB.

    ``box`` is (left, right, bottom, top, back, front) in 0–1 of the current
    bounds. Axes match look: +X right, +Y up, +Z front.
    """
    action = (action or "remove").strip().lower()
    if action not in {"remove", "keep"}:
        raise CarveError("action must be remove or keep.")
    mesh = mesh.copy()
    n_faces = int(len(mesh.faces))
    if n_faces == 0:
        raise CarveError("Mesh has no faces to carve.")
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    span = np.maximum(hi - lo, 1e-9)
    left, right, bottom, top, back, front = box
    box_lo = lo + span * np.array([left, bottom, back], dtype=np.float64)
    box_hi = lo + span * np.array([right, top, front], dtype=np.float64)
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    inside = np.all((centers >= box_lo) & (centers <= box_hi), axis=1)
    keep_mask = inside if action == "keep" else ~inside
    kept = int(keep_mask.sum())
    dropped = n_faces - kept
    if dropped <= 0:
        raise CarveError("Nothing in that region. look, then tighten/shift the box or side.")
    if kept < max(min_keep_faces, int(min_keep_ratio * n_faces)):
        raise CarveError(
            f"Carve would leave {kept}/{n_faces} faces — too aggressive. "
            "restore_mesh if already saved, then use a smaller amount or a tighter box."
        )
    cx, cy, cz = carve_constrained_axes(box)
    if action == "remove" and sum((cx, cy, cz)) < 2:
        raise CarveError(
            "A full-side slab also cuts the armrest and legs. "
            "For a backrest wing: side=right|left, amount=0.08–0.14, bottom=0.45, front=0.55. "
            "Always set a second bound (bottom/top or back/front)."
        )
    before_left, before_right = _side_band_area(mesh)
    mesh.update_faces(keep_mask)
    mesh.remove_unreferenced_vertices()
    if drop_crumbs:
        mesh = _drop_tiny_components(mesh)
    if action == "remove" and cx:
        _assert_carve_keeps_sides(mesh, before_left, before_right)
    logger.info(
        "carve_region: action=%s dropped %d/%d faces box=(L%.2f R%.2f B%.2f T%.2f Bk%.2f F%.2f)",
        action,
        dropped,
        n_faces,
        left,
        right,
        bottom,
        top,
        back,
        front,
    )
    return mesh, {
        "action": action,
        "faces_before": n_faces,
        "faces_after": int(len(mesh.faces)),
        "faces_dropped": dropped,
        "box": box,
    }


def _side_band_area(mesh: trimesh.Trimesh) -> tuple[float, float]:
    """Mid-height face area left vs right of the remaining centroid."""
    if len(mesh.faces) == 0:
        return 0.0, 0.0
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    lo, hi = np.asarray(mesh.bounds, dtype=np.float64)
    yspan = float(max(hi[1] - lo[1], 1e-9))
    mid = (centers[:, 1] >= lo[1] + 0.30 * yspan) & (centers[:, 1] <= lo[1] + 0.78 * yspan)
    if not bool(mid.any()):
        return 0.0, 0.0
    cx = float(centers[mid][:, 0].mean())
    left = mid & (centers[:, 0] < cx)
    right = mid & (centers[:, 0] >= cx)
    return float(areas[left].sum()), float(areas[right].sum())


def _assert_carve_keeps_sides(
    mesh: trimesh.Trimesh,
    before_left: float,
    before_right: float,
) -> None:
    """Refuse a left/right cut that deleted an armrest while leaving the other."""
    left, right = _side_band_area(mesh)
    before_ratio = min(before_left, before_right) / max(before_left, before_right, 1e-9)
    after_ratio = min(left, right) / max(left, right, 1e-9)
    if after_ratio >= 0.62 or after_ratio >= before_ratio - 0.08:
        return
    raise CarveError(
        "Cut chopped a structural side (armrest/leg/seat). "
        "Tighten the box: for a backrest wing use bottom~0.45 and front~0.55, amount 0.08-0.14."
    )


def _drop_tiny_components(mesh: trimesh.Trimesh, *, min_ratio: float = 0.02) -> trimesh.Trimesh:
    """Drop leftover crumbs after a cut; keep every substantial component."""
    if len(mesh.faces) == 0:
        return mesh
    try:
        if len(getattr(mesh, "face_adjacency", [])) == 0:
            return mesh
        components = list(
            trimesh.graph.connected_components(
                mesh.face_adjacency,
                min_len=1,
                nodes=np.arange(len(mesh.faces)),
            )
        )
    except Exception:
        return mesh
    if len(components) <= 1:
        return mesh
    sizes = [len(c) for c in components]
    threshold = max(20, int(min_ratio * max(sizes)))
    keep_idx = [c for c in components if len(c) >= threshold]
    if not keep_idx:
        return mesh
    kept_n = int(sum(len(c) for c in keep_idx))
    if kept_n >= len(mesh.faces):
        return mesh
    mask = np.zeros(len(mesh.faces), dtype=bool)
    mask[np.concatenate([np.asarray(c, dtype=np.int64) for c in keep_idx])] = True
    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()
    return mesh


def remove_needle_faces(mesh: trimesh.Trimesh, min_edge_mm: float = 0.08) -> trimesh.Trimesh:
    """Remove faces with tiny edges or extreme aspect (classic recon 'needles')."""
    mesh = mesh.copy()
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        return mesh
    min_edge_mm = max(float(min_edge_mm), 1e-4)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    e0 = np.linalg.norm(verts[faces[:, 0]] - verts[faces[:, 1]], axis=1)
    e1 = np.linalg.norm(verts[faces[:, 1]] - verts[faces[:, 2]], axis=1)
    e2 = np.linalg.norm(verts[faces[:, 2]] - verts[faces[:, 0]], axis=1)
    min_e = np.minimum(np.minimum(e0, e1), e2)
    max_e = np.maximum(np.maximum(e0, e1), e2)
    aspect = max_e / np.maximum(min_e, 1e-12)
    # Absolute + relative floors so dense but valid surfaces are kept.
    rel_floor = float(np.percentile(min_e, 1))
    edge_floor = max(min_edge_mm, rel_floor)
    keep = (min_e >= edge_floor) & (aspect < 60.0)
    # Never delete more than ~25% in one pass.
    if float(keep.mean()) < 0.75:
        edge_floor = max(min_edge_mm * 0.25, float(np.percentile(min_e, 0.5)))
        keep = (min_e >= edge_floor) & (aspect < 100.0)
    if float(keep.mean()) < 0.75:
        keep = aspect < 120.0
    removed = int((~keep).sum())
    if removed <= 0:
        return mesh
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    logger.info("remove_needle_faces: removed %d faces (edge_floor=%.4f mm)", removed, edge_floor)
    return mesh


def try_make_watertight(mesh: trimesh.Trimesh, *, max_faces_for_fill: int = 200_000) -> trimesh.Trimesh:
    """Light repair + optional hole fill (skip on huge nets to avoid explosion)."""
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
    try:
        if max_faces_for_fill > 0 and (not mesh.is_watertight) and len(mesh.faces) <= int(max_faces_for_fill):
            trimesh.repair.fill_holes(mesh)
    except Exception as exc:
        logger.warning("fill_holes failed: %s", exc)
    return mesh


def _faces_ok(mesh: trimesh.Trimesh, baseline: int, *, min_faces: int = 500, min_ratio: float = 0.05) -> bool:
    n = len(mesh.faces) if mesh is not None else 0
    if n < min_faces:
        return False
    if baseline > 0 and n < int(baseline * min_ratio):
        return False
    return True


def repair_reconstruction_mesh(
    mesh: trimesh.Trimesh,
    *,
    target_faces: int = 120_000,
    smooth_iters: int = 2,
    min_edge_mm: float = 0.08,
    close_holes: bool = True,
    voxel_mm: float = 0.0,
) -> trimesh.Trimesh:
    """
    Light post-process for ComfyUI / Hunyuan meshes.

    Preserve detail by default. Voxel remesh only when explicitly enabled and the
    mesh looks pathological (tiny edges / huge face count + open).
    """
    baseline = max(len(mesh.faces), 1)
    current = mesh

    def _step(label: str, fn, *, min_ratio: float = 0.05) -> bool:
        nonlocal current
        before = current
        try:
            nxt = fn(current)
        except Exception as exc:
            logger.warning("repair step %s failed: %s", label, exc)
            return False
        if nxt is None or not _faces_ok(nxt, baseline, min_ratio=min_ratio):
            logger.warning(
                "repair step %s rejected (%s faces, baseline %s)",
                label,
                0 if nxt is None else len(nxt.faces),
                baseline,
            )
            current = before
            return False
        current = nxt
        return True

    _step("basic_repair", lambda m: try_make_watertight(m, max_faces_for_fill=0))

    # Voxel only for pathological open/spiky nets. Good ComfyUI meshes stay intact.
    sealed = bool(getattr(current, "is_watertight", False))
    needs_voxel = False
    if close_holes and voxel_mm and float(voxel_mm) > 0 and not sealed:
        try:
            edges = current.edges_unique_length
            min_e = float(np.min(edges)) if len(edges) else 1.0
        except Exception:
            min_e = 1.0
        needs_voxel = min_e < max(float(min_edge_mm) * 0.5, 0.02) or len(current.faces) > 400_000
        if not needs_voxel:
            logger.info(
                "repair_reconstruction: skip voxel (open but not pathological; min_edge=%.4f faces=%d)",
                min_e,
                len(current.faces),
            )

    if needs_voxel:
        pitches = []
        base = float(voxel_mm)
        for p in (base, base * 1.5, max(base * 2.0, 2.0)):
            if p not in pitches:
                pitches.append(p)
        source = current
        best = None
        for pitch in pitches:
            logger.info("repair_reconstruction: voxel remesh @ %.3f mm", pitch)
            try:
                candidate = remesh_voxel(source, pitch)
            except Exception as exc:
                logger.warning("voxel remesh @ %.3f failed: %s", pitch, exc)
                continue
            if candidate is None or not _faces_ok(candidate, baseline, min_ratio=0.02):
                logger.warning(
                    "voxel remesh @ %.3f rejected (%s faces)",
                    pitch,
                    0 if candidate is None else len(candidate.faces),
                )
                continue
            best = candidate
            if bool(getattr(candidate, "is_watertight", False)):
                current = candidate
                sealed = True
                logger.info(
                    "voxel remesh sealed @ %.3f mm (%d faces)",
                    pitch,
                    len(candidate.faces),
                )
                break
        if not sealed and best is not None:
            current = best
            logger.warning("voxel remesh did not seal; keeping best candidate (%d faces)", len(best.faces))

        if current is not source:
            try:
                target_h = float(np.max(np.asarray(source.extents, dtype=float)))
                current = normalize_height_mm(current, target_h)
                current = orient_upright(current)
            except Exception as exc:
                logger.debug("re-normalize after voxel skipped: %s", exc)

    if sealed and smooth_iters > 0:
        _step("smooth_sealed", lambda m: smooth_mesh(m, iterations=max(1, min(smooth_iters, 2))))
    elif not sealed:
        # Light cleanup without destroying ComfyUI detail.
        _step("keep_components", lambda m: keep_largest_component(m, single=True))
        _step("remove_needles", lambda m: remove_needle_faces(m, min_edge_mm=min_edge_mm))
        if close_holes:
            fill_cap = max(int(target_faces or 120_000) * 2, 200_000)
            _step("fill_holes", lambda m: try_make_watertight(m, max_faces_for_fill=fill_cap))
            if not bool(getattr(current, "is_watertight", False)):
                def _pml(m: trimesh.Trimesh) -> trimesh.Trimesh:
                    from mesh_forge.ops.repair import repair_with_pymeshlab

                    return repair_with_pymeshlab(m, smooth_iters=0)

                _step("pymeshlab", _pml)
        if smooth_iters > 0:
            _step("smooth", lambda m: smooth_mesh(m, iterations=min(smooth_iters, 2)))

    target = int(target_faces) if target_faces else 0
    if target > 0 and len(current.faces) > int(target * 1.15):
        before_decimate = current
        if _step("decimate", lambda m: decimate(m, target)):
            # Decimate can open a sealed mesh — revert if it did.
            if sealed and not bool(getattr(current, "is_watertight", False)):
                logger.warning("decimate re-opened mesh; reverting")
                current = before_decimate
            else:
                try:
                    trimesh.grouping.merge_vertices(current, digits_vertex=5)
                    current.remove_unreferenced_vertices()
                    current._cache.clear()
                except Exception:
                    pass

    # Never run needle/component filters on a sealed mesh — they reopen holes.
    if not bool(getattr(current, "is_watertight", False)):
        _step("remove_needles_final", lambda m: remove_needle_faces(m, min_edge_mm=min_edge_mm * 0.5))
        _step("keep_components_final", lambda m: keep_largest_component(m, single=True))

    try:
        trimesh.repair.fix_normals(current)
    except Exception:
        pass
    try:
        current.merge_vertices()
        current.remove_unreferenced_vertices()
    except Exception as exc:
        logger.debug("merge_vertices skipped: %s", exc)
    try:
        # Refresh cached topology flags after merge.
        current._cache.clear()
    except Exception:
        pass
    logger.info(
        "repair_reconstruction: %d -> %d faces, watertight=%s",
        baseline,
        len(current.faces),
        bool(getattr(current, "is_watertight", False)),
    )
    return current


def _boundary_vertex_indices(mesh: trimesh.Trimesh) -> np.ndarray:
    try:
        once = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
        if len(once) == 0:
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.asarray(mesh.edges_sorted[once], dtype=np.int64).reshape(-1))
    except Exception:
        return np.zeros(0, dtype=np.int64)


def _clamp_vertex_motion(
    mesh: trimesh.Trimesh,
    original: np.ndarray,
    diag: float,
    *,
    max_frac: float = 0.012,
) -> trimesh.Trimesh:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.shape != original.shape or not np.isfinite(verts).all():
        mesh.vertices = original
        return mesh
    delta = verts - original
    dist = np.linalg.norm(delta, axis=1)
    max_step = max(max_frac * max(diag, 1e-6), 1e-4)
    over = dist > max_step
    if np.any(over):
        scale = np.ones(len(dist), dtype=np.float64)
        scale[over] = max_step / np.maximum(dist[over], 1e-12)
        mesh.vertices = original + delta * scale[:, None]
        logger.info("smooth: clamped %d spike/pit vertices (max_step=%.4f)", int(over.sum()), max_step)
    return mesh


def smooth_mesh(mesh: trimesh.Trimesh, iterations: int = 2) -> trimesh.Trimesh:
    """Smooth without growing needles, pits, or holes."""
    iterations = max(0, min(int(iterations), 5))
    if iterations <= 0:
        return mesh
    mesh = _weld_mesh_vertices(mesh)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return mesh
    iters = iterations if iterations % 2 == 0 else iterations + 1
    original = np.asarray(mesh.vertices, dtype=np.float64).copy()
    extents0 = np.asarray(mesh.extents, dtype=float)
    diag = float(np.linalg.norm(extents0)) if extents0.size else 0.0
    boundary = _boundary_vertex_indices(mesh)
    area0 = float(getattr(mesh, "area", 0.0) or 0.0)

    used_pml = False
    try:
        from mesh_forge.ops.repair import smooth_with_pymeshlab

        nxt = smooth_with_pymeshlab(mesh, iterations=max(1, iterations))
        if nxt is not None and len(nxt.vertices) > 0 and len(nxt.faces) > 0:
            mesh = nxt
            used_pml = True
    except Exception as exc:
        logger.debug("pymeshlab smooth skipped: %s", exc)

    if not used_pml:
        try:
            trimesh.smoothing.filter_taubin(mesh, lamb=0.33, nu=0.34, iterations=iters)
        except Exception as exc:
            logger.warning("taubin smooth failed (%s); trying laplacian", exc)
            try:
                mesh.vertices = original
                trimesh.smoothing.filter_laplacian(
                    mesh, lamb=0.2, iterations=min(iterations, 3), volume_constraint=False
                )
            except Exception:
                mesh.vertices = original
                return mesh

    if not used_pml and len(mesh.vertices) == len(original):
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        if verts.shape != original.shape or not np.isfinite(verts).all():
            mesh.vertices = original
            return mesh
        if len(boundary) > 0:
            verts[boundary] = original[boundary]
        mesh.vertices = verts
        mesh = _clamp_vertex_motion(mesh, original, diag, max_frac=0.012)

    if float(np.max(mesh.extents)) > 1.25 * max(float(np.max(extents0)), 1e-6):
        logger.warning("smooth: bbox exploded; reverting")
        if not used_pml and len(mesh.vertices) == len(original):
            mesh.vertices = original
        return mesh
    area1 = float(getattr(mesh, "area", 0.0) or 0.0)
    if area0 > 0 and area1 > 1.2 * area0:
        logger.warning("smooth: surface area blew up; reverting")
        if not used_pml and len(mesh.vertices) == len(original):
            mesh.vertices = original
    return mesh


def _weld_mesh_vertices(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Index coincident vertices so faces share edges.

    STL dumps store 3 unique verts per triangle. Quadric decimation then has
    nothing to collapse and just deletes faces, leaving a gappy triangle spray.
    """
    mesh = mesh.copy()
    before = len(mesh.vertices)
    try:
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
    except Exception as exc:
        logger.debug("merge_vertices skipped: %s", exc)
    if len(getattr(mesh, "face_adjacency", [])) == 0:
        try:
            trimesh.grouping.merge_vertices(mesh, digits_vertex=5)
            mesh.remove_unreferenced_vertices()
            mesh._cache.clear()
        except Exception as exc:
            logger.debug("digits merge skipped: %s", exc)
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    after = len(mesh.vertices)
    if after < before:
        logger.info("weld: %d -> %d vertices", before, after)
    return mesh


def _mesh_from_arrays(vertices, faces) -> trimesh.Trimesh:
    out = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    try:
        out.update_faces(out.nondegenerate_faces())
        out.remove_unreferenced_vertices()
    except Exception:
        pass
    return out


def _decimate_fast_simplification(mesh: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
    from fast_simplification import simplify

    vertices, faces = simplify(
        points=np.asarray(mesh.vertices, dtype=np.float64),
        triangles=np.asarray(mesh.faces, dtype=np.int32),
        target_count=target,
    )
    return _mesh_from_arrays(vertices, faces)


def _decimate_open3d(mesh: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Decimate requires fast-simplification or open3d") from exc

    o3 = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    simplified = o3.simplify_quadric_decimation(target_number_of_triangles=target)
    return _mesh_from_arrays(np.asarray(simplified.vertices), np.asarray(simplified.triangles))


def _decimate_looks_valid(source: trimesh.Trimesh, result: trimesh.Trimesh) -> bool:
    """Reject the classic 'deleted isolated triangles' failure mode."""
    if result is None or len(result.faces) < 4 or len(result.vertices) < 4:
        return False
    src_ext = float(np.max(source.extents)) if len(source.vertices) else 0.0
    out_ext = float(np.max(result.extents)) if len(result.vertices) else 0.0
    if src_ext > 0 and out_ext < 0.5 * src_ext:
        return False
    if src_ext > 0 and out_ext > 1.25 * src_ext:
        return False
    src_area = float(getattr(source, "area", 0.0) or 0.0)
    out_area = float(getattr(result, "area", 0.0) or 0.0)
    if src_area > 0 and out_area < 0.7 * src_area:
        return False
    # Isolated triangles: ~3 unique verts per face and no shared edges.
    if len(result.vertices) >= 2.8 * len(result.faces):
        try:
            adj = len(getattr(result, "face_adjacency", []))
        except Exception:
            adj = 0
        if adj == 0:
            return False
    return True


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Reduce face count. Uses fast-simplification / Open3D; never passes face count as percent."""
    n = int(len(mesh.faces))
    target = int(target_faces)
    if target <= 0 or n <= target:
        return mesh
    target = max(4, min(target, n - 1))

    prepared = _weld_mesh_vertices(mesh)
    n = int(len(prepared.faces))
    if n <= target:
        return prepared
    target = max(4, min(target, n - 1))

    result = None
    try:
        result = _decimate_fast_simplification(prepared, target)
    except Exception as exc:
        logger.warning("fast_simplification failed (%s); trying Open3D", exc)

    if result is None or not _decimate_looks_valid(prepared, result):
        if result is not None:
            logger.warning(
                "fast_simplification fragmented the mesh (%d verts / %d faces); trying Open3D",
                len(result.vertices),
                len(result.faces),
            )
        try:
            result = _decimate_open3d(prepared, target)
        except Exception as exc:
            logger.warning("Open3D decimate failed: %s", exc)
            result = None

    if result is None or not _decimate_looks_valid(prepared, result):
        logger.warning("decimate failed to preserve the surface; leaving mesh unchanged")
        return mesh
    try:
        trimesh.repair.fix_normals(result)
    except Exception:
        pass
    return result


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
                if len(mesh.faces) < 500_000:
                    trimesh.repair.fill_holes(mesh)
                else:
                    # Large nets: try PyMeshLab hole close when available.
                    try:
                        from mesh_forge.ops.repair import repair_with_pymeshlab

                        mesh = repair_with_pymeshlab(mesh, smooth_iters=0)
                    except Exception as exc:
                        logger.warning("fill_holes skipped on large mesh (%s faces): %s", len(mesh.faces), exc)
            except Exception as exc:
                logger.warning("fill_holes failed: %s", exc)
        elif name == "remove_needles":
            min_edge = float(op.get("min_edge_mm", 0.08))
            mesh = remove_needle_faces(mesh, min_edge_mm=min_edge)
    return save_mesh(mesh, out_path)
