from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

logger = logging.getLogger("mesh_forge.geometry")


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if hasattr(g, "vertices") and len(g.vertices) > 0]
        if not geoms:
            raise ValueError(f"No geometry in mesh file: {path}")
        loaded = trimesh.util.concatenate(geoms)
    if len(loaded.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {path}")
    return loaded


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
    """Drop floaters / duplicate Hunyuan bodies; keep the primary figurine.

    ``single=True`` (default): keep the largest component **plus** nearby
    fragments that belong to the same body (legs/ears often disconnect). Far
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
    if largest <= 4 and len(components) > 32:
        logger.warning(
            "keep_largest_component: fragmented adjacency (%d parts, largest=%d); skip",
            len(components),
            largest,
        )
        return mesh

    if single:
        keep_idx = _select_primary_body_faces(mesh, components, largest_i)
    else:
        threshold = max(50, int(min_ratio * largest))
        keep_idx = np.concatenate([c for c in components if len(c) >= threshold])

    if len(keep_idx) >= len(mesh.faces):
        return mesh
    mask = np.zeros(len(mesh.faces), dtype=bool)
    mask[np.asarray(keep_idx, dtype=np.int64)] = True
    dropped = int((~mask).sum())
    kept_n = 0
    for c in components:
        idx = np.asarray(c, dtype=np.int64)
        if idx.size and bool(mask[idx].all()):
            kept_n += 1
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
    """Largest shell + nearby fragments; drop far twin / paper-thin debris."""
    seed = components[largest_i]
    seed_mesh = mesh.submesh([seed], append=True)
    seed_centroid = np.asarray(seed_mesh.centroid, dtype=np.float64)
    seed_bounds = np.asarray(seed_mesh.bounds, dtype=np.float64)
    diag = float(np.linalg.norm(seed_mesh.extents))
    pad = max(0.35 * diag, 1e-6)
    lo, hi = seed_bounds[0] - pad, seed_bounds[1] + pad
    seed_faces = len(seed)
    min_keep = max(200, int(0.01 * seed_faces))

    kept: list[np.ndarray] = [np.asarray(seed, dtype=np.int64)]
    for i, faces in enumerate(components):
        if i == largest_i:
            continue
        n = len(faces)
        if n < min_keep:
            continue
        part = mesh.submesh([faces], append=True)
        extents = np.asarray(part.extents, dtype=np.float64)
        # Paper-thin sheets / walls
        if float(np.min(extents)) < 0.03 * max(float(np.max(extents)), 1e-9):
            continue
        centroid = np.asarray(part.centroid, dtype=np.float64)
        # Far twin of similar size (classic Hunyuan double body)
        if n >= 0.45 * seed_faces:
            dist = float(np.linalg.norm(centroid - seed_centroid))
            if dist > 0.45 * diag:
                continue
        if not (np.all(centroid >= lo) and np.all(centroid <= hi)):
            continue
        kept.append(np.asarray(faces, dtype=np.int64))
    return np.concatenate(kept)


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


def smooth_mesh(mesh: trimesh.Trimesh, iterations: int = 2) -> trimesh.Trimesh:
    iterations = max(0, min(int(iterations), 5))
    if iterations <= 0:
        return mesh
    # volume_constraint blows up on open / non-manifold scans
    try:
        return trimesh.smoothing.filter_laplacian(
            mesh, lamb=0.5, iterations=iterations, volume_constraint=False
        )
    except TypeError:
        return trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=iterations)
    except Exception:
        return mesh


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Reduce face count. Uses fast-simplification / Open3D; never passes face count as percent."""
    n = int(len(mesh.faces))
    target = int(target_faces)
    if target <= 0 or n <= target:
        return mesh
    # Keep at least a tiny mesh
    target = max(4, min(target, n - 1))

    try:
        from fast_simplification import simplify

        vertices, faces = simplify(
            points=np.asarray(mesh.vertices, dtype=np.float64),
            triangles=np.asarray(mesh.faces, dtype=np.int32),
            target_count=target,
        )
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    except Exception as exc:
        logger.warning("fast_simplification failed (%s); trying Open3D", exc)

    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "Decimate requires fast-simplification or open3d"
        ) from exc

    o3 = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    simplified = o3.simplify_quadric_decimation(target_number_of_triangles=target)
    return trimesh.Trimesh(
        vertices=np.asarray(simplified.vertices),
        faces=np.asarray(simplified.triangles),
        process=False,
    )


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
                if len(mesh.faces) < 200_000:
                    trimesh.repair.fill_holes(mesh)
                else:
                    logger.warning("fill_holes skipped on large mesh (%s faces)", len(mesh.faces))
            except Exception as exc:
                logger.warning("fill_holes failed: %s", exc)
    return save_mesh(mesh, out_path)
