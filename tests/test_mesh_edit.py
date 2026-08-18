from __future__ import annotations

import unittest
from pathlib import Path

import trimesh

from mesh_forge.ops.edit import (
    EditError,
    add_primitive_in_region,
    extract_in_region,
    fill_in_region,
    join_in_region,
    match_in_region,
    offset_in_region,
    remove_in_region,
    remove_near_pick,
    restore_patch_in_region,
    smooth_in_region,
    split_in_region,
)
from mesh_forge.ops.region import RegionError, parse_region, region_box


def _box(center, extents, subdivisions=2) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    for _ in range(subdivisions):
        mesh = mesh.subdivide()
    mesh.apply_translation(center)
    return mesh


class RegionTests(unittest.TestCase):
    def test_aliases(self) -> None:
        self.assertEqual(parse_region("спинка"), "back")
        self.assertEqual(parse_region("ножки"), "legs")

    def test_unknown(self) -> None:
        with self.assertRaises(RegionError):
            parse_region("wing")

    def test_right_is_not_a_full_slab(self) -> None:
        box = region_box("right")
        left, right, bottom, top, back, front = box
        self.assertGreater(left, 0.5)
        self.assertGreater(bottom, 0.2)
        self.assertLess(front, 0.8)

    def test_infer_and_pick_box(self) -> None:
        from mesh_forge.ops.region import infer_region, pick_box

        self.assertEqual(infer_region(0.9, 0.8, 0.2), "right")
        self.assertEqual(infer_region(0.5, 0.25, 0.5), "legs")
        box = pick_box(0.8, 0.7, 0.3, 0.1)
        self.assertGreater(box[0], 0.6)
        self.assertLess(box[1], 0.95)


class EditOpTests(unittest.TestCase):
    def test_remove_strategy_classifier(self) -> None:
        from mesh_forge.ops.remove import classify_removal_strategy

        self.assertEqual(classify_removal_strategy("удали лишнюю ножку стула"), "protrusion_cut")
        self.assertEqual(classify_removal_strategy("удали островок сбоку"), "island_drop")
        self.assertEqual(classify_removal_strategy("удали полукруг от подола юбки"), "edge_trim")
        self.assertEqual(classify_removal_strategy("удали лепестковый отросток на юбке справа"), "hem_flap_trim")
        self.assertEqual(classify_removal_strategy("удали пятно на поверхности"), "surface_patch")

    def test_build_auto_remove_proposal_protrusion_cut(self) -> None:
        import numpy as np

        from mesh_forge.ops.remove import build_auto_remove_proposal

        body = _box([0.0, 0.5, 0.0], [1.4, 1.0, 1.0], subdivisions=2)
        wing = _box([1.05, 0.9, -0.2], [0.45, 0.4, 0.3], subdivisions=1)
        src = trimesh.util.concatenate([body, wing])
        result = build_auto_remove_proposal(src, "удали лишний выступ справа")
        self.assertEqual(result["strategy"], "protrusion_cut")
        self.assertGreater(int(np.asarray(result["mask"], dtype=bool).sum()), 0)
        self.assertLess(int(len(result["mesh"].faces)), int(len(src.faces)))

    def test_build_auto_remove_proposal_edge_trim(self) -> None:
        from mesh_forge.ops.remove import build_auto_remove_proposal

        skirt = trimesh.creation.box(extents=[1.2, 0.8, 0.8]).subdivide().subdivide()
        before = int(len(skirt.faces))
        result = build_auto_remove_proposal(skirt, "удали полукруг от подола юбки справа")
        self.assertEqual(result["strategy"], "edge_trim")
        self.assertLess(int(len(result["mesh"].faces)), before)

    def test_build_auto_remove_proposal_skirt_prefers_lower_lobe_over_arm(self) -> None:
        import numpy as np

        from mesh_forge.ops.remove import build_auto_remove_proposal

        body = _box([0.0, 0.5, 0.0], [1.0, 1.2, 0.8], subdivisions=1)
        arm = _box([0.78, 0.82, 0.05], [0.22, 0.18, 0.18], subdivisions=1)
        petal = _box([0.76, 0.18, 0.02], [0.18, 0.12, 0.12], subdivisions=1)
        src = trimesh.util.concatenate([body, arm, petal])
        centers = np.asarray(src.triangles_center, dtype=np.float64)
        arm_faces = centers[:, 1] > 0.65
        petal_faces = centers[:, 1] < 0.30
        result = build_auto_remove_proposal(src, "удали лепестковый отросток на юбке справа")
        mask = np.asarray(result["mask"], dtype=bool)
        self.assertEqual(result["strategy"], "hem_flap_trim")
        self.assertGreater(int((mask & petal_faces).sum()), int((mask & arm_faces).sum()))

    def test_protrusion_candidates_for_skirt_include_multi_seed_unions(self) -> None:
        import numpy as np
        from unittest.mock import patch

        from mesh_forge.ops.remove import _protrusion_candidates

        src = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        n = int(len(src.faces))
        m1 = np.zeros(n, dtype=bool)
        m2 = np.zeros(n, dtype=bool)
        m3 = np.zeros(n, dtype=bool)
        m1[:2] = True
        m2[2:4] = True
        m3[4:6] = True
        with patch("mesh_forge.ops.remove.knife_lump_faces", return_value=np.zeros(n, dtype=bool)), patch(
            "mesh_forge.ops.remove._semantic_tip_seeds",
            return_value=[1, 2, 3],
        ), patch(
            "mesh_forge.ops.remove._grow_lump_from_tip",
            side_effect=[m1, m2, m3],
        ):
            candidates = _protrusion_candidates(
                src,
                "right",
                along="bottom",
                describe="удали лепестковый отросток на юбке справа",
            )
        sizes = sorted(int(np.asarray(mask, dtype=bool).sum()) for mask in candidates if np.any(mask))
        self.assertGreaterEqual(len(sizes), 2)
        self.assertGreater(sizes[-1], sizes[0])

    def test_remove_right_drops_outer_blob(self) -> None:
        body = _box([0.0, 0.5, 0.0], [1.4, 1.0, 1.0], subdivisions=2)
        wing = _box([1.05, 0.9, -0.2], [0.45, 0.4, 0.3], subdivisions=1)
        src = trimesh.util.concatenate([body, wing])
        before = float(src.bounds[1][0])
        out, stats = remove_in_region(src, "right")
        self.assertGreater(stats["faces_dropped"], 0)
        self.assertLess(float(out.bounds[1][0]), before - 0.05)

    def test_pick_remove_drops_petal_not_body(self) -> None:
        body = _box([0.0, 0.5, 0.0], [1.0, 1.2, 0.8], subdivisions=2)
        petal = _box([0.72, 0.22, 0.0], [0.16, 0.10, 0.10], subdivisions=2)
        src = trimesh.util.concatenate([body, petal])
        lo, hi = src.bounds
        span = hi - lo
        c = petal.centroid
        pick = [
            float((c[0] - lo[0]) / span[0]),
            float((c[1] - lo[1]) / span[1]),
            float((c[2] - lo[2]) / span[2]),
        ]
        before = int(len(src.faces))
        out, stats = remove_near_pick(src, pick)
        self.assertGreater(stats["faces_dropped"], 0)
        self.assertLess(stats["faces_dropped"], int(0.25 * before))
        self.assertGreater(float(out.extents[1]), 1.0)

    def test_knife_plane_is_mesh_axis_not_camera(self) -> None:
        from mesh_forge.ops.region import knife_plane

        mesh = _box([0.0, 0.5, 0.0], [2.0, 1.0, 1.0], subdivisions=0)
        origin, normal, label = knife_plane(mesh, "right", at=0.80)
        self.assertIn("knife:right", label)
        self.assertLess(abs(float(normal[0]) + 1.0), 1e-9)
        self.assertLess(abs(float(normal[1])), 1e-9)
        self.assertLess(abs(float(normal[2])), 1e-9)
        lo, hi = mesh.bounds
        expected_x = float(lo[0] + 0.80 * (hi[0] - lo[0]))
        self.assertAlmostEqual(float(origin[0]), expected_x, places=5)

    def test_topo_hit_and_remove_face(self) -> None:
        from mesh_forge.ops.edit import remove_topo
        from mesh_forge.ops.topo import face_mask_for_topo, hit_topology

        mesh = _box([0.0, 0.5, 0.0], [1.0, 1.0, 1.0], subdivisions=2)
        lo, hi = mesh.bounds
        span = hi - lo
        c = mesh.triangles_center[0]
        nx, ny, nz = (c - lo) / span
        topo = hit_topology(mesh, float(nx), float(ny), float(nz), kind="face")
        self.assertGreaterEqual(int(topo["face"]), 0)
        self.assertGreaterEqual(int(topo["vertex"]), 0)
        self.assertEqual(len(topo["edge"]), 2)
        topo["hops"] = 0
        before = int(len(mesh.faces))
        self.assertEqual(int(face_mask_for_topo(mesh, topo).sum()), 1)
        out, stats = remove_topo(mesh, topo)
        self.assertEqual(int(stats["faces_dropped"]), 1)
        self.assertEqual(int(len(out.faces)), before - 1)

    def test_topo_remove_vertex_drops_incident_faces(self) -> None:
        from mesh_forge.ops.edit import remove_topo
        from mesh_forge.ops.topo import face_mask_for_topo, topology_from_ids

        mesh = _box([0.0, 0.5, 0.0], [1.0, 1.0, 1.0], subdivisions=1)
        topo = topology_from_ids(mesh, kind="vertex", vertex=0)
        topo["hops"] = 0
        drop = int(face_mask_for_topo(mesh, topo).sum())
        self.assertGreater(drop, 1)
        out, stats = remove_topo(mesh, topo)
        self.assertEqual(int(stats["faces_dropped"]), drop)
        self.assertGreater(int(len(out.faces)), 8)

    def test_viewport_hit_front_center(self) -> None:
        from mesh_forge.ops.topo import viewport_hit

        mesh = _box([0.0, 0.5, 0.0], [1.0, 1.0, 1.0], subdivisions=1)
        topo = viewport_hit(mesh, camera="front", x=0.5, y=0.5, hops=2)
        self.assertGreaterEqual(int(topo["face"]), 0)
        self.assertGreaterEqual(int(topo["faces"]), 1)

    def test_viewport_off_frame_snaps_to_silhouette(self) -> None:
        from mesh_forge.ops.topo import viewport_hit

        mesh = _box([0.0, 0.5, 0.0], [1.0, 1.0, 1.0], subdivisions=1)
        topo = viewport_hit(mesh, camera="front", x=0.99, y=0.5, hops=1)
        self.assertGreaterEqual(int(topo["face"]), 0)

    def test_viewport_uses_look_yaw_offset_for_named_view(self) -> None:
        from mesh_forge.ops.topo import viewport_hit
        from mesh_forge.tools.look import parse_look_shots

        mesh = _box([0.0, 0.5, 0.0], [1.0, 1.0, 1.0], subdivisions=1)
        shot = parse_look_shots("right", yaw=45.0)[0]
        self.assertEqual(shot.camera, "custom")
        self.assertAlmostEqual(float(shot.yaw or 0.0), 135.0)
        topo = viewport_hit(mesh, views="right", yaw=45.0, x=0.5, y=0.5, hops=1)
        self.assertGreaterEqual(int(topo["face"]), 0)

    def test_grow_patch_is_more_than_one_face_but_not_all(self) -> None:
        from mesh_forge.ops.topo import face_mask_for_topo, topology_from_ids

        mesh = _box([0.0, 0.5, 0.0], [1.0, 1.0, 1.0], subdivisions=3)
        topo = topology_from_ids(mesh, kind="face", face=0)
        topo["hops"] = 8
        n = int(face_mask_for_topo(mesh, topo).sum())
        self.assertGreater(n, 1)
        self.assertLess(n, int(0.25 * len(mesh.faces)))

    def test_knife_drops_outer_lump_not_body(self) -> None:
        body = _box([0.0, 0.5, 0.0], [1.0, 1.2, 0.8], subdivisions=2)
        petal = _box([0.72, 0.22, 0.0], [0.16, 0.10, 0.10], subdivisions=2)
        src = trimesh.util.concatenate([body, petal])
        out, stats = remove_in_region(src, "", knife="right", at=0.80)
        self.assertLess(float(out.bounds[1][0]), float(src.bounds[1][0]) - 0.05)
        self.assertGreater(float(out.extents[1]), 1.0)
        self.assertGreater(int(len(out.faces)), 20)

    def test_knife_on_connected_skirt_keeps_body(self) -> None:
        import numpy as np

        skirt = trimesh.creation.cylinder(radius=0.45, height=0.9, sections=32)
        skirt.apply_translation([0.0, 0.45, 0.0])
        verts = np.asarray(skirt.vertices, dtype=np.float64)
        idx = int(np.argmax(verts[:, 0]))
        verts[idx, 0] += 0.28
        skirt.vertices = verts
        before_x = float(skirt.bounds[1][0])
        out, stats = remove_in_region(skirt, "", knife="right", at=0.88)
        self.assertLess(float(out.bounds[1][0]), before_x - 0.02)
        self.assertGreater(int(len(out.faces)), int(0.80 * len(skirt.faces)))
        self.assertLess(int(stats["faces_dropped"]), int(0.20 * len(skirt.faces)))

    def test_pick_on_connected_skirt_keeps_body(self) -> None:
        import numpy as np

        skirt = trimesh.creation.cylinder(radius=0.45, height=0.9, sections=32)
        skirt.apply_translation([0.0, 0.45, 0.0])
        verts = np.asarray(skirt.vertices, dtype=np.float64)
        idx = int(np.argmax(verts[:, 0]))
        verts[idx, 0] += 0.28
        skirt.vertices = verts
        lo, hi = skirt.bounds
        span = np.maximum(hi - lo, 1e-9)
        tip = verts[idx]
        pick = [float((tip[i] - lo[i]) / span[i]) for i in range(3)]
        before = int(len(skirt.faces))
        out, stats = remove_near_pick(skirt, pick)
        self.assertGreater(stats["faces_dropped"], 0)
        self.assertLess(stats["faces_dropped"], int(0.08 * before))
        self.assertGreater(int(len(out.faces)), int(0.90 * before))

    def test_click_wins_over_seat_region(self) -> None:
        body = _box([0.0, 0.5, 0.0], [1.0, 1.2, 0.8], subdivisions=2)
        petal = _box([0.72, 0.22, 0.0], [0.16, 0.10, 0.10], subdivisions=2)
        src = trimesh.util.concatenate([body, petal])
        lo, hi = src.bounds
        span = hi - lo
        c = petal.centroid
        pick = [
            float((c[0] - lo[0]) / span[0]),
            float((c[1] - lo[1]) / span[1]),
            float((c[2] - lo[2]) / span[2]),
        ]
        before = int(len(src.faces))
        out, stats = remove_in_region(src, "seat", pick=pick, protect_sides=True)
        self.assertLess(stats["faces_dropped"], int(0.15 * before))
        self.assertGreater(float(out.extents[1]), 1.0)

    def test_fill_closes_top_hole(self) -> None:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
        mesh.apply_translation([0.0, 0.5, 0.0])
        centers = mesh.triangles_center
        y_cut = float(mesh.bounds[0][1] + 0.82 * (mesh.bounds[1][1] - mesh.bounds[0][1]))
        mesh.update_faces(centers[:, 1] < y_cut)
        mesh.remove_unreferenced_vertices()
        before = int(len(mesh.faces))
        out, stats = fill_in_region(mesh, "top")
        self.assertGreaterEqual(int(len(out.faces)), before)
        self.assertGreaterEqual(stats["faces_added"], 0)

    def test_split_severs_bridge(self) -> None:
        dummy = _box([0.0, 0.75, 0.0], [0.4, 0.5, 0.4], subdivisions=1)
        left = _box([-0.55, 0.12, 0.2], [0.35, 0.22, 0.35], subdivisions=1)
        right = _box([0.55, 0.12, 0.2], [0.35, 0.22, 0.35], subdivisions=1)
        bar = _box([0.0, 0.12, 0.2], [0.8, 0.06, 0.06], subdivisions=1)
        src = trimesh.util.concatenate([dummy, left, right, bar])
        out, stats = split_in_region(src, "legs")
        self.assertGreater(stats["faces_dropped"], 0)
        self.assertGreater(int(len(out.faces)), 20)

    def test_join_welds_nearby_parts(self) -> None:
        a = _box([-0.12, 0.25, 0.0], [0.22, 0.4, 0.22], subdivisions=1)
        b = _box([0.12, 0.25, 0.0], [0.22, 0.4, 0.22], subdivisions=1)
        src = trimesh.util.concatenate([a, b])
        out, _ = join_in_region(src, "legs")
        self.assertGreater(int(len(out.faces)), 0)

    def test_match_mirror_copies_larger_side(self) -> None:
        left = _box([-0.45, 0.22, 0.15], [0.28, 0.4, 0.28], subdivisions=2)
        right = _box([0.45, 0.18, 0.15], [0.12, 0.22, 0.12], subdivisions=1)
        seat = _box([0.0, 0.55, 0.1], [1.1, 0.12, 0.6], subdivisions=1)
        src = trimesh.util.concatenate([left, right, seat])
        before_right = float(src.bounds[1][0])
        out, stats = match_in_region(src, "legs", "mirror")
        self.assertEqual(stats["how"], "mirror")
        self.assertGreater(float(out.area), 0.8 * float(src.area))
        self.assertGreater(float(out.bounds[1][0]), 0.2)
        self.assertLessEqual(abs(float(out.bounds[1][0]) - before_right), 0.6)

    def test_match_height_needs_two_parts(self) -> None:
        src = _box([0.0, 0.2, 0.0], [0.4, 0.3, 0.4], subdivisions=1)
        with self.assertRaises(EditError):
            match_in_region(src, "legs", "height")

    def test_match_flat_seat(self) -> None:
        import numpy as np

        frame = _box([0.0, 0.5, 0.0], [1.2, 1.0, 1.0], subdivisions=1)
        seat = _box([0.0, 0.45, 0.25], [0.8, 0.08, 0.55], subdivisions=2)
        seat.vertices[:, 1] += 0.05 * (seat.vertices[:, 0] ** 2)
        src = trimesh.util.concatenate([frame, seat])
        out, stats = match_in_region(src, "seat", "flat")
        self.assertEqual(stats["how"], "flat")
        lo, hi = out.bounds
        y0 = float(lo[1] + 0.28 * (hi[1] - lo[1]))
        y1 = float(lo[1] + 0.58 * (hi[1] - lo[1]))
        ys = out.vertices[:, 1]
        band = ys[(ys >= y0) & (ys <= y1)]
        self.assertGreater(len(band), 8)
        self.assertLess(float(np.ptp(band)), 0.12)

    def test_smooth_region_keeps_outside(self) -> None:
        src = _box([0.0, 0.5, 0.0], [1.0, 1.0, 1.0], subdivisions=2)
        before = src.vertices.copy()
        out = smooth_in_region(src, "top", 2)
        self.assertEqual(len(out.vertices), len(before))

    def test_extract_pick_keeps_remainder(self) -> None:
        body = _box([0.0, 0.5, 0.0], [1.0, 1.2, 0.8], subdivisions=2)
        petal = _box([0.72, 0.22, 0.0], [0.16, 0.10, 0.10], subdivisions=2)
        src = trimesh.util.concatenate([body, petal])
        lo, hi = src.bounds
        span = hi - lo
        c = petal.centroid
        pick = [
            float((c[0] - lo[0]) / span[0]),
            float((c[1] - lo[1]) / span[1]),
            float((c[2] - lo[2]) / span[2]),
        ]
        rest, piece, stats = extract_in_region(src, "right", pick=pick)
        self.assertGreater(stats["faces_extracted"], 0)
        self.assertGreater(int(len(rest.faces)), int(len(piece.faces)))
        self.assertGreater(float(rest.extents[1]), 1.0)

    def test_offset_inflates_region(self) -> None:
        src = _box([0.0, 0.5, 0.0], [1.0, 1.0, 1.0], subdivisions=2)
        before = float(src.extents[1])
        out, stats = offset_in_region(src, "top", 0.08)
        self.assertGreater(float(out.extents[1]), before)
        self.assertAlmostEqual(stats["amount"], 0.08)

    def test_add_cylinder_increases_faces(self) -> None:
        src = _box([0.0, 0.5, 0.0], [1.0, 0.4, 1.0], subdivisions=1)
        before = int(len(src.faces))
        out, stats = add_primitive_in_region(src, "legs", "cylinder")
        self.assertGreater(int(len(out.faces)), before)
        self.assertEqual(stats["shape"], "cylinder")

    def test_restore_patch_fills_hole(self) -> None:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
        mesh.apply_translation([0.0, 0.5, 0.0])
        centers = mesh.triangles_center
        y_cut = float(mesh.bounds[0][1] + 0.82 * (mesh.bounds[1][1] - mesh.bounds[0][1]))
        mesh.update_faces(centers[:, 1] < y_cut)
        mesh.remove_unreferenced_vertices()
        before = int(len(mesh.faces))
        out, stats = restore_patch_in_region(mesh, "top")
        self.assertGreaterEqual(int(len(out.faces)), before)
        self.assertIn(stats.get("how"), {"fill", "mirror"})


class MeshMaskStoreTests(unittest.TestCase):
    def test_mask_roundtrip_and_carve(self) -> None:
        import tempfile
        from pathlib import Path

        import numpy as np

        from mesh_forge.chat.store import ChatStore
        from mesh_forge.ops.geometry import carve_faces

        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
        drop_n = 12
        idx = np.arange(drop_n, dtype=np.int32)
        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("mask").id
            store.set_mesh_mask(chat, "body.stl", idx)
            meta = store.get_meta(chat)
            self.assertEqual(int(meta.mesh_mask["count"]), drop_n)
            loaded = store.load_mesh_mask(chat, n_faces=len(mesh.faces), mesh_name="body.stl")
            self.assertIsNotNone(loaded)
            self.assertEqual(int(loaded.sum()), drop_n)
            out, stats = carve_faces(mesh, loaded, min_keep_ratio=0.1, min_keep_faces=4, drop_crumbs=False)
            self.assertEqual(int(stats["faces_dropped"]), drop_n)
            self.assertEqual(int(stats.get("faces_extra") or 0), 0)
            self.assertEqual(int(len(out.faces)), int(len(mesh.faces)) - drop_n)
            wrong = store.load_mesh_mask(chat, n_faces=len(mesh.faces), mesh_name="other.stl")
            self.assertIsNone(wrong)

    def test_new_mask_resets_look_seen(self) -> None:
        import tempfile
        from pathlib import Path

        import numpy as np

        from mesh_forge.chat.store import ChatStore

        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("mask").id
            store.set_look_view(chat, views="right")
            store.set_look_view(chat, views="front")
            self.assertEqual(store.get_meta(chat).look_view.get("seen"), ["right", "front"])
            store.set_mesh_mask(chat, "body.stl", np.array([1, 2, 3], dtype=np.int32))
            self.assertEqual(store.get_meta(chat).look_view.get("seen"), [])
            self.assertEqual(store.get_meta(chat).look_view.get("views"), "front")

    def test_clear_mesh_target_keeps_mask(self) -> None:
        import tempfile
        from pathlib import Path

        import numpy as np

        from mesh_forge.chat.store import ChatStore

        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("mask").id
            store.apply_mesh_topo(chat, {"kind": "face", "face": 12, "vertex": 3, "edge": [1, 2], "nx": 0.5, "ny": 0.5, "nz": 0.5})
            store.set_mesh_mask(chat, "body.stl", np.array([1, 2, 3], dtype=np.int32))
            store.clear_mesh_target(chat)
            meta = store.get_meta(chat)
            self.assertEqual(meta.mesh_pick, [])
            self.assertEqual(meta.mesh_topo, {})
            self.assertEqual(int(meta.mesh_mask.get("count") or 0), 3)

    def test_mask_state_roundtrip_and_mesh_change_clears_it(self) -> None:
        import tempfile
        from pathlib import Path

        from mesh_forge.chat.store import ChatStore

        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("mask").id
            files = store.files_dir(chat)
            source = files / "body.stl"
            source.write_bytes(b"solid empty\nendsolid empty\n")
            edited = files / "body_fixed.stl"
            edited.write_bytes(b"solid empty\nendsolid empty\n")
            store.set_current_mesh(chat, source, role="source")
            store.set_mask_state(chat, {"proposal_status": "ready", "candidate_faces": 24})
            self.assertEqual(store.mask_state(chat).get("proposal_status"), "ready")
            store.set_current_mesh(chat, edited, role="edit")
            self.assertEqual(store.mask_state(chat), {})

    def test_removal_state_roundtrip_and_mesh_change_clears_it(self) -> None:
        import tempfile
        from pathlib import Path

        from mesh_forge.chat.store import ChatStore

        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("mask").id
            files = store.files_dir(chat)
            source = files / "body.stl"
            source.write_bytes(b"solid empty\nendsolid empty\n")
            edited = files / "body_fixed.stl"
            edited.write_bytes(b"solid empty\nendsolid empty\n")
            store.set_current_mesh(chat, source, role="source")
            store.set_removal_state(chat, {"strategy": "protrusion_cut", "proposal_status": "ready"})
            self.assertEqual(store.removal_state(chat).get("strategy"), "protrusion_cut")
            store.set_current_mesh(chat, edited, role="edit")
            self.assertEqual(store.removal_state(chat), {})

    def test_visible_lump_stays_on_stick(self) -> None:
        from mesh_forge.ops.topo import grow_visible_lump
        from mesh_forge.render import _camera_eye_target, _seat_for_viewer

        body = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        body.apply_translation([0.0, 0.5, 0.0])
        stick = trimesh.creation.box(extents=[0.8, 0.14, 0.14])
        stick.apply_translation([0.9, 0.5, 0.0])
        mesh = trimesh.util.concatenate([body, stick])
        verts, faces, extent = _seat_for_viewer(mesh)
        eye, target = _camera_eye_target(extent, "right", pad=1.0, zoom=1.0)
        seed = int(mesh.triangles_center[:, 0].argmax())
        mask = grow_visible_lump(
            mesh,
            seed,
            seated_verts=verts,
            seated_faces=faces,
            eye=eye,
            target=target,
        )
        self.assertGreaterEqual(int(mask.sum()), 4)
        self.assertLess(int(mask.sum()), int(0.55 * len(mesh.faces)))

    def test_screen_lump_finds_petal_not_body(self) -> None:
        from mesh_forge.ops.topo import grow_screen_lump, viewport_hit
        from mesh_forge.render import _camera_eye_target, _seat_for_viewer

        body = _box([0.0, 0.5, 0.0], [1.0, 1.2, 0.8], subdivisions=2)
        petal = _box([0.72, 0.22, 0.0], [0.16, 0.10, 0.10], subdivisions=2)
        mesh = trimesh.util.concatenate([body, petal])
        mesh.merge_vertices()
        verts, faces, extent = _seat_for_viewer(mesh)
        eye, target = _camera_eye_target(extent, "right", pad=1.0, zoom=1.5)
        topo = viewport_hit(mesh, views="right", x=0.78, y=0.58, zoom=1.5)
        seed = int(topo["face"])
        mask = grow_screen_lump(
            mesh,
            seed,
            seated_verts=verts,
            seated_faces=faces,
            eye=eye,
            target=target,
            x=0.78,
            y=0.58,
        )
        petal_faces = mesh.triangles_center[:, 0] > 0.42
        self.assertGreaterEqual(int((mask & petal_faces).sum()), 8)

    def test_screen_lump_covers_dense_petal(self) -> None:
        import numpy as np
        from mesh_forge.ops.topo import grow_screen_lump
        from mesh_forge.render import _camera_eye_target, _project_points, _seat_for_viewer

        body = trimesh.creation.icosphere(subdivisions=4, radius=0.55)
        body.apply_translation([0.0, 0.55, 0.0])
        petal = trimesh.creation.icosphere(subdivisions=3, radius=0.18)
        petal.apply_translation([0.68, 0.28, 0.0])
        mesh = trimesh.util.concatenate([body, petal])
        mesh.merge_vertices()
        verts, faces, extent = _seat_for_viewer(mesh)
        eye, target = _camera_eye_target(extent, "right", pad=1.0, zoom=1.5)
        centers = verts[faces].mean(axis=1)
        petal_faces = centers[:, 0] > 0.42
        xy, _ = _project_points(centers, eye, target, 512)
        hit = xy[petal_faces].mean(axis=0)
        x = float(hit[0] / 511.0)
        y = float(hit[1] / 511.0)
        seed = int(np.where(petal_faces)[0][0])
        mask = grow_screen_lump(
            mesh,
            seed,
            seated_verts=verts,
            seated_faces=faces,
            eye=eye,
            target=target,
            x=x,
            y=y,
        )
        caught = int((mask & petal_faces).sum())
        self.assertGreaterEqual(caught, int(0.40 * int(petal_faces.sum())))
        self.assertLess(int((mask & ~petal_faces).sum()), int(0.15 * len(mesh.faces)))

    def test_screen_lump_does_not_paint_occluded_body(self) -> None:
        import numpy as np
        from mesh_forge.ops.topo import grow_screen_lump
        from mesh_forge.render import _camera_eye_target, _project_points, _seat_for_viewer

        near = trimesh.creation.box(extents=[0.2, 0.8, 0.8])
        near.apply_translation([0.7, 0.4, 0.0])
        far = trimesh.creation.box(extents=[0.2, 0.8, 0.8])
        far.apply_translation([-0.7, 0.4, 0.0])
        mesh = trimesh.util.concatenate([near, far])
        mesh.merge_vertices()
        verts, faces, extent = _seat_for_viewer(mesh)
        eye, target = _camera_eye_target(extent, "right", pad=1.0, zoom=1.2)
        centers = verts[faces].mean(axis=1)
        near_faces = centers[:, 0] > 0.0
        xy, _ = _project_points(centers, eye, target, 512)
        hit = xy[near_faces].mean(axis=0)
        x = float(np.clip(hit[0] / 511.0, 0.0, 1.0))
        y = float(np.clip(hit[1] / 511.0, 0.0, 1.0))
        seed = int(np.where(near_faces)[0][0])
        mask = grow_screen_lump(
            mesh,
            seed,
            seated_verts=verts,
            seated_faces=faces,
            eye=eye,
            target=target,
            x=x,
            y=y,
        )
        self.assertGreaterEqual(int((mask & near_faces).sum()), 2)
        self.assertEqual(int((mask & ~near_faces).sum()), 0)

    def test_paint_screen_region_uses_image_box_not_mesh_walk(self) -> None:
        import numpy as np
        from mesh_forge.ops.topo import paint_screen_region
        from mesh_forge.render import _camera_eye_target, _project_points, _seat_for_viewer

        body = _box([0.0, 0.7, 0.0], [0.7, 0.9, 0.6], subdivisions=2)
        skirt = _box([0.0, 0.18, 0.0], [1.1, 0.32, 0.9], subdivisions=2)
        petal = _box([0.82, 0.14, 0.05], [0.30, 0.08, 0.08], subdivisions=2)
        n_petal = int(len(petal.faces))
        mesh = trimesh.util.concatenate([body, skirt, petal])
        verts, faces, extent = _seat_for_viewer(mesh)
        eye, target = _camera_eye_target(extent, "front", pad=1.0, zoom=1.2)
        centers = verts[faces].mean(axis=1)
        petal_faces = np.zeros(len(faces), dtype=bool)
        petal_faces[-n_petal:] = True
        xy, _ = _project_points(centers, eye, target, 512)
        pts = xy[petal_faces]
        hit = pts.mean(axis=0)
        x = float(np.clip(hit[0] / 511.0, 0.0, 1.0))
        y = float(np.clip(hit[1] / 511.0, 0.0, 1.0))
        x0 = float(np.clip(pts[:, 0].min() / 511.0 - 0.02, 0.0, 1.0))
        x1 = float(np.clip(pts[:, 0].max() / 511.0 + 0.02, 0.0, 1.0))
        y0 = float(np.clip(pts[:, 1].min() / 511.0 - 0.02, 0.0, 1.0))
        y1 = float(np.clip(pts[:, 1].max() / 511.0 + 0.02, 0.0, 1.0))
        mask = paint_screen_region(
            mesh, verts, faces, eye, target, x0=x0, y0=y0, x1=x1, y1=y1, side="right"
        )
        self.assertGreaterEqual(int((mask & petal_faces).sum()), int(0.45 * int(petal_faces.sum())))
        self.assertLess(int((mask & ~petal_faces).sum()), int(0.20 * len(mesh.faces)))

    def test_complete_visual_mask_keeps_click_component(self) -> None:
        import numpy as np
        from mesh_forge.ops.topo import complete_visual_mask, paint_screen_region
        from mesh_forge.render import _camera_eye_target, _project_points, _seat_for_viewer

        body = _box([0.0, 0.7, 0.0], [0.7, 0.9, 0.6], subdivisions=2)
        skirt = _box([0.0, 0.18, 0.0], [1.1, 0.32, 0.9], subdivisions=2)
        petal = _box([0.82, 0.14, 0.05], [0.30, 0.08, 0.08], subdivisions=2)
        n_petal = int(len(petal.faces))
        mesh = trimesh.util.concatenate([body, skirt, petal])
        verts, faces, extent = _seat_for_viewer(mesh)
        eye, target = _camera_eye_target(extent, "front", pad=1.0, zoom=1.2)
        centers = verts[faces].mean(axis=1)
        petal_faces = np.zeros(len(faces), dtype=bool)
        petal_faces[-n_petal:] = True
        xy, _ = _project_points(centers, eye, target, 512)
        pts = xy[petal_faces]
        hit = pts.mean(axis=0)
        x = float(np.clip(hit[0] / 511.0, 0.0, 1.0))
        y = float(np.clip(hit[1] / 511.0, 0.0, 1.0))
        x0 = float(np.clip(pts[:, 0].min() / 511.0 - 0.02, 0.0, 1.0))
        x1 = float(np.clip(pts[:, 0].max() / 511.0 + 0.02, 0.0, 1.0))
        y0 = float(np.clip(pts[:, 1].min() / 511.0 - 0.02, 0.0, 1.0))
        y1 = float(np.clip(pts[:, 1].max() / 511.0 + 0.02, 0.0, 1.0))
        visual = paint_screen_region(
            mesh, verts, faces, eye, target, x0=x0, y0=y0, x1=x1, y1=y1, side="right"
        )
        visual[0] = True
        seed = int(np.flatnonzero(petal_faces)[0])
        mask = complete_visual_mask(
            mesh,
            verts,
            faces,
            visual,
            eye=eye,
            target=target,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            seed=seed,
            aim_x=x,
            aim_y=y,
        )
        self.assertTrue(bool(mask[seed]))
        self.assertFalse(bool(mask[0]))
        self.assertGreaterEqual(int((mask & petal_faces).sum()), int(0.50 * n_petal))

    def test_paint_region_ignores_hidden_centroids_in_box(self) -> None:
        import numpy as np
        from mesh_forge.ops.topo import paint_screen_region
        from mesh_forge.render import _camera_eye_target, _project_points, _seat_for_viewer

        body = _box([0.0, 0.7, 0.0], [0.7, 0.9, 0.6], subdivisions=2)
        skirt = _box([0.0, 0.18, 0.0], [1.1, 0.32, 0.9], subdivisions=2)
        petal = _box([0.82, 0.14, 0.05], [0.30, 0.08, 0.08], subdivisions=2)
        n_petal = int(len(petal.faces))
        mesh = trimesh.util.concatenate([body, skirt, petal])
        verts, faces, extent = _seat_for_viewer(mesh)
        eye, target = _camera_eye_target(extent, "front", pad=1.0, zoom=1.2)
        centers = verts[faces].mean(axis=1)
        petal_faces = np.zeros(len(faces), dtype=bool)
        petal_faces[-n_petal:] = True
        xy, depth = _project_points(centers, eye, target, 512)
        pts = xy[petal_faces]
        x0 = float(np.clip(pts[:, 0].min() / 511.0 - 0.10, 0.0, 1.0))
        x1 = float(np.clip(pts[:, 0].max() / 511.0 + 0.10, 0.0, 1.0))
        y0 = float(np.clip(pts[:, 1].min() / 511.0 - 0.10, 0.0, 1.0))
        y1 = float(np.clip(pts[:, 1].max() / 511.0 + 0.10, 0.0, 1.0))
        size = 512
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
        skirt_in_box = in_box & ~petal_faces
        mask = paint_screen_region(
            mesh, verts, faces, eye, target, x0=x0, y0=y0, x1=x1, y1=y1, side="right"
        )
        self.assertGreater(int(skirt_in_box.sum()), 0)
        self.assertGreater(int((mask & petal_faces).sum()), 0)
        # Raster-first: do not paint every centroid in the widened box (old bug ~100 skirt faces).
        self.assertLess(int(mask.sum()), int(0.55 * int(in_box.sum())))

    def test_mask_from_view_observations_prefers_cross_view_petal(self) -> None:
        import numpy as np
        from mesh_forge.ops.topo import mask_from_view_observations
        from mesh_forge.render import _camera_eye_target, _project_points, _seat_for_viewer

        body = _box([0.0, 0.7, 0.0], [0.7, 0.9, 0.6], subdivisions=2)
        skirt = _box([0.0, 0.18, 0.0], [1.1, 0.32, 0.9], subdivisions=2)
        petal = _box([0.82, 0.14, 0.05], [0.30, 0.08, 0.08], subdivisions=2)
        n_petal = int(len(petal.faces))
        mesh = trimesh.util.concatenate([body, skirt, petal])
        verts, faces, extent = _seat_for_viewer(mesh)
        petal_faces = np.zeros(len(faces), dtype=bool)
        petal_faces[-n_petal:] = True
        observations = []
        for camera in ("front", "right"):
            eye, target = _camera_eye_target(extent, camera, pad=1.0, zoom=1.3)
            centers = verts[faces].mean(axis=1)
            xy, _ = _project_points(centers, eye, target, 512)
            pts = xy[petal_faces]
            observations.append(
                {
                    "view": camera,
                    "visible": True,
                    "confidence": 0.9,
                    "kind": "protrusion",
                    "touchesBody": True,
                    "x0": float(np.clip(pts[:, 0].min() / 511.0 - 0.03, 0.0, 1.0)),
                    "y0": float(np.clip(pts[:, 1].min() / 511.0 - 0.03, 0.0, 1.0)),
                    "x1": float(np.clip(pts[:, 0].max() / 511.0 + 0.03, 0.0, 1.0)),
                    "y1": float(np.clip(pts[:, 1].max() / 511.0 + 0.03, 0.0, 1.0)),
                    "eye": eye,
                    "target": target,
                }
            )
        mask, scores = mask_from_view_observations(mesh, verts, faces, observations)
        self.assertGreater(float(np.maximum(scores, 0.0).sum()), 0.0)
        self.assertGreaterEqual(int((mask & petal_faces).sum()), int(0.45 * n_petal))
        self.assertLess(int((mask & ~petal_faces).sum()), int(0.20 * len(mesh.faces)))

    def test_apply_review_refinement_shrinks_and_grows(self) -> None:
        import numpy as np

        from mesh_forge.render import _seat_for_viewer
        from mesh_forge.tools.mask_mesh import _apply_review_refinement

        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
        verts, faces, _ = _seat_for_viewer(mesh)
        mask = np.zeros(len(mesh.faces), dtype=bool)
        mask[:24] = True
        shrunk = _apply_review_refinement(mesh, mask, "too_much", verts=verts, faces=faces)
        grown = _apply_review_refinement(mesh, mask, "too_little", verts=verts, faces=faces)
        self.assertLessEqual(int(shrunk.sum()), int(mask.sum()))
        self.assertGreaterEqual(int(grown.sum()), int(mask.sum()))

    def test_review_view_pack_uses_focus_only(self) -> None:
        from types import SimpleNamespace

        from mesh_forge.tools.mask_mesh import (
            _active_click_topo,
            _auto_acceptance_failure,
            _mask_view_pack,
            _review_view_pack,
            _score_observations,
            _semantic_focus_view,
        )

        self.assertEqual(_mask_view_pack("right"), ["right", "front", "back"])
        self.assertEqual(_mask_view_pack("front"), ["front", "right", "left"])
        self.assertEqual(_review_view_pack("right"), ["right"])
        self.assertEqual(_review_view_pack("front"), ["front"])
        scored = _score_observations(
            "right",
            [
                {"view": "back", "visible": True, "confidence": 0.95},
                {"view": "right", "visible": True, "confidence": 0.70},
                {"view": "front", "visible": True, "confidence": 0.99},
            ],
            limit=2,
        )
        self.assertEqual([obs["view"] for obs in scored], ["right", "front"])
        complementary = _score_observations(
            "right",
            [
                {"view": "back", "visible": True, "confidence": 0.99},
                {"view": "right", "visible": True, "confidence": 0.70},
                {"view": "front", "visible": True, "confidence": 0.80},
            ],
            limit=2,
        )
        self.assertEqual([obs["view"] for obs in complementary], ["right", "front"])

        class _Store:
            def active_mesh_topo(self, _chat_id):
                return {"mesh": "mesh.stl", "face": 7, "vertex": 1}

        ctx = SimpleNamespace(deps=SimpleNamespace(chat_id="c", store=_Store()))
        self.assertEqual(int(_active_click_topo(ctx, "mesh.stl")["face"]), 7)
        self.assertEqual(_active_click_topo(ctx, "other.stl"), {})
        self.assertEqual(_semantic_focus_view("лепесток справа на юбке", "viewer"), "right")
        self.assertEqual(_semantic_focus_view("отросток сзади", "viewer"), "back")
        self.assertEqual(
            _auto_acceptance_failure(
                {"is_slab": False, "outward_score": 0.7, "largest_component_faces": 14, "area_frac": 0.01},
                {"verdict": "tiny_spot"},
            ),
            "tiny spot",
        )
        self.assertEqual(
            _auto_acceptance_failure(
                {"is_slab": True, "outward_score": 0.7, "largest_component_faces": 14, "area_frac": 0.01},
                {"verdict": "ok"},
            ),
            "flat skirt patch",
        )

    def test_build_auto_mask_stops_after_confident_first_pass(self) -> None:
        import numpy as np
        from types import SimpleNamespace

        from mesh_forge.tools import mask_mesh as mm

        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        ctx = SimpleNamespace(
            deps=SimpleNamespace(chat_id="c", store=None, emit_event=lambda *_args, **_kwargs: None)
        )
        src = Path("mesh.stl")
        calls: list[str] = []

        orig_render = mm._render_detection_pack
        orig_comfy = mm._detect_multi_view_with_comfy
        orig_detect = mm._detect_multi_view
        orig_mask = mm.mask_from_view_observations
        orig_review = mm._review_mask_candidate
        try:
            mm._render_detection_pack = lambda *args, **kwargs: [{"view": "right"}]
            mm._detect_multi_view_with_comfy = lambda *args, **kwargs: []

            def fake_detect(*args, **kwargs):
                calls.append("detect")
                return [
                    {"view": "right", "visible": True, "confidence": 0.9, "x0": 0.6, "y0": 0.3, "x1": 0.9, "y1": 0.8},
                    {"view": "front", "visible": True, "confidence": 0.8, "x0": 0.5, "y0": 0.2, "x1": 0.85, "y1": 0.75},
                ]

            def fake_mask(*args, **kwargs):
                mask = np.zeros(len(mesh.faces), dtype=bool)
                mask[:4] = True
                scores = np.ones(len(mesh.faces), dtype=float)
                return mask, scores

            def fake_review(*args, **kwargs):
                calls.append("review")
                return {"verdict": "ok", "confidence": 0.8}, ["right"]

            mm._detect_multi_view = fake_detect
            mm.mask_from_view_observations = fake_mask
            mm._review_mask_candidate = fake_review

            result = mm._build_auto_mask(
                ctx,
                mesh,
                src,
                target="petal",
                focus_view="right",
                yaw=None,
                pitch=None,
                zoom=1.0,
            )
        finally:
            mm._render_detection_pack = orig_render
            mm._detect_multi_view_with_comfy = orig_comfy
            mm._detect_multi_view = orig_detect
            mm.mask_from_view_observations = orig_mask
            mm._review_mask_candidate = orig_review

        self.assertEqual(calls, ["detect", "review"])
        self.assertEqual(result["review_views"], ["right"])
        self.assertEqual(int(result["mask"].sum()), 4)

    def test_mask_bbox_from_preview_extracts_normalized_box(self) -> None:
        import tempfile
        from pathlib import Path

        from PIL import Image

        from mesh_forge.tools.mask_mesh import _mask_bbox_from_preview

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mask.png"
            img = Image.new("L", (100, 80), 0)
            for x in range(20, 71):
                for y in range(10, 51):
                    img.putpixel((x, y), 255)
            img.save(path)
            bbox = _mask_bbox_from_preview(path)

        self.assertIsNotNone(bbox)
        assert bbox is not None
        self.assertAlmostEqual(float(bbox["x0"]), 20 / 99.0, places=2)
        self.assertAlmostEqual(float(bbox["x1"]), 70 / 99.0, places=2)
        self.assertAlmostEqual(float(bbox["y0"]), 10 / 79.0, places=2)
        self.assertAlmostEqual(float(bbox["y1"]), 50 / 79.0, places=2)

    def test_detect_multi_view_with_comfy_uses_segmentation_when_enabled(self) -> None:
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch

        import numpy as np
        from PIL import Image

        from mesh_forge.config import AppConfig, SegmentationConfig
        from mesh_forge.domain import ImageArtifact, SegmentationArtifact
        from mesh_forge.tools.mask_mesh import _detect_multi_view_with_comfy

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            render_path = root / "render.png"
            Image.new("RGB", (64, 64), (0, 0, 0)).save(render_path)
            mask_path = root / "mask.png"
            mask = Image.new("L", (64, 64), 0)
            for x in range(16, 49):
                for y in range(12, 45):
                    mask.putpixel((x, y), 255)
            mask.save(mask_path)
            vis_path = root / "vis.png"
            Image.new("RGB", (64, 64), (255, 0, 0)).save(vis_path)

            class _Store:
                def files_dir(self, _chat_id):
                    return root

                def artifact_from_path(self, _chat_id, path, label="", view=""):
                    return {"path": str(path), "label": label, "view": view}

            emitted = []
            ctx = SimpleNamespace(
                deps=SimpleNamespace(
                    chat_id="c",
                    store=_Store(),
                    emit_event=lambda *_args, **_kwargs: None,
                    emit_artifact=lambda art, **_kwargs: emitted.append(art),
                    files_dir=lambda: root,
                )
            )
            records = [
                {
                    "view": "right",
                    "path": render_path,
                    "eye": np.array([1.0, 0.0, 0.0]),
                    "target": np.array([0.0, 0.0, 0.0]),
                    "zoom": 1.2,
                },
                {
                    "view": "front",
                    "path": render_path,
                    "eye": np.array([0.0, 0.0, 1.0]),
                    "target": np.array([0.0, 0.0, 0.0]),
                    "zoom": 1.05,
                },
                {
                    "view": "back",
                    "path": render_path,
                    "eye": np.array([0.0, 0.0, -1.0]),
                    "target": np.array([0.0, 0.0, 0.0]),
                    "zoom": 1.05,
                },
            ]
            cfg = AppConfig(segmentation=SegmentationConfig(enabled=True, max_views=4))

            class _Client:
                def segment_view_by_text(self, *_args, **_kwargs):
                    return SegmentationArtifact(
                        mask=ImageArtifact(path=mask_path, label="mask"),
                        visualization=ImageArtifact(path=vis_path, label="seg"),
                        boxes=[{"x0": 0.62, "y0": 0.48, "x1": 0.78, "y1": 0.66}],
                        scores=[0.91],
                    )

            with patch("mesh_forge.config.load_config", return_value=cfg), patch(
                "mesh_forge.adapters.comfyui_client.load_config", return_value=cfg
            ), patch("mesh_forge.adapters.comfyui_client.ComfyUiClient", return_value=_Client()):
                observations = _detect_multi_view_with_comfy(ctx, "right skirt flap", records)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["view"], "right")
        self.assertAlmostEqual(float(observations[0]["x0"]), 0.62, places=3)
        self.assertLess(float(observations[0]["x1"]) - float(observations[0]["x0"]), 0.25)
        self.assertGreaterEqual(len(emitted), 2)

    def test_comfy_bbox_too_broad_is_ignored(self) -> None:
        from mesh_forge.tools.mask_mesh import _bbox_is_too_broad

        self.assertTrue(_bbox_is_too_broad({"x0": 0.37, "y0": 0.17, "x1": 0.63, "y1": 0.77}))
        self.assertFalse(_bbox_is_too_broad({"x0": 0.62, "y0": 0.48, "x1": 0.78, "y1": 0.66}))

    def test_emit_mask_preview_copies_nested_comfy_files_into_chat_files(self) -> None:
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        from PIL import Image

        from mesh_forge.agent.deps import ChatDeps
        from mesh_forge.chat.store import ChatStore
        from mesh_forge.tools.mask_mesh import _emit_mask_preview

        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("mask").id
            nested = store.files_dir(chat) / "segmentation" / "segmentation"
            nested.mkdir(parents=True)
            src = nested / "visualization.png"
            Image.new("RGB", (8, 8), (255, 0, 0)).save(src)
            emitted: list = []
            ctx = SimpleNamespace(
                deps=ChatDeps(
                    chat_id=chat,
                    store=store,
                    emit=lambda event: emitted.append(event),
                )
            )
            _emit_mask_preview(ctx, src, label="маска · comfy right", view="right")

            self.assertEqual(len(emitted), 1)
            artifact = emitted[0]["artifact"]
            served = store.resolve_file(chat, artifact["name"])
            self.assertTrue(served.is_file())
            self.assertEqual(served.parent, store.files_dir(chat))
            self.assertNotEqual(served.name, "visualization.png")

    def test_mask_geometry_metrics_marks_flat_patch(self) -> None:
        import numpy as np

        from mesh_forge.ops.topo import mask_geometry_metrics

        skirt = trimesh.creation.box(extents=[1.1, 0.32, 0.9])
        skirt.apply_translation([0.0, 0.18, 0.0])
        verts = np.asarray(skirt.vertices, dtype=np.float64)
        faces = np.asarray(skirt.faces, dtype=np.int64)
        centers = verts[faces].mean(axis=1)
        patch = centers[:, 0] > np.percentile(centers[:, 0], 75)
        metrics = mask_geometry_metrics(skirt, patch, verts=verts, faces=faces)
        self.assertGreater(int(metrics["faces"]), 0)
        self.assertGreater(float(metrics["area_frac"]), 0.01)

    def test_connected_protrusion_covers_whole_stick(self) -> None:
        from mesh_forge.ops.topo import grow_visible_lump
        from mesh_forge.render import _camera_eye_target, _seat_for_viewer

        body = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        stick = trimesh.creation.box(extents=[1.2, 0.16, 0.16])
        stick.apply_translation([0.75, 0.0, 0.0])
        mesh = trimesh.util.concatenate([body, stick])
        mesh.merge_vertices()
        verts, faces, extent = _seat_for_viewer(mesh)
        eye, target = _camera_eye_target(extent, "right", pad=1.0, zoom=1.0)
        seed = int(mesh.triangles_center[:, 0].argmax())
        mask = grow_visible_lump(
            mesh,
            seed,
            seated_verts=verts,
            seated_faces=faces,
            eye=eye,
            target=target,
        )
        stick_faces = mesh.triangles_center[:, 0] > 0.42
        self.assertGreater(int((mask & stick_faces).sum()), int(0.55 * max(int(stick_faces.sum()), 1)))
        self.assertLess(int(mask.sum()), int(0.55 * len(mesh.faces)))

    def test_mask_from_view_observations_grows_connected_protrusion_from_seed(self) -> None:
        import numpy as np

        from mesh_forge.ops.topo import mask_from_view_observations
        from mesh_forge.render import _camera_eye_target, _project_points, _seat_for_viewer

        body = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        stick = trimesh.creation.box(extents=[1.2, 0.16, 0.16])
        stick.apply_translation([0.75, 0.0, 0.0])
        mesh = trimesh.util.concatenate([body, stick])
        mesh.merge_vertices()
        verts, faces, extent = _seat_for_viewer(mesh)
        centers = verts[faces].mean(axis=1)
        protrusion_faces = centers[:, 0] > 0.42
        observations = []
        for camera in ("right", "front", "back"):
            eye, target = _camera_eye_target(extent, camera, pad=1.0, zoom=1.2 if camera == "right" else 1.05)
            xy, _ = _project_points(centers, eye, target, 512)
            pts = xy[protrusion_faces]
            observations.append(
                {
                    "view": camera,
                    "visible": True,
                    "confidence": 0.95 if camera != "front" else 0.82,
                    "kind": "protrusion",
                    "touchesBody": True,
                    "x0": float(np.clip(pts[:, 0].min() / 511.0 - 0.03, 0.0, 1.0)),
                    "y0": float(np.clip(pts[:, 1].min() / 511.0 - 0.03, 0.0, 1.0)),
                    "x1": float(np.clip(pts[:, 0].max() / 511.0 + 0.03, 0.0, 1.0)),
                    "y1": float(np.clip(pts[:, 1].max() / 511.0 + 0.03, 0.0, 1.0)),
                    "eye": eye,
                    "target": target,
                }
            )
        mask, _scores = mask_from_view_observations(mesh, verts, faces, observations)
        self.assertGreaterEqual(int(mask.sum()), 8)
        self.assertGreater(
            int((mask & protrusion_faces).sum()),
            int(0.55 * max(int(protrusion_faces.sum()), 1)),
        )
        self.assertLess(int(mask.sum()), int(0.55 * len(mesh.faces)))

    def test_mask_from_view_observations_handles_connected_flat_flap(self) -> None:
        import numpy as np

        from mesh_forge.ops.topo import mask_from_view_observations
        from mesh_forge.render import _camera_eye_target, _project_points, _seat_for_viewer

        mesh = trimesh.creation.box(extents=[1.0, 1.0, 0.8])
        verts0 = np.asarray(mesh.vertices, dtype=np.float64)
        flap_idx = np.where(
            (verts0[:, 0] > 0.45)
            & (verts0[:, 1] < 0.10)
            & (verts0[:, 2] > -0.05)
        )[0]
        verts0[flap_idx, 0] += 0.34
        verts0[flap_idx, 2] += 0.10
        mesh.vertices = verts0
        verts, faces, extent = _seat_for_viewer(mesh)
        centers = verts[faces].mean(axis=1)
        flap_faces = np.any(np.isin(faces, flap_idx), axis=1)
        observations = []
        for camera in ("right", "front", "back"):
            eye, target = _camera_eye_target(extent, camera, pad=1.0, zoom=1.25 if camera == "right" else 1.05)
            xy, _ = _project_points(centers, eye, target, 512)
            pts = xy[flap_faces]
            observations.append(
                {
                    "view": camera,
                    "visible": True,
                    "confidence": 0.95 if camera != "front" else 0.5,
                    "kind": "protrusion",
                    "touchesBody": True,
                    "x0": float(np.clip(pts[:, 0].min() / 511.0 - 0.03, 0.0, 1.0)),
                    "y0": float(np.clip(pts[:, 1].min() / 511.0 - 0.03, 0.0, 1.0)),
                    "x1": float(np.clip(pts[:, 0].max() / 511.0 + 0.03, 0.0, 1.0)),
                    "y1": float(np.clip(pts[:, 1].max() / 511.0 + 0.03, 0.0, 1.0)),
                    "eye": eye,
                    "target": target,
                }
            )
        mask, _scores = mask_from_view_observations(mesh, verts, faces, observations)
        self.assertGreaterEqual(int(mask.sum()), 4)
        self.assertGreater(
            int((mask & flap_faces).sum()),
            int(0.45 * max(int(flap_faces.sum()), 1)),
        )
        self.assertLess(int(mask.sum()), int(0.70 * len(mesh.faces)))

    def test_spike_picks_petal_not_skirt_panel(self) -> None:
        from mesh_forge.ops.topo import grow_visible_lump, silhouette_spike_face
        from mesh_forge.render import _camera_eye_target, _seat_for_viewer

        body = _box([0.0, 0.7, 0.0], [0.7, 0.9, 0.6], subdivisions=2)
        skirt = _box([0.0, 0.18, 0.0], [1.1, 0.32, 0.9], subdivisions=2)
        petal = _box([0.82, 0.14, 0.05], [0.30, 0.08, 0.08], subdivisions=2)
        mesh = trimesh.util.concatenate([body, skirt, petal])
        mesh.merge_vertices()
        verts, faces, extent = _seat_for_viewer(mesh)
        eye, target = _camera_eye_target(extent, "front", pad=1.0, zoom=1.2)
        seed = silhouette_spike_face(
            verts, faces, eye, target, side="right", y_lo=40.0, y_hi=100.0
        )
        self.assertGreaterEqual(seed, 0)
        centers = verts[faces].mean(axis=1)
        petal_faces = centers[:, 0] > 0.52
        self.assertTrue(bool(petal_faces[seed]))
        mask = grow_visible_lump(
            mesh,
            seed,
            seated_verts=verts,
            seated_faces=faces,
            eye=eye,
            target=target,
        )
        self.assertGreaterEqual(int((mask & petal_faces).sum()), 6)
        self.assertLess(int((mask & ~petal_faces).sum()), int(0.22 * len(mesh.faces)))

    def test_wide_hem_flap_is_a_lobe_from_front(self) -> None:
        from mesh_forge.ops.topo import (
            grow_visible_lump,
            mask_silhouette_cameras,
            silhouette_extra_face,
            silhouette_lobe_face,
        )
        from mesh_forge.render import _camera_eye_target, _seat_for_viewer

        body = _box([0.0, 0.7, 0.0], [0.7, 0.9, 0.6], subdivisions=2)
        skirt = _box([0.0, 0.22, 0.0], [1.0, 0.36, 0.9], subdivisions=2)
        flap = _box([0.78, 0.02, 0.0], [0.46, 0.22, 0.07], subdivisions=2)
        mesh = trimesh.util.concatenate([body, skirt, flap])
        mesh.merge_vertices()
        verts, faces, extent = _seat_for_viewer(mesh)
        eye, target = _camera_eye_target(extent, "front", pad=1.0, zoom=1.2)
        seed = silhouette_lobe_face(
            verts, faces, eye, target, side="right", y_lo=40.0, y_hi=100.0
        )
        self.assertGreaterEqual(seed, 0)
        extra = silhouette_extra_face(
            verts, faces, eye, target, side="right", y_lo=40.0, y_hi=100.0
        )
        self.assertGreaterEqual(extra, 0)
        self.assertEqual(mask_silhouette_cameras("лепесток справа на юбке", "right")[0], "front")
        centers = verts[faces].mean(axis=1)
        flap_faces = centers[:, 0] > 0.55
        self.assertTrue(bool(flap_faces[seed]))
        mask = grow_visible_lump(
            mesh,
            seed,
            seated_verts=verts,
            seated_faces=faces,
            eye=eye,
            target=target,
        )
        self.assertGreaterEqual(int((mask & flap_faces).sum()), 6)
        self.assertLess(int(mask.sum()), int(0.45 * len(mesh.faces)))

    def test_small_connected_flap_is_kept(self) -> None:
        import numpy as np
        from mesh_forge.ops.topo import grow_visible_lump, mask_is_tiny
        from mesh_forge.render import _camera_eye_target, _seat_for_viewer

        body = _box([0.0, 0.5, 0.0], [0.8, 1.0, 0.7], subdivisions=1)
        flap = trimesh.creation.box(extents=[0.32, 0.10, 0.05])
        flap.apply_translation([0.58, 0.12, 0.0])
        mesh = trimesh.util.concatenate([body, flap])
        verts, faces, extent = _seat_for_viewer(mesh)
        eye, target = _camera_eye_target(extent, "front", pad=1.0, zoom=1.2)
        seed = int(np.argmax(verts[faces].mean(axis=1)[:, 0]))
        mask = grow_visible_lump(
            mesh,
            seed,
            seated_verts=verts,
            seated_faces=faces,
            eye=eye,
            target=target,
        )
        self.assertGreaterEqual(int(mask.sum()), 6)
        self.assertLess(int(mask.sum()), int(0.45 * len(mesh.faces)))
        self.assertFalse(mask_is_tiny(mask, verts, faces))

    def test_mask_is_tiny_rejects_two_crumbs(self) -> None:
        import numpy as np
        from mesh_forge.ops.topo import mask_is_tiny

        mesh = trimesh.creation.icosphere(subdivisions=4, radius=0.5)
        mask = np.zeros(len(mesh.faces), dtype=bool)
        mask[0] = True
        mask[1] = True
        self.assertTrue(mask_is_tiny(mask, mesh.vertices, mesh.faces))

    def test_side_camera_is_not_required_for_right_flap(self) -> None:
        from mesh_forge.ops.topo import mask_silhouette_cameras

        self.assertEqual(
            mask_silhouette_cameras("лепестковый отросток справа на юбке", "right"),
            ["front", "viewer", "right"],
        )

    def test_smooth_box_has_no_silhouette_spike(self) -> None:
        from mesh_forge.ops.topo import silhouette_spike_face
        from mesh_forge.render import _camera_eye_target, _seat_for_viewer

        mesh = _box([0.0, 0.5, 0.0], [1.0, 1.0, 1.0], subdivisions=2)
        verts, faces, extent = _seat_for_viewer(mesh)
        eye, target = _camera_eye_target(extent, "front", pad=1.0, zoom=1.0)
        self.assertEqual(silhouette_spike_face(verts, faces, eye, target, side="right"), -1)
        from mesh_forge.ops.topo import silhouette_lobe_face

        self.assertEqual(silhouette_lobe_face(verts, faces, eye, target, side="right"), -1)

    def test_silhouette_extreme_picks_petal_when_aim_misses(self) -> None:
        import numpy as np
        from mesh_forge.ops.topo import grow_visible_lump, mask_aim_side, silhouette_extreme_face
        from mesh_forge.render import _camera_eye_target, _seat_for_viewer

        body = _box([0.0, 0.5, 0.0], [1.0, 1.2, 0.8], subdivisions=2)
        petal = _box([0.78, 0.18, 0.0], [0.18, 0.12, 0.12], subdivisions=2)
        mesh = trimesh.util.concatenate([body, petal])
        mesh.merge_vertices()
        verts, faces, extent = _seat_for_viewer(mesh)
        eye, target = _camera_eye_target(extent, "front", pad=1.0, zoom=1.2)
        self.assertEqual(mask_aim_side("плоский треугольный артефакт на юбке справа"), "right")
        seed = silhouette_extreme_face(verts, faces, eye, target, side="right", y_lo=40.0, y_hi=100.0)
        centers = verts[faces].mean(axis=1)
        petal_faces = centers[:, 0] > 0.45
        self.assertTrue(bool(petal_faces[seed]))
        mask = grow_visible_lump(
            mesh,
            seed,
            seated_verts=verts,
            seated_faces=faces,
            eye=eye,
            target=target,
        )
        self.assertGreaterEqual(int((mask & petal_faces).sum()), 8)
        self.assertLess(int((mask & ~petal_faces).sum()), int(0.25 * len(mesh.faces)))

    def test_visible_lump_grows_unwelded_stick(self) -> None:
        import numpy as np
        from mesh_forge.ops.topo import grow_visible_lump
        from mesh_forge.render import _camera_eye_target, _seat_for_viewer

        body = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        body.apply_translation([0.0, 0.5, 0.0])
        stick = trimesh.creation.box(extents=[0.9, 0.14, 0.14])
        stick.apply_translation([0.9, 0.5, 0.0])
        welded = trimesh.util.concatenate([body, stick])
        soup = trimesh.Trimesh(
            vertices=np.asarray(welded.vertices)[welded.faces].reshape(-1, 3),
            faces=np.arange(len(welded.faces) * 3, dtype=np.int64).reshape(-1, 3),
            process=False,
        )
        verts, faces, extent = _seat_for_viewer(soup)
        eye, target = _camera_eye_target(extent, "right", pad=1.0, zoom=1.0)
        seed = int(verts[faces].mean(axis=1)[:, 0].argmax())
        mask = grow_visible_lump(
            soup,
            seed,
            seated_verts=verts,
            seated_faces=faces,
            eye=eye,
            target=target,
        )
        self.assertGreaterEqual(int(mask.sum()), 4)

    def test_erode_and_dilate_change_mask_size(self) -> None:
        from mesh_forge.ops.topo import dilate_face_mask, erode_face_mask, face_mask_for_topo, topology_from_ids

        mesh = _box([0.0, 0.5, 0.0], [1.0, 1.0, 1.0], subdivisions=3)
        topo = topology_from_ids(mesh, kind="face", face=0)
        topo["hops"] = 6
        seed = face_mask_for_topo(mesh, topo)
        grown = dilate_face_mask(mesh, seed, hops=2)
        self.assertGreater(int(grown.sum()), int(seed.sum()))
        shrunk = erode_face_mask(mesh, grown, hops=1)
        self.assertLess(int(shrunk.sum()), int(grown.sum()))
        self.assertGreaterEqual(int(shrunk.sum()), 1)

    def test_keep_outward_blob_drops_inner_island(self) -> None:
        import numpy as np
        from mesh_forge.ops.topo import keep_outward_blob
        from mesh_forge.render import _seat_for_viewer

        body = _box([0.0, 0.5, 0.0], [1.0, 1.0, 1.0], subdivisions=2)
        petal = _box([0.85, 0.2, 0.0], [0.2, 0.12, 0.12], subdivisions=2)
        mesh = trimesh.util.concatenate([body, petal])
        mesh.merge_vertices()
        verts, faces, _ = _seat_for_viewer(mesh)
        centers = verts[faces].mean(axis=1)
        mask = (centers[:, 0] > 0.45) | (np.linalg.norm(centers, axis=1) < 0.2)
        kept = keep_outward_blob(mesh, mask, seated_verts=verts, seated_faces=faces)
        self.assertGreater(int((kept & (centers[:, 0] > 0.45)).sum()), 0)
        self.assertLess(int(kept.sum()), int(mask.sum()))

    def test_parse_mask_review_verdicts(self) -> None:
        from mesh_forge.backends.lmstudio import parse_mask_review

        ok = parse_mask_review(
            '{"verdict":"ok","confidence":0.9,"note":"отросток","views":"right","x":0.7,"y":0.6}'
        )
        self.assertEqual(ok["verdict"], "ok")
        self.assertAlmostEqual(ok["x"], 0.7)
        over = parse_mask_review('```json\n{"verdict":"too_much","confidence":0.8,"note":"юбка"}\n```')
        self.assertEqual(over["verdict"], "too_much")
        self.assertEqual(parse_mask_review("not json"), {})

    def test_remove_extra_surface_patch_failure_clears_stale_proposal(self) -> None:
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch

        from mesh_forge.agent.deps import ChatDeps
        from mesh_forge.chat.store import ChatStore
        from mesh_forge.ops.edit import EditError
        from mesh_forge.tools.remove_extra import RemoveExtra

        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("remove").id
            mesh_path = store.files_dir(chat) / "body.stl"
            trimesh.creation.box(extents=[1.0, 1.0, 1.0]).export(mesh_path)
            store.set_current_mesh(chat, mesh_path, role="source")
            store.set_removal_state(
                chat,
                {"strategy": "protrusion_cut", "proposal_status": "ready", "proposal_mesh": "old.stl"},
            )
            ctx = SimpleNamespace(deps=ChatDeps(chat_id=chat, store=store))
            with patch("mesh_forge.tools.remove_extra.MaskMesh.run", return_value="mask_mesh failed"), patch(
                "mesh_forge.tools.remove_extra.carve_painted_mask",
                side_effect=EditError("no mask"),
            ):
                note = RemoveExtra().run(ctx, describe="удали пятно на поверхности")
            self.assertIn("mask_mesh failed", note)
            self.assertEqual(store.removal_state(chat), {})

    def test_remove_extra_apply_blocked_after_user_rejection(self) -> None:
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        from mesh_forge.agent.deps import ChatDeps
        from mesh_forge.chat.models import UiMessage
        from mesh_forge.chat.store import ChatStore
        from mesh_forge.tools.remove_extra import RemoveExtra

        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("remove").id
            mesh_path = store.files_dir(chat) / "body.stl"
            trimesh.creation.box(extents=[1.0, 1.0, 1.0]).export(mesh_path)
            store.set_current_mesh(chat, mesh_path, role="source")
            proposal = store.files_dir(chat) / "proposal.stl"
            trimesh.creation.box(extents=[0.9, 1.0, 1.0]).export(proposal)
            store.set_removal_state(
                chat,
                {
                    "strategy": "protrusion_cut",
                    "proposal_status": "ready",
                    "mesh": mesh_path.name,
                    "proposal_mesh": proposal.name,
                },
            )
            store.save_messages(
                chat,
                [
                    UiMessage(id="u1", role="user", content="убери лепесток справа"),
                    UiMessage(id="u2", role="user", content="нет"),
                ],
            )
            ctx = SimpleNamespace(deps=ChatDeps(chat_id=chat, store=store))
            note = RemoveExtra().run(ctx, apply=True, mesh_ref=mesh_path.name)
            self.assertIn("apply blocked", note)
            self.assertEqual(store.removal_state(chat), {})

    def test_remove_extra_uses_geometry_without_sam3_first(self) -> None:
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch

        import numpy as np

        from mesh_forge.agent.deps import ChatDeps
        from mesh_forge.chat.store import ChatStore
        from mesh_forge.config import AppConfig, SegmentationConfig
        from mesh_forge.tools.remove_extra import RemoveExtra

        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("remove").id
            mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0]).subdivide()
            mesh_path = store.files_dir(chat) / "body.stl"
            mesh.export(mesh_path)
            store.set_current_mesh(chat, mesh_path, role="source")
            ctx = SimpleNamespace(deps=ChatDeps(chat_id=chat, store=store))
            n = int(len(mesh.faces))
            geometry = np.zeros(n, dtype=bool)
            geometry[: max(8, n // 8)] = True
            seen: dict[str, object] = {}

            def _fake_rerank(_ctx, _src, _mesh, result, **_kwargs):
                seen["candidates"] = list(result.get("candidate_masks") or [])
                return None

            with patch(
                "mesh_forge.config.load_config",
                return_value=AppConfig(segmentation=SegmentationConfig(enabled=True)),
            ), patch(
                "mesh_forge.tools.remove_extra.build_auto_remove_proposal",
                return_value={
                    "strategy": "protrusion_cut",
                    "mask": geometry,
                    "candidate_masks": [geometry],
                    "mesh": mesh,
                    "note": "geometry",
                    "stats": {"faces_dropped": int(geometry.sum())},
                    "debug_candidates": [],
                },
            ), patch(
                "mesh_forge.tools.remove_extra._rerank_protrusion_candidates",
                side_effect=_fake_rerank,
            ), patch(
                "mesh_forge.tools.remove_extra.emit_masked_mesh_view",
            ), patch(
                "mesh_forge.tools.remove_extra.save_mesh_artifact",
                return_value=SimpleNamespace(name="proposal.stl"),
            ):
                note = RemoveExtra().run(ctx, describe="убери лепесток справа", views="right")

        self.assertIn("protrusion_cut", note)
        self.assertEqual(len(seen.get("candidates") or []), 1)
        first = np.asarray((seen.get("candidates") or [None])[0], dtype=bool)
        self.assertTrue(np.array_equal(first, geometry))

    def test_rerank_protrusion_candidates_prefers_vlm_ok_candidate(self) -> None:
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch

        import numpy as np

        from mesh_forge.agent.deps import ChatDeps
        from mesh_forge.chat.store import ChatStore
        from mesh_forge.tools.remove_extra import _rerank_protrusion_candidates

        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("rerank").id
            mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0]).subdivide()
            mesh_path = store.files_dir(chat) / "body.stl"
            mesh.export(mesh_path)
            ctx = SimpleNamespace(deps=ChatDeps(chat_id=chat, store=store))
            n = int(len(mesh.faces))
            cand1 = np.zeros(n, dtype=bool)
            cand2 = np.zeros(n, dtype=bool)
            cand1[: max(1, n // 4)] = True
            cand2[max(1, n // 2) : max(2, (3 * n) // 4)] = True

            def _fake_render(_src, preview, **_kwargs):
                preview.write_bytes(b"png")

            class _FakeClient:
                def __init__(self) -> None:
                    self.calls = 0

                def review_mesh_mask(self, _images, *, target):
                    self.calls += 1
                    if self.calls == 1:
                        return {"verdict": "wrong", "confidence": 0.9, "note": target}
                    return {"verdict": "ok", "confidence": 0.9, "note": target}

            result = {
                "strategy": "protrusion_cut",
                "mask": cand1,
                "candidate_masks": [cand1, cand2],
                "mesh": mesh,
                "note": "base",
                "stats": {},
            }
            with patch("mesh_forge.render.render_mesh_preview", side_effect=_fake_render), patch(
                "mesh_forge.tools.remove_extra.LMStudioClient",
                return_value=_FakeClient(),
            ):
                updated = _rerank_protrusion_candidates(
                    ctx,
                    mesh_path,
                    mesh,
                    result,
                    target="лепесток на юбке",
                    focus_view="right",
                    zoom=1.0,
                )
            self.assertIsNotNone(updated)
            self.assertTrue(np.array_equal(np.asarray(updated["mask"], dtype=bool), cand2))

    def test_rerank_protrusion_candidates_can_union_partial_masks(self) -> None:
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch

        import numpy as np

        from mesh_forge.agent.deps import ChatDeps
        from mesh_forge.chat.store import ChatStore
        from mesh_forge.tools.remove_extra import _rerank_protrusion_candidates

        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("rerank").id
            mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0]).subdivide()
            mesh_path = store.files_dir(chat) / "body.stl"
            mesh.export(mesh_path)
            ctx = SimpleNamespace(deps=ChatDeps(chat_id=chat, store=store))
            n = int(len(mesh.faces))
            cand1 = np.zeros(n, dtype=bool)
            cand2 = np.zeros(n, dtype=bool)
            cand1[: max(1, n // 4)] = True
            cand2[max(1, n // 4) : max(2, n // 2)] = True
            union = np.asarray(cand1 | cand2, dtype=bool)

            def _fake_review(_ctx, _src, _mesh, mask, **_kwargs):
                if np.array_equal(np.asarray(mask, dtype=bool), cand1):
                    return "partial", 2.8
                if np.array_equal(np.asarray(mask, dtype=bool), cand2):
                    return "too_little", 2.7
                if np.array_equal(np.asarray(mask, dtype=bool), union):
                    return "ok", 4.8
                return "wrong", 0.5

            result = {
                "strategy": "protrusion_cut",
                "mask": cand1,
                "candidate_masks": [cand1, cand2],
                "mesh": mesh,
                "note": "base",
                "stats": {},
            }
            with patch("mesh_forge.tools.remove_extra._review_mask_candidate_for_remove_extra", side_effect=_fake_review):
                updated = _rerank_protrusion_candidates(
                    ctx,
                    mesh_path,
                    mesh,
                    result,
                    target="лепесток на юбке",
                    focus_view="right",
                    zoom=1.0,
                )
            self.assertIsNotNone(updated)
            self.assertTrue(np.array_equal(np.asarray(updated["mask"], dtype=bool), union))

    def test_rerank_protrusion_candidates_can_expand_partial_mask(self) -> None:
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch

        import numpy as np

        from mesh_forge.agent.deps import ChatDeps
        from mesh_forge.chat.store import ChatStore
        from mesh_forge.tools.remove_extra import _rerank_protrusion_candidates

        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("rerank").id
            mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0]).subdivide()
            mesh_path = store.files_dir(chat) / "body.stl"
            mesh.export(mesh_path)
            ctx = SimpleNamespace(deps=ChatDeps(chat_id=chat, store=store))
            n = int(len(mesh.faces))
            cand1 = np.zeros(n, dtype=bool)
            expanded = np.zeros(n, dtype=bool)
            cand1[:2] = True
            expanded[:4] = True

            def _fake_review(_ctx, _src, _mesh, mask, **_kwargs):
                current = np.asarray(mask, dtype=bool)
                if np.array_equal(current, cand1):
                    return "partial", 2.8
                if np.array_equal(current, expanded):
                    return "ok", 4.9
                return "wrong", 0.5

            result = {
                "strategy": "protrusion_cut",
                "mask": cand1,
                "candidate_masks": [cand1],
                "mesh": mesh,
                "note": "base",
                "stats": {},
            }
            with patch("mesh_forge.tools.remove_extra._review_mask_candidate_for_remove_extra", side_effect=_fake_review), patch(
                "mesh_forge.tools.remove_extra.dilate_face_mask",
                return_value=expanded,
            ):
                updated = _rerank_protrusion_candidates(
                    ctx,
                    mesh_path,
                    mesh,
                    result,
                    target="лепесток на юбке",
                    focus_view="right",
                    zoom=1.0,
                )
            self.assertIsNotNone(updated)
            self.assertTrue(np.array_equal(np.asarray(updated["mask"], dtype=bool), expanded))


class ToolRegistryTests(unittest.TestCase):
    def test_exposed_edit_tools(self) -> None:
        from mesh_forge.tools import ALL_TOOLS

        names = {t.name for t in ALL_TOOLS}
        for name in (
            "mask_mesh",
            "remove_mesh",
            "fill_mesh",
            "split_mesh",
            "join_mesh",
            "match_mesh",
            "remesh_mesh",
            "smooth_mesh",
            "look",
            "restore_mesh",
            "regen_mesh",
            "extract_mesh",
            "offset_mesh",
            "add_mesh",
            "restore_patch",
        ):
            self.assertIn(name, names)
        for hidden in ("carve_mesh", "decimate_mesh", "inspect_mesh", "repair_mesh"):
            self.assertNotIn(hidden, names)


if __name__ == "__main__":
    unittest.main()
