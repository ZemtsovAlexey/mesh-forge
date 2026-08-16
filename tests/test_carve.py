from __future__ import annotations

import unittest

import trimesh

from mesh_forge.ops.geometry import (
    CarveError,
    carve_region,
    resolve_carve_box,
)


def _box(center, extents, subdivisions=1) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    for _ in range(subdivisions):
        mesh = mesh.subdivide()
    mesh.apply_translation(center)
    return mesh


def _chair_with_wing() -> trimesh.Trimesh:
    body = _box([0.0, 0.5, 0.0], [2.0, 1.0, 1.0], subdivisions=2)
    wing = _box([1.35, 0.85, -0.15], [0.5, 0.5, 0.35], subdivisions=1)
    return trimesh.util.concatenate([body, wing])


def _chair_with_armrests_and_wing() -> trimesh.Trimesh:
    body = _box([0.0, 0.45, 0.0], [1.4, 0.9, 0.9], subdivisions=2)
    left_arm = _box([-0.75, 0.55, 0.15], [0.28, 0.55, 0.7], subdivisions=1)
    right_arm = _box([0.75, 0.55, 0.15], [0.28, 0.55, 0.7], subdivisions=1)
    wing = _box([1.25, 0.95, -0.2], [0.45, 0.45, 0.3], subdivisions=1)
    return trimesh.util.concatenate([body, left_arm, right_arm, wing])


class CarveTests(unittest.TestCase):
    def test_side_slab_maps_right(self) -> None:
        box = resolve_carve_box(side="справа", amount=0.2)
        self.assertAlmostEqual(box[0], 0.8)
        self.assertAlmostEqual(box[1], 1.0)

    def test_side_plus_height_narrows_box(self) -> None:
        box = resolve_carve_box(side="right", amount=0.2, bottom=0.5, front=0.5)
        self.assertAlmostEqual(box[0], 0.8)
        self.assertAlmostEqual(box[2], 0.5)
        self.assertAlmostEqual(box[5], 0.5)

    def test_requires_region(self) -> None:
        with self.assertRaises(CarveError):
            resolve_carve_box()

    def test_full_side_slab_rejected(self) -> None:
        src = _chair_with_armrests_and_wing()
        box = resolve_carve_box(side="right", amount=0.22)
        with self.assertRaises(CarveError) as ctx:
            carve_region(src, box, action="remove", min_keep_faces=50)
        self.assertIn("armrest", str(ctx.exception).lower())

    def test_removes_right_wing_keeps_body(self) -> None:
        src = _chair_with_wing()
        box = resolve_carve_box(side="right", amount=0.22, bottom=0.5, front=0.7)
        out, stats = carve_region(src, box, action="remove", min_keep_faces=50)
        self.assertLess(stats["faces_after"], stats["faces_before"])
        self.assertGreater(float(out.extents[0]), 1.6)
        self.assertLess(float(out.bounds[1][0]), 1.12)
        self.assertGreater(float(out.area), 0.7 * float(_box([0.0, 0.5, 0.0], [2.0, 1.0, 1.0]).area))

    def test_local_box_spares_armrest(self) -> None:
        src = _chair_with_armrests_and_wing()
        before_right = float(src.bounds[1][0])
        box = resolve_carve_box(side="right", amount=0.18, bottom=0.55, front=0.55)
        out, _ = carve_region(src, box, action="remove", min_keep_faces=50)
        self.assertLess(float(out.bounds[1][0]), before_right - 0.05)
        self.assertGreater(float(out.bounds[1][0]), 0.7)

    def test_wide_right_cut_does_not_chop_armrest(self) -> None:
        src = _chair_with_armrests_and_wing()
        box = resolve_carve_box(left=0.55, bottom=0.15)
        with self.assertRaises(CarveError) as ctx:
            carve_region(src, box, action="remove", min_keep_faces=50)
        self.assertIn("structural", str(ctx.exception).lower())

    def test_keep_crops_along_x(self) -> None:
        src = trimesh.creation.box(extents=[2.0, 2.0, 2.0]).subdivide().subdivide()
        box = resolve_carve_box(left=0.25, right=0.75)
        out, stats = carve_region(src, box, action="keep", min_keep_faces=20, min_keep_ratio=0.05)
        self.assertLess(stats["faces_after"], stats["faces_before"])
        self.assertLess(float(out.extents[0]), 1.3)
        self.assertGreater(float(out.extents[1]), 1.6)

    def test_refuses_empty_cut(self) -> None:
        src = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        box = resolve_carve_box(left=0.9, right=1.0, bottom=0.9, top=1.0, back=0.9, front=1.0)
        with self.assertRaises(CarveError):
            carve_region(src, box, action="remove", min_keep_faces=1, min_keep_ratio=0.0)

    def test_refuses_deleting_almost_everything(self) -> None:
        src = trimesh.creation.box(extents=[4.0, 1.0, 1.0]).subdivide().subdivide()
        box = resolve_carve_box(right=0.9)
        with self.assertRaises(CarveError):
            carve_region(src, box, action="remove", min_keep_ratio=0.5, min_keep_faces=10)

    def test_tool_is_registered(self) -> None:
        from mesh_forge.tools import ALL_TOOLS

        names = {t.name for t in ALL_TOOLS}
        self.assertIn("carve_mesh", names)


if __name__ == "__main__":
    unittest.main()
