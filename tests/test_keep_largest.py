from __future__ import annotations

import unittest

import numpy as np
import trimesh

from mesh_forge.ops.geometry import keep_largest_component


def _box(center, extents, subdivisions=2) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    if subdivisions > 0:
        mesh = mesh.subdivide()
        if subdivisions > 1:
            mesh = mesh.subdivide()
    mesh.apply_translation(center)
    return mesh


class KeepLargestTests(unittest.TestCase):
    def test_keeps_thin_nearby_panel(self) -> None:
        frame = _box([0, 0.5, 0], [1.0, 1.0, 1.0])
        seat = _box([0, 0.55, 0], [0.7, 0.04, 0.7])
        combined = trimesh.util.concatenate([frame, seat])
        out = keep_largest_component(combined, single=True)
        self.assertGreater(float(out.area), 0.9 * float(combined.area))
        self.assertGreater(len(out.faces), len(frame.faces))

    def test_drops_far_floater(self) -> None:
        body = _box([0, 0.5, 0], [1.0, 1.0, 1.0])
        floater = _box([8.0, 0.5, 0], [0.15, 0.15, 0.15], subdivisions=1)
        combined = trimesh.util.concatenate([body, floater])
        out = keep_largest_component(combined, single=True)
        self.assertLess(len(out.faces), len(combined.faces))
        self.assertGreater(float(out.area), 0.9 * float(body.area))

    def test_keeps_cracked_surface_patches(self) -> None:
        body = _box([0, 0.5, 0], [1.0, 1.0, 1.0])
        patches = [_box([0.1 * i, 0.55, 0], [0.08, 0.02, 0.08], subdivisions=0) for i in range(-3, 4)]
        combined = trimesh.util.concatenate([body, *patches])
        out = keep_largest_component(combined, single=True)
        self.assertGreater(len(out.faces), len(body.faces) + 10)


if __name__ == "__main__":
    unittest.main()
