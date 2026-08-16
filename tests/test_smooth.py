from __future__ import annotations

import unittest

import numpy as np
import trimesh

from mesh_forge.ops.geometry import smooth_mesh


class SmoothTests(unittest.TestCase):
    def test_sphere_stays_stable(self) -> None:
        src = trimesh.creation.icosphere(subdivisions=3)
        out = smooth_mesh(src, iterations=4)
        self.assertEqual(len(out.faces), len(src.faces))
        self.assertLess(float(np.max(out.extents)), 1.15 * float(np.max(src.extents)))
        self.assertGreater(float(out.area), 0.85 * float(src.area))

    def test_existing_spike_does_not_grow(self) -> None:
        src = trimesh.creation.icosphere(subdivisions=3)
        verts = np.asarray(src.vertices, dtype=np.float64).copy()
        radial = np.linalg.norm(verts[0])
        verts[0] = verts[0] / max(radial, 1e-9) * radial * 1.8
        src.vertices = verts
        before = float(np.linalg.norm(src.vertices[0]))
        out = smooth_mesh(src, iterations=4)
        after = float(np.linalg.norm(out.vertices[0]))
        self.assertLessEqual(after, before * 1.05)
        self.assertLessEqual(float(np.max(out.extents)), float(np.max(src.extents)) * 1.05)

    def test_does_not_grow_holes(self) -> None:
        src = trimesh.creation.icosphere(subdivisions=3)
        src.update_faces(np.arange(len(src.faces) - 8))
        src.remove_unreferenced_vertices()
        before = len(trimesh.grouping.group_rows(src.edges_sorted, require_count=1))
        out = smooth_mesh(src, iterations=4)
        after = len(trimesh.grouping.group_rows(out.edges_sorted, require_count=1))
        self.assertLessEqual(after, before)

    def test_sphere_volume_does_not_collapse(self) -> None:
        src = trimesh.creation.icosphere(subdivisions=3)
        out = smooth_mesh(src, iterations=4)
        self.assertGreater(abs(float(out.volume)), 0.85 * abs(float(src.volume)))


if __name__ == "__main__":
    unittest.main()
