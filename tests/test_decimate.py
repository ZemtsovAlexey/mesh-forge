from __future__ import annotations

import unittest

import numpy as np
import trimesh

from mesh_forge.ops.geometry import decimate


def _unindex(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """STL-style dump: 3 unique vertices per triangle, no shared indices."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)[faces.reshape(-1)]
    new_faces = np.arange(len(verts), dtype=np.int64).reshape((-1, 3))
    return trimesh.Trimesh(vertices=verts, faces=new_faces, process=False)


class DecimateTests(unittest.TestCase):
    def test_unwelded_sphere_keeps_surface(self) -> None:
        src = trimesh.creation.icosphere(subdivisions=4)
        dumped = _unindex(src)
        self.assertEqual(len(dumped.vertices), 3 * len(dumped.faces))
        self.assertEqual(len(getattr(dumped, "face_adjacency", [])), 0)

        target = max(200, len(dumped.faces) // 4)
        out = decimate(dumped, target)

        self.assertLessEqual(len(out.faces), target + 8)
        self.assertGreater(len(out.faces), target * 0.5)
        self.assertGreater(float(out.area), 0.7 * float(src.area))
        self.assertGreater(len(getattr(out, "face_adjacency", [])), 0)
        self.assertLess(len(out.vertices), 2.5 * len(out.faces))

    def test_half_reduction_keeps_area(self) -> None:
        src = trimesh.creation.icosphere(subdivisions=4)
        dumped = _unindex(src)
        out = decimate(dumped, max(1000, len(dumped.faces) // 2))
        self.assertGreater(float(out.area), 0.85 * float(src.area))
        self.assertGreater(len(getattr(out, "face_adjacency", [])), 0)

    def test_already_welded_mesh(self) -> None:
        src = trimesh.creation.icosphere(subdivisions=3)
        target = max(80, len(src.faces) // 3)
        out = decimate(src, target)
        self.assertLessEqual(len(out.faces), target + 8)
        self.assertGreater(float(out.area), 0.7 * float(src.area))

    def test_exploded_bounds_are_invalid(self) -> None:
        from mesh_forge.ops.geometry import _decimate_looks_valid

        src = trimesh.creation.icosphere(subdivisions=2)
        exploded = src.copy()
        verts = np.asarray(exploded.vertices, dtype=np.float64)
        verts[0] = verts[0] + np.array([20.0, 0.0, 0.0])
        exploded.vertices = verts
        self.assertFalse(_decimate_looks_valid(src, exploded))


if __name__ == "__main__":
    unittest.main()
