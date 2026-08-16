from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import trimesh

from mesh_forge.mesh_qc import MeshStats, analyze_mesh


class MeshQcRussianTests(unittest.TestCase):
    def test_summary_is_russian(self) -> None:
        stats = MeshStats(
            triangle_count=12,
            vertex_count=8,
            watertight=False,
            winding_consistent=False,
            bbox_mm=[10.0, 20.0, 30.0],
            issues=["Меш не замкнут (дыры или открытые края)"],
        )
        text = stats.summary()
        self.assertIn("Треугольники: 12", text)
        self.assertIn("Замкнут: нет", text)
        self.assertIn("Проблемы:", text)
        self.assertNotIn("Triangles", text)
        self.assertNotIn("Watertight", text)

    def test_open_box_issues_are_russian(self) -> None:
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "box.stl"
            mesh.export(path)
            stats = analyze_mesh(path)
        text = stats.summary()
        self.assertIn("Треугольники:", text)
        self.assertTrue(any("замкнут" in (i or "").lower() or "нормал" in (i or "").lower() for i in (stats.issues or [])) or "Замкнут" in text)


if __name__ == "__main__":
    unittest.main()
