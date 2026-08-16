from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import trimesh

from mesh_forge.mesh_qc import MeshStats, analyze_mesh
from mesh_forge.tools.inspect_mesh import _component_line, _inspect_advice


class InspectAdviceTests(unittest.TestCase):
    def test_open_mesh_is_not_a_repair_order(self) -> None:
        stats = MeshStats(triangle_count=100, vertex_count=50, watertight=False)
        advice = _inspect_advice(stats, bodies=200)
        self.assertIn("restore_mesh", advice)
        self.assertIn("не повод для repair", advice)
        self.assertNotIn("generate_image", advice)
        self.assertNotIn("images_to_mesh", advice)

    def test_exported_box_has_no_problem_list(self) -> None:
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "box.stl"
            mesh.export(path)
            stats = analyze_mesh(path)
        self.assertNotIn("Проблемы:", stats.summary())

    def test_many_patches_are_not_separate_bodies(self) -> None:
        line = _component_line(27473)
        self.assertIn("Лоскутов: 27473", line)
        self.assertIn("не 27473 отдельных объектов", line)


if __name__ == "__main__":
    unittest.main()
