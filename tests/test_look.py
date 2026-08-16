from __future__ import annotations

import unittest

from mesh_forge.tools.look import default_mesh_look, parse_look_shots


class LookShotsTests(unittest.TestCase):
    def test_default_is_overview(self) -> None:
        shots = parse_look_shots("")
        self.assertEqual(shots, [("viewer", 1.0, "")])

    def test_orbit_is_four_sides(self) -> None:
        shots = parse_look_shots("orbit")
        self.assertEqual([c for c, _, _ in shots], ["front", "left", "back", "right"])

    def test_comma_list_and_aliases(self) -> None:
        shots = parse_look_shots("спереди, сверху")
        self.assertEqual([c for c, _, _ in shots], ["front", "top"])

    def test_region_raises_zoom(self) -> None:
        shots = parse_look_shots("overview", region="спинка")
        self.assertEqual(len(shots), 1)
        camera, zoom, region = shots[0]
        self.assertEqual(camera, "viewer")
        self.assertEqual(region, "backrest")
        self.assertGreater(zoom, 1.5)

    def test_detail_without_region_uses_few_closeups(self) -> None:
        shots = parse_look_shots("detail", zoom=2.0)
        self.assertGreaterEqual(len(shots), 2)
        self.assertLessEqual(len(shots), 4)
        self.assertTrue(all(z == 2.0 for _, z, _ in shots))

    def test_caps_at_four(self) -> None:
        shots = parse_look_shots("orbit,top,viewer")
        self.assertEqual(len(shots), 4)

    def test_default_mesh_look_compares_sides(self) -> None:
        views, question = default_mesh_look()
        self.assertIn("left", views)
        self.assertIn("right", views)
        self.assertIn("restore_mesh", question)

    def test_default_mesh_look_keeps_explicit_views(self) -> None:
        views, question = default_mesh_look("front", question="спинка?")
        self.assertEqual(views, "front")
        self.assertEqual(question, "спинка?")


if __name__ == "__main__":
    unittest.main()
