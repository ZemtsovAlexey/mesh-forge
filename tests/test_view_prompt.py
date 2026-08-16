from __future__ import annotations

import unittest

from mesh_forge.adapters.comfyui_client import isolation_view_prompt


class IsolationPromptTests(unittest.TestCase):
    def test_adds_single_object_and_full_frame(self) -> None:
        out = isolation_view_prompt("elegant wooden chair", "color")
        self.assertTrue(out.startswith("elegant wooden chair,"))
        self.assertIn("one item only", out)
        self.assertIn("fully in frame", out)
        self.assertIn("plain contrasting studio backdrop", out)
        self.assertNotIn("matte clay", out)

    def test_front_asks_for_orthographic_eye_level(self) -> None:
        out = isolation_view_prompt("wooden chair", "color", view="front")
        self.assertIn("orthographic front elevation", out)
        self.assertIn("eye-level camera looking straight on", out)
        self.assertIn("no three-quarter", out)
        self.assertIn("flat ground plane", out)
        self.assertNotIn("left profile", out)

    def test_side_view_uses_profile_camera(self) -> None:
        left = isolation_view_prompt("wooden chair", "color", view="left")
        self.assertIn("orthographic left profile", left)
        self.assertIn("true side view", left)
        back = isolation_view_prompt("wooden chair", "color", view="back")
        self.assertIn("orthographic rear elevation", back)

    def test_clay_adds_matte_not_photoreal(self) -> None:
        out = isolation_view_prompt("wooden chair", "clay")
        self.assertIn("matte clay", out)
        self.assertIn("no photoreal materials", out)
        self.assertIn("orthographic front elevation", out)


if __name__ == "__main__":
    unittest.main()
