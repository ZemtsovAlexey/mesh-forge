from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from mesh_forge.config import load_config, normalize_reasoning_effort, save_config, update_llm_settings


class ReasoningEffortTests(unittest.TestCase):
    def test_normalize_aliases(self) -> None:
        self.assertEqual(normalize_reasoning_effort("low"), "low")
        self.assertEqual(normalize_reasoning_effort("HIGH"), "high")
        self.assertEqual(normalize_reasoning_effort("extra-high"), "xhigh")
        self.assertEqual(normalize_reasoning_effort("nope"), "medium")
        self.assertEqual(normalize_reasoning_effort(None), "medium")

    def test_persists_in_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("llm:\n  planner_model: a\n  vision_model: b\n", encoding="utf-8")
            cfg = load_config(path)
            self.assertEqual(cfg.llm.reasoning_effort, "medium")
            cfg.llm.reasoning_effort = "xhigh"
            save_config(cfg)
            again = load_config(path)
            self.assertEqual(again.llm.reasoning_effort, "xhigh")

    def test_look_completion_sends_effort_without_temperature(self) -> None:
        from mesh_forge.backends.lmstudio import completion_kwargs

        kwargs = completion_kwargs(
            model="vl",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            effort="low",
        )
        self.assertEqual(kwargs["reasoning_effort"], "low")
        self.assertTrue(kwargs["stream"])
        self.assertNotIn("temperature", kwargs)

    def test_update_llm_settings_effort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "llm:\n  base_url: http://127.0.0.1:1234/v1\n  api_key: k\n"
                "  planner_model: a\n  vision_model: b\n",
                encoding="utf-8",
            )
            previous = os.environ.get("MESHFORGE_CONFIG")
            os.environ["MESHFORGE_CONFIG"] = str(path)
            try:
                updated = update_llm_settings(reasoning_effort="extra_high")
                self.assertEqual(updated.llm.reasoning_effort, "xhigh")
                self.assertEqual(load_config(path).llm.reasoning_effort, "xhigh")
            finally:
                if previous is None:
                    os.environ.pop("MESHFORGE_CONFIG", None)
                else:
                    os.environ["MESHFORGE_CONFIG"] = previous
