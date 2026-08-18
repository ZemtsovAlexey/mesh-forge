from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mesh_forge.config import load_config, save_config, segmentation_segmenter_base_url


class SegmentationConfigTests(unittest.TestCase):
    def test_persists_segmentation_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "comfyui:\n  base_url: http://127.0.0.1:8188\n"
                "segmentation:\n  enabled: true\n  detector_model: IDEA-Research/grounding-dino-tiny\n",
                encoding="utf-8",
            )
            cfg = load_config(path)
            self.assertTrue(cfg.segmentation.enabled)
            self.assertEqual(cfg.segmentation.provider, "comfyui")
            self.assertEqual(cfg.segmentation.detector, "groundingdino")
            self.assertEqual(cfg.segmentation.segmenter, "sam3")
            cfg.segmentation.max_views = 5
            save_config(cfg)
            again = load_config(path)
            self.assertEqual(again.segmentation.max_views, 5)
            self.assertEqual(again.segmentation.detector_model, "IDEA-Research/grounding-dino-tiny")

    def test_segmenter_base_url_falls_back_to_comfy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "comfyui:\n  base_url: http://192.168.0.22:8188\n"
                "segmentation:\n  enabled: true\n",
                encoding="utf-8",
            )
            cfg = load_config(path)
            self.assertEqual(segmentation_segmenter_base_url(cfg), "http://192.168.0.22:8188")
            cfg.segmentation.segmenter_base_url = "http://127.0.0.1:9191"
            self.assertEqual(segmentation_segmenter_base_url(cfg), "http://127.0.0.1:9191")


if __name__ == "__main__":
    unittest.main()
