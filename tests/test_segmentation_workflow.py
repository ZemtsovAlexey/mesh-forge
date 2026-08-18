from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from mesh_forge.adapters.comfyui_client import ComfyUiClient
from mesh_forge.config import AppConfig, ComfyUIConfig, SegmentationConfig


class SegmentationWorkflowTests(unittest.TestCase):
    def test_load_native_segmentation_workflow_replaces_placeholders(self) -> None:
        cfg = AppConfig(
            comfyui=ComfyUIConfig(base_url="http://127.0.0.1:8188"),
            segmentation=SegmentationConfig(
                workflow_segment="",
                mask_threshold=0.33,
                segmenter_model="sam3.1_multiplex_fp16.safetensors",
            ),
        )
        with patch("mesh_forge.adapters.comfyui_client.load_config", return_value=cfg):
            client = ComfyUiClient()

        workflow_path = Path(__file__).resolve().parents[1] / "mesh_forge" / "workflows" / "seg_text_view.json"
        workflow = client._load_segmentation_text_workflow(
            workflow_path,
            uploaded_image="meshforge/test/input.png",
            prompt="red skirt flap",
            confidence_threshold=None,
            max_detections=2,
        )

        self.assertEqual(workflow["1"]["inputs"]["image"], "meshforge/test/input.png")
        self.assertEqual(workflow["12"]["class_type"], "CheckpointLoaderSimple")
        self.assertEqual(workflow["12"]["inputs"]["ckpt_name"], "sam3.1_multiplex_fp16.safetensors")
        self.assertEqual(workflow["13"]["inputs"]["text"], "red skirt flap")
        self.assertEqual(workflow["15"]["class_type"], "SAM3_Detect")
        self.assertEqual(workflow["15"]["inputs"]["threshold"], 0.33)

    def test_load_custom_sam3_grounding_workflow_still_works(self) -> None:
        cfg = AppConfig(
            comfyui=ComfyUIConfig(base_url="http://127.0.0.1:8188"),
            segmentation=SegmentationConfig(
                detector_dtype="float16",
                mask_threshold=0.33,
            ),
        )
        with patch("mesh_forge.adapters.comfyui_client.load_config", return_value=cfg):
            client = ComfyUiClient()

        workflow_path = (
            Path(__file__).resolve().parents[1] / "mesh_forge" / "workflows" / "seg_text_view_custom.json"
        )
        workflow = client._load_segmentation_text_workflow(
            workflow_path,
            uploaded_image="meshforge/test/input.png",
            prompt="red skirt flap",
            confidence_threshold=None,
            max_detections=2,
        )
        self.assertEqual(workflow["12"]["inputs"]["precision"], "fp16")
        self.assertEqual(workflow["15"]["inputs"]["text_prompt"], "red skirt flap")
        self.assertEqual(workflow["15"]["inputs"]["max_detections"], 2)

    def test_bind_native_checkpoint_picks_sam3_from_object_info(self) -> None:
        cfg = AppConfig(
            comfyui=ComfyUIConfig(base_url="http://127.0.0.1:8188"),
            segmentation=SegmentationConfig(segmenter_model="sam3.1_multiplex_fp16.safetensors"),
        )
        with patch("mesh_forge.adapters.comfyui_client.load_config", return_value=cfg):
            client = ComfyUiClient()
        workflow = {
            "12": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "missing.safetensors"}},
            "15": {"class_type": "SAM3_Detect", "inputs": {}},
        }

        class _Resp:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "SAM3_Detect": {},
                    "CheckpointLoaderSimple": {
                        "input": {
                            "required": {
                                "ckpt_name": [["hunyuan.safetensors", "sam3.1_multiplex_fp16.safetensors"]]
                            }
                        }
                    },
                }

        class _Http:
            def get(self, _url: str):
                return _Resp()

        bound = client._bind_native_sam3_checkpoint(_Http(), workflow)
        self.assertEqual(bound["12"]["inputs"]["ckpt_name"], "sam3.1_multiplex_fp16.safetensors")

    def test_bind_native_checkpoint_errors_when_missing(self) -> None:
        cfg = AppConfig(comfyui=ComfyUIConfig(base_url="http://192.168.0.22:8188"))
        with patch("mesh_forge.adapters.comfyui_client.load_config", return_value=cfg):
            client = ComfyUiClient()
        workflow = {
            "12": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"}},
            "15": {"class_type": "SAM3_Detect", "inputs": {}},
        }

        class _Resp:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "SAM3_Detect": {},
                    "CheckpointLoaderSimple": {
                        "input": {"required": {"ckpt_name": [["hunyuan3d-dit-v2-mv_fp16.safetensors"]]}}
                    },
                }

        class _Http:
            def get(self, _url: str):
                return _Resp()

        with self.assertRaisesRegex(RuntimeError, "no SAM3 checkpoint"):
            client._bind_native_sam3_checkpoint(_Http(), workflow)

    def test_parse_sam3_boxes_from_history_json(self) -> None:
        cfg = AppConfig(comfyui=ComfyUIConfig(base_url="http://127.0.0.1:8188"))
        with patch("mesh_forge.adapters.comfyui_client.load_config", return_value=cfg):
            client = ComfyUiClient()
        boxes, scores = client._parse_sam3_detections(
            {
                "15": {
                    "boxes": ["[[10, 20, 90, 80]]"],
                    "scores": ["[0.88]"],
                }
            },
            grounding_node="15",
            width=100,
            height=100,
        )
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0]["x0"], 0.1, places=3)
        self.assertAlmostEqual(boxes[0]["y1"], 0.8, places=3)
        self.assertAlmostEqual(scores[0], 0.88, places=3)

    def test_parse_native_sam3_xywh_boxes(self) -> None:
        cfg = AppConfig(comfyui=ComfyUIConfig(base_url="http://127.0.0.1:8188"))
        with patch("mesh_forge.adapters.comfyui_client.load_config", return_value=cfg):
            client = ComfyUiClient()
        boxes, scores = client._parse_sam3_detections(
            {
                "15": {
                    "bboxes": [[{"x": 10, "y": 20, "width": 80, "height": 60, "score": 0.77}]],
                }
            },
            grounding_node="15",
            width=100,
            height=100,
        )
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0]["x0"], 0.1, places=3)
        self.assertAlmostEqual(boxes[0]["x1"], 0.9, places=3)
        self.assertAlmostEqual(scores[0], 0.77, places=3)


if __name__ == "__main__":
    unittest.main()
