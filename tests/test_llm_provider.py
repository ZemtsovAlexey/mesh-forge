from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mesh_forge.config import (
    AppConfig,
    ComfyUIConfig,
    GPUConfig,
    LLMConfig,
    llm_display_name,
    llm_http_timeout,
    llm_uses_gpu,
    load_config,
    normalize_llm_provider,
    update_llm_settings,
)
from mesh_forge.runtime.gpu_handoff import queues_are_split, switch_vram


class LlmProviderTests(unittest.TestCase):
    def test_normalize_aliases(self) -> None:
        self.assertEqual(normalize_llm_provider("lmstudio"), "lmstudio")
        self.assertEqual(normalize_llm_provider("LM Studio"), "lmstudio")
        self.assertEqual(normalize_llm_provider("openai"), "openai")
        self.assertEqual(normalize_llm_provider("aitunnel"), "openai")
        self.assertEqual(normalize_llm_provider("openai-compatible"), "openai")

    def test_infers_openai_from_url(self) -> None:
        self.assertEqual(
            normalize_llm_provider(None, "https://api.aitunnel.ru/v1"),
            "openai",
        )
        self.assertEqual(
            normalize_llm_provider("", "http://127.0.0.1:1234/v1"),
            "lmstudio",
        )

    def test_remote_openai_skips_gpu(self) -> None:
        cfg = AppConfig(
            llm=LLMConfig(
                provider="openai",
                base_url="https://api.aitunnel.ru/v1",
                api_key="sk-test",
            ),
            comfyui=ComfyUIConfig(base_url="http://127.0.0.1:8188"),
        )
        self.assertFalse(llm_uses_gpu(cfg))
        self.assertEqual(llm_display_name(cfg), "AI Tunnel")
        self.assertEqual(llm_http_timeout(cfg), 600.0)
        self.assertTrue(queues_are_split(cfg))

    def test_lmstudio_still_uses_gpu(self) -> None:
        cfg = AppConfig(
            llm=LLMConfig(base_url="http://127.0.0.1:1234/v1"),
            comfyui=ComfyUIConfig(base_url="http://127.0.0.1:8188"),
        )
        self.assertTrue(llm_uses_gpu(cfg))
        self.assertEqual(llm_display_name(cfg), "LM Studio")
        self.assertEqual(llm_http_timeout(cfg), 90.0)
        self.assertFalse(queues_are_split(cfg))

    def test_update_llm_settings_persists_provider(self) -> None:
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
                updated = update_llm_settings(
                    provider="openai",
                    base_url="https://api.aitunnel.ru/v1",
                    api_key="sk-test",
                    planner_model="gpt-5.6-luna",
                    vision_model="gpt-5.6-luna",
                )
                self.assertEqual(updated.llm.provider, "openai")
                self.assertEqual(updated.llm.planner_model, "gpt-5.6-luna")
                again = load_config(path)
                self.assertEqual(again.llm.provider, "openai")
                self.assertEqual(again.llm.base_url, "https://api.aitunnel.ru/v1")
            finally:
                if previous is None:
                    os.environ.pop("MESHFORGE_CONFIG", None)
                else:
                    os.environ["MESHFORGE_CONFIG"] = previous

    def test_switch_vram_skips_remote_openai(self) -> None:
        cfg = AppConfig(
            llm=LLMConfig(
                provider="openai",
                base_url="https://api.aitunnel.ru/v1",
                api_key="sk-test",
            ),
            comfyui=ComfyUIConfig(base_url="http://127.0.0.1:8188"),
            gpu=GPUConfig(sequential_models=True, shared_gpu=True),
        )
        with (
            patch("mesh_forge.runtime.gpu_handoff.load_config", return_value=cfg),
            patch("mesh_forge.runtime.gpu_handoff.httpx.Client") as client_cls,
        ):
            switch_vram("llm", "comfy")
        client_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
