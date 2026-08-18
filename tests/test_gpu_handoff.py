from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from mesh_forge.config import AppConfig, ComfyUIConfig, GPUConfig, LLMConfig
from mesh_forge.runtime.gpu_handoff import switch_vram
from mesh_forge.runtime.gpu_scheduler import GpuScheduler


def _stats(torch_used: int, vram_free: int = 0) -> dict:
    return {
        "devices": [
            {
                "type": "cuda",
                "torch_vram_total": torch_used + 100,
                "torch_vram_free": 100,
                "vram_free": vram_free,
            }
        ]
    }


class GpuHandoffTests(unittest.TestCase):
    def test_switch_to_llm_posts_free_and_waits_for_vram_drop(self) -> None:
        cfg = AppConfig(
            llm=LLMConfig(base_url="http://127.0.0.1:1234/v1"),
            comfyui=ComfyUIConfig(base_url="http://127.0.0.1:8188"),
            gpu=GPUConfig(sequential_models=True, shared_gpu=True),
        )
        client = MagicMock()
        high = MagicMock(status_code=200)
        high.json.return_value = _stats(6 * 1024**3)
        low = MagicMock(status_code=200)
        low.json.return_value = _stats(200 * 1024**2, vram_free=6 * 1024**3)
        free_ok = MagicMock(status_code=200)
        interrupt_ok = MagicMock(status_code=200)
        client.get.side_effect = [high, low]
        client.post.return_value = free_ok

        def post(url, **kwargs):
            if url.endswith("/interrupt"):
                return interrupt_ok
            return free_ok

        client.post.side_effect = post
        cm = MagicMock()
        cm.__enter__.return_value = client
        cm.__exit__.return_value = False

        with (
            patch("mesh_forge.runtime.gpu_handoff.load_config", return_value=cfg),
            patch("mesh_forge.runtime.gpu_handoff.httpx.Client", return_value=cm),
            patch("mesh_forge.runtime.gpu_handoff.time.sleep"),
            patch("mesh_forge.runtime.gpu_handoff._restart_local_comfyui") as restart,
        ):
            switch_vram("comfy", "llm")

        posted = [call.args[0] for call in client.post.call_args_list]
        self.assertTrue(any(url.endswith("/free") for url in posted))
        restart.assert_not_called()

    def test_first_llm_lease_triggers_handoff(self) -> None:
        calls: list[tuple] = []

        def handoff(from_kind, to_kind):
            calls.append((from_kind, to_kind))

        cfg = AppConfig(
            llm=LLMConfig(base_url="http://127.0.0.1:1234/v1"),
            comfyui=ComfyUIConfig(base_url="http://127.0.0.1:8188"),
            gpu=GPUConfig(sequential_models=True, shared_gpu=True),
        )
        scheduler = GpuScheduler(handoff=handoff)
        with (
            patch("mesh_forge.runtime.gpu_scheduler.queues_are_split", return_value=False),
            patch("mesh_forge.config.load_config", return_value=cfg),
        ):
            with scheduler.acquire("LM Studio", kind="llm"):
                pass
        self.assertEqual(calls, [(None, "llm")])

    def test_second_llm_lease_skips_handoff(self) -> None:
        calls: list[tuple] = []

        def handoff(from_kind, to_kind):
            calls.append((from_kind, to_kind))

        cfg = AppConfig(
            llm=LLMConfig(base_url="http://127.0.0.1:1234/v1"),
            comfyui=ComfyUIConfig(base_url="http://127.0.0.1:8188"),
            gpu=GPUConfig(sequential_models=True, shared_gpu=True),
        )
        scheduler = GpuScheduler(handoff=handoff)
        with (
            patch("mesh_forge.runtime.gpu_scheduler.queues_are_split", return_value=False),
            patch("mesh_forge.config.load_config", return_value=cfg),
        ):
            with scheduler.acquire("LM Studio", kind="llm"):
                pass
            with scheduler.acquire("LM Studio", kind="llm"):
                pass
        self.assertEqual(calls, [(None, "llm")])


if __name__ == "__main__":
    unittest.main()
