from __future__ import annotations

from pathlib import Path

from mesh_forge.domain import GenerationJob, TextToMeshResult

from .comfyui_client import ComfyUiClient


class ComfyTextToMeshBackend:
    def __init__(self, client: ComfyUiClient | None = None) -> None:
        self.client = client or ComfyUiClient()

    def available(self) -> bool:
        return self.client.health_check()

    def generate(self, prompt: str, work_dir: Path, job: GenerationJob) -> TextToMeshResult:
        work_dir.mkdir(parents=True, exist_ok=True)
        if not prompt.strip():
            raise ValueError("ComfyTextToMeshBackend requires a text prompt")
        return self.client.run_text_to_mesh(
            prompt.strip(),
            work_dir,
            project_id=job.project_id,
            count=job.options.view_count,
        )
