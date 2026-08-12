from __future__ import annotations

from pathlib import Path

from mesh_forge.domain import GenerationJob, ImageSet, MeshArtifact

from .comfyui_client import ComfyUiClient


class ComfyImageToMeshBackend:
    def __init__(self, client: ComfyUiClient | None = None) -> None:
        self.client = client or ComfyUiClient()

    def available(self) -> bool:
        return self.client.health_check()

    def reconstruct(self, images: ImageSet, work_dir: Path, job: GenerationJob) -> MeshArtifact:
        work_dir.mkdir(parents=True, exist_ok=True)
        if not images:
            raise ValueError("ComfyImageToMeshBackend requires at least one image")
        return self.client.run_images_to_mesh(
            images,
            work_dir,
            project_id=job.project_id,
        )
