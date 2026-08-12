from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mesh_forge.adapters import ComfyUiClient, LMStudioClient
from mesh_forge.application import PipelineRunner
from mesh_forge.config import load_config
from mesh_forge.domain import GenerationJob, JobOptions
from mesh_forge.manifest import ProjectManifest
from mesh_forge.runtime import get_gpu_scheduler

logger = logging.getLogger("mesh_forge.orchestrator")


class Orchestrator:
    def __init__(self) -> None:
        self.reload_config()

    def reload_config(self) -> None:
        self.config = load_config()
        self.llm = LMStudioClient(self.config)
        self.runner = PipelineRunner()
        self.comfyui = ComfyUiClient()

    def run_job(self, manifest: ProjectManifest, job: GenerationJob) -> tuple[ProjectManifest, str]:
        result = self.runner.run(manifest, job)
        logger.info("run_job project=%s plan=%s", manifest.id, result.plan.summary)
        return result.manifest, result.message

    def create_photo(self, manifest: ProjectManifest, image_path: Path, **kwargs: Any) -> tuple[ProjectManifest, str]:
        job = GenerationJob(
            project_id=manifest.id,
            image_paths=[image_path],
            options=JobOptions(
                remove_bg=bool(kwargs.get("remove_bg", True)),
                solidify_mm=float(kwargs.get("solidify_mm", 0.0)),
            ),
        )
        return self.run_job(manifest, job)

    def create_scan(self, manifest: ProjectManifest, scan_path: Path, **kwargs: Any) -> tuple[ProjectManifest, str]:
        job = GenerationJob(
            project_id=manifest.id,
            source_mesh=scan_path,
            options=JobOptions(
                scan_mode=str(kwargs.get("mode", "light")),
                smooth_iters=int(kwargs.get("smooth_iters", 1)),
                solidify_mm=float(kwargs.get("solidify_mm", 0.0)),
            ),
        )
        return self.run_job(manifest, job)

    def create_text(self, manifest: ProjectManifest, prompt: str, **kwargs: Any) -> tuple[ProjectManifest, str]:
        job = GenerationJob(
            project_id=manifest.id,
            prompt=prompt,
            options=JobOptions(
                solidify_mm=float(kwargs.get("solidify_mm", 0.0)),
                view_count=int(kwargs.get("view_count", self.config.comfyui.view_count)),
            ),
        )
        return self.run_job(manifest, job)

    def edit_text(
        self,
        manifest: ProjectManifest,
        instruction: str,
        *,
        apply_solidify: float = 0.0,
    ) -> tuple[ProjectManifest, str]:
        current = manifest.current_mesh_path()
        if not current:
            raise ValueError("No mesh to edit")
        job = GenerationJob(
            project_id=manifest.id,
            prompt=instruction,
            source_mesh=current,
            options=JobOptions(solidify_mm=float(apply_solidify)),
        )
        return self.run_job(manifest, job)

    def edit_photo(
        self,
        manifest: ProjectManifest,
        instruction: str,
        ref_image: Path | None,
    ) -> tuple[ProjectManifest, str]:
        current = manifest.current_mesh_path()
        if not current:
            raise ValueError("No mesh to edit")
        job = GenerationJob(
            project_id=manifest.id,
            prompt=instruction,
            image_paths=[ref_image] if ref_image else [],
            source_mesh=current,
            options=JobOptions(),
        )
        return self.run_job(manifest, job)

    def system_status(self) -> dict[str, Any]:
        return {
            "lmstudio": self.llm.health_check(),
            "comfyui": self.comfyui.health_check(),
        }

    def status_text(self) -> str:
        s = self.system_status()
        lines = [f"{k}: {'OK' if v else 'missing'}" for k, v in s.items()]
        lines.append(get_gpu_scheduler().status_text())
        return "\n".join(lines)
