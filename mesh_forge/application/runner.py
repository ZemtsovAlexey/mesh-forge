from __future__ import annotations

import base64
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mesh_forge import progress as prog
from mesh_forge.adapters import ComfyImageToMeshBackend, ComfyTextToMeshBackend, ComfyUiClient, LMStudioClient
from mesh_forge.application.planner import JobPlan, JobPlanner
from mesh_forge.application.project_service import ProjectService
from mesh_forge.config import load_config
from mesh_forge.domain import GenerationJob, ImageArtifact, ImageSet, MeshArtifact, PipelineStepType
from mesh_forge.manifest import ProjectManifest
from mesh_forge.mesh_qc import analyze_mesh
from mesh_forge.processing import MeshProcessingService
from mesh_forge.render import render_mesh_preview

logger = logging.getLogger("mesh_forge.runner")


@dataclass
class RunResult:
    manifest: ProjectManifest
    message: str
    plan: JobPlan
    notes: list[str] = field(default_factory=list)
    operations: list[dict] = field(default_factory=list)
    instruction: str | None = None
    reference: str | None = None


class PipelineRunner:
    def __init__(
        self,
        planner: JobPlanner | None = None,
        project_service: ProjectService | None = None,
        mesh_processing: MeshProcessingService | None = None,
        image_generator: ComfyUiClient | None = None,
    ) -> None:
        self.config = load_config()
        self.planner = planner or JobPlanner()
        self.project_service = project_service or ProjectService()
        self.mesh_processing = mesh_processing or MeshProcessingService()
        self.image_generator = image_generator or ComfyUiClient()
        self.text_to_mesh = ComfyTextToMeshBackend(self.image_generator)
        self.image_to_mesh = ComfyImageToMeshBackend(self.image_generator)
        self.llm = LMStudioClient(self.config)

    @staticmethod
    def _solidify_note(solidify_mm: float) -> str | None:
        if solidify_mm > 0:
            return (
                f"Solidify {solidify_mm:g} mm was requested, but Blender is no longer part of "
                "the runtime; set wall thickness in your slicer."
            )
        return None

    def run(self, manifest: ProjectManifest, job: GenerationJob) -> RunResult:
        plan = self.planner.plan(job)
        work_dir = manifest.root / "work" / f"v{manifest.current_version + 1}_{plan.branch}"
        work_dir.mkdir(parents=True, exist_ok=True)
        image_set = self._stage_images(job.image_paths, work_dir / "input_images")
        current_mesh = job.source_mesh
        final_mesh: Path | None = None
        notes: list[str] = [plan.summary]
        operations: list[dict] = []
        reference = image_set.primary().name if image_set else None
        user_prompt = (job.options.user_prompt or "").strip()
        generation_prompt = (job.options.generation_prompt or job.prompt or "").strip()
        if generation_prompt:
            job.prompt = generation_prompt
        instruction = user_prompt or generation_prompt or None
        raw_mesh: MeshArtifact | None = None
        version_artifacts: list[dict[str, Any]] = []

        logger.info("run job project=%s inputs=%s plan=%s", manifest.id, job.describe_inputs(), plan.summary)
        if user_prompt:
            notes.append(f"User prompt: {user_prompt}")
        if generation_prompt and generation_prompt != user_prompt:
            notes.append(f"Generation prompt (EN): {generation_prompt}")
        for step in plan.steps:
            if step.step_type == PipelineStepType.TEXT_TO_MESH:
                prog.update(manifest.id, 12, "concept")
                generated = self.text_to_mesh.generate(job.prompt, work_dir / "text_to_mesh", job)
                image_set = generated.views
                raw_mesh = generated.mesh
                notes.append(f"Generated {len(image_set.items)} named views and reconstructed mesh in ComfyUI")
                reference = ", ".join(image_set.labels())
            elif step.step_type == PipelineStepType.GENERATE_VIEWS:
                prog.update(manifest.id, 6, step.label)
                image_set = self.image_generator.generate_views(
                    job.prompt,
                    work_dir / "generated_views",
                    count=int(step.params.get("count", job.options.view_count)),
                    project_id=manifest.id,
                )
                notes.append(f"Generated {len(image_set.items)} reference views in ComfyUI")
                reference = image_set.primary().name
            elif step.step_type == PipelineStepType.RECONSTRUCT_MESH:
                prog.update(manifest.id, 18, "mesh")
                raw_mesh = self.image_to_mesh.reconstruct(image_set, work_dir / "image_to_mesh", job)
                notes.append(f"Reconstructed mesh in ComfyUI from {len(image_set.items)} image(s)")
            elif step.step_type == PipelineStepType.FINALIZE_RECONSTRUCTION:
                if raw_mesh is None:
                    raise RuntimeError("Reconstruction result is missing")
                prog.update(manifest.id, 88, "finalize")
                solidify_mm = float(step.params.get("solidify_mm", 0.0))
                final_mesh = self.mesh_processing.finalize_reconstruction(
                    raw_mesh.path,
                    work_dir / "finalize",
                    solidify_mm=solidify_mm,
                )
                note = self._solidify_note(solidify_mm)
                if note:
                    notes.append(note)
            elif step.step_type == PipelineStepType.CLEANUP_MESH:
                if current_mesh is None:
                    raise RuntimeError("Cleanup requires input mesh")
                prog.update(manifest.id, 20, step.label)
                solidify_mm = float(step.params.get("solidify_mm", 0.0))
                final_mesh = self.mesh_processing.cleanup_mesh(
                    current_mesh,
                    work_dir / "cleanup",
                    mode=str(step.params.get("mode", "light")),
                    smooth_iters=int(step.params.get("smooth_iters", 1)),
                    solidify_mm=solidify_mm,
                )
                note = self._solidify_note(solidify_mm)
                if note:
                    notes.append(note)
            elif step.step_type == PipelineStepType.ANALYZE_REFERENCE:
                if current_mesh is None or not image_set:
                    raise RuntimeError("Reference analysis requires mesh and image")
                prog.update(manifest.id, 16, step.label)
                instruction = self._build_reference_instruction(
                    mesh_path=current_mesh,
                    ref_image=image_set.primary(),
                    base_instruction=str(step.params.get("instruction", "")),
                    work_dir=work_dir / "analysis",
                )
                notes.append("Reference image analyzed with VLM")
            elif step.step_type == PipelineStepType.EDIT_MESH:
                if current_mesh is None:
                    raise RuntimeError("Edit requires mesh input")
                resolved_instruction = str(step.params.get("instruction") or instruction or "").strip()
                if not resolved_instruction:
                    raise ValueError("Edit instruction is empty")
                prog.update(manifest.id, 24, step.label)
                edited_artifact, operations, summary = self._apply_edit(
                    mesh_path=current_mesh,
                    instruction=resolved_instruction,
                    work_dir=work_dir / "edit",
                    solidify_mm=float(step.params.get("solidify_mm", 0.0)),
                )
                final_mesh = edited_artifact.path
                instruction = resolved_instruction
                if summary:
                    notes.append(summary)
                note = self._solidify_note(float(step.params.get("solidify_mm", 0.0)))
                if note:
                    notes.append(note)

        if final_mesh is None:
            raise RuntimeError("Pipeline produced no output mesh")

        version_artifacts.extend(self._build_view_artifacts(image_set))
        if raw_mesh is not None:
            version_artifacts.append(self._mesh_artifact_entry(raw_mesh))
        version_artifacts.append(
            {
                "path": final_mesh,
                "kind": "mesh",
                "label": "mesh_final",
                "stage": "finalize",
                "source": plan.branch,
            }
        )
        self.project_service.add_result(
            manifest,
            final_mesh,
            branch=plan.branch,
            action=plan.action,
            instruction=self._history_instruction(user_prompt, generation_prompt, instruction),
            ref=reference,
            ops=operations,
            artifacts=version_artifacts,
        )
        note_block = "\n".join(notes[1:]).strip()
        message = f"{plan.summary} complete."
        if note_block:
            message += f"\n\n{note_block}"
        return RunResult(
            manifest=manifest,
            message=message,
            plan=plan,
            notes=notes,
            operations=operations,
            instruction=instruction,
            reference=reference,
        )

    def _history_instruction(
        self,
        user_prompt: str,
        generation_prompt: str,
        fallback: str | None,
    ) -> str | None:
        user_prompt = (user_prompt or "").strip()
        generation_prompt = (generation_prompt or "").strip()
        if user_prompt and generation_prompt and user_prompt != generation_prompt:
            return f"user: {user_prompt}\ngeneration: {generation_prompt}"
        if generation_prompt:
            return generation_prompt
        if user_prompt:
            return user_prompt
        return fallback

    def _stage_images(self, image_paths: list[Path], target_dir: Path) -> ImageSet:
        target_dir.mkdir(parents=True, exist_ok=True)
        items: list[ImageArtifact] = []
        for index, src in enumerate(image_paths, start=1):
            suffix = src.suffix.lower() or ".png"
            dest = target_dir / f"image_{index}{suffix}"
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
            else:
                dest = src
            items.append(ImageArtifact(path=dest, label=dest.stem, role="reference", stage="input"))
        return ImageSet(items=items)

    def _build_reference_instruction(
        self,
        *,
        mesh_path: Path,
        ref_image: Path,
        base_instruction: str,
        work_dir: Path,
    ) -> str:
        preview = work_dir / "mesh_preview.png"
        render_mesh_preview(mesh_path, preview)
        preview_b64 = base64.b64encode(preview.read_bytes()).decode()
        analysis = self.llm.describe_image_diff(
            f"Compare current mesh preview with the reference image. User wants: {base_instruction}",
            preview_b64,
            ref_image,
        )
        return f"{analysis}\n\nUser instruction: {base_instruction}".strip()

    def _apply_edit(
        self,
        *,
        mesh_path: Path,
        instruction: str,
        work_dir: Path,
        solidify_mm: float,
    ) -> tuple[MeshArtifact, list[dict], str]:
        stats = analyze_mesh(mesh_path)
        plan = self.llm.plan_edit(instruction, stats.to_dict())
        operations = list(plan.get("operations", []))
        edited = self.mesh_processing.apply_edit_operations(
            mesh_path,
            operations,
            work_dir,
            solidify_mm=solidify_mm,
        )
        summary = str(plan.get("summary", "")).strip()
        return MeshArtifact(path=edited, source="edit", notes=summary), operations, summary

    def _build_view_artifacts(self, images: ImageSet) -> list[dict[str, Any]]:
        return [
            {
                "path": item.path,
                "kind": "image",
                "label": item.label or item.path.stem,
                "stage": item.stage,
                "source": item.role,
            }
            for item in images.items
        ]

    def _mesh_artifact_entry(self, artifact: MeshArtifact) -> dict[str, Any]:
        return {
            "path": artifact.path,
            "kind": "mesh",
            "label": artifact.label,
            "stage": artifact.stage,
            "source": artifact.source,
        }
