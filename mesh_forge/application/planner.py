from __future__ import annotations

from dataclasses import dataclass, field

from mesh_forge.domain import GenerationJob, PipelineStep, PipelineStepType


@dataclass
class JobPlan:
    branch: str
    action: str
    summary: str
    steps: list[PipelineStep] = field(default_factory=list)


class JobPlanner:
    def plan(self, job: GenerationJob) -> JobPlan:
        if not (job.has_prompt() or job.has_images() or job.has_mesh()):
            raise ValueError("Job must contain text, images, or mesh")

        # Semantic regen: rewrite via ComfyUI text→mesh, not filter ops on existing mesh.
        if job.options.semantic_regen and job.has_prompt():
            return JobPlan(
                branch="regen_edit",
                action="semantic_edit",
                summary="Regenerate mesh from semantic edit brief with ComfyUI",
                steps=[
                    PipelineStep(
                        PipelineStepType.TEXT_TO_MESH,
                        "ComfyUI semantic regen",
                        {"count": job.options.view_count},
                    ),
                    PipelineStep(
                        PipelineStepType.FINALIZE_RECONSTRUCTION,
                        "Подготовка STL",
                        {"solidify_mm": job.options.solidify_mm},
                    ),
                ],
            )

        if job.has_mesh():
            if job.has_images():
                return JobPlan(
                    branch="edit",
                    action="multimodal_edit" if job.has_prompt() else "photo_edit",
                    summary="Edit existing mesh from reference image and prompt",
                    steps=[
                        PipelineStep(
                            PipelineStepType.ANALYZE_REFERENCE,
                            "Анализ референса",
                            {"instruction": job.prompt.strip() or "match reference image"},
                        ),
                        PipelineStep(
                            PipelineStepType.EDIT_MESH,
                            "Правка mesh",
                            {"solidify_mm": job.options.solidify_mm},
                        ),
                    ],
                )
            if job.has_prompt():
                # Text + existing mesh without semantic_regen flag: treat as semantic regen
                # (chat path sets the flag; legacy callers also get ComfyUI regen, not filters).
                return JobPlan(
                    branch="regen_edit",
                    action="semantic_edit",
                    summary="Regenerate mesh from text correction with ComfyUI",
                    steps=[
                        PipelineStep(
                            PipelineStepType.TEXT_TO_MESH,
                            "ComfyUI semantic regen",
                            {"count": job.options.view_count},
                        ),
                        PipelineStep(
                            PipelineStepType.FINALIZE_RECONSTRUCTION,
                            "Подготовка STL",
                            {"solidify_mm": job.options.solidify_mm},
                        ),
                    ],
                )
            return JobPlan(
                branch="scan",
                action="repair",
                summary="Clean and validate input mesh",
                steps=[
                    PipelineStep(
                        PipelineStepType.CLEANUP_MESH,
                        "Очистка mesh",
                        {
                            "mode": job.options.scan_mode,
                            "smooth_iters": job.options.smooth_iters,
                            "solidify_mm": job.options.solidify_mm,
                        },
                    )
                ],
            )

        if job.has_images():
            return JobPlan(
                branch="multimodal" if job.has_prompt() else "photo",
                action="create",
                summary="Reconstruct from images with ComfyUI",
                steps=[
                    PipelineStep(
                        PipelineStepType.RECONSTRUCT_MESH,
                        "ComfyUI images → mesh",
                        {},
                    ),
                    PipelineStep(
                        PipelineStepType.FINALIZE_RECONSTRUCTION,
                        "Подготовка STL",
                        {"solidify_mm": job.options.solidify_mm},
                    ),
                ],
            )

        return JobPlan(
            branch="text",
            action="create",
            summary="Generate mesh from text with ComfyUI",
            steps=[
                PipelineStep(
                    PipelineStepType.TEXT_TO_MESH,
                    "ComfyUI text → mesh",
                    {"count": job.options.view_count},
                ),
                PipelineStep(
                    PipelineStepType.FINALIZE_RECONSTRUCTION,
                    "Подготовка STL",
                    {"solidify_mm": job.options.solidify_mm},
                ),
            ],
        )
