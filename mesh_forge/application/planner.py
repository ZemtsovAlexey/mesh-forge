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

        # Guided edit: preserve identity via img2img from anchor view.
        if job.options.guided_edit and job.has_prompt():
            return JobPlan(
                branch="guided_edit",
                action="guided_edit",
                summary="Guided edit from anchor view with ComfyUI",
                steps=[
                    PipelineStep(
                        PipelineStepType.GUIDED_EDIT,
                        "ComfyUI guided edit",
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
                # Text + existing mesh: geometry/filter edit unless semantic_regen is set.
                return JobPlan(
                    branch="edit",
                    action="geometry_edit",
                    summary="Edit existing mesh with geometry operations",
                    steps=[
                        PipelineStep(
                            PipelineStepType.EDIT_MESH,
                            "Правка mesh",
                            {
                                "instruction": job.prompt.strip(),
                                "solidify_mm": job.options.solidify_mm,
                                "operations": list(job.options.planned_ops or []),
                            },
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
