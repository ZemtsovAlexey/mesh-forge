from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProjectSummary(BaseModel):
    id: str
    name: str
    current_version: int
    has_mesh: bool


class ArtifactInfo(BaseModel):
    kind: str
    path: str
    label: str
    stage: str = ""
    source: str = ""


class VersionInfo(BaseModel):
    version: int
    branch: str
    action: str
    instruction: str | None = None
    qc: dict[str, Any] | None = None
    artifacts: list[ArtifactInfo] = Field(default_factory=list)
    created_at: str


class ProjectDetail(BaseModel):
    id: str
    name: str
    current_version: int
    has_mesh: bool
    mesh_url: str | None = None
    versions: list[VersionInfo]


class CreateProjectRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class RenameProjectRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class DuplicateProjectRequest(BaseModel):
    name: str | None = Field(default=None, max_length=120)


class PipelineImageInfo(BaseModel):
    label: str
    path: str = ""
    stage: str = ""
    url: str = ""


class PipelineStateInfo(BaseModel):
    pipeline: str = "idle"
    step: str = "idle"
    status: str = "idle"
    brief_en: str = ""
    user_prompt: str = ""
    message: str = ""
    error: str | None = None
    quality_ok: bool = True
    can_continue: bool = False
    can_redo: bool = False
    updated_at: str = ""
    images: list[PipelineImageInfo] = Field(default_factory=list)


class PipelineRedoRequest(BaseModel):
    step: str = "front"
    brief_en: str | None = None
    solidify_mm: float | None = None


class OperationResult(BaseModel):
    message: str
    project: ProjectDetail
    qc_report: str | None = None
    pipeline: PipelineStateInfo | None = None


class GpuQueueEntry(BaseModel):
    kind: str
    label: str
    project_id: str | None = None
    position: int = 0


class GpuQueueInfo(BaseModel):
    active: GpuQueueEntry | None = None
    waiting: list[GpuQueueEntry] = Field(default_factory=list)


class SystemStatus(BaseModel):
    services: dict[str, bool]
    status_text: str
    gpu: GpuQueueInfo = Field(default_factory=GpuQueueInfo)


class LLMSettings(BaseModel):
    base_url: str
    api_key: str
    planner_model: str
    vision_model: str


class LLMSettingsUpdate(BaseModel):
    base_url: str
    api_key: str = "lm-studio"
    planner_model: str
    vision_model: str


class LLMModelsResponse(BaseModel):
    models: list[str]
    status: str
    planner_model: str | None = None
    vision_model: str | None = None


class GenerationPresetInfo(BaseModel):
    label: str
    checkpoint: str
    mesh_checkpoint: str
    image_checkpoint: str = ""
    steps: int
    cfg: float
    mesh_steps: int
    mesh_cfg: float = 4.0
    mesh_guidance: float = 3.5


class GenerationKnobs(BaseModel):
    """Tunable generation knobs (checkpoints + sampling + mesh)."""
    checkpoint: str = "sd_xl_turbo_1.0_fp16.safetensors"
    mesh_checkpoint: str = "hunyuan3d-dit-v2-mv-turbo_fp16.safetensors"
    image_checkpoint: str = "hunyuan3d-dit-v2-mv-turbo_fp16.safetensors"
    zero123_checkpoint: str = "stable_zero123.ckpt"
    width: int = 768
    height: int = 768
    steps: int = 8
    cfg: float = 1.5
    view_denoise: float = 0.58
    view_denoise_turbo: float = 0.72
    view_sampler: str = "euler"
    view_scheduler: str = "sgm_uniform"
    zero123_width: int = 256
    zero123_height: int = 256
    zero123_steps: int = 20
    zero123_cfg: float = 3.0
    zero123_sampler: str = "euler"
    zero123_scheduler: str = "normal"
    zero123_elevation: float = 0.0
    zero123_azimuth_left: float = -90.0
    zero123_azimuth_back: float = 180.0
    zero123_azimuth_right: float = 90.0
    mesh_steps: int = 20
    mesh_cfg: float = 4.0
    mesh_guidance: float = 3.5
    mesh_resolution: int = 3072
    mesh_octree_resolution: int = 256
    mesh_num_chunks: int = 8000


class GenerationActiveInfo(BaseModel):
    checkpoint: str
    mesh_checkpoint: str
    image_checkpoint: str
    zero123_checkpoint: str = "stable_zero123.ckpt"
    view_consistency: str = "img2img"
    mesh_postprocess: bool = True
    steps: int
    cfg: float
    mesh_steps: int
    mesh_cfg: float
    mesh_guidance: float
    knobs: GenerationKnobs = Field(default_factory=GenerationKnobs)
    view_style: str = "clay"


class GenerationSettings(BaseModel):
    quality_preset: str
    view_consistency: str = "img2img"
    view_style: str = "clay"
    mesh_postprocess: bool = True
    view_modes: dict[str, dict[str, str]] = Field(default_factory=dict)
    view_styles: dict[str, dict[str, str]] = Field(default_factory=dict)
    presets: dict[str, GenerationPresetInfo]
    active: GenerationActiveInfo
    knobs: GenerationKnobs = Field(default_factory=GenerationKnobs)
    missing_checkpoints: list[str] = Field(default_factory=list)
    downloaded_checkpoints: list[str] = Field(default_factory=list)
    download_errors: list[str] = Field(default_factory=list)


class GenerationSettingsUpdate(BaseModel):
    quality_preset: str
    view_consistency: str = "img2img"
    view_style: str = "clay"
    mesh_postprocess: bool = True
    knobs: GenerationKnobs | None = None
    download_missing: bool = True


class ExportInfo(BaseModel):
    report: str
    print_ready: bool = False
    download_url: str | None = None


class ProgressInfo(BaseModel):
    project_id: str
    operation: str
    percent: float
    stage: str
    active: bool
    error: str | None = None
    elapsed_sec: float = 0.0


class ChatMessageInfo(BaseModel):
    role: str
    content: str
    created_at: str = ""
    id: str = ""
    kind: str = "text"
    ref_ids: list[str] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


class NotebookEntryInfo(BaseModel):
    id: str
    kind: str = "note"
    title: str = ""
    summary: str = ""
    step: str = ""
    brief_en: str = ""
    user_prompt: str = ""
    version: int | None = None
    images: list[dict[str, Any]] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""


class ChatStateInfo(BaseModel):
    messages: list[ChatMessageInfo] = Field(default_factory=list)
    mode: str = "create"
    status: str = "idle"
    intent: str = "create"
    draft_prompt_en: str = ""
    edit_brief_en: str = ""
    user_prompt: str = ""
    ready: bool = False
    questions: list[str] = Field(default_factory=list)
    assistant_message: str = ""
    planned_ops: list[dict[str, Any]] = Field(default_factory=list)
    pipeline: PipelineStateInfo | None = None
    notebook: list[NotebookEntryInfo] = Field(default_factory=list)


class ChatMessageRequest(BaseModel):
    text: str = ""


class ChatConfirmRequest(BaseModel):
    solidify_mm: float = 0.0
    mode: str = "light"
    smooth_iters: int = 1
    remove_bg: bool = True
