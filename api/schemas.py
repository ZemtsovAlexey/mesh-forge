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


class OperationResult(BaseModel):
    message: str
    project: ProjectDetail
    qc_report: str | None = None


class SystemStatus(BaseModel):
    services: dict[str, bool]
    status_text: str


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
    steps: int
    cfg: float
    mesh_steps: int


class GenerationActiveInfo(BaseModel):
    checkpoint: str
    mesh_checkpoint: str
    image_checkpoint: str
    steps: int
    cfg: float
    mesh_steps: int
    mesh_cfg: float
    mesh_guidance: float


class GenerationSettings(BaseModel):
    quality_preset: str
    presets: dict[str, GenerationPresetInfo]
    active: GenerationActiveInfo
    missing_checkpoints: list[str] = Field(default_factory=list)
    downloaded_checkpoints: list[str] = Field(default_factory=list)
    download_errors: list[str] = Field(default_factory=list)


class GenerationSettingsUpdate(BaseModel):
    quality_preset: str
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


class ChatMessageRequest(BaseModel):
    text: str = ""


class ChatConfirmRequest(BaseModel):
    solidify_mm: float = 0.0
    mode: str = "light"
    smooth_iters: int = 1
    remove_bg: bool = True
