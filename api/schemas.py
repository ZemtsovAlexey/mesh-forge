from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatSummaryOut(BaseModel):
    id: str
    title: str
    updated_at: str
    has_mesh: bool = False


class ArtifactOut(BaseModel):
    id: str
    kind: str
    name: str
    label: str = ""
    url: str = ""
    view: str = ""


class ToolCallOut(BaseModel):
    id: str
    name: str
    title: str = ""
    status: str = "running"
    args: dict[str, Any] = Field(default_factory=dict)
    knobs: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    progress: float = 0
    stage: str = ""
    artifacts: list[ArtifactOut] = Field(default_factory=list)


class MessageBlockOut(BaseModel):
    kind: str = "text"
    text: str = ""
    tool_id: str = ""


class MessageOut(BaseModel):
    id: str
    role: str
    content: str = ""
    created_at: str = ""
    attachments: list[ArtifactOut] = Field(default_factory=list)
    tools: list[ToolCallOut] = Field(default_factory=list)
    artifacts: list[ArtifactOut] = Field(default_factory=list)
    blocks: list[MessageBlockOut] = Field(default_factory=list)
    reply_to: str = ""
    reply_artifact_ids: list[str] = Field(default_factory=list)


class ChatDetail(BaseModel):
    id: str
    title: str
    created_at: str = ""
    updated_at: str = ""
    current_mesh: str = ""
    messages: list[MessageOut] = Field(default_factory=list)


class CreateChatRequest(BaseModel):
    title: str = "Новый чат"


class RenameChatRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class GpuQueueEntry(BaseModel):
    kind: str
    label: str
    project_id: str | None = None
    position: int = 0


class GpuQueueInfo(BaseModel):
    active: GpuQueueEntry | None = None
    waiting: list[GpuQueueEntry] = Field(default_factory=list)
    shared: bool = True
    actives: list[GpuQueueEntry] = Field(default_factory=list)
    llm_host: str = ""
    comfy_host: str = ""


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
    status: str = ""
    planner_model: str | None = None
    vision_model: str | None = None


class GenerationDefaults(BaseModel):
    quality_preset: str = "draft"
    view_style: str = "clay"


class ComfyUISettings(BaseModel):
    enabled: bool = True
    base_url: str = "http://127.0.0.1:8188"


class ComfyUISettingsUpdate(BaseModel):
    base_url: str
    enabled: bool | None = None


class ComfyUIProbeResponse(BaseModel):
    ok: bool
    base_url: str
    status: str = ""
