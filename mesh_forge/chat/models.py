from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Artifact(BaseModel):
    id: str
    kind: Literal["image", "mesh", "mesh_preview", "file"]
    name: str
    label: str = ""
    url: str = ""
    view: str = ""


class ToolCallRecord(BaseModel):
    id: str
    name: str
    title: str = ""
    status: Literal["running", "ok", "error"] = "running"
    args: dict[str, Any] = Field(default_factory=dict)
    knobs: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    progress: float = 0
    stage: str = ""
    thinking: str = ""
    artifacts: list[Artifact] = Field(default_factory=list)


class MessageBlock(BaseModel):
    kind: Literal["text", "tool", "thinking"] = "text"
    text: str = ""
    tool_id: str = ""


class UiMessage(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str = ""
    created_at: str = ""
    attachments: list[Artifact] = Field(default_factory=list)
    tools: list[ToolCallRecord] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    blocks: list[MessageBlock] = Field(default_factory=list)
    reply_to: str = ""
    reply_artifact_ids: list[str] = Field(default_factory=list)
    mesh_region: str = ""
    mesh_pick: list[float] = Field(default_factory=list)
    mesh_topo: dict[str, Any] = Field(default_factory=dict)


class ChatMeta(BaseModel):
    id: str
    title: str = "Новый чат"
    created_at: str = ""
    updated_at: str = ""
    current_mesh: str = ""
    source_mesh: str = ""
    previous_mesh: str = ""
    mesh_region: str = ""
    mesh_pick: list[float] = Field(default_factory=list)
    mesh_topo: dict[str, Any] = Field(default_factory=dict)
    look_view: dict[str, Any] = Field(default_factory=dict)
    mesh_mask: dict[str, Any] = Field(default_factory=dict)
    mask_state: dict[str, Any] = Field(default_factory=dict)
    removal_state: dict[str, Any] = Field(default_factory=dict)
    mesh_history: list[str] = Field(default_factory=list)
    title_locked: bool = False


class ChatSummary(BaseModel):
    id: str
    title: str
    updated_at: str
    has_mesh: bool = False
