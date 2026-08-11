from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProjectSummary(BaseModel):
    id: str
    name: str
    current_version: int
    has_mesh: bool


class VersionInfo(BaseModel):
    version: int
    branch: str
    action: str
    instruction: str | None = None
    qc: dict[str, Any] | None = None
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


class TextCreateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    mode: str = "mechanical"


class TextEditRequest(BaseModel):
    instruction: str = Field(min_length=1)
    solidify_mm: float = 0.0


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
