from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PipelineStepType(StrEnum):
    TEXT_TO_MESH = "text_to_mesh"
    GUIDED_EDIT = "guided_edit"
    GENERATE_VIEWS = "generate_views"
    RECONSTRUCT_MESH = "reconstruct_mesh"
    FINALIZE_RECONSTRUCTION = "finalize_reconstruction"
    CLEANUP_MESH = "cleanup_mesh"
    ANALYZE_REFERENCE = "analyze_reference"
    EDIT_MESH = "edit_mesh"


@dataclass
class PipelineStep:
    step_type: PipelineStepType
    label: str
    params: dict[str, Any] = field(default_factory=dict)
