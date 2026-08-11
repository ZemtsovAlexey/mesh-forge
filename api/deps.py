from __future__ import annotations

from functools import lru_cache

from mesh_forge.manifest import ProjectManifest
from mesh_forge.orchestrator import Orchestrator


@lru_cache(maxsize=1)
def get_orchestrator() -> Orchestrator:
    return Orchestrator()


def load_project(project_id: str) -> ProjectManifest:
    return ProjectManifest.load(project_id)
