from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class JobOptions:
    backend: str = "auto"
    remove_bg: bool = True
    solidify_mm: float = 0.0
    scan_mode: str = "light"
    smooth_iters: int = 1
    use_current_mesh: bool = False
    view_count: int = 4


@dataclass
class GenerationJob:
    project_id: str
    prompt: str = ""
    image_paths: list[Path] = field(default_factory=list)
    source_mesh: Path | None = None
    options: JobOptions = field(default_factory=JobOptions)

    def has_prompt(self) -> bool:
        return bool(self.prompt.strip())

    def has_images(self) -> bool:
        return bool(self.image_paths)

    def has_mesh(self) -> bool:
        return self.source_mesh is not None

    def describe_inputs(self) -> str:
        parts: list[str] = []
        if self.has_prompt():
            parts.append("text")
        if self.has_images():
            parts.append(f"images:{len(self.image_paths)}")
        if self.has_mesh():
            parts.append("mesh")
        return ", ".join(parts) or "empty"
