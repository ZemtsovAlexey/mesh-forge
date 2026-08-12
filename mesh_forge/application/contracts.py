from __future__ import annotations

from pathlib import Path
from typing import Protocol

from mesh_forge.domain import GenerationJob, ImageSet, MeshArtifact, TextToMeshResult


class IImageGenerator(Protocol):
    def generate_views(self, prompt: str, work_dir: Path, *, count: int) -> ImageSet: ...


class IMeshReconstructor(Protocol):
    def reconstruct(self, images: ImageSet, work_dir: Path, job: GenerationJob) -> MeshArtifact: ...


class ITextToMeshBackend(Protocol):
    def generate(self, prompt: str, work_dir: Path, job: GenerationJob) -> TextToMeshResult: ...


class IImageToMeshBackend(Protocol):
    def reconstruct(self, images: ImageSet, work_dir: Path, job: GenerationJob) -> MeshArtifact: ...


class IMeshEditor(Protocol):
    def apply(self, mesh_path: Path, instruction: str, work_dir: Path, *, solidify_mm: float) -> tuple[MeshArtifact, list[dict], str]: ...
