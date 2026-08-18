from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ImageArtifact:
    path: Path
    label: str = ""
    role: str = "reference"
    stage: str = "input"


@dataclass
class ImageSet:
    items: list[ImageArtifact] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.items)

    def paths(self) -> list[Path]:
        return [item.path for item in self.items]

    def labels(self) -> list[str]:
        return [item.label or item.path.stem for item in self.items]

    def get(self, label: str) -> ImageArtifact | None:
        normalized = label.strip().lower()
        for item in self.items:
            if (item.label or item.path.stem).strip().lower() == normalized:
                return item
        return None

    def primary(self) -> Path:
        if not self.items:
            raise ValueError("ImageSet is empty")
        return self.items[0].path


@dataclass
class MeshArtifact:
    path: Path
    source: str
    notes: str = ""
    label: str = "mesh"
    stage: str = "mesh"


@dataclass
class SegmentationArtifact:
    mask: ImageArtifact
    visualization: ImageArtifact
    label: str = "segmentation"
    stage: str = "segmentation"
    boxes: list[dict[str, float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)


@dataclass
class TextToMeshResult:
    views: ImageSet
    mesh: MeshArtifact
