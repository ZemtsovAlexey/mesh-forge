from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mesh_forge.manifest import ProjectManifest

logger = logging.getLogger("mesh_forge.pipeline_run")


@dataclass
class PipelineImage:
    label: str
    path: str  # relative to project root
    stage: str = "views"


@dataclass
class PipelineRunState:
    pipeline: str = "text_stepped"  # text_stepped | photo_gated
    step: str = "idle"  # idle | front | views | photo | mesh | done
    status: str = "idle"  # idle | ready | error
    brief_en: str = ""
    user_prompt: str = ""
    solidify_mm: float = 0.0
    remove_bg: bool = True
    work_dir: str = ""  # relative
    images: list[PipelineImage] = field(default_factory=list)
    message: str = ""
    error: str | None = None
    quality_ok: bool = True
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "step": self.step,
            "status": self.status,
            "brief_en": self.brief_en,
            "user_prompt": self.user_prompt,
            "solidify_mm": self.solidify_mm,
            "remove_bg": self.remove_bg,
            "work_dir": self.work_dir,
            "images": [asdict(img) for img in self.images],
            "message": self.message,
            "error": self.error,
            "quality_ok": self.quality_ok,
            "updated_at": self.updated_at,
            "can_continue": (
                self.status == "ready"
                and self.quality_ok
                and self.step in {"front", "views", "photo"}
            ),
            "can_redo": self.status == "ready" and self.step in {"front", "views", "photo"},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineRunState":
        images = [
            PipelineImage(
                label=str(item.get("label") or "image"),
                path=str(item.get("path") or ""),
                stage=str(item.get("stage") or "views"),
            )
            for item in (data.get("images") or [])
            if isinstance(item, dict) and item.get("path")
        ]
        return cls(
            pipeline=str(data.get("pipeline") or "text_stepped"),
            step=str(data.get("step") or "idle"),
            status=str(data.get("status") or "idle"),
            brief_en=str(data.get("brief_en") or ""),
            user_prompt=str(data.get("user_prompt") or ""),
            solidify_mm=float(data.get("solidify_mm") or 0.0),
            remove_bg=bool(data.get("remove_bg", True)),
            work_dir=str(data.get("work_dir") or ""),
            images=images,
            message=str(data.get("message") or ""),
            error=(str(data["error"]) if data.get("error") else None),
            quality_ok=bool(data.get("quality_ok", True)),
            updated_at=str(data.get("updated_at") or datetime.now(timezone.utc).isoformat()),
        )


def pipeline_path(manifest: ProjectManifest) -> Path:
    return manifest.root / "pipeline.json"


def load_pipeline(manifest: ProjectManifest) -> PipelineRunState:
    path = pipeline_path(manifest)
    if not path.is_file():
        return PipelineRunState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return PipelineRunState.from_dict(data if isinstance(data, dict) else {})
    except Exception as exc:
        logger.warning("Failed to read pipeline state %s: %s", path, exc)
        return PipelineRunState()


def save_pipeline(manifest: ProjectManifest, state: PipelineRunState) -> PipelineRunState:
    state.updated_at = datetime.now(timezone.utc).isoformat()
    path = pipeline_path(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def clear_pipeline(manifest: ProjectManifest) -> None:
    path = pipeline_path(manifest)
    if path.is_file():
        path.unlink(missing_ok=True)


def work_dir_for(manifest: ProjectManifest, name: str = "stepped") -> Path:
    path = manifest.root / "work" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel_to_project(manifest: ProjectManifest, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(manifest.root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def abs_from_rel(manifest: ProjectManifest, rel: str) -> Path:
    return (manifest.root / rel).resolve()


def copy_into(work: Path, src: Path, name: str) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    dest = work / name
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest
