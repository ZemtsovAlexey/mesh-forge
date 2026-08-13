from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mesh_forge.manifest import ProjectManifest

logger = logging.getLogger("mesh_forge.notebook")

MAX_ENTRIES = 80


@dataclass
class NotebookEntry:
    id: str
    kind: str  # pipeline_step | version | draft | note | message_apply
    title: str
    summary: str = ""
    step: str = ""
    brief_en: str = ""
    user_prompt: str = ""
    version: int | None = None
    images: list[dict[str, str]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NotebookEntry":
        images_raw = data.get("images") or []
        images = [img for img in images_raw if isinstance(img, dict)]
        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        version = data.get("version")
        return cls(
            id=str(data.get("id") or _new_id()),
            kind=str(data.get("kind") or "note"),
            title=str(data.get("title") or "Запись"),
            summary=str(data.get("summary") or ""),
            step=str(data.get("step") or ""),
            brief_en=str(data.get("brief_en") or ""),
            user_prompt=str(data.get("user_prompt") or ""),
            version=int(version) if version is not None else None,
            images=images,
            meta=meta,
            created_at=str(data.get("created_at") or datetime.now(timezone.utc).isoformat()),
        )


def _new_id() -> str:
    return uuid.uuid4().hex[:10]


def notebook_path(manifest: ProjectManifest) -> Path:
    return manifest.root / "notebook.json"


def load_notebook(manifest: ProjectManifest) -> list[NotebookEntry]:
    path = notebook_path(manifest)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data.get("entries") if isinstance(data, dict) else data
        if not isinstance(raw, list):
            return []
        return [NotebookEntry.from_dict(item) for item in raw if isinstance(item, dict)]
    except Exception as exc:
        logger.warning("Failed to read notebook %s: %s", path, exc)
        return []


def save_notebook(manifest: ProjectManifest, entries: list[NotebookEntry]) -> list[NotebookEntry]:
    path = notebook_path(manifest)
    trimmed = entries[-MAX_ENTRIES:]
    payload = {"entries": [e.to_dict() for e in trimmed]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return trimmed


def append_entry(manifest: ProjectManifest, entry: NotebookEntry) -> NotebookEntry:
    entries = load_notebook(manifest)
    entries.append(entry)
    save_notebook(manifest, entries)
    return entry


def add_note(
    manifest: ProjectManifest,
    *,
    title: str,
    summary: str = "",
    kind: str = "note",
    brief_en: str = "",
    user_prompt: str = "",
    step: str = "",
    version: int | None = None,
    images: list[dict[str, str]] | None = None,
    meta: dict[str, Any] | None = None,
) -> NotebookEntry:
    entry = NotebookEntry(
        id=_new_id(),
        kind=kind,
        title=title.strip() or "Запись",
        summary=(summary or "").strip(),
        step=step,
        brief_en=(brief_en or "").strip(),
        user_prompt=(user_prompt or "").strip(),
        version=version,
        images=list(images or []),
        meta=dict(meta or {}),
    )
    return append_entry(manifest, entry)


def snapshot_pipeline(manifest: ProjectManifest, state: Any, *, title: str | None = None) -> NotebookEntry | None:
    """Record current pipeline gate as a notebook entry."""
    try:
        data = state.to_dict() if hasattr(state, "to_dict") else dict(state or {})
    except Exception:
        return None
    step = str(data.get("step") or "")
    if step in {"", "idle"}:
        return None
    images = [
        {"label": str(img.get("label") or ""), "path": str(img.get("path") or ""), "stage": str(img.get("stage") or "")}
        for img in (data.get("images") or [])
        if isinstance(img, dict)
    ]
    brief = str(data.get("brief_en") or "")
    auto_title = title or f"Шаг {step}"
    return add_note(
        manifest,
        kind="pipeline_step",
        title=auto_title,
        summary=str(data.get("message") or data.get("error") or ""),
        step=step,
        brief_en=brief,
        user_prompt=str(data.get("user_prompt") or ""),
        images=images,
        meta={
            "pipeline": data.get("pipeline"),
            "status": data.get("status"),
            "quality_ok": data.get("quality_ok"),
        },
    )


def snapshot_version(manifest: ProjectManifest, version: int, *, instruction: str = "") -> NotebookEntry:
    return add_note(
        manifest,
        kind="version",
        title=f"Версия v{version}",
        summary=(instruction or "").strip()[:240],
        version=int(version),
        brief_en=(instruction or "").strip(),
        meta={"version": int(version)},
    )


def notebook_payload(manifest: ProjectManifest, *, limit: int = 20) -> list[dict[str, Any]]:
    entries = load_notebook(manifest)
    return [e.to_dict() for e in entries[-limit:]]


def notebook_summary_for_llm(manifest: ProjectManifest, *, limit: int = 12) -> str:
    entries = load_notebook(manifest)[-limit:]
    if not entries:
        return "(notebook empty)"
    lines: list[str] = []
    for e in entries:
        bit = f"- [{e.id}] {e.kind}/{e.step or '-'} · {e.title}"
        if e.brief_en:
            bit += f" · brief: {e.brief_en[:120]}"
        if e.summary:
            bit += f" · {e.summary[:80]}"
        if e.version is not None:
            bit += f" · v{e.version}"
        lines.append(bit)
    return "\n".join(lines)
