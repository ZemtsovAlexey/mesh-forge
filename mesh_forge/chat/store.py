from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mesh_forge.chat.models import Artifact, ChatMeta, ChatSummary, UiMessage
from mesh_forge.config import load_config

logger = logging.getLogger("mesh_forge.chat.store")

_SAFE_NAME = re.compile(r"[^a-zA-Z0-9._-]+")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(n: int = 12) -> str:
    return uuid.uuid4().hex[:n]


class ChatStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else load_config().projects_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def chat_dir(self, chat_id: str) -> Path:
        path = (self.root / chat_id).resolve()
        if not str(path).startswith(str(self.root.resolve())):
            raise FileNotFoundError("Invalid chat id")
        return path

    def files_dir(self, chat_id: str) -> Path:
        path = self.chat_dir(chat_id) / "files"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _meta_path(self, chat_id: str) -> Path:
        return self.chat_dir(chat_id) / "meta.json"

    def _messages_path(self, chat_id: str) -> Path:
        return self.chat_dir(chat_id) / "ui.json"

    def _agent_path(self, chat_id: str) -> Path:
        return self.chat_dir(chat_id) / "agent_messages.json"

    def list_chats(self) -> list[ChatSummary]:
        items: list[ChatSummary] = []
        if not self.root.is_dir():
            return items
        for child in self.root.iterdir():
            meta_path = child / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = ChatMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            items.append(
                ChatSummary(
                    id=meta.id,
                    title=meta.title,
                    updated_at=meta.updated_at or meta.created_at,
                    has_mesh=bool(meta.current_mesh),
                )
            )
        items.sort(key=lambda c: c.updated_at, reverse=True)
        return items

    def create_chat(self, title: str = "Новый чат") -> ChatMeta:
        chat_id = _new_id()
        now = _now()
        meta = ChatMeta(id=chat_id, title=title.strip() or "Новый чат", created_at=now, updated_at=now)
        self.chat_dir(chat_id).mkdir(parents=True, exist_ok=True)
        self.files_dir(chat_id)
        self._meta_path(chat_id).write_text(meta.model_dump_json(indent=2), encoding="utf-8")
        self._messages_path(chat_id).write_text("[]", encoding="utf-8")
        self._agent_path(chat_id).write_text("[]", encoding="utf-8")
        return meta

    def get_meta(self, chat_id: str) -> ChatMeta:
        path = self._meta_path(chat_id)
        if not path.is_file():
            raise FileNotFoundError(f"Chat not found: {chat_id}")
        return ChatMeta.model_validate_json(path.read_text(encoding="utf-8"))

    def save_meta(self, meta: ChatMeta) -> None:
        meta.updated_at = _now()
        self._meta_path(meta.id).write_text(meta.model_dump_json(indent=2), encoding="utf-8")

    def rename(self, chat_id: str, title: str) -> ChatMeta:
        meta = self.get_meta(chat_id)
        meta.title = title.strip() or meta.title
        meta.title_locked = True
        self.save_meta(meta)
        return meta

    def delete(self, chat_id: str) -> None:
        path = self.chat_dir(chat_id)
        if not path.is_dir():
            raise FileNotFoundError(f"Chat not found: {chat_id}")
        import shutil

        shutil.rmtree(path)

    def load_messages(self, chat_id: str) -> list[UiMessage]:
        path = self._messages_path(chat_id)
        if not path.is_file():
            return []
        raw = json.loads(path.read_text(encoding="utf-8") or "[]")
        return [UiMessage.model_validate(item) for item in raw]

    def save_messages(self, chat_id: str, messages: list[UiMessage]) -> None:
        payload = [m.model_dump() for m in messages]
        self._messages_path(chat_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        meta = self.get_meta(chat_id)
        self.save_meta(meta)

    def load_agent_messages(self, chat_id: str) -> list[Any]:
        path = self._agent_path(chat_id)
        if not path.is_file():
            return []
        try:
            from pydantic_ai.messages import ModelMessagesTypeAdapter

            return list(ModelMessagesTypeAdapter.validate_json(path.read_bytes()))
        except Exception:
            logger.exception("failed to load agent messages for %s", chat_id)
            return []

    def save_agent_messages(self, chat_id: str, messages: list[Any]) -> None:
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        data = ModelMessagesTypeAdapter.dump_json(messages, indent=2)
        self._agent_path(chat_id).write_bytes(data)

    def current_mesh(self, chat_id: str) -> Path | None:
        meta = self.get_meta(chat_id)
        if not meta.current_mesh:
            return None
        path = self.resolve_file(chat_id, meta.current_mesh)
        return path if path.is_file() else None

    def set_current_mesh(self, chat_id: str, path: Path) -> None:
        meta = self.get_meta(chat_id)
        meta.current_mesh = path.name
        self.save_meta(meta)

    def maybe_set_title(self, chat_id: str, text: str) -> None:
        meta = self.get_meta(chat_id)
        if meta.title_locked:
            return
        if meta.title and meta.title != "Новый чат":
            return
        line = (text or "").strip().splitlines()[0] if text else ""
        if not line:
            return
        meta.title = line[:60]
        self.save_meta(meta)

    def resolve_file(self, chat_id: str, name: str) -> Path:
        safe = Path(name).name
        if not safe or safe != name.replace("\\", "/").split("/")[-1]:
            raise FileNotFoundError("Invalid file name")
        path = (self.files_dir(chat_id) / safe).resolve()
        root = self.files_dir(chat_id).resolve()
        if not str(path).startswith(str(root)):
            raise FileNotFoundError("Invalid file name")
        return path

    def new_file(self, chat_id: str, filename: str) -> Path:
        stem = _SAFE_NAME.sub("_", Path(filename).stem) or "file"
        suffix = Path(filename).suffix.lower() or ".bin"
        name = f"{_new_id(8)}_{stem}{suffix}"
        return self.files_dir(chat_id) / name

    def save_bytes(self, chat_id: str, filename: str, data: bytes) -> Path:
        dest = self.new_file(chat_id, filename)
        dest.write_bytes(data)
        return dest

    def list_files(self, chat_id: str) -> list[Artifact]:
        files = []
        for path in sorted(self.files_dir(chat_id).iterdir()):
            if not path.is_file():
                continue
            files.append(self.artifact_from_path(chat_id, path))
        return files

    def artifact_from_path(
        self,
        chat_id: str,
        path: Path,
        *,
        label: str = "",
        view: str = "",
    ) -> Artifact:
        suffix = path.suffix.lower()
        kind = "file"
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            kind = "image"
        elif suffix in {".stl", ".obj", ".glb", ".gltf"}:
            kind = "mesh"
        return Artifact(
            id=path.name,
            kind=kind,
            name=path.name,
            label=label or path.stem,
            url=f"/api/chats/{chat_id}/files/{path.name}",
            view=view,
        )

    def resolve_ref(self, chat_id: str, ref: str) -> Path:
        ref = (ref or "").strip()
        if not ref:
            raise FileNotFoundError("Empty artifact ref")
        name = Path(ref).name
        path = self.resolve_file(chat_id, name)
        if path.is_file():
            return path
        for item in self.files_dir(chat_id).iterdir():
            if item.stem == ref or item.name == ref:
                return item
        raise FileNotFoundError(f"Artifact not found: {ref}")
