from __future__ import annotations

import json
import logging
import re
import threading
import time
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


def _write_atomic(path: Path, text: str, *, retries: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(max(1, int(retries))):
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            try:
                tmp.replace(path)
            except PermissionError:
                # Windows: ui.json may be open in the IDE while the agent persists SSE.
                path.write_text(text, encoding="utf-8")
                tmp.unlink(missing_ok=True)
            return
        except PermissionError as exc:
            last_err = exc
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            time.sleep(0.04 * (attempt + 1))
        except OSError as exc:
            last_err = exc
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise
    if last_err is not None:
        raise last_err


def _parse_chat_meta(text: str) -> ChatMeta:
    raw = (text or "").strip()
    try:
        return ChatMeta.model_validate_json(raw)
    except Exception:
        obj, _end = json.JSONDecoder().raw_decode(raw)
        return ChatMeta.model_validate(obj)


def _new_id(n: int = 12) -> str:
    return uuid.uuid4().hex[:n]


class ChatStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else load_config().projects_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

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
                meta = _parse_chat_meta(meta_path.read_text(encoding="utf-8"))
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
        _write_atomic(self._meta_path(chat_id), meta.model_dump_json(indent=2))
        _write_atomic(self._messages_path(chat_id), "[]")
        _write_atomic(self._agent_path(chat_id), "[]")
        return meta

    def get_meta(self, chat_id: str) -> ChatMeta:
        path = self._meta_path(chat_id)
        if not path.is_file():
            raise FileNotFoundError(f"Chat not found: {chat_id}")
        with self._lock:
            return _parse_chat_meta(path.read_text(encoding="utf-8"))

    def save_meta(self, meta: ChatMeta) -> None:
        meta.updated_at = _now()
        path = self._meta_path(meta.id)
        text = meta.model_dump_json(indent=2)
        with self._lock:
            _write_atomic(path, text)

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
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        with self._lock:
            _write_atomic(self._messages_path(chat_id), text)
            meta = _parse_chat_meta(self._meta_path(chat_id).read_text(encoding="utf-8"))
            meta.updated_at = _now()
            _write_atomic(self._meta_path(meta.id), meta.model_dump_json(indent=2))

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
        return self._mesh_path(chat_id, self.get_meta(chat_id).current_mesh)

    def source_mesh(self, chat_id: str) -> Path | None:
        meta = self.get_meta(chat_id)
        path = self._mesh_path(chat_id, meta.source_mesh)
        if path:
            return path
        for art in self.list_files(chat_id):
            if art.kind != "mesh":
                continue
            label = (art.label or "").strip().lower()
            if label == "mesh" or str(art.name).endswith("_mesh.stl"):
                found = self._mesh_path(chat_id, art.name)
                if found:
                    return found
        return None

    def previous_mesh(self, chat_id: str) -> Path | None:
        return self._mesh_path(chat_id, self.get_meta(chat_id).previous_mesh)

    def mesh_history(self, chat_id: str) -> list[str]:
        return list(self.get_meta(chat_id).mesh_history or [])

    def set_current_mesh(self, chat_id: str, path: Path, *, role: str = "edit") -> None:
        meta = self.get_meta(chat_id)
        name = path.name
        kind = (role or "edit").strip().lower()
        first = not meta.source_mesh and not meta.current_mesh
        if kind == "source" or (kind == "edit" and first):
            meta.source_mesh = name
            meta.current_mesh = name
            if kind == "source":
                meta.mesh_history = []
                meta.previous_mesh = ""
        else:
            if meta.current_mesh and meta.current_mesh != name:
                self._push_history(meta, meta.current_mesh)
            if not meta.source_mesh:
                meta.source_mesh = meta.previous_mesh or name
            meta.current_mesh = name
        self._clear_pick(meta)
        self.save_meta(meta)

    def _push_history(self, meta: ChatMeta, name: str) -> None:
        if not name:
            return
        hist = [item for item in (meta.mesh_history or []) if item]
        if not hist or hist[-1] != name:
            hist.append(name)
        meta.mesh_history = hist[-20:]
        meta.previous_mesh = meta.mesh_history[-1]

    def restore_mesh(self, chat_id: str, to: str = "previous") -> Path:
        meta = self.get_meta(chat_id)
        wanted = (to or "previous").strip().lower()
        if wanted == "source":
            target = self.source_mesh(chat_id)
            if target is None:
                raise FileNotFoundError("Нет меша для отката (previous/source пусты).")
            if meta.current_mesh and meta.current_mesh != target.name:
                meta.previous_mesh = meta.current_mesh
                meta.current_mesh = target.name
                self._clear_pick(meta)
                self.save_meta(meta)
            return target
        hist = [item for item in (meta.mesh_history or []) if item]
        target: Path | None = None
        if hist:
            name = hist.pop()
            target = self._mesh_path(chat_id, name)
            meta.mesh_history = hist
        if target is None:
            target = self.previous_mesh(chat_id) or self.source_mesh(chat_id)
        if target is None:
            raise FileNotFoundError("Нет меша для отката (previous/source пусты).")
        if meta.current_mesh and meta.current_mesh != target.name:
            meta.previous_mesh = meta.current_mesh
            meta.current_mesh = target.name
            self._clear_pick(meta)
            self.save_meta(meta)
        return target

    def _clear_pick(self, meta: ChatMeta) -> None:
        meta.mesh_region = ""
        meta.mesh_pick = []
        meta.mesh_topo = {}
        meta.mesh_mask = {}
        meta.mask_state = {}
        meta.removal_state = {}

    def set_mesh_pick(
        self,
        chat_id: str,
        nx: float,
        ny: float,
        nz: float,
        radius: float = 0.022,
        *,
        kind: str | None = None,
        mesh_ref: str | None = None,
        hops: int = 12,
    ) -> ChatMeta:
        from mesh_forge.ops.geometry import load_mesh
        from mesh_forge.ops.region import infer_region
        from mesh_forge.ops.topo import hit_topology

        meta = self.get_meta(chat_id)
        nx = max(0.0, min(1.0, float(nx)))
        ny = max(0.0, min(1.0, float(ny)))
        nz = max(0.0, min(1.0, float(nz)))
        radius = max(0.015, min(0.05, float(radius)))
        hops = max(1, min(int(hops), 24))
        meta.mesh_region = infer_region(nx, ny, nz)
        meta.mesh_pick = [nx, ny, nz, radius]
        meta.mesh_topo = {}
        target = None
        if mesh_ref and str(mesh_ref).strip():
            try:
                target = self.resolve_ref(chat_id, str(mesh_ref).strip())
            except FileNotFoundError:
                target = None
        if target is None:
            target = self.current_mesh(chat_id)
        if target is not None:
            try:
                topo = hit_topology(load_mesh(target), nx, ny, nz, kind=kind)
                meta.mesh_topo = {
                    "kind": topo["kind"],
                    "vertex": int(topo["vertex"]),
                    "face": int(topo["face"]),
                    "edge": [int(x) for x in topo["edge"]],
                    "mesh": target.name,
                    "hops": hops,
                }
                meta.mesh_pick = [float(topo["nx"]), float(topo["ny"]), float(topo["nz"]), radius]
            except Exception:
                meta.mesh_topo = {}
        self.save_meta(meta)
        return meta

    def set_look_view(
        self,
        chat_id: str,
        *,
        views: str,
        yaw: float | None = None,
        pitch: float | None = None,
        zoom: float = 1.0,
    ) -> ChatMeta:
        meta = self.get_meta(chat_id)
        prev = dict(meta.look_view or {})
        seen = [str(v) for v in (prev.get("seen") or []) if v]
        cam = str(views or "").split(",")[0].strip()
        if cam and cam not in seen:
            seen.append(cam)
        meta.look_view = {
            **prev,
            "views": str(views or ""),
            "yaw": yaw,
            "pitch": pitch,
            "zoom": float(zoom or 1.0),
            "seen": seen[-8:],
        }
        self.save_meta(meta)
        return meta

    def set_viewport_aim(
        self,
        chat_id: str,
        x: float,
        y: float,
        *,
        views: str = "",
        yaw: float | None = None,
        pitch: float | None = None,
        zoom: float | None = None,
    ) -> ChatMeta:
        meta = self.get_meta(chat_id)
        lv = dict(meta.look_view or {})
        lv["aim_x"] = max(0.0, min(1.0, float(x)))
        lv["aim_y"] = max(0.0, min(1.0, float(y)))
        if views:
            lv["views"] = str(views)
        if yaw is not None:
            lv["yaw"] = yaw
        if pitch is not None:
            lv["pitch"] = pitch
        if zoom is not None:
            lv["zoom"] = float(zoom)
        meta.look_view = lv
        meta.mesh_pick = []
        meta.mesh_topo = {}
        meta.mesh_region = ""
        self.save_meta(meta)
        return meta

    def set_mesh_mask(self, chat_id: str, mesh_name: str, face_idx) -> ChatMeta:
        import numpy as np

        meta = self.get_meta(chat_id)
        idx = np.asarray(face_idx, dtype=np.int32).reshape(-1)
        idx = idx[(idx >= 0)]
        dest = self.new_file(chat_id, "mesh_mask.npz")
        np.savez_compressed(dest, faces=idx)
        meta.mesh_mask = {
            "mesh": str(mesh_name or meta.current_mesh or ""),
            "file": dest.name,
            "count": int(idx.size),
        }
        state = dict(meta.mask_state or {})
        state["mesh"] = str(mesh_name or meta.current_mesh or "")
        state["candidate_faces"] = int(idx.size)
        meta.mask_state = state
        lv = dict(meta.look_view or {})
        lv["seen"] = []
        meta.look_view = lv
        self.save_meta(meta)
        return meta

    def load_mesh_mask(self, chat_id: str, *, n_faces: int, mesh_name: str = ""):
        import numpy as np

        meta = self.get_meta(chat_id)
        info = dict(meta.mesh_mask or {})
        if not info:
            return None
        wanted = str(mesh_name or meta.current_mesh or "")
        stored = str(info.get("mesh") or "")
        if wanted and stored and stored != wanted:
            return None
        name = str(info.get("file") or "")
        if not name:
            return None
        path = self.files_dir(chat_id) / name
        if not path.is_file():
            return None
        try:
            data = np.load(path)
            idx = np.asarray(data["faces"], dtype=np.int64).reshape(-1)
        except Exception:
            return None
        n = int(n_faces)
        mask = np.zeros(n, dtype=bool)
        if n <= 0:
            return mask
        valid = idx[(idx >= 0) & (idx < n)]
        mask[valid] = True
        return mask

    def mask_state(self, chat_id: str) -> dict[str, Any]:
        return dict(self.get_meta(chat_id).mask_state or {})

    def set_mask_state(self, chat_id: str, state: dict[str, Any] | None) -> ChatMeta:
        meta = self.get_meta(chat_id)
        meta.mask_state = dict(state or {})
        self.save_meta(meta)
        return meta

    def update_mask_state(self, chat_id: str, **fields: Any) -> ChatMeta:
        meta = self.get_meta(chat_id)
        state = dict(meta.mask_state or {})
        for key, value in fields.items():
            if value is None:
                state.pop(key, None)
            else:
                state[key] = value
        meta.mask_state = state
        self.save_meta(meta)
        return meta

    def clear_mask_state(self, chat_id: str, *, keep_mask: bool = False) -> ChatMeta:
        meta = self.get_meta(chat_id)
        meta.mask_state = {}
        if not keep_mask:
            meta.mesh_mask = {}
        self.save_meta(meta)
        return meta

    def removal_state(self, chat_id: str) -> dict[str, Any]:
        return dict(self.get_meta(chat_id).removal_state or {})

    def set_removal_state(self, chat_id: str, state: dict[str, Any] | None) -> ChatMeta:
        meta = self.get_meta(chat_id)
        meta.removal_state = dict(state or {})
        self.save_meta(meta)
        return meta

    def update_removal_state(self, chat_id: str, **fields: Any) -> ChatMeta:
        meta = self.get_meta(chat_id)
        state = dict(meta.removal_state or {})
        for key, value in fields.items():
            if value is None:
                state.pop(key, None)
            else:
                state[key] = value
        meta.removal_state = state
        self.save_meta(meta)
        return meta

    def clear_removal_state(self, chat_id: str) -> ChatMeta:
        meta = self.get_meta(chat_id)
        meta.removal_state = {}
        self.save_meta(meta)
        return meta

    def apply_mesh_topo(self, chat_id: str, topo: dict, *, mesh_name: str = "") -> ChatMeta:
        from mesh_forge.ops.region import infer_region

        meta = self.get_meta(chat_id)
        hops = max(1, min(int(topo.get("hops") or 12), 24))
        stored = {
            "kind": str(topo.get("kind") or "face"),
            "vertex": int(topo.get("vertex", -1)),
            "face": int(topo.get("face", -1)),
            "edge": [int(x) for x in (topo.get("edge") or [])[:2]],
            "mesh": mesh_name or str(topo.get("mesh") or meta.current_mesh or ""),
            "hops": hops,
            "faces": int(topo.get("faces") or 0),
        }
        nx = float(topo.get("nx", 0.5))
        ny = float(topo.get("ny", 0.5))
        nz = float(topo.get("nz", 0.5))
        meta.mesh_region = infer_region(nx, ny, nz)
        meta.mesh_pick = [nx, ny, nz, 0.022]
        meta.mesh_topo = stored
        self.save_meta(meta)
        return meta

    def set_mesh_region(self, chat_id: str, region: str, *, keep_pick: bool = False) -> ChatMeta:
        from mesh_forge.ops.region import parse_region

        meta = self.get_meta(chat_id)
        meta.mesh_region = parse_region(region)
        if not keep_pick:
            meta.mesh_pick = []
            meta.mesh_topo = {}
        self.save_meta(meta)
        return meta

    def clear_mesh_pick(self, chat_id: str) -> ChatMeta:
        meta = self.get_meta(chat_id)
        self._clear_pick(meta)
        self.save_meta(meta)
        return meta

    def clear_mesh_target(self, chat_id: str) -> ChatMeta:
        """Drop click/topo from composer without clearing a painted delete mask."""
        meta = self.get_meta(chat_id)
        meta.mesh_region = ""
        meta.mesh_pick = []
        meta.mesh_topo = {}
        self.save_meta(meta)
        return meta

    def active_mesh_target(self, chat_id: str) -> tuple[str, list[float]]:
        """Pending click in meta, else the latest user message's click.

        A look() that stored only mesh_region must not hide the click on the message.
        """
        meta = self.get_meta(chat_id)
        if meta.mesh_pick and len(meta.mesh_pick) >= 3:
            return meta.mesh_region, list(meta.mesh_pick)
        for msg in reversed(self.load_messages(chat_id)):
            if msg.role != "user":
                continue
            if msg.mesh_pick and len(msg.mesh_pick) >= 3:
                return msg.mesh_region or "", list(msg.mesh_pick)
            if msg.mesh_region:
                return msg.mesh_region, []
            break
        if meta.mesh_region:
            return meta.mesh_region, []
        return "", []

    def active_mesh_topo(self, chat_id: str) -> dict:
        meta = self.get_meta(chat_id)
        if meta.mesh_topo and (int(meta.mesh_topo.get("face", -1)) >= 0 or int(meta.mesh_topo.get("vertex", -1)) >= 0):
            return dict(meta.mesh_topo)
        for msg in reversed(self.load_messages(chat_id)):
            if msg.role != "user":
                continue
            topo = dict(msg.mesh_topo or {})
            if topo and (int(topo.get("face", -1)) >= 0 or int(topo.get("vertex", -1)) >= 0):
                return topo
            break
        return {}

    def _mesh_path(self, chat_id: str, name: str) -> Path | None:
        if not (name or "").strip():
            return None
        try:
            path = self.resolve_file(chat_id, name)
        except FileNotFoundError:
            return None
        return path if path.is_file() else None

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
            if not path.is_file() or path.name.endswith(".preview.png"):
                continue
            files.append(self.artifact_from_path(chat_id, path))
        return files

    def mesh_preview_path(self, mesh_path: Path) -> Path:
        return mesh_path.with_name(f"{mesh_path.name}.preview.png")

    def ensure_mesh_preview(self, chat_id: str, mesh_path: Path, *, mesh: Any = None) -> Path:
        resolved = mesh_path if mesh_path.is_file() else self.resolve_file(chat_id, mesh_path.name)
        if resolved.suffix.lower() not in {".stl", ".obj", ".glb", ".gltf"}:
            raise FileNotFoundError("Not a mesh")
        dest = self.mesh_preview_path(resolved)
        if dest.is_file() and dest.stat().st_size > 0 and dest.stat().st_mtime >= resolved.stat().st_mtime:
            return dest
        from mesh_forge.render import render_mesh_preview

        render_mesh_preview(resolved, dest, size=384, mesh=mesh)
        if not dest.is_file() or dest.stat().st_size <= 0:
            raise FileNotFoundError("Preview render failed")
        return dest

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
