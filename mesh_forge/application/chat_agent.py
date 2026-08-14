from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from mesh_forge.adapters import LMStudioClient
from mesh_forge.application.chat_results import enrich_chat_payload, post_result_message
from mesh_forge.application.notebook import (
    add_note,
    load_notebook,
    notebook_payload,
    notebook_summary_for_llm,
)
from mesh_forge.application.pipeline_run import load_pipeline
from mesh_forge.application.prompt_chat import (
    ChatArtifact,
    ChatMessage,
    ChatState,
    PromptChatService,
    _format_assistant_with_questions,
    _looks_english,
)
from mesh_forge.manifest import ProjectManifest
from mesh_forge import progress as prog
from mesh_forge.progress import OperationCancelled
from mesh_forge.render import render_mesh_preview

logger = logging.getLogger("mesh_forge.chat_agent")

MAX_TOOL_ROUNDS = 5
MAX_LOOK_IMAGES = 4
MAX_LOOK_NOTE = 1200
LOOK_MESH_PREVIEW = "agent_look_mesh.png"
LOOK_TOOL_NAMES = {
    "look",
    "look_at",
    "look_at_mesh",
    "look_at_views",
    "look_at_photos",
    "look_at_front",
}

AGENT_SYSTEM = """You are MeshForge, a local 3D mesh agent in chat (no UI buttons).
You see full chat history, notebook, and live pipeline as TEXT only.
You do NOT see images or the mesh unless you call look.
Reply with ONLY valid JSON:
{
  "assistant_message": "short reply in the user's language",
  "tool_calls": [{"name": "tool_name", "args": {}}],
  "done": true
}

Tools:
- look {target?: "auto"|"mesh"|"front"|"views"|"photos", question?: str} — inspect visuals with the vision model. Use on a later turn when the user comments on appearance or attached photos. NEVER call look in the same turn as run_front / continue_pipeline / redo_step / run_photo_preview / run_pending_edit.
- ask_user {questions: [str]} — clarify only
- set_draft_prompt {prompt_en: str, ready?: bool} — internal EN subject for Comfy; if ready=true starts front.
  The EN text is NEVER shown to the user — translation is an internal front step.
- run_front {} — generate front (translates subject to English inside the pipeline)
- continue_pipeline {} — next step (views after front, mesh after views/photo)
- redo_step {step?: "front"|"views", brief_en?: str} — regenerate current gate (new chat message, old kept)
- run_photo_preview {} — photo→preview gate (needs uploaded photos)
- prepare_geometry_edit {instruction?: str} — plan cleanup then run it
- prepare_guided_edit {instruction?: str} — plan gentle edit then run it
- prepare_semantic_edit {instruction?: str} — plan full regen then run it
- run_pending_edit {} — execute prepared geometry/guided/semantic job
- apply_message {message_id: str, mode: "main"|"revise"}
- apply_notebook {entry_id: str, mode: "main"|"revise"}
- notebook_write {title: str, summary?: str, brief_en?: str}
- get_status {}
- noop {}

Rules:
- Never show English prompts, translations, or draft_prompt_en in assistant_message.
- assistant_message is only short user-language talk (questions / status). When starting front, keep it empty or one short line like «Генерирую…» — the front result message will appear separately.
- YOU decide every action via tool_calls from context + chat history. No shortcuts — read pipeline.can_continue / can_redo, visuals, refs, and the latest user message.
- New create: set_draft_prompt ready=true (starts front) when subject is clear. Do not also call run_front or look in that same turn.
- pipeline.can_continue + user wants next step → continue_pipeline. User asks to fix/redo current gate → redo_step (usually front). User attached photos (visuals.new_photos) → run_photo_preview.
- User message has ref_ids or asks to apply/revise a cited message/notebook → apply_message or apply_notebook, then run if ready.
- Results appear as separate chat messages with images/mesh preview; never overwrite history.
- After look, treat the observation as ground truth. It is saved on those images as message.look in history — reuse it. Call look again only if the latest result has no look (regenerated on a previous turn) or the user asks about a new visual detail.
- Set done=false after look so you can act on it.
- Prefer one heavy run tool per turn. After a generation tool, stop.
"""


def _new_msg_id() -> str:
    return uuid.uuid4().hex[:10]


def _clip_look_note(text: str, limit: int = MAX_LOOK_NOTE) -> str:
    note = (text or "").strip()
    if len(note) <= limit:
        return note
    return note[: limit - 1].rstrip() + "…"


def _path_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def _strip_en_draft_leak(message: str, draft_en: str) -> str:
    """Remove accidental English draft dumps from user-facing chat text."""
    msg = (message or "").strip()
    draft = (draft_en or "").strip()
    if not msg:
        return msg
    if draft and draft in msg:
        msg = msg.replace(draft, "").strip(" \n:-—")
    lowered = msg.lower()
    leak_markers = (
        "draft_prompt",
        "english prompt",
        "английский промпт",
        "промпт для comfy",
        "prepared english",
        "подготовил английский",
    )
    if any(m in lowered for m in leak_markers):
        # Keep only the first short sentence in the user's language if possible.
        first = msg.split("\n")[0].strip()
        if any(m in first.lower() for m in leak_markers):
            return "Генерирую…"
        return first
    return msg


class ChatAgentService:
    """LLM agent with tools; pipeline runs become durable chat result messages."""

    def __init__(self, llm: LMStudioClient | None = None) -> None:
        self.llm = llm or LMStudioClient()
        self.chat = PromptChatService(self.llm)
        self._results_posted = False

    def post_message(
        self,
        manifest: ProjectManifest,
        text: str,
        *,
        has_images: bool = False,
        ref_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        text = (text or "").strip()
        ref_ids = [str(r).strip() for r in (ref_ids or []) if str(r).strip()]
        if not text and not has_images and not ref_ids:
            raise ValueError("Message text is empty")

        self._results_posted = False
        state = self.chat.load(manifest)
        self._ensure_message_ids(state)
        has_mesh = manifest.current_mesh_path() is not None
        state.mode = "edit" if has_mesh else "create"
        prog.start(manifest.id, "chat", "Агент…")

        try:
            return self._post_message_inner(
                manifest,
                state,
                text=text,
                has_images=has_images,
                ref_ids=ref_ids,
                has_mesh=has_mesh,
            )
        except OperationCancelled:
            if not self._results_posted:
                post_result_message(
                    manifest,
                    "⏹ Остановлено.",
                    kind="text",
                )
                self._results_posted = True
            prog.finish(manifest.id, ok=False, error="Остановлено")
            return self._finalize(manifest, self.chat.load(manifest))
        except Exception:
            prog.finish(manifest.id, ok=False, error="Ошибка")
            raise
        finally:
            job = prog.get(manifest.id)
            if job and job.active:
                prog.finish(manifest.id, ok=True)

    def _post_message_inner(
        self,
        manifest: ProjectManifest,
        state: ChatState,
        *,
        text: str,
        has_images: bool,
        ref_ids: list[str],
        has_mesh: bool,
    ) -> dict[str, Any]:
        prog.raise_if_cancelled(manifest.id)

        if text or ref_ids or has_images:
            state.messages.append(
                ChatMessage(
                    id=_new_msg_id(),
                    role="user",
                    content=text or ("(фото)" if has_images else "(ссылка на сообщение)"),
                    ref_ids=list(ref_ids),
                )
            )
            if text:
                if not state.user_prompt:
                    state.user_prompt = text
                else:
                    state.user_prompt = f"{state.user_prompt}\n{text}".strip()

        if has_images:
            self._attach_upload_images(manifest, state)
            self.chat.save(manifest, state)

        tool_trace: list[dict[str, Any]] = []
        for _round in range(MAX_TOOL_ROUNDS):
            prog.raise_if_cancelled(manifest.id)
            decision = self._decide(manifest, state, tool_trace)
            assistant_message = str(decision.get("assistant_message") or "").strip()
            tool_calls = decision.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                tool_calls = []
            done = bool(decision.get("done", True)) or not tool_calls

            if not tool_calls:
                if assistant_message:
                    state.assistant_message = assistant_message
                elif not state.assistant_message:
                    state.assistant_message = "Чем помочь с моделью?"
                break

            ran_look = False
            for call in tool_calls:
                prog.raise_if_cancelled(manifest.id)
                if not isinstance(call, dict):
                    continue
                name = str(call.get("name") or "").strip()
                args = call.get("args") if isinstance(call.get("args"), dict) else {}
                if self._results_posted:
                    tool_trace.append(
                        {
                            "name": name,
                            "args": args,
                            "result": "skipped: generation already posted",
                        }
                    )
                    continue
                if name.lower() in LOOK_TOOL_NAMES:
                    ran_look = True
                # Persist draft/intent before long Comfy runs
                self.chat.save(manifest, state)
                obs = self._run_tool(manifest, state, name, args, latest_text=text)
                tool_trace.append({"name": name, "args": args, "result": obs})
                # Reload state after tools that save via pipeline helpers
                state = self.chat.load(manifest)

            if assistant_message and not self._results_posted:
                state.assistant_message = _strip_en_draft_leak(
                    assistant_message,
                    state.draft_prompt_en,
                )
            if self._results_posted:
                break
            if done and not ran_look:
                break

        if not self._results_posted:
            if not state.assistant_message:
                state.assistant_message = "Готово." if state.ready else "Уточните, пожалуйста."
            if state.questions and not state.ready:
                state.assistant_message = _format_assistant_with_questions(
                    state.assistant_message,
                    state.questions,
                    ready=False,
                    draft="",
                )
            # Never dump EN draft into the chat bubble.
            state.assistant_message = _strip_en_draft_leak(
                state.assistant_message,
                state.draft_prompt_en,
            )
            state.status = "ready" if state.ready else ("clarifying" if state.questions else state.status or "idle")
            state.messages.append(
                ChatMessage(id=_new_msg_id(), role="assistant", content=state.assistant_message)
            )
        return self._finalize(manifest, state)

    def restart_from_message(self, manifest: ProjectManifest, message_id: str) -> dict[str, Any]:
        """Truncate chat to message_id and re-run from that user turn (or nearest prior user)."""
        from mesh_forge.application.pipeline_run import PipelineRunState, save_pipeline

        mid = (message_id or "").strip()
        if not mid:
            raise ValueError("message_id is empty")

        state = self.chat.load(manifest)
        self._ensure_message_ids(state)
        idx = next((i for i, m in enumerate(state.messages) if m.id == mid), -1)
        if idx < 0:
            raise ValueError(f"Message not found: {mid}")

        # Find the user message at or before the clicked one — that becomes the restart point.
        user_idx = idx
        while user_idx >= 0 and state.messages[user_idx].role != "user":
            user_idx -= 1
        if user_idx < 0:
            raise ValueError("Нет пользовательского сообщения для перезапуска")

        target = state.messages[user_idx]
        kept = state.messages[:user_idx]  # history before the turn we re-run
        text = (target.content or "").strip()
        refs = list(target.ref_ids or [])
        if not text and not refs:
            raise ValueError("Пустое сообщение — нечего перезапускать")

        state.messages = kept
        state.ready = False
        state.status = "idle"
        state.questions = []
        state.assistant_message = ""
        state.draft_prompt_en = ""
        state.edit_brief_en = ""
        state.planned_ops = []
        state.intent = "create"
        # Rebuild user_prompt from remaining user texts (optional context accumulation).
        user_bits = [m.content.strip() for m in kept if m.role == "user" and m.content.strip()]
        state.user_prompt = "\n".join(user_bits)
        self.chat.save(manifest, state)

        # Drop active pipeline gate so the re-run starts clean.
        save_pipeline(manifest, PipelineRunState())

        add_note(
            manifest,
            kind="note",
            title="Перезапуск чата",
            summary=f"С сообщения {target.id}: {(text or '')[:120]}",
            user_prompt=text,
        )

        return self.post_message(manifest, text, ref_ids=refs)

    def _finalize(self, manifest: ProjectManifest, state: ChatState) -> dict[str, Any]:
        if self._results_posted:
            # Pipeline helpers already wrote chat.json — don't clobber with stale in-memory state.
            state = self.chat.load(manifest)
        else:
            self.chat.save(manifest, state)
        data = state.to_dict()
        data["notebook"] = notebook_payload(manifest)
        from mesh_forge.application.stepped_pipeline import pipeline_payload

        data["pipeline"] = pipeline_payload(manifest)
        return enrich_chat_payload(manifest, data)

    def _ensure_message_ids(self, state: ChatState) -> None:
        for msg in state.messages:
            if not msg.id:
                msg.id = _new_msg_id()

    def _resolve_refs(
        self,
        state: ChatState,
        manifest: ProjectManifest,
        ref_ids: list[str],
    ) -> list[dict[str, str]]:
        by_id = {m.id: m for m in state.messages if m.id}
        out: list[dict[str, str]] = []
        for rid in ref_ids:
            if rid in by_id:
                m = by_id[rid]
                out.append({"id": rid, "label": f"chat:{m.role}", "content": (m.content or "")[:500]})
                continue
            for entry in load_notebook(manifest):
                if entry.id == rid:
                    content = entry.brief_en or entry.summary or entry.title
                    out.append({"id": rid, "label": f"notebook:{entry.kind}", "content": content[:500]})
                    break
        return out

    def _visual_inventory(
        self,
        manifest: ProjectManifest,
        state: ChatState,
    ) -> dict[str, Any]:
        """What the look tool can inspect right now (paths stay server-side)."""
        views: list[str] = []
        photos: list[str] = []
        has_front = False
        has_mesh_preview = False
        looked = {"mesh": False, "front": False, "views": False, "photos": False}
        seen_views_msg = False
        seen_mesh_msg = False
        seen_front = False
        seen_photos = False
        for msg in reversed(state.messages):
            view_arts: list[ChatArtifact] = []
            mesh_arts: list[ChatArtifact] = []
            front_arts: list[ChatArtifact] = []
            photo_arts: list[ChatArtifact] = []
            for art in msg.artifacts or []:
                label = (art.label or "").lower()
                if art.kind in {"mesh_preview", "mesh"}:
                    mesh_arts.append(art)
                    has_mesh_preview = has_mesh_preview or art.kind == "mesh_preview"
                elif art.kind == "image":
                    if label == "front" or art.stage == "front":
                        has_front = True
                        front_arts.append(art)
                    if label in {"front", "left", "back", "right"} or art.stage == "views":
                        if label not in views:
                            views.append(label or art.stage)
                        view_arts.append(art)
                    if art.stage in {"upload", "photo"} or label.startswith("photo") or label in {
                        "preview",
                        "photo",
                    }:
                        if (art.label or label) not in photos:
                            photos.append(art.label or label)
                        photo_arts.append(art)
            if not seen_views_msg and view_arts:
                looked["views"] = any(bool((a.caption or "").strip()) for a in view_arts)
                seen_views_msg = True
            if not seen_mesh_msg and mesh_arts:
                looked["mesh"] = any(bool((a.caption or "").strip()) for a in mesh_arts)
                seen_mesh_msg = True
            if not seen_front and front_arts:
                looked["front"] = any(bool((a.caption or "").strip()) for a in front_arts)
                seen_front = True
            if not seen_photos and photo_arts:
                looked["photos"] = any(bool((a.caption or "").strip()) for a in photo_arts)
                seen_photos = True
        latest_user = next((m for m in reversed(state.messages) if m.role == "user"), None)
        new_photos = [
            a.label or a.path
            for a in (latest_user.artifacts if latest_user else [])
            if a.kind == "image" and a.stage == "upload"
        ]
        mesh = manifest.current_mesh_path()
        has_mesh = bool(mesh and mesh.is_file()) or has_mesh_preview
        if has_mesh and not seen_mesh_msg:
            looked["mesh"] = False
        return {
            "mesh": has_mesh,
            "front": has_front,
            "views": views[:4],
            "photos": photos[:4],
            "new_photos": new_photos,
            "looked": looked,
        }

    def _pending_upload_paths(self, manifest: ProjectManifest) -> list[Path]:
        upload_dir = manifest.root / "work" / "chat_uploads"
        if not upload_dir.is_dir():
            return []
        return sorted([p for p in upload_dir.iterdir() if p.is_file()], key=lambda p: p.name)

    def _artifact_abs(self, manifest: ProjectManifest, art: ChatArtifact) -> Path | None:
        if not art.path:
            return None
        path = manifest.root / art.path
        return path if path.is_file() else None

    def _recent_chat_images(
        self,
        manifest: ProjectManifest,
        state: ChatState,
        *,
        kinds: set[str] | None = None,
        labels: set[str] | None = None,
        stages: set[str] | None = None,
        limit: int = MAX_LOOK_IMAGES,
        one_message: bool = False,
    ) -> list[tuple[str, Path]]:
        found: list[tuple[str, Path]] = []
        seen: set[str] = set()
        for msg in reversed(state.messages):
            msg_hits = 0
            for art in msg.artifacts or []:
                if kinds and art.kind not in kinds:
                    continue
                label = (art.label or "").lower()
                if labels and label not in labels:
                    continue
                if stages and (art.stage or "") not in stages and label not in stages:
                    continue
                path = self._artifact_abs(manifest, art)
                if path is None:
                    continue
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                found.append((art.label or label or art.kind, path))
                msg_hits += 1
                if len(found) >= limit:
                    return found
            if one_message and msg_hits:
                return found
        return found

    def _resolve_look_images(
        self,
        manifest: ProjectManifest,
        state: ChatState,
        target: str,
    ) -> tuple[str, list[tuple[str, Path]]]:
        wanted = (target or "auto").strip().lower()
        if wanted not in {"auto", "mesh", "front", "views", "photos"}:
            wanted = "auto"
        inventory = self._visual_inventory(manifest, state)
        pending = self._pending_upload_paths(manifest)
        combine_mesh_photos = False

        if wanted == "auto":
            if inventory.get("new_photos") and inventory["mesh"]:
                wanted = "mesh"
                combine_mesh_photos = True
            elif inventory["mesh"]:
                wanted = "mesh"
            elif inventory["views"]:
                wanted = "views"
            elif inventory["front"]:
                wanted = "front"
            elif inventory.get("new_photos") or inventory["photos"] or pending:
                wanted = "photos"
            else:
                wanted = "mesh"

        images: list[tuple[str, Path]] = []
        if wanted == "mesh":
            mesh = manifest.current_mesh_path()
            if mesh and mesh.is_file():
                out = manifest.root / "work" / "chat_edit" / "agent_look_mesh.png"
                try:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    render_mesh_preview(mesh, out)
                    images.append(("mesh", out))
                except Exception as exc:
                    logger.warning("Look mesh preview failed: %s", exc)
            if not images:
                images = self._recent_chat_images(
                    manifest, state, kinds={"mesh_preview"}, limit=1
                )
            if combine_mesh_photos:
                room = MAX_LOOK_IMAGES - len(images)
                if room > 0:
                    images.extend(self._photo_look_images(manifest, state, pending, limit=room))
                wanted = "auto"
        elif wanted == "front":
            images = self._recent_chat_images(
                manifest,
                state,
                kinds={"image"},
                labels={"front"},
                limit=1,
            )
            if not images:
                images = self._recent_chat_images(
                    manifest, state, kinds={"image"}, stages={"front"}, limit=1
                )
        elif wanted == "views":
            images = self._recent_chat_images(
                manifest,
                state,
                kinds={"image"},
                labels={"front", "left", "back", "right"},
                limit=MAX_LOOK_IMAGES,
                one_message=True,
            )
            if not images:
                images = self._recent_chat_images(
                    manifest,
                    state,
                    kinds={"image"},
                    stages={"views", "front"},
                    limit=MAX_LOOK_IMAGES,
                    one_message=True,
                )
        else:  # photos
            images = self._photo_look_images(manifest, state, pending, limit=MAX_LOOK_IMAGES)

        return wanted, images[:MAX_LOOK_IMAGES]

    def _photo_look_images(
        self,
        manifest: ProjectManifest,
        state: ChatState,
        pending: list[Path],
        *,
        limit: int,
    ) -> list[tuple[str, Path]]:
        latest_user = next((m for m in reversed(state.messages) if m.role == "user"), None)
        from_msg: list[tuple[str, Path]] = []
        if latest_user:
            for art in latest_user.artifacts or []:
                if art.kind != "image" or art.stage != "upload":
                    continue
                path = self._artifact_abs(manifest, art)
                if path is None:
                    continue
                from_msg.append((art.label or path.name, path))
                if len(from_msg) >= limit:
                    return from_msg
        if from_msg:
            return from_msg[:limit]
        images: list[tuple[str, Path]] = []
        for idx, src in enumerate(pending, start=1):
            if len(images) >= limit:
                return images
            images.append((f"photo_{idx}", src))
        if len(images) < limit:
            images.extend(
                self._recent_chat_images(
                    manifest,
                    state,
                    kinds={"image"},
                    stages={"upload", "photo"},
                    limit=limit - len(images),
                )
            )
        return images[:limit]

    def _tool_look(
        self,
        manifest: ProjectManifest,
        state: ChatState,
        *,
        target: str,
        question: str,
    ) -> str:
        resolved, images = self._resolve_look_images(manifest, state, target)
        available = self._visual_inventory(manifest, state)
        if not images:
            return json.dumps(
                {"error": "nothing to look at", "target": resolved, "available": available},
                ensure_ascii=False,
            )
        labels = {
            "mesh": "mesh",
            "front": "front",
            "views": "проекции",
            "photos": "фото",
            "auto": "кадры",
        }
        prog.update(manifest.id, 8, f"Смотрю на {labels.get(resolved, resolved)}…")
        try:
            observation = self.llm.inspect_images(images, question=question)
        except Exception as exc:
            logger.warning("Look failed: %s", exc)
            return json.dumps(
                {"error": str(exc), "target": resolved, "seen": [label for label, _ in images]},
                ensure_ascii=False,
            )
        if not observation:
            return json.dumps(
                {"error": "vision returned empty", "target": resolved, "seen": [label for label, _ in images]},
                ensure_ascii=False,
            )
        note = _clip_look_note(observation)
        self._remember_look(manifest, state, images, note)
        self.chat.save(manifest, state)
        return json.dumps(
            {
                "target": resolved,
                "seen": [label for label, _ in images],
                "observation": note,
                "cached": True,
            },
            ensure_ascii=False,
        )

    def _remember_look(
        self,
        manifest: ProjectManifest,
        state: ChatState,
        images: list[tuple[str, Path]],
        note: str,
    ) -> None:
        """Bind a look observation to the artifacts that were actually inspected."""
        if not note:
            return
        looked_keys: set[str] = set()
        mesh_looked = False
        photos_looked = False
        for label, path in images:
            looked_keys.add(_path_key(path))
            if label == "mesh" or path.name == LOOK_MESH_PREVIEW:
                mesh_looked = True
            if label.startswith("photo") or "chat_uploads" in path.parts:
                photos_looked = True

        for msg in state.messages:
            for art in msg.artifacts or []:
                if art.kind not in {"image", "mesh_preview", "mesh"}:
                    continue
                abs_path = self._artifact_abs(manifest, art)
                if abs_path is None:
                    continue
                if _path_key(abs_path) in looked_keys:
                    art.caption = note

        if mesh_looked:
            for msg in reversed(state.messages):
                targets = [a for a in (msg.artifacts or []) if a.kind == "mesh_preview"]
                if not targets:
                    targets = [a for a in (msg.artifacts or []) if a.kind == "mesh"]
                if targets:
                    for art in targets:
                        art.caption = note
                    break

        if photos_looked:
            latest_user = next((m for m in reversed(state.messages) if m.role == "user"), None)
            if latest_user:
                for art in latest_user.artifacts or []:
                    if art.kind == "image" and art.stage == "upload":
                        art.caption = note

    def _decide(
        self,
        manifest: ProjectManifest,
        state: ChatState,
        tool_trace: list[dict[str, Any]],
    ) -> dict[str, Any]:
        has_mesh = manifest.current_mesh_path() is not None
        pipe = load_pipeline(manifest)
        pipe_data = pipe.to_dict()
        history = []
        for m in state.messages[-24:]:
            item: dict[str, Any] = {
                "role": m.role,
                "content": m.content,
                "id": m.id,
                "kind": m.kind,
            }
            if m.ref_ids:
                item["ref_ids"] = m.ref_ids
            if m.artifacts:
                arts_out: list[dict[str, str]] = []
                looks: list[str] = []
                for a in m.artifacts:
                    if a.kind not in {"image", "mesh_preview", "mesh"}:
                        continue
                    arts_out.append({"label": a.label, "kind": a.kind})
                    cap = (a.caption or "").strip()
                    if cap and cap not in looks:
                        looks.append(cap)
                if arts_out:
                    item["artifacts"] = arts_out
                if looks:
                    item["look"] = looks[0] if len(looks) == 1 else looks
            history.append(item)

        latest_user = next((m for m in reversed(state.messages) if m.role == "user"), None)
        context = {
            "has_mesh": has_mesh,
            "mode": state.mode,
            "intent": state.intent,
            "ready": state.ready,
            "draft_prompt_en": state.draft_prompt_en,
            "edit_brief_en": state.edit_brief_en,
            "pipeline": {
                "step": pipe_data.get("step"),
                "status": pipe_data.get("status"),
                "brief_en": pipe_data.get("brief_en"),
                "message": pipe_data.get("message"),
                "can_continue": pipe_data.get("can_continue"),
                "can_redo": pipe_data.get("can_redo"),
            },
            "notebook": notebook_summary_for_llm(manifest),
            "tool_results": tool_trace[-6:],
            "visuals": self._visual_inventory(manifest, state),
        }
        if latest_user and latest_user.ref_ids:
            context["refs"] = self._resolve_refs(state, manifest, latest_user.ref_ids)
        user_blob = (
            "Context JSON:\n"
            + json.dumps(context, ensure_ascii=False)
            + "\n\nChat history:\n"
            + json.dumps(history, ensure_ascii=False)
            + "\n\nDecide next assistant_message and optional tool_calls."
            + " Use pipeline.can_continue / can_redo and visuals.new_photos — no keyword shortcuts."
            + " Call look only when the user asks about appearance or attached photos."
            + " Do not look at a result you just generated; wait for the next user message."
            + " message.look is a cached observation; visuals.looked says whether the latest files already have one."
            + " If context.refs is set, the user cited prior messages/notebook — use apply_message or apply_notebook."
        )

        try:
            model = self.llm.config.llm.planner_model
            messages = [
                {"role": "system", "content": AGENT_SYSTEM},
                {"role": "user", "content": user_blob},
            ]

            text = self.llm.chat(model, messages, temperature=0.2)
            from mesh_forge.backends.lmstudio import _parse_json_response

            data = _parse_json_response(text)
            if isinstance(data, dict) and not data.get("parse_error"):
                return data
        except Exception as exc:
            logger.warning("Agent decide failed: %s", exc)

        if not has_mesh:
            try:
                result = self.llm.clarify_or_enhance(
                    [{"role": m.role, "content": m.content} for m in state.messages if m.role in {"user", "assistant"}]
                )
                draft = str(result.get("draft_prompt_en") or "").strip()
                calls = []
                if draft and result.get("ready"):
                    calls.append({"name": "set_draft_prompt", "args": {"prompt_en": draft, "ready": True}})
                    reply = ""  # front result message will appear; no EN dump
                elif draft:
                    calls.append({"name": "set_draft_prompt", "args": {"prompt_en": draft, "ready": False}})
                    reply = str(result.get("assistant_message") or "").strip() or "Ок."
                else:
                    reply = str(result.get("assistant_message") or "").strip() or "Ок."
                questions = [str(q) for q in (result.get("questions") or []) if str(q).strip()][:3]
                if questions and not (draft and result.get("ready")):
                    calls.append({"name": "ask_user", "args": {"questions": questions}})
                return {
                    "assistant_message": _strip_en_draft_leak(reply, draft),
                    "tool_calls": calls,
                    "done": True,
                }
            except Exception as exc:
                logger.warning("Fallback clarify failed: %s", exc)
        return {
            "assistant_message": "Не удалось обработать запрос. Попробуйте переформулировать.",
            "tool_calls": [],
            "done": True,
        }

    def _run_tool(
        self,
        manifest: ProjectManifest,
        state: ChatState,
        name: str,
        args: dict[str, Any],
        *,
        latest_text: str,
    ) -> str:
        name = (name or "").strip().lower()
        try:
            if name in {"noop", ""}:
                return "ok"
            if name in LOOK_TOOL_NAMES:
                aliases = {
                    "look_at_mesh": "mesh",
                    "look_at_views": "views",
                    "look_at_photos": "photos",
                    "look_at_front": "front",
                }
                target = aliases.get(name) or str(args.get("target") or "auto").strip()
                question = str(args.get("question") or args.get("focus") or latest_text or "").strip()
                return self._tool_look(manifest, state, target=target, question=question)
            if name == "ask_user":
                qs = [str(q).strip() for q in (args.get("questions") or []) if str(q).strip()][:3]
                state.questions = qs
                state.ready = False
                state.status = "clarifying"
                return f"questions={len(qs)}"
            if name == "set_draft_prompt":
                prompt = str(args.get("prompt_en") or "").strip()
                if not prompt:
                    return "error: empty prompt_en"
                if not _looks_english(prompt):
                    prompt = self.llm.ensure_english_subject(prompt)
                state.intent = "create"
                state.mode = "create"
                state.draft_prompt_en = prompt
                state.edit_brief_en = ""
                state.planned_ops = []
                state.questions = []
                ready = bool(args.get("ready", True)) and bool(prompt)
                state.ready = ready
                state.status = "ready" if ready else "clarifying"
                add_note(
                    manifest,
                    kind="draft",
                    title="Черновик промпта",
                    brief_en=prompt,
                    user_prompt=state.user_prompt,
                    summary="set_draft_prompt",
                )
                self.chat.save(manifest, state)
                if ready:
                    return self._tool_run_front(manifest, state)
                return f"draft set ready={ready}"
            if name == "run_front":
                return self._tool_run_front(manifest, state)
            if name == "continue_pipeline":
                return self._tool_continue(manifest, state)
            if name == "redo_step":
                step = str(args.get("step") or "").strip() or None
                brief = str(args.get("brief_en") or "").strip() or None
                return self._tool_redo(manifest, state, step=step, brief_en=brief)
            if name == "run_photo_preview":
                return self._tool_run_photo_preview(manifest, state)
            if name == "prepare_geometry_edit":
                instruction = str(args.get("instruction") or latest_text or state.user_prompt).strip()
                from mesh_forge.mesh_qc import analyze_mesh

                mesh = manifest.current_mesh_path()
                if not mesh or not mesh.is_file():
                    return "error: no mesh"
                stats = analyze_mesh(mesh)
                plan = self.llm.plan_edit(instruction, stats.to_dict())
                operations = list(plan.get("operations") or []) or [
                    {"op": "remove_needles"},
                    {"op": "smooth", "iterations": 2},
                    {"op": "fill_holes"},
                ]
                summary = str(plan.get("summary") or "").strip() or "Geometry cleanup."
                state.intent = "geometry_edit"
                state.mode = "edit"
                state.edit_brief_en = summary
                state.draft_prompt_en = ""
                state.planned_ops = operations
                state.ready = True
                state.status = "ready"
                state.questions = []
                self.chat.save(manifest, state)
                return self._tool_run_pending_edit(manifest, state)
            if name == "prepare_guided_edit":
                instruction = str(args.get("instruction") or latest_text or "").strip()
                data = self.chat._handle_guided_edit(manifest, state, instruction)
                loaded = ChatState.from_dict(data)
                self._copy_edit_state(state, loaded)
                if state.messages and state.messages[-1].role == "assistant":
                    state.messages.pop()
                self.chat.save(manifest, state)
                return self._tool_run_pending_edit(manifest, state)
            if name == "prepare_semantic_edit":
                instruction = str(args.get("instruction") or latest_text or "").strip()
                data = self.chat._handle_semantic_edit(manifest, state, instruction)
                loaded = ChatState.from_dict(data)
                self._copy_edit_state(state, loaded)
                if state.messages and state.messages[-1].role == "assistant":
                    state.messages.pop()
                self.chat.save(manifest, state)
                return self._tool_run_pending_edit(manifest, state)
            if name == "run_pending_edit":
                return self._tool_run_pending_edit(manifest, state)
            if name == "apply_message":
                mid = str(args.get("message_id") or "").strip()
                mode = str(args.get("mode") or "main").strip().lower()
                msg = next((m for m in state.messages if m.id == mid), None)
                if not msg:
                    return f"error: message {mid} not found"
                return self._apply_source(
                    manifest, state, msg.content or "", mode=mode, latest_text=latest_text, label=f"msg:{mid}"
                )
            if name == "apply_notebook":
                eid = str(args.get("entry_id") or "").strip()
                mode = str(args.get("mode") or "main").strip().lower()
                entry = next((e for e in load_notebook(manifest) if e.id == eid), None)
                if not entry:
                    return f"error: notebook {eid} not found"
                source = (entry.brief_en or entry.user_prompt or entry.summary or entry.title).strip()
                return self._apply_source(
                    manifest, state, source, mode=mode, latest_text=latest_text, label=f"nb:{eid}"
                )
            if name == "notebook_write":
                title = str(args.get("title") or "Заметка").strip()
                entry = add_note(
                    manifest,
                    kind="note",
                    title=title,
                    summary=str(args.get("summary") or "").strip(),
                    brief_en=str(args.get("brief_en") or "").strip(),
                    user_prompt=state.user_prompt,
                )
                return f"noted {entry.id}"
            if name == "get_status":
                pipe = load_pipeline(manifest).to_dict()
                return json.dumps(
                    {
                        "pipeline_step": pipe.get("step"),
                        "pipeline_status": pipe.get("status"),
                        "can_continue": pipe.get("can_continue"),
                        "can_redo": pipe.get("can_redo"),
                        "notebook": notebook_summary_for_llm(manifest, limit=6),
                        "ready": state.ready,
                        "intent": state.intent,
                    },
                    ensure_ascii=False,
                )
            return f"error: unknown tool {name}"
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            return f"error: {exc}"

    def _attach_upload_images(self, manifest: ProjectManifest, state: ChatState) -> None:
        """Copy uploaded chat photos onto the latest user message (no vision caption)."""
        paths = self._pending_upload_paths(manifest)
        if not paths:
            return
        user_msg = next((m for m in reversed(state.messages) if m.role == "user"), None)
        if user_msg is None:
            return
        from mesh_forge.application.chat_results import _copy_into_media

        arts: list[ChatArtifact] = list(user_msg.artifacts or [])
        existing = {(a.label, a.path) for a in arts}
        for idx, src in enumerate(paths, start=1):
            label = f"photo_{idx}"
            try:
                rel = _copy_into_media(
                    manifest, user_msg.id or _new_msg_id(), src, f"{label}{src.suffix or '.png'}"
                )
            except Exception:
                rel = ""
            if rel and (label, rel) not in existing:
                arts.append(
                    ChatArtifact(
                        kind="image",
                        label=label,
                        path=rel,
                        stage="upload",
                    )
                )
        user_msg.artifacts = arts

    def _copy_edit_state(self, state: ChatState, loaded: ChatState) -> None:
        state.intent = loaded.intent
        state.mode = loaded.mode
        state.draft_prompt_en = loaded.draft_prompt_en
        state.edit_brief_en = loaded.edit_brief_en
        state.ready = loaded.ready
        state.status = loaded.status
        state.planned_ops = loaded.planned_ops
        state.questions = loaded.questions
        state.assistant_message = loaded.assistant_message
        state.messages = loaded.messages

    def _tool_run_front(self, manifest: ProjectManifest, state: ChatState) -> str:
        from mesh_forge.application.stepped_pipeline import start_text_front

        last_user = ""
        for m in reversed(state.messages):
            if m.role == "user" and (m.content or "").strip():
                last_user = m.content.strip()
                break
        # Prefer the internal EN subject; raw chat often contains purpose words
        # ("для 3d печати") that SD paints as a scene.
        source = (state.draft_prompt_en or last_user or state.user_prompt or "").strip()
        if not source:
            return "error: empty subject"
        pipe = start_text_front(
            manifest,
            brief_en=source,
            user_prompt=state.user_prompt or last_user or source,
            solidify_mm=0.0,
        )
        # Keep translated brief only as internal state (not shown as its own chat turn).
        if pipe.brief_en:
            state.draft_prompt_en = pipe.brief_en
        self._results_posted = True
        state.ready = False
        state.status = "pipeline"
        return f"front status={pipe.status} quality_ok={pipe.quality_ok}"

    def _tool_continue(self, manifest: ProjectManifest, state: ChatState) -> str:
        from mesh_forge.application.stepped_pipeline import continue_pipeline

        pipe = continue_pipeline(manifest)
        self._results_posted = True
        state.ready = False
        state.status = "done" if pipe.step == "done" else "pipeline"
        return f"continued to step={pipe.step} status={pipe.status}"

    def _tool_redo(
        self,
        manifest: ProjectManifest,
        state: ChatState,
        *,
        step: str | None = None,
        brief_en: str | None = None,
    ) -> str:
        from mesh_forge.application.stepped_pipeline import redo_step

        if brief_en:
            state.draft_prompt_en = brief_en
            self.chat.save(manifest, state)
        pipe = redo_step(manifest, step=step or "front", brief_en=brief_en)
        self._results_posted = True
        state.ready = False
        state.status = "pipeline"
        return f"redid step={pipe.step} status={pipe.status}"

    def _tool_run_photo_preview(self, manifest: ProjectManifest, state: ChatState) -> str:
        from mesh_forge.application.stepped_pipeline import start_photo_gate

        upload_dir = manifest.root / "work" / "chat_uploads"
        paths = sorted([p for p in upload_dir.iterdir() if p.is_file()], key=lambda p: p.name) if upload_dir.is_dir() else []
        if not paths:
            return "error: no uploaded photos"
        pipe = start_photo_gate(
            manifest,
            paths,
            user_prompt=state.user_prompt or "",
            solidify_mm=0.0,
            remove_bg=True,
        )
        self._results_posted = True
        state.ready = False
        state.status = "pipeline"
        state.intent = "create"
        return f"photo preview status={pipe.status}"

    def _tool_run_pending_edit(self, manifest: ProjectManifest, state: ChatState) -> str:
        from api.deps import get_orchestrator
        from api.services import confirm_project_chat

        self.chat.save(manifest, state)
        result = confirm_project_chat(get_orchestrator(), manifest)
        mesh = manifest.current_mesh_path()
        msg = result.message if hasattr(result, "message") else str(result)
        post_result_message(
            manifest,
            msg or "Правка готова.",
            mesh_path=mesh,
            kind="edit",
        )
        self._results_posted = True
        return f"edit done: {msg}"

    def _apply_source(
        self,
        manifest: ProjectManifest,
        state: ChatState,
        source: str,
        *,
        mode: str,
        latest_text: str,
        label: str,
    ) -> str:
        if not source:
            return "error: empty source"
        has_mesh = manifest.current_mesh_path() is not None
        if mode == "revise" and latest_text:
            combined = f"{source}\n\nRevision: {latest_text}"
        else:
            combined = source
        if has_mesh and mode == "revise":
            brief = self.llm.ensure_english_subject(combined[:400])
            state.intent = "guided_edit"
            state.mode = "edit"
            state.edit_brief_en = brief
            state.draft_prompt_en = brief
        else:
            prompt = self.llm.ensure_english_subject(
                combined.split("\n")[0][:300] if mode == "main" else combined[:400]
            )
            if not _looks_english(prompt):
                prompt = self.chat._fallback_draft_en(combined)
            state.intent = "create" if not has_mesh else "semantic_edit"
            state.mode = "create" if not has_mesh else "edit"
            if state.intent == "create":
                state.draft_prompt_en = prompt
                state.edit_brief_en = ""
            else:
                state.edit_brief_en = prompt
                state.draft_prompt_en = prompt
        state.ready = True
        state.status = "ready"
        state.questions = []
        state.planned_ops = []
        add_note(
            manifest,
            kind="message_apply",
            title=f"Применено {label} ({mode})",
            brief_en=state.draft_prompt_en or state.edit_brief_en,
            user_prompt=latest_text or source[:200],
            summary=source[:200],
        )
        return f"applied {label} mode={mode} intent={state.intent}"
