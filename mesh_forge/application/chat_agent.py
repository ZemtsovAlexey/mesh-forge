from __future__ import annotations

import base64
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

MAX_TOOL_ROUNDS = 4

AGENT_SYSTEM = """You are MeshForge, a local 3D mesh agent in chat (no UI buttons).
You see full chat history, notebook, and live pipeline.
Reply with ONLY valid JSON:
{
  "assistant_message": "short reply in the user's language",
  "tool_calls": [{"name": "tool_name", "args": {}}],
  "done": true
}

Tools:
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
- No "Generate/Next/Redo" buttons — YOU call run_*/continue/redo tools.
- If subject is clear for a new object: set_draft_prompt ready=true (starts front).
- If user says дальше/ок/продолжай → continue_pipeline.
- If user says переделай / сделай иначе → redo_step (optionally with new brief).
- Results appear as separate chat messages with images/mesh preview; never overwrite history.
- Image captions appear as [видение] in history — treat them as hints.
- When images are attached to this turn (mesh preview / views / photos), LOOK at them — that is what the mesh/object looks like. Prefer attached images over captions.
- Prefer one heavy run tool per turn.
"""


def _new_msg_id() -> str:
    return uuid.uuid4().hex[:10]


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


def _wants_continue(text: str) -> bool:
    lowered = (text or "").lower().strip()
    if lowered in {"ок", "окей", "ok", "okay", "да", "yes", "хорошо", "ага", "угу", "next", "go"}:
        return True
    return any(
        k in lowered
        for k in (
            "дальше",
            "далее",
            "продолж",
            "собери mesh",
            "сделай mesh",
            "в mesh",
            "проекц",
            "continue",
            "go on",
            "proceed",
        )
    )


def _wants_redo(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        k in lowered
        for k in ("переделай", "перегенер", "заново front", "redo", "ещё раз", "еще раз")
    )


def _wants_generate(text: str) -> bool:
    lowered = (text or "").lower().strip()
    return any(
        k in lowered
        for k in (
            "генерир",
            "сгенерир",
            "запускай",
            "запусти",
            "поехали",
            "generate",
            "run it",
            "start generation",
        )
    )


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
                    caption_images=False,
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

        if text or ref_ids:
            state.messages.append(
                ChatMessage(
                    id=_new_msg_id(),
                    role="user",
                    content=text or "(ссылка на сообщение)",
                    ref_ids=list(ref_ids),
                )
            )
            if text:
                if not state.user_prompt:
                    state.user_prompt = text
                else:
                    state.user_prompt = f"{state.user_prompt}\n{text}".strip()

        # Photos → caption for agent context, then start preview (no confirm button)
        if has_images and not has_mesh:
            self._attach_upload_captions(manifest, state)
            self.chat.save(manifest, state)
            obs = self._tool_run_photo_preview(manifest, state)
            if not self._results_posted:
                state.assistant_message = obs
                state.messages.append(ChatMessage(id=_new_msg_id(), role="assistant", content=obs))
            return self._finalize(manifest, state)

        # Photos with existing mesh: still caption so agent can reason about the reference.
        if has_images and has_mesh:
            self._attach_upload_captions(manifest, state)
            self.chat.save(manifest, state)

        forced = self._maybe_force_apply(manifest, state, text=text, ref_ids=ref_ids)
        if forced:
            # After apply-as-main, generate if create draft is ready
            if state.ready and state.intent == "create" and state.draft_prompt_en:
                self.chat.save(manifest, state)
                self._tool_run_front(manifest, state)
            elif state.ready and state.intent in {"geometry_edit", "guided_edit", "semantic_edit"}:
                self.chat.save(manifest, state)
                self._tool_run_pending_edit(manifest, state)
            return self._finalize(manifest, state)

        if self._maybe_force_pipeline(manifest, state, text=text):
            return self._finalize(manifest, state)

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

            for call in tool_calls:
                prog.raise_if_cancelled(manifest.id)
                if not isinstance(call, dict):
                    continue
                name = str(call.get("name") or "").strip()
                args = call.get("args") if isinstance(call.get("args"), dict) else {}
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
            if done or self._results_posted:
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
            # If we only prepared an internal draft and are about to leave ready without
            # having run front (shouldn't happen often), don't spam translation text.
            if state.ready and state.intent == "create" and state.draft_prompt_en:
                # Auto-run front instead of announcing the EN prompt.
                self.chat.save(manifest, state)
                self._tool_run_front(manifest, state)
                return self._finalize(manifest, state)
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

    def _maybe_force_pipeline(self, manifest: ProjectManifest, state: ChatState, *, text: str) -> bool:
        pipe = load_pipeline(manifest).to_dict()
        if _wants_continue(text) and pipe.get("can_continue"):
            self.chat.save(manifest, state)
            self._tool_continue(manifest, state)
            return True
        if _wants_redo(text) and pipe.get("can_redo"):
            step = "views" if pipe.get("step") == "views" else "front"
            if pipe.get("pipeline") == "photo_gated":
                step = "front"
            brief = None
            # If user also gave a correction, use as new brief
            if len((text or "").split()) > 2 and state.draft_prompt_en:
                brief = None
            self.chat.save(manifest, state)
            self._tool_redo(manifest, state, step=step, brief_en=brief)
            return True
        if (
            _wants_generate(text)
            and state.draft_prompt_en
            and state.intent == "create"
            and pipe.get("step") in {"", "idle", "done", None}
        ):
            self.chat.save(manifest, state)
            self._tool_run_front(manifest, state)
            return True
        if _wants_generate(text) and state.ready and state.intent in {
            "geometry_edit",
            "guided_edit",
            "semantic_edit",
        }:
            self.chat.save(manifest, state)
            self._tool_run_pending_edit(manifest, state)
            return True
        return False

    def _maybe_force_apply(
        self,
        manifest: ProjectManifest,
        state: ChatState,
        *,
        text: str,
        ref_ids: list[str],
    ) -> bool:
        if not ref_ids:
            return False
        lowered = (text or "").lower()
        wants_main = any(k in lowered for k in ("основн", "примени", "main", "apply"))
        wants_revise = any(k in lowered for k in ("передел", "revise", "на основан", "на основании"))
        if not wants_main and not wants_revise:
            return False
        mode = "revise" if wants_revise else "main"
        rid = ref_ids[0]
        msg = next((m for m in state.messages if m.id == rid), None)
        if msg:
            source = msg.content or ""
            self._apply_source(manifest, state, source, mode=mode, latest_text=text, label=f"msg:{rid}")
        else:
            entry = next((e for e in load_notebook(manifest) if e.id == rid), None)
            if not entry:
                return False
            source = (entry.brief_en or entry.user_prompt or entry.summary or entry.title).strip()
            self._apply_source(manifest, state, source, mode=mode, latest_text=text, label=f"nb:{rid}")
        draft = state.draft_prompt_en or state.edit_brief_en
        # Do not echo EN draft into chat — front/edit will translate & run.
        state.assistant_message = (
            "Применил как основу — запускаю."
            if mode == "main"
            else "Пересобрал на основе ссылки — запускаю."
        )
        state.status = "ready" if state.ready else state.status
        # Skip chat bubble if we immediately run a heavy tool (caller may generate).
        if not (state.ready and state.intent in {"create", "geometry_edit", "guided_edit", "semantic_edit"}):
            state.messages.append(
                ChatMessage(id=_new_msg_id(), role="assistant", content=state.assistant_message)
            )
        _ = draft
        return True

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

    def _collect_vision_images(
        self,
        manifest: ProjectManifest,
        state: ChatState,
    ) -> list[tuple[str, Path]]:
        """Pick recent mesh/view/photo images for multimodal agent decide."""
        found: list[tuple[str, Path]] = []
        seen: set[str] = set()

        def _add(label: str, path: Path) -> None:
            key = str(path.resolve()) if path.is_file() else ""
            if not key or key in seen:
                return
            seen.add(key)
            found.append((label, path))

        # Newest mesh_preview from chat artifacts
        for msg in reversed(state.messages):
            for art in msg.artifacts or []:
                if art.kind != "mesh_preview" or not art.path:
                    continue
                _add(f"mesh_preview:{art.label or 'preview'}", manifest.root / art.path)
            if any(lbl.startswith("mesh_preview:") for lbl, _ in found):
                break

        # Fresh render of current mesh if none in history
        if not any(lbl.startswith("mesh_preview:") for lbl, _ in found):
            mesh = manifest.current_mesh_path()
            if mesh and mesh.is_file():
                out = manifest.root / "work" / "chat_edit" / "agent_mesh_preview.png"
                try:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    render_mesh_preview(mesh, out)
                    _add("mesh_preview:current", out)
                except Exception as exc:
                    logger.warning("Agent mesh preview render failed: %s", exc)

        # Recent front / photo / upload views
        for msg in reversed(state.messages[-10:]):
            for art in msg.artifacts or []:
                if art.kind != "image" or not art.path:
                    continue
                label = (art.label or "").lower()
                if label in {"front", "preview", "photo", "photo_1", "image_1"} or art.stage in {
                    "front",
                    "photo",
                    "upload",
                }:
                    _add(f"image:{art.label or label}", manifest.root / art.path)
            if len(found) >= 4:
                break

        return found[:4]

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
                item["artifacts"] = [
                    {
                        "label": a.label,
                        "kind": a.kind,
                        "caption": a.caption,
                    }
                    for a in m.artifacts
                    if a.kind in {"image", "mesh_preview"}
                ]
            history.append(item)

        vision_images = self._collect_vision_images(manifest, state)
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
            "attached_images": [label for label, _ in vision_images],
        }
        user_blob = (
            "Context JSON:\n"
            + json.dumps(context, ensure_ascii=False)
            + "\n\nChat history:\n"
            + json.dumps(history, ensure_ascii=False)
            + "\n\nDecide next assistant_message and optional tool_calls."
        )
        if vision_images:
            user_blob += (
                "\n\nAttached images below are what you can SEE "
                "(mesh preview and/or views/photos). Use them as ground truth for shape/pose/defects."
            )

        try:
            if vision_images:
                content: list[dict[str, Any]] = [{"type": "text", "text": user_blob}]
                for label, path in vision_images:
                    raw = path.read_bytes()
                    b64 = base64.b64encode(raw).decode()
                    suffix = path.suffix.lower()
                    mime = {
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".webp": "image/webp",
                        ".png": "image/png",
                    }.get(suffix, "image/png")
                    content.append({"type": "text", "text": f"[image {label}]"})
                    content.append(
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                    )
                model = self.llm.config.llm.vision_model
                messages = [
                    {"role": "system", "content": AGENT_SYSTEM},
                    {"role": "user", "content": content},
                ]
            else:
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

    def _attach_upload_captions(self, manifest: ProjectManifest, state: ChatState) -> None:
        """Describe uploaded chat photos and attach captions to the latest user message."""
        upload_dir = manifest.root / "work" / "chat_uploads"
        if not upload_dir.is_dir():
            return
        paths = sorted([p for p in upload_dir.iterdir() if p.is_file()], key=lambda p: p.name)
        if not paths:
            return
        user_msg = next((m for m in reversed(state.messages) if m.role == "user"), None)
        if user_msg is None:
            return
        from mesh_forge.application.chat_results import _caption_path, _copy_into_media

        captions: list[str] = []
        arts: list[ChatArtifact] = list(user_msg.artifacts or [])
        for idx, src in enumerate(paths, start=1):
            label = f"photo_{idx}"
            try:
                rel = _copy_into_media(manifest, user_msg.id or _new_msg_id(), src, f"{label}{src.suffix or '.png'}")
            except Exception:
                rel = ""
            caption = _caption_path(src, label=label, kind="upload")
            if caption:
                captions.append(caption)
            if rel:
                arts.append(
                    ChatArtifact(
                        kind="image",
                        label=label,
                        path=rel,
                        stage="upload",
                        caption=caption,
                    )
                )
        user_msg.artifacts = arts
        if captions:
            block = "\n".join(f"- {c}" for c in captions)
            extra = f"\n\n[видение фото]\n{block}"
            if "[видение" not in (user_msg.content or ""):
                user_msg.content = f"{(user_msg.content or '').rstrip()}{extra}".strip()

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

        # Prefer the user's wording; English translation is done inside start_text_front.
        last_user = ""
        for m in reversed(state.messages):
            if m.role == "user" and (m.content or "").strip():
                last_user = m.content.strip()
                break
        source = (last_user or state.user_prompt or state.draft_prompt_en or "").strip()
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
