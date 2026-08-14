from __future__ import annotations

import base64
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mesh_forge.adapters import LMStudioClient
from mesh_forge.manifest import ProjectManifest
from mesh_forge.render import render_mesh_preview

logger = logging.getLogger("mesh_forge.prompt_chat")


def _new_message_id() -> str:
    return uuid.uuid4().hex[:10]


@dataclass
class ChatArtifact:
    kind: str  # image | mesh_preview | mesh
    label: str
    path: str  # relative to project root
    stage: str = ""
    caption: str = ""  # cached look note for this image/mesh version

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "path": self.path,
            "stage": self.stage,
            "caption": self.caption,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatArtifact":
        return cls(
            kind=str(data.get("kind") or "image"),
            label=str(data.get("label") or "file"),
            path=str(data.get("path") or ""),
            stage=str(data.get("stage") or ""),
            caption=str(data.get("caption") or ""),
        )


@dataclass
class ChatMessage:
    role: str
    content: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    id: str = field(default_factory=_new_message_id)
    kind: str = "text"  # text | system | status | front | views | photo | mesh | result | edit
    ref_ids: list[str] = field(default_factory=list)
    artifacts: list[ChatArtifact] = field(default_factory=list)


@dataclass
class ChatState:
    messages: list[ChatMessage] = field(default_factory=list)
    mode: str = "create"  # create | edit
    status: str = "idle"  # idle | clarifying | ready
    intent: str = "create"
    draft_prompt_en: str = ""
    edit_brief_en: str = ""
    user_prompt: str = ""
    ready: bool = False
    questions: list[str] = field(default_factory=list)
    assistant_message: str = ""
    planned_ops: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "created_at": m.created_at,
                    "id": m.id,
                    "kind": m.kind,
                    "ref_ids": list(m.ref_ids),
                    "artifacts": [a.to_dict() for a in m.artifacts],
                }
                for m in self.messages
            ],
            "mode": self.mode,
            "status": self.status,
            "intent": self.intent,
            "draft_prompt_en": self.draft_prompt_en,
            "edit_brief_en": self.edit_brief_en,
            "user_prompt": self.user_prompt,
            "ready": self.ready,
            "questions": list(self.questions),
            "assistant_message": self.assistant_message,
            "planned_ops": list(self.planned_ops),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatState":
        messages: list[ChatMessage] = []
        for item in data.get("messages") or []:
            if not isinstance(item, dict):
                continue
            refs_raw = item.get("ref_ids") or []
            ref_ids = [str(r) for r in refs_raw if str(r).strip()]
            arts_raw = item.get("artifacts") or []
            artifacts = [
                ChatArtifact.from_dict(a) for a in arts_raw if isinstance(a, dict) and a.get("path")
            ]
            messages.append(
                ChatMessage(
                    role=str(item.get("role", "user")),
                    content=str(item.get("content", "")),
                    created_at=str(item.get("created_at") or datetime.now(timezone.utc).isoformat()),
                    id=str(item.get("id") or _new_message_id()),
                    kind=str(item.get("kind") or "text"),
                    ref_ids=ref_ids,
                    artifacts=artifacts,
                )
            )
        ops_raw = data.get("planned_ops") or []
        planned_ops = [op for op in ops_raw if isinstance(op, dict)]
        return cls(
            messages=messages,
            mode=str(data.get("mode") or "create"),
            status=str(data.get("status") or "idle"),
            intent=str(data.get("intent") or "create"),
            draft_prompt_en=str(data.get("draft_prompt_en") or ""),
            edit_brief_en=str(data.get("edit_brief_en") or ""),
            user_prompt=str(data.get("user_prompt") or ""),
            ready=bool(data.get("ready")),
            questions=[str(q) for q in (data.get("questions") or [])],
            assistant_message=str(data.get("assistant_message") or ""),
            planned_ops=planned_ops,
        )


class PromptChatService:
    def __init__(self, llm: LMStudioClient | None = None) -> None:
        self.llm = llm or LMStudioClient()

    def chat_path(self, manifest: ProjectManifest) -> Path:
        return manifest.root / "chat.json"

    def load(self, manifest: ProjectManifest) -> ChatState:
        path = self.chat_path(manifest)
        if not path.is_file():
            return ChatState()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ChatState.from_dict(data if isinstance(data, dict) else {})
        except Exception as exc:
            logger.warning("Failed to read chat state %s: %s", path, exc)
            return ChatState()

    def save(self, manifest: ProjectManifest, state: ChatState) -> ChatState:
        path = self.chat_path(manifest)
        path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return state

    def get(self, manifest: ProjectManifest) -> dict[str, Any]:
        return self.load(manifest).to_dict()

    def reset(self, manifest: ProjectManifest) -> dict[str, Any]:
        return self.save(manifest, ChatState()).to_dict()

    def post_message(
        self,
        manifest: ProjectManifest,
        text: str,
        *,
        has_images: bool = False,
        ref_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        from mesh_forge.application.chat_agent import ChatAgentService

        return ChatAgentService(self.llm).post_message(
            manifest,
            text,
            has_images=has_images,
            ref_ids=ref_ids,
        )

    def post_message_legacy(
        self,
        manifest: ProjectManifest,
        text: str,
        *,
        has_images: bool = False,
    ) -> dict[str, Any]:
        text = (text or "").strip()
        if not text and not has_images:
            raise ValueError("Message text is empty")

        state = self.load(manifest)
        has_mesh = manifest.current_mesh_path() is not None
        state.mode = "edit" if has_mesh else "create"
        if text:
            state.messages.append(ChatMessage(role="user", content=text))
            if not state.user_prompt:
                state.user_prompt = text
            else:
                state.user_prompt = f"{state.user_prompt}\n{text}".strip()

        if has_images and not has_mesh:
            # Image-to-mesh does not need EN draft; ready to confirm reconstruction.
            state.intent = "create"
            state.ready = True
            state.status = "ready"
            state.questions = []
            state.draft_prompt_en = ""
            state.edit_brief_en = ""
            state.assistant_message = (
                "Понял: будет реконструкция из фото через ComfyUI. "
                "Текст (если есть) сохранится в историю. Нажмите «Запустить»."
            )
            state.messages.append(ChatMessage(role="assistant", content=state.assistant_message))
            return self.save(manifest, state).to_dict()

        recreate = _looks_like_recreate(text)
        if has_mesh and not recreate:
            if _looks_like_geometry_repair(text):
                return self._handle_geometry_edit(manifest, state, text)
            if _looks_like_full_regen(text):
                return self._handle_semantic_edit(manifest, state, text)
            return self._handle_guided_edit(manifest, state, text)
        if recreate:
            state.mode = "create"
            state.intent = "create"
        return self._handle_create(manifest, state)

    def _handle_create(self, manifest: ProjectManifest, state: ChatState) -> dict[str, Any]:
        history = [{"role": m.role, "content": m.content} for m in state.messages if m.role in {"user", "assistant"}]
        clarifying_rounds = sum(1 for m in state.messages if m.role == "assistant")
        force_draft = clarifying_rounds >= 2 or _user_wants_to_stop_clarifying(state.messages)

        result = self.llm.clarify_or_enhance(history)
        state.intent = str(result.get("intent") or "create")
        state.questions = [str(q).strip() for q in (result.get("questions") or []) if str(q).strip()][:3]
        user_text = state.user_prompt or _last_user_text(state.messages)
        state.edit_brief_en = ""
        state.planned_ops = []
        state.ready = bool(result.get("ready"))

        if (force_draft or state.ready) and user_text:
            state.draft_prompt_en = self.llm.ensure_english_subject(user_text)
            state.questions = []
            state.ready = bool(state.draft_prompt_en)
            if force_draft and not bool(result.get("ready")):
                state.assistant_message = (
                    "Ок, собираю промпт из того что уже есть — можно запускать. "
                    "Если нужно иначе, напишите правку после генерации."
                )
            else:
                state.assistant_message = _format_assistant_with_questions(
                    str(result.get("assistant_message") or "").strip(),
                    [],
                    ready=state.ready,
                    draft=state.draft_prompt_en,
                )
        else:
            state.draft_prompt_en = ""
            state.ready = False
            state.assistant_message = _format_assistant_with_questions(
                str(result.get("assistant_message") or "").strip(),
                state.questions,
                ready=False,
                draft="",
            )
            if not state.questions:
                state.questions = [
                    "Что именно моделируем? (объект / персонаж / здание…)",
                    "Какой стиль? (мультяшный / реалистичный / clay / lowpoly…)",
                    "Какие детали обязательны?",
                ]
                state.assistant_message = _format_assistant_with_questions(
                    "Уточните, пожалуйста:",
                    state.questions,
                    ready=False,
                    draft="",
                )

        state.status = "ready" if state.ready else "clarifying"
        state.messages.append(ChatMessage(role="assistant", content=state.assistant_message))
        return self.save(manifest, state).to_dict()

    def _fallback_draft_en(self, user_text: str) -> str:
        subject = (user_text or "a simple object").strip()
        subject = subject.split("\n")[0].strip()[:200]
        return self.llm.ensure_english_subject(subject)

    def _handle_geometry_edit(self, manifest: ProjectManifest, state: ChatState, text: str) -> dict[str, Any]:
        mesh = manifest.current_mesh_path()
        if not mesh or not mesh.is_file():
            raise ValueError("No current mesh to edit")
        if not text:
            raise ValueError("Describe the geometry fix")

        from mesh_forge.mesh_qc import analyze_mesh

        stats = analyze_mesh(mesh)
        plan = self.llm.plan_edit(text, stats.to_dict())
        operations = list(plan.get("operations") or [])
        if not operations:
            # Deterministic fallback for cleanup wording when LLM returns nothing.
            operations = [
                {"op": "remove_needles"},
                {"op": "smooth", "iterations": 2},
                {"op": "fill_holes"},
            ]
            summary = "Remove needles/spikes, light smooth, try to fill holes."
        else:
            summary = str(plan.get("summary") or "").strip() or "Geometry cleanup on current mesh."

        op_names = ", ".join(str(op.get("op") or "?") for op in operations)
        state.intent = "geometry_edit"
        state.edit_brief_en = summary
        state.draft_prompt_en = ""
        state.ready = True
        state.status = "ready"
        state.questions = []
        state.assistant_message = (
            "Это геометрическая очистка текущего mesh (не перегенерация через ComfyUI).\n"
            f"План: {op_names}.\n"
            f"{summary}\n"
            "Нажмите «Запустить»."
        )
        state.messages.append(ChatMessage(role="assistant", content=state.assistant_message))
        # Stash ops in edit_brief for confirm (JSON-ish via summary + store on state)
        # Persist ops by encoding into edit_brief_en prefix that confirm can parse,
        # or better: store on ChatState. Add planned_ops field.
        state.planned_ops = operations
        return self.save(manifest, state).to_dict()

    def _handle_guided_edit(self, manifest: ProjectManifest, state: ChatState, text: str) -> dict[str, Any]:
        mesh = manifest.current_mesh_path()
        if not mesh or not mesh.is_file():
            raise ValueError("No current mesh to edit")
        if not text:
            raise ValueError("Describe the guided change")

        # Prefer vision brief when possible; fall back to short EN draft.
        work_dir = manifest.root / "work" / "chat_edit"
        work_dir.mkdir(parents=True, exist_ok=True)
        preview = work_dir / "mesh_preview.png"
        render_mesh_preview(mesh, preview)
        preview_b64 = base64.b64encode(preview.read_bytes()).decode()

        prior = ""
        for version in reversed(manifest.versions):
            if version.instruction:
                prior = version.instruction
                break

        ref_photo = manifest.find_reference_photo()
        ref_b64 = None
        if ref_photo and ref_photo.is_file():
            ref_b64 = base64.b64encode(ref_photo.read_bytes()).decode()

        result = self.llm.interpret_mesh_edit(
            instruction=text,
            mesh_preview_b64=preview_b64,
            prior_prompt=prior,
            prefer_guided=True,
            reference_photo_b64=ref_b64,
        )
        intent = str(result.get("intent") or "guided_edit").strip().lower()
        if intent == "geometry_edit":
            return self._handle_geometry_edit(manifest, state, text)
        if intent == "semantic_edit" and _looks_like_full_regen(text):
            return self._handle_semantic_edit(manifest, state, text)

        brief = self.llm.ensure_english_subject(text)
        if not brief:
            brief = str(result.get("edit_brief_en") or "").strip()
        if not brief:
            brief = _fallback_draft_en(text)

        view_anchor = manifest.find_view_anchor()
        if view_anchor and view_anchor.is_file():
            anchor_note = f"Якорь: clay-вид {view_anchor.name}."
        else:
            anchor_note = "Якорь: bake clay-front с текущего mesh (фото-референс не используется как img2img)."

        state.intent = "guided_edit"
        state.edit_brief_en = brief
        state.draft_prompt_en = brief
        state.planned_ops = []
        state.ready = True
        state.status = "ready"
        state.questions = []
        state.assistant_message = _format_assistant_with_questions(
            str(result.get("assistant_message") or "").strip() or "Понял щадящую правку.",
            [],
            ready=True,
            draft=brief,
            ready_fallback=(
                f"Щадящая правка от текущего mesh/вида (не полная перегенерация). {anchor_note} "
                "Можно запускать."
            ),
            empty_fallback="Не понял, что именно добавить/убрать.",
        )
        state.messages.append(ChatMessage(role="assistant", content=state.assistant_message))
        return self.save(manifest, state).to_dict()

    def _handle_semantic_edit(self, manifest: ProjectManifest, state: ChatState, text: str) -> dict[str, Any]:
        mesh = manifest.current_mesh_path()
        if not mesh or not mesh.is_file():
            raise ValueError("No current mesh to edit")
        if not text:
            raise ValueError("Describe the semantic change to make")

        # Safety net: LLM/vision may still get geometry requests without heuristic hit.
        if _looks_like_geometry_repair(text):
            return self._handle_geometry_edit(manifest, state, text)
        if not _looks_like_full_regen(text):
            return self._handle_guided_edit(manifest, state, text)

        work_dir = manifest.root / "work" / "chat_edit"
        work_dir.mkdir(parents=True, exist_ok=True)
        preview = work_dir / "mesh_preview.png"
        render_mesh_preview(mesh, preview)
        preview_b64 = base64.b64encode(preview.read_bytes()).decode()

        prior = ""
        for version in reversed(manifest.versions):
            if version.instruction:
                prior = version.instruction
                break

        result = self.llm.interpret_mesh_edit(
            instruction=text,
            mesh_preview_b64=preview_b64,
            prior_prompt=prior,
            prefer_guided=False,
        )
        if str(result.get("intent") or "") == "geometry_edit":
            return self._handle_geometry_edit(manifest, state, text)
        if str(result.get("intent") or "") == "guided_edit":
            return self._handle_guided_edit(manifest, state, text)

        state.intent = "semantic_edit"
        state.assistant_message = str(result.get("assistant_message") or "").strip()
        state.questions = [str(q).strip() for q in (result.get("questions") or []) if str(q).strip()][:3]
        brief = self.llm.ensure_english_subject(text)
        if not brief:
            brief = str(result.get("edit_brief_en") or "").strip()
        state.edit_brief_en = brief
        state.draft_prompt_en = brief
        state.planned_ops = []
        state.ready = bool(brief)
        state.status = "ready" if state.ready else "clarifying"
        state.assistant_message = _format_assistant_with_questions(
            state.assistant_message,
            state.questions,
            ready=state.ready,
            draft=state.edit_brief_en,
            ready_fallback=(
                "Полная перегенерация через ComfyUI (форма/детали с нуля). "
                "Сходство с текущим mesh не гарантируется. Можно запускать."
            ),
            empty_fallback="Не понял, что именно исправить в модели.",
        )
        state.messages.append(ChatMessage(role="assistant", content=state.assistant_message))
        return self.save(manifest, state).to_dict()


def _looks_like_recreate(text: str) -> bool:
    lowered = (text or "").lower()
    markers = (
        "сделай заново",
        "с нуля",
        "заново сделай",
        "перегенерируй с нуля",
        "start over",
        "from scratch",
        "regenerate from scratch",
    )
    return any(m in lowered for m in markers)


def _looks_like_full_regen(text: str) -> bool:
    """Major redesign that should not try to preserve the current mesh identity."""
    if _looks_like_recreate(text):
        return True
    lowered = (text or "").lower()
    markers = (
        "полность переделай",
        "полностью переделай",
        "другую позу",
        "другая поза",
        "совсем другой",
        "совершенно другой",
        "change pose",
        "different pose",
        "completely different",
        "full regenerate",
        "перегенерируй",
        "regenerate",
    )
    return any(m in lowered for m in markers)


def _looks_like_geometry_repair(text: str) -> bool:
    """True when the user asks for mesh cleanup, not a semantic redesign."""
    lowered = (text or "").lower()
    markers = (
        "шум",
        "шипы",
        "шип",
        "дыр",
        "игол",
        "игл",
        "спайк",
        "сглад",
        "задела",
        "заделай",
        "закрой дыр",
        "убрать шум",
        "убери шум",
        "артефакт",
        "остры",
        "неровн",
        "починить",
        "почисти",
        "очист",
        "ремонт",
        "manifold",
        "watertight",
        "non-manifold",
        "noise",
        "spike",
        "needle",
        "hole",
        "holes",
        "smooth",
        "cleanup",
        "clean up",
        "repair",
        "fix holes",
        "fill holes",
        "jagged",
        "artifacts",
    )
    if not any(m in lowered for m in markers):
        return False
    # Explicit redesign still wins even if cleanup words appear.
    semantic_override = (
        "перегенерируй",
        "заново",
        "с нуля",
        "другую форму",
        "другая поза",
        "добавь",
        "убери лап",
        "убери хвост",
        "лишнюю лап",
        "extra limb",
        "extra leg",
        "change pose",
        "regenerate",
    )
    return not any(m in lowered for m in semantic_override)


def _last_user_text(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user" and msg.content.strip():
            return msg.content.strip()
    return ""


def _user_wants_to_stop_clarifying(messages: list[ChatMessage]) -> bool:
    text = _last_user_text(messages).lower()
    markers = (
        "ты ничего не спросил",
        "каких",
        "просто сделай",
        "без вопросов",
        "давай так",
        "хватает",
        "достаточно",
        "генерируй",
        "запускай",
        "just generate",
        "no questions",
        "go ahead",
    )
    return any(m in text for m in markers)


def _brief_looks_generic_blob(brief: str) -> bool:
    lowered = (brief or "").lower()
    markers = (
        "grey box",
        "gray box",
        "boxy structure",
        "wood grain",
        "generic",
        "simple block",
        "clay blob",
    )
    return any(m in lowered for m in markers)


def _guided_brief_from_context(user_text: str, prior: str) -> str:
    """Build a safer guided brief when vision invents a grey blob subject."""
    fix = (user_text or "").strip().replace("\n", " ")
    base = ""
    if prior:
        # Prefer generation: line if present.
        for line in prior.splitlines():
            low = line.strip().lower()
            if low.startswith("generation:"):
                base = line.split(":", 1)[1].strip()
                break
        if not base:
            for line in prior.splitlines():
                low = line.strip().lower()
                if low.startswith("user:"):
                    continue
                if line.strip():
                    base = line.strip()
                    break
    if base and _looks_english(base):
        return f"{base.rstrip('.')} ; change: {fix}."
    if _looks_english(fix):
        return fix
    return f"Same object as before; apply this change: {fix}."


def _looks_english(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    ascii_letters = sum(1 for c in letters if ord(c) < 128)
    return ascii_letters / len(letters) > 0.85


def _format_assistant_with_questions(
    message: str,
    questions: list[str],
    *,
    ready: bool,
    draft: str,
    ready_fallback: str = "Генерирую…",
    empty_fallback: str = "Нужно больше деталей об объекте.",
) -> str:
    msg = (message or "").strip()
    if ready:
        return msg or ready_fallback
    qs = [q for q in questions if q.strip()]
    if qs:
        block = "\n".join(f"- {q}" for q in qs)
        # Avoid "here are questions" with nothing after — always attach the list.
        if all(q in msg for q in qs):
            return msg
        if not msg or msg.endswith(":") or "вопрос" in msg.lower() or "question" in msg.lower():
            base = msg.rstrip() or "Уточните, пожалуйста:"
            return f"{base}\n{block}"
        return f"{msg}\n{block}"
    if draft:
        return msg or ready_fallback
    return msg or empty_fallback
