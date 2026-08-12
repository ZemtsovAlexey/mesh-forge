from __future__ import annotations

import base64
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mesh_forge.adapters import LMStudioClient
from mesh_forge.manifest import ProjectManifest
from mesh_forge.render import render_mesh_preview

logger = logging.getLogger("mesh_forge.prompt_chat")


@dataclass
class ChatMessage:
    role: str
    content: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [asdict(m) for m in self.messages],
            "mode": self.mode,
            "status": self.status,
            "intent": self.intent,
            "draft_prompt_en": self.draft_prompt_en,
            "edit_brief_en": self.edit_brief_en,
            "user_prompt": self.user_prompt,
            "ready": self.ready,
            "questions": list(self.questions),
            "assistant_message": self.assistant_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatState":
        messages = [
            ChatMessage(
                role=str(item.get("role", "user")),
                content=str(item.get("content", "")),
                created_at=str(item.get("created_at") or datetime.now(timezone.utc).isoformat()),
            )
            for item in (data.get("messages") or [])
            if isinstance(item, dict)
        ]
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
            return self._handle_semantic_edit(manifest, state, text)
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
        state.draft_prompt_en = str(result.get("draft_prompt_en") or "").strip()
        state.edit_brief_en = ""
        state.ready = bool(result.get("ready")) and bool(state.draft_prompt_en)

        if force_draft and not state.ready:
            # Break empty clarify loops: synthesize a usable EN draft from user text.
            state.draft_prompt_en = _fallback_draft_en(state.user_prompt or _last_user_text(state.messages))
            state.questions = []
            state.ready = bool(state.draft_prompt_en)
            state.assistant_message = (
                "Ок, собираю промпт из того что уже есть — можно запускать. "
                "Если нужно иначе, напишите правку после генерации."
            )
        else:
            state.assistant_message = _format_assistant_with_questions(
                str(result.get("assistant_message") or "").strip(),
                state.questions,
                ready=state.ready,
                draft=state.draft_prompt_en,
            )
            if not state.ready and not state.questions:
                state.questions = [
                    "Какая поза? (сидит / стоит / динамичная)",
                    "Стиль: matte clay фигурка или другой?",
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

    def _handle_semantic_edit(self, manifest: ProjectManifest, state: ChatState, text: str) -> dict[str, Any]:
        mesh = manifest.current_mesh_path()
        if not mesh or not mesh.is_file():
            raise ValueError("No current mesh to edit")
        if not text:
            raise ValueError("Describe the semantic change to make")

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
        )
        state.intent = "semantic_edit"
        state.assistant_message = str(result.get("assistant_message") or "").strip()
        state.questions = [str(q).strip() for q in (result.get("questions") or []) if str(q).strip()][:3]
        state.edit_brief_en = str(result.get("edit_brief_en") or "").strip()
        state.draft_prompt_en = state.edit_brief_en
        state.ready = bool(result.get("ready")) and bool(state.edit_brief_en)
        state.status = "ready" if state.ready else "clarifying"
        state.assistant_message = _format_assistant_with_questions(
            state.assistant_message,
            state.questions,
            ready=state.ready,
            draft=state.edit_brief_en,
            ready_fallback=(
                "Понял смысловую правку. Это будет перегенерация через ComfyUI "
                "(не фильтры). Можно запускать."
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


def _fallback_draft_en(user_text: str) -> str:
    subject = (user_text or "a small figurine").strip()
    subject = subject.split("\n")[0].strip()[:200]
    # Keep faithful and short; ComfyUI wrapper adds clay/studio style.
    return subject if _looks_english(subject) else f"A desktop figurine: {subject}."


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
    ready_fallback: str = "Подготовил английский промпт для ComfyUI. Можно запускать.",
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
