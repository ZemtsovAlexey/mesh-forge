from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from mesh_forge.config import AppConfig, load_config

DEFAULT_CREATE_QUESTIONS = [
    "Какая поза? (сидит / стоит / динамичная)",
    "Стиль: matte clay фигурка или другой?",
    "Какие детали обязательны? (уши, хвост, основание…)",
]

PLANNER_SYSTEM = """You are a 3D mesh operations planner for 3D printing.
Given a user instruction and optional mesh stats, output ONLY valid JSON:
{
  "operations": [
    {"op": "scale_axis", "axis": "z", "value_mm": 100},
    {"op": "scale_uniform", "factor": 1.1},
    {"op": "solidify", "thickness_mm": 3},
    {"op": "decimate", "target_faces": 200000},
    {"op": "smooth", "iterations": 2}
  ],
  "summary": "short explanation"
}
Allowed ops: scale_axis (x|y|z + value_mm), scale_uniform (factor), solidify (thickness_mm),
decimate (target_faces), smooth (iterations), fill_holes, remesh_voxel (voxel_mm).

Rules:
- Prefer minimal ops. For "remove spikes / needles / fix edges" use smooth (1-3) then optional
  decimate — do NOT remesh_voxel unless the user explicitly asks to remesh.
- remesh_voxel is destructive on large scans (bbox > 200 mm). If you must remesh, set
  voxel_mm to at least max(bbox)/150 (often 5–20 mm), never 0.5 mm on meter-scale parts.
- fill_holes only on small meshes (< 100k faces) or when user asks to close holes.
- Never invent dimensions; use mesh stats bbox when scaling.
If you cannot map the request, return empty operations and explain in summary."""


class LMStudioClient:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.client = OpenAI(
            base_url=self.config.llm.base_url,
            api_key=self.config.llm.api_key,
        )

    def chat(self, model: str, messages: list[dict[str, Any]], temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def plan_edit(self, instruction: str, mesh_stats: dict[str, Any] | None = None) -> dict[str, Any]:
        user_content = f"Instruction: {instruction}\n"
        if mesh_stats:
            user_content += f"Mesh stats: {json.dumps(mesh_stats, ensure_ascii=False)}\n"
        text = self.chat(
            self.config.llm.planner_model,
            [
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": user_content},
            ],
        )
        return _parse_json_response(text)

    def enhance_prompt(self, prompt: str, mode: str = "organic") -> str:
        system = (
            "Improve this 3D object description for generation. "
            "Output only the improved prompt, one paragraph."
        )
        if mode == "mechanical":
            system += " Focus on dimensions in mm, simple printable geometry."
        return self.chat(
            self.config.llm.planner_model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        ).strip()

    def clarify_or_enhance(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        system = """You help prepare a short English subject line for text-to-3D image generation.
Reply with ONLY valid JSON:
{
  "intent": "create",
  "assistant_message": "short reply in the user's language",
  "questions": ["up to 3 short clarifying questions"],
  "ready": false,
  "draft_prompt_en": "",
  "confidence": 0.0
}

Rules for draft_prompt_en:
- Faithful translation of what the user asked — do NOT reinvent or over-interpret.
- ONE short sentence (max ~35 words). Example:
  user: "настольная фигурка потягивающегося и зевающего кота"
  draft: "A desktop figurine of a stretching and yawning cat."
- Keep pose/action words from the user (stretching, yawning, sitting…).
- Do NOT add: watertight, manifold, multiview, identical from all angles, clay/matte finish,
  studio lighting, printable, silhouette, no holes, no fur, debris, geometric form, or similar
  tech/quality boilerplate (that is added elsewhere).
- Do NOT invent extra anatomy details the user did not mention (paw positions, tail curl, tongue…).
- Do NOT invent numeric dimensions unless the user gave them.

Clarify rules:
- Prefer ready=true when the subject is clear. Ask questions ONLY if critical.
- If ready=false: questions MUST contain 1-3 real question strings; assistant_message lists them.
- If ready=true: questions=[], fill draft_prompt_en.
- Never map the request to mesh filters (smooth/decimate/solidify).
- If the user complains that you asked nothing / wants to proceed, set ready=true with a
  faithful short draft from the conversation so far."""
        text = self.chat(
            self.config.llm.planner_model,
            [{"role": "system", "content": system}, *messages],
            temperature=0.2,
        )
        data = _parse_json_response(text)
        if data.get("parse_error"):
            return {
                "intent": "create",
                "assistant_message": str(data.get("summary") or text)[:500],
                "questions": list(DEFAULT_CREATE_QUESTIONS),
                "ready": False,
                "draft_prompt_en": "",
                "confidence": 0.0,
            }
        data["intent"] = "create"
        questions = [str(q).strip() for q in (data.get("questions") or []) if str(q).strip()][:3]
        draft = _sanitize_create_draft(str(data.get("draft_prompt_en") or "").strip())
        ready = bool(data.get("ready")) and bool(draft)
        assistant = str(data.get("assistant_message") or "").strip()

        # Empty "I'll ask questions" responses are useless — force defaults or draft.
        if not ready and not questions:
            questions = list(DEFAULT_CREATE_QUESTIONS)
        if ready:
            questions = []
        data["questions"] = questions
        data["draft_prompt_en"] = draft
        data["ready"] = ready
        data["assistant_message"] = assistant
        return data

    def interpret_mesh_edit(
        self,
        *,
        instruction: str,
        mesh_preview_b64: str,
        prior_prompt: str = "",
    ) -> dict[str, Any]:
        system = """You analyze a 3D mesh preview and a user edit request.
This is SEMANTIC geometry correction (e.g. remove an extra limb), NOT mesh filters.
Reply with ONLY valid JSON:
{
  "intent": "semantic_edit",
  "assistant_message": "short reply in the user's language",
  "questions": [],
  "ready": false,
  "edit_brief_en": "",
  "confidence": 0.0
}

Rules:
- edit_brief_en: one English paragraph describing the corrected object for text-to-3D regeneration.
  Include the fix explicitly (e.g. "fix: remove the extra left front paw") and keep desired pose/style.
- If unclear what to change, ready=false and ask up to 3 questions.
- Do NOT suggest smooth/decimate/solidify/remesh filters."""
        user_text = f"User edit request: {instruction.strip()}"
        if prior_prompt.strip():
            user_text += f"\nPrior creation prompt/context: {prior_prompt.strip()}"
        content: list[dict[str, Any]] = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{mesh_preview_b64}"}},
        ]
        text = self.chat(
            self.config.llm.vision_model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            temperature=0.2,
        )
        data = _parse_json_response(text)
        if data.get("parse_error"):
            return {
                "intent": "semantic_edit",
                "assistant_message": str(data.get("summary") or text)[:500],
                "questions": ["Что именно нужно изменить в модели?"],
                "ready": False,
                "edit_brief_en": "",
                "confidence": 0.0,
            }
        data["intent"] = "semantic_edit"
        data["questions"] = list(data.get("questions") or [])[:3]
        data["edit_brief_en"] = str(data.get("edit_brief_en") or "").strip()
        data["ready"] = bool(data.get("ready")) and bool(data["edit_brief_en"])
        data["assistant_message"] = str(data.get("assistant_message") or "").strip()
        return data

    def generate_openscad(self, prompt: str) -> str:
        system = (
            "Write OpenSCAD code for a 3D-printable object. "
            "Use only basic primitives and difference/union. "
            "Output ONLY OpenSCAD code, no markdown."
        )
        return self.chat(
            self.config.llm.planner_model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        ).strip()

    def describe_image_diff(self, instruction: str, mesh_preview_b64: str | None, ref_image_path: Path | None) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
        if mesh_preview_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{mesh_preview_b64}"},
            })
        if ref_image_path and ref_image_path.is_file():
            b64 = base64.b64encode(ref_image_path.read_bytes()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        return self.chat(
            self.config.llm.vision_model,
            [{"role": "user", "content": content}],
        )

    def health_check(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return model IDs exposed by LM Studio (loaded models)."""
        try:
            response = self.client.models.list()
            ids = [m.id for m in response.data if getattr(m, "id", None)]
            return sorted(set(ids))
        except Exception:
            return []

    def models_status(self) -> str:
        if not self.health_check():
            return "LM Studio API недоступен. Запустите Local Server в LM Studio."
        models = self.list_models()
        if not models:
            return (
                "API отвечает, но моделей нет. Загрузите модель в LM Studio "
                "(Chat → Load model) и обновите список."
            )
        lines = [f"Доступно моделей: {len(models)}", ""]
        for mid in models:
            mark = []
            if mid == self.config.llm.planner_model:
                mark.append("planner")
            if mid == self.config.llm.vision_model:
                mark.append("vision")
            suffix = f" ← {', '.join(mark)}" if mark else ""
            lines.append(f"  • {mid}{suffix}")
        return "\n".join(lines)


def _parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"operations": [], "summary": text, "parse_error": True}


_DRAFT_BOILERPLATE_RE = re.compile(
    r"\b("
    r"watertight|manifold|multiview|identical character from all angles|"
    r"matte clay(?: material)?(?: finish)?|smooth closed surface|"
    r"no holes|no floating debris|photoreal fur|studio lighting|"
    r"3d printable|printable figurine|clean silhouette|"
    r"suitable for (?:multiview )?reconstruction|simple geometric form|"
    r"optimized for 3d|clean manifold surface|closed surface without"
    r")\b[^.]*\.?",
    re.IGNORECASE,
)


def _sanitize_create_draft(draft: str) -> str:
    """Strip quality/tech boilerplate the LLM loves to invent; keep subject short."""
    text = (draft or "").strip()
    if not text:
        return ""
    # Drop sentences that are mostly boilerplate.
    parts = re.split(r"(?<=[.!?])\s+", text)
    kept: list[str] = []
    for part in parts:
        cleaned = _DRAFT_BOILERPLATE_RE.sub("", part).strip(" ,;")
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        if len(cleaned) < 8:
            continue
        if _DRAFT_BOILERPLATE_RE.search(part) and len(cleaned) < 24:
            continue
        kept.append(cleaned)
    text = " ".join(kept).strip() or draft.strip()
    # Prefer the first sentence — later ones are usually invented polish.
    text = re.split(r"(?<=[.!?])\s+", text)[0].strip()
    # If still a comma-stack of invented details, keep the core clause(s).
    words = text.split()
    if len(words) > 22 and "," in text:
        head = text.split(",")[0].strip()
        if len(head.split()) >= 6:
            text = head.rstrip(".,; ") + "."
            words = text.split()
    if len(words) > 28:
        text = " ".join(words[:28]).rstrip(",;") + "."
    return text
