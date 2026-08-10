from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from mesh_forge.config import AppConfig, load_config

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
