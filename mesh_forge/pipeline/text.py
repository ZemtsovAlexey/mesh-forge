from __future__ import annotations

from pathlib import Path

from mesh_forge.backends.lmstudio import LMStudioClient
from mesh_forge.backends.openscad import openscad_available, render_scad_to_stl


def create_from_text(
    prompt: str,
    work_dir: Path,
    *,
    mode: str = "mechanical",
    use_llm: bool = True,
) -> tuple[Path, str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    notes = ""

    if mode == "mechanical":
        if not openscad_available():
            raise RuntimeError("OpenSCAD not configured. Set paths.openscad in config.yaml")
        scad_code = _fallback_scad(prompt)
        if use_llm:
            try:
                client = LMStudioClient()
                enhanced = client.enhance_prompt(prompt, mode="mechanical")
                scad_code = client.generate_openscad(enhanced)
                notes = f"Prompt enhanced.\n\nOpenSCAD:\n{scad_code[:2000]}"
            except Exception as exc:
                notes = f"LLM unavailable ({exc}), using fallback box."
        out = work_dir / "text_openscad.stl"
        render_scad_to_stl(scad_code, out)
        return out, notes

    raise NotImplementedError("Organic text-to-3D (TRELLIS) not yet implemented. Use mechanical mode.")


def _fallback_scad(prompt: str) -> str:
    return f"""// Fallback: simple box — replace via LLM
// Prompt: {prompt[:80]}
cube([40, 40, 20], center=true);
"""
