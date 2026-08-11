from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mesh_forge.backends.lmstudio import LMStudioClient
from mesh_forge.manifest import ProjectManifest, add_version
from mesh_forge.mesh_qc import analyze_mesh
from mesh_forge.ops.geometry import apply_operations
from mesh_forge.pipeline.photo import create_from_photo
from mesh_forge.pipeline.scan import create_from_scan
from mesh_forge.pipeline.text import create_from_text
from mesh_forge.render import render_mesh_preview

logger = logging.getLogger("mesh_forge.orchestrator")


class Orchestrator:
    def __init__(self):
        self.reload_config()

    def reload_config(self) -> None:
        from mesh_forge.config import load_config

        self.config = load_config()
        self.llm = LMStudioClient(self.config)

    def create_photo(
        self, manifest: ProjectManifest, image_path: Path, **kwargs: Any
    ) -> tuple[ProjectManifest, str]:
        work = manifest.root / "work" / f"v{manifest.current_version + 1}_photo"
        logger.info("create_photo project=%s image=%s work=%s kwargs=%s", manifest.id, image_path, work, kwargs)
        mesh = create_from_photo(image_path, work, project_id=manifest.id, **kwargs)
        logger.info("create_photo project=%s mesh=%s", manifest.id, mesh)
        add_version(manifest, mesh, branch="photo", action="create")
        return manifest, "Photo → 3D complete"

    def create_scan(
        self, manifest: ProjectManifest, scan_path: Path, **kwargs: Any
    ) -> tuple[ProjectManifest, str]:
        work = manifest.root / "work" / f"v{manifest.current_version + 1}_scan"
        logger.info("create_scan project=%s scan=%s work=%s kwargs=%s", manifest.id, scan_path, work, kwargs)
        mesh = create_from_scan(scan_path, work, **kwargs)
        logger.info("create_scan project=%s mesh=%s", manifest.id, mesh)
        add_version(manifest, mesh, branch="scan", action="create")
        return manifest, "Scan cleanup complete"

    def create_text(
        self, manifest: ProjectManifest, prompt: str, **kwargs: Any
    ) -> tuple[ProjectManifest, str]:
        work = manifest.root / "work" / f"v{manifest.current_version + 1}_text"
        logger.info("create_text project=%s work=%s kwargs=%s", manifest.id, work, kwargs)
        mesh, notes = create_from_text(prompt, work, **kwargs)
        logger.info("create_text project=%s mesh=%s", manifest.id, mesh)
        add_version(manifest, mesh, branch="text", action="create", instruction=prompt)
        return manifest, notes or "Text → 3D complete"

    def edit_text(
        self, manifest: ProjectManifest, instruction: str, *, apply_solidify: float = 0.0
    ) -> tuple[ProjectManifest, str]:
        current = manifest.current_mesh_path()
        if not current:
            raise ValueError("No mesh to edit")

        stats = analyze_mesh(current)
        plan = self.llm.plan_edit(instruction, stats.to_dict())
        ops = plan.get("operations", [])
        work = manifest.root / "work" / f"v{manifest.current_version + 1}_edit"
        work.mkdir(parents=True, exist_ok=True)
        edited = work / "edited.stl"
        apply_operations(current, ops, edited)

        if apply_solidify > 0:
            from mesh_forge.backends.blender import repair_and_export
            final = work / "edited_solid.stl"
            repair_and_export(edited, final, solidify_mm=apply_solidify)
            edited = final

        add_version(
            manifest, edited,
            branch="edit", action="text_edit",
            instruction=instruction,
            ops=ops,
        )
        summary = plan.get("summary", "")
        return manifest, f"Applied {len(ops)} operations.\n{summary}"

    def edit_photo(
        self, manifest: ProjectManifest,
        instruction: str,
        ref_image: Path | None,
    ) -> tuple[ProjectManifest, str]:
        current = manifest.current_mesh_path()
        if not current:
            raise ValueError("No mesh to edit")
        if not ref_image:
            return self.edit_text(manifest, instruction)

        preview = manifest.root / "work" / "preview.png"
        render_mesh_preview(current, preview)
        import base64
        b64 = base64.b64encode(preview.read_bytes()).decode()
        analysis = self.llm.describe_image_diff(
            f"Compare current mesh (1st image) with reference (2nd). User wants: {instruction}",
            b64, ref_image,
        )
        combined = f"{analysis}\n\nUser instruction: {instruction}"
        return self.edit_text(manifest, combined)

    def system_status(self) -> dict[str, Any]:
        from mesh_forge.backends.blender import blender_available
        from mesh_forge.backends.openscad import openscad_available
        from mesh_forge.backends.triposr import triposr_available

        return {
            "lmstudio": self.llm.health_check(),
            "triposr": triposr_available(),
            "blender": blender_available(),
            "openscad": openscad_available(),
        }

    def status_text(self) -> str:
        s = self.system_status()
        lines = [f"{k}: {'OK' if v else 'missing'}" for k, v in s.items()]
        return "\n".join(lines)
