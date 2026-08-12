from __future__ import annotations

from pathlib import Path

from mesh_forge.manifest import ProjectManifest, add_version


class ProjectService:
    def add_result(
        self,
        manifest: ProjectManifest,
        mesh_path: Path,
        *,
        branch: str,
        action: str,
        instruction: str | None = None,
        ref: str | None = None,
        ops: list[dict] | None = None,
        artifacts: list[dict] | None = None,
    ) -> ProjectManifest:
        ext = mesh_path.suffix if mesh_path.suffix else ".stl"
        add_version(
            manifest,
            mesh_path,
            branch=branch,
            action=action,
            instruction=instruction,
            ref=ref,
            ops=ops,
            ext=ext,
            artifacts=artifacts,
        )
        return manifest
