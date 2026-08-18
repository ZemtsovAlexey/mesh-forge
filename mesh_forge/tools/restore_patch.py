from __future__ import annotations

from typing import Literal

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.ops.edit import EditError, restore_patch_in_region
from mesh_forge.ops.geometry import load_mesh
from mesh_forge.ops.region import RegionError
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import LOOK_AFTER, resolve_edit_target, resolve_mesh, save_mesh_artifact

_Region = Literal["legs", "seat", "back", "left", "right", "top", "bottom", "front"]


class RestorePatch(MeshTool):
    title = "Восстановить участок"
    heavy = True

    def run(
        self,
        ctx: RunContext[ChatDeps],
        region: _Region | None = None,
        mesh_ref: str | None = None,
    ) -> str:
        """Rebuild a missing patch in region: close holes, else copy the intact side.

        Not a full remesh and not generate_image. Extra blob → remove_mesh.
        """
        src = resolve_mesh(ctx, mesh_ref)
        try:
            label, box, _ = resolve_edit_target(ctx, region)
            mesh, stats = restore_patch_in_region(load_mesh(src), label, box=box)
        except (RegionError, EditError) as exc:
            return f"restore_patch skipped: {exc} Click or pass region."
        except Exception as extra:
            return f"restore_patch failed: {extra}. restore_mesh(to='previous')."
        art = save_mesh_artifact(ctx, mesh, "patched.stl", label="patched")
        how = stats.get("how", "fill")
        return (
            f"Restored patch {stats.get('region', label)} how={how} on {src.name} "
            f"→ {art.name}. {LOOK_AFTER}"
        )
