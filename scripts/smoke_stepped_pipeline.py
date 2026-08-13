"""Smoke-test stepped pipeline state machine without a long Comfy run.

Uses grey clay-like PNGs and mocks ComfyUiClient methods.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image

from mesh_forge.application.stepped_pipeline import (
    continue_pipeline,
    pipeline_payload,
    redo_step,
    start_photo_gate,
    start_text_front,
)
from mesh_forge.domain import ImageArtifact, ImageSet, MeshArtifact
from mesh_forge.manifest import create_project, delete_project


def _clay_png(path: Path, size: int = 64) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), (180, 180, 180)).save(path)
    return path


def _views(work: Path) -> ImageSet:
    items = []
    for label in ("front", "left", "back", "right"):
        p = _clay_png(work / f"{label}.png")
        items.append(ImageArtifact(path=p, label=label, role="view", stage="views"))
    return ImageSet(items=items)


def main() -> int:
    manifest = create_project("smoke-stepped")
    try:
        front_dir = manifest.root / "tmp_front"
        views_dir = manifest.root / "tmp_views"
        mesh_path = manifest.root / "tmp_mesh" / "mesh.stl"
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        # Minimal ASCII STL triangle
        mesh_path.write_text(
            "solid smoke\n"
            " facet normal 0 0 1\n"
            "  outer loop\n"
            "   vertex 0 0 0\n"
            "   vertex 1 0 0\n"
            "   vertex 0 1 0\n"
            "  endloop\n"
            " endfacet\n"
            "endsolid smoke\n",
            encoding="utf-8",
        )

        fake_client = MagicMock()
        fake_client.generate_front.side_effect = lambda prompt, work_dir, project_id=None: _views(front_dir)
        fake_client.generate_views_from_front.side_effect = (
            lambda prompt, front_image, work_dir, project_id=None: _views(views_dir)
        )
        fake_client.mesh_from_views.side_effect = (
            lambda views, work_dir, project_id=None: MeshArtifact(
                path=mesh_path, source="mock", label="mesh_raw", stage="mesh"
            )
        )

        with patch("mesh_forge.application.stepped_pipeline.ComfyUiClient", return_value=fake_client), patch(
            "mesh_forge.application.stepped_pipeline.MeshProcessingService"
        ) as mesh_svc:
            mesh_svc.return_value.finalize_reconstruction.side_effect = (
                lambda mesh_path, work_dir, solidify_mm=0.0: Path(mesh_path)
            )

            s1 = start_text_front(
                manifest,
                brief_en="a small toy house",
                user_prompt="домик",
            )
            assert s1.step == "front" and s1.to_dict()["can_continue"], s1.to_dict()
            print("front OK", pipeline_payload(manifest, s1)["can_continue"])

            s2 = continue_pipeline(manifest)
            assert s2.step == "views" and s2.to_dict()["can_continue"], s2.to_dict()
            print("views OK", len(s2.images))

            s_redo = redo_step(manifest, step="views")
            assert s_redo.step == "views", s_redo
            print("redo views OK")

            s3 = continue_pipeline(manifest)
            assert s3.step == "done", s3
            assert manifest.current_mesh_path() is not None
            print("mesh OK", manifest.current_mesh_path())

            photo = manifest.root / "photo_in.png"
            Image.new("RGB", (64, 64), (220, 40, 40)).save(photo)
            sp = start_photo_gate(manifest, [photo], user_prompt="photo", remove_bg=False)
            assert sp.pipeline == "photo_gated" and sp.step == "photo"
            print("photo gate OK", sp.message)

        print("SMOKE PASSED")
        return 0
    finally:
        try:
            delete_project(manifest.id)
        except Exception as exc:
            print("cleanup warn", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
