from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from mesh_forge.render import render_mesh_front_clay, render_mesh_preview


def _save_mesh(mesh: trimesh.Trimesh, folder: Path, name: str = "mesh.stl") -> Path:
    path = folder / name
    mesh.export(path)
    return path


def _clay_object(img: np.ndarray) -> np.ndarray:
    """Clay object pixels, ignoring the dark ground grid."""
    r = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    b = img[:, :, 2].astype(np.int16)
    return (r > 70) & (r > b + 8) & (g > b)


def _not_background(img: np.ndarray) -> np.ndarray:
    bg = img[2, 2].astype(np.int16)
    return np.any(np.abs(img.astype(np.int16) - bg) > 18, axis=2)


class RenderPreviewTests(unittest.TestCase):
    def test_box_preview_is_solid_not_spiky(self) -> None:
        mesh = trimesh.creation.box(extents=[1.0, 1.2, 0.8])
        with tempfile.TemporaryDirectory() as tmp:
            src = _save_mesh(mesh, Path(tmp))
            out = Path(tmp) / "look.png"
            render_mesh_preview(src, out, size=256)
            img = np.asarray(Image.open(out).convert("RGB"))
        fg = _clay_object(img)
        ys, xs = np.where(fg)
        self.assertGreater(len(ys), 800)
        bbox = (int(ys.max() - ys.min()) + 1) * (int(xs.max() - xs.min()) + 1)
        fill = float(fg.sum()) / float(bbox)
        self.assertGreater(fill, 0.35)
        aspect = (xs.max() - xs.min() + 1) / max(ys.max() - ys.min() + 1, 1)
        self.assertGreater(float(aspect), 0.45)
        self.assertLess(float(aspect), 2.2)

    def test_preview_sits_on_ground_grid(self) -> None:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
        mesh.apply_translation([0.0, 0.5, 0.0])
        with tempfile.TemporaryDirectory() as tmp:
            src = _save_mesh(mesh, Path(tmp))
            out = Path(tmp) / "look.png"
            render_mesh_preview(src, out, size=256)
            img = np.asarray(Image.open(out).convert("RGB"))
        fg = _clay_object(img)
        ys, xs = np.where(fg)
        self.assertGreater(len(ys), 400)
        # Object should occupy the middle of the frame, not a corner spike.
        cy = float(ys.mean()) / img.shape[0]
        cx = float(xs.mean()) / img.shape[1]
        self.assertGreater(cx, 0.35)
        self.assertLess(cx, 0.65)
        self.assertGreater(cy, 0.30)
        self.assertLess(cy, 0.75)

    def test_dense_sphere_stays_compact(self) -> None:
        mesh = trimesh.creation.icosphere(subdivisions=4)
        with tempfile.TemporaryDirectory() as tmp:
            src = _save_mesh(mesh, Path(tmp))
            out = Path(tmp) / "look.png"
            render_mesh_preview(src, out, size=256)
            img = np.asarray(Image.open(out).convert("RGB"))
        fg = _clay_object(img)
        ys, xs = np.where(fg)
        self.assertGreater(len(ys), 1500)
        bbox = (int(ys.max() - ys.min()) + 1) * (int(xs.max() - xs.min()) + 1)
        fill = float(fg.sum()) / float(bbox)
        self.assertGreater(fill, 0.45)

    def test_front_clay_writes_png(self) -> None:
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        with tempfile.TemporaryDirectory() as tmp:
            src = _save_mesh(mesh, Path(tmp))
            out = Path(tmp) / "front.png"
            render_mesh_front_clay(src, out, size=128)
            img = np.asarray(Image.open(out).convert("RGB"))
        self.assertEqual(img.shape[0], 128)
        self.assertGreater(int(_not_background(img).sum()), 80)

    def test_side_view_differs_from_overview(self) -> None:
        mesh = trimesh.creation.box(extents=[0.4, 1.2, 0.8])
        with tempfile.TemporaryDirectory() as tmp:
            src = _save_mesh(mesh, Path(tmp))
            overview = Path(tmp) / "ov.png"
            side = Path(tmp) / "side.png"
            render_mesh_preview(src, overview, size=160, camera="viewer")
            render_mesh_preview(src, side, size=160, camera="left")
            a = np.asarray(Image.open(overview).convert("RGB"))
            b = np.asarray(Image.open(side).convert("RGB"))
        self.assertGreater(float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)))), 4.0)

    def test_zoom_fills_more_of_the_frame(self) -> None:
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        with tempfile.TemporaryDirectory() as tmp:
            src = _save_mesh(mesh, Path(tmp))
            wide = Path(tmp) / "wide.png"
            close = Path(tmp) / "close.png"
            render_mesh_preview(src, wide, size=160, camera="front", zoom=1.0)
            render_mesh_preview(src, close, size=160, camera="front", zoom=2.4, region="top")
            a = _clay_object(np.asarray(Image.open(wide).convert("RGB")))
            b = _clay_object(np.asarray(Image.open(close).convert("RGB")))
        self.assertGreater(int(b.sum()), int(a.sum()) * 1.08)

    def test_preview_keeps_full_mesh(self) -> None:
        mesh = trimesh.creation.icosphere(subdivisions=5)
        with tempfile.TemporaryDirectory() as tmp:
            src = _save_mesh(mesh, Path(tmp))
            from mesh_forge.render import load_render_mesh

            loaded = load_render_mesh(src)
        self.assertGreaterEqual(len(loaded.faces), len(mesh.faces) * 0.9)

    def test_high_poly_sphere_is_smooth_not_crystalline(self) -> None:
        mesh = trimesh.creation.icosphere(subdivisions=4)
        with tempfile.TemporaryDirectory() as tmp:
            src = _save_mesh(mesh, Path(tmp))
            out = Path(tmp) / "look.png"
            render_mesh_preview(src, out, size=192)
            img = np.asarray(Image.open(out).convert("RGB"))
        fg = _clay_object(img)
        colors = img[fg]
        unique = len(np.unique(colors.reshape(-1, 3), axis=0))
        self.assertGreater(unique, 80)


if __name__ == "__main__":
    unittest.main()
