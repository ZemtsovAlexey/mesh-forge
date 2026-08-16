from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.chat.store import ChatStore
from mesh_forge.ops.background import cut_background, has_alpha
from mesh_forge.tools.remove_background import RemoveBackground, _pick_images


def _rgb(path: Path, color: tuple[int, int, int] = (200, 40, 40)) -> Path:
    Image.new("RGB", (24, 24), color).save(path)
    return path


def _rgba(path: Path, alpha: int) -> Path:
    Image.new("RGBA", (24, 24), (200, 40, 40, alpha)).save(path)
    return path


def _png_bytes(*, alpha: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (24, 24), (10, 20, 30, alpha)).save(buf, format="PNG")
    return buf.getvalue()


class BackgroundOpsTests(unittest.TestCase):
    def test_has_alpha_detects_transparency(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.assertFalse(has_alpha(_rgb(root / "rgb.png")))
            self.assertFalse(has_alpha(_rgba(root / "opaque.png", 255)))
            self.assertTrue(has_alpha(_rgba(root / "cut.png", 80)))

    def test_cut_skips_rembg_when_already_transparent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            src = _rgba(root / "src.png", 40)
            dest = root / "out.png"
            with patch("rembg.remove", side_effect=AssertionError("rembg should not run")):
                cut_background(src, dest)
            out = Image.open(dest)
            self.assertEqual(out.mode, "RGBA")
            self.assertLess(out.getchannel("A").getextrema()[0], 255)

    def test_cut_writes_rgba_from_rembg(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            src = _rgb(root / "src.jpg")
            dest = root / "out.png"
            with patch("rembg.remove", return_value=_png_bytes(alpha=90)) as mocked:
                cut_background(src, dest)
            mocked.assert_called_once()
            out = Image.open(dest)
            self.assertEqual(out.mode, "RGBA")
            self.assertEqual(out.getchannel("A").getextrema()[0], 90)


class RemoveBackgroundToolTests(unittest.TestCase):
    def test_picks_named_refs_and_drops_extra(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            store = ChatStore(Path(raw))
            meta = store.create_chat()
            names = []
            for view in ("front", "left", "back", "right", "extra"):
                dest = store.new_file(meta.id, f"{view}.png")
                _rgb(dest)
                names.append(dest.name)
            ctx = SimpleNamespace(
                deps=ChatDeps(chat_id=meta.id, store=store),
            )
            picked, dropped = _pick_images(ctx, names)
            self.assertEqual(len(picked), 4)
            self.assertEqual([label for label, _ in picked], ["front", "left", "back", "right"])
            self.assertEqual(dropped, [names[4]])

    def test_emits_cutout_artifact(self) -> None:
        events: list[dict] = []
        with tempfile.TemporaryDirectory() as raw:
            store = ChatStore(Path(raw))
            meta = store.create_chat()
            src = store.new_file(meta.id, "front.png")
            _rgb(src)
            deps = ChatDeps(chat_id=meta.id, store=store, emit=lambda event: events.append(event))
            ctx = SimpleNamespace(deps=deps)

            def fake_cut(source: Path, dest: Path) -> Path:
                Image.new("RGBA", (24, 24), (200, 40, 40, 50)).save(dest)
                return dest

            with (
                patch("mesh_forge.tools.remove_background.cut_background", fake_cut),
                patch("mesh_forge.tools.remove_background.has_alpha", return_value=False),
            ):
                note = RemoveBackground().run(ctx, images=[src.name])
        self.assertIn("Cutout 1", note)
        arts = [e["artifact"] for e in events if e.get("type") == "artifact"]
        self.assertEqual(len(arts), 1)
        self.assertEqual(arts[0]["view"], "front")
        self.assertEqual(arts[0]["kind"], "image")
        self.assertIn("front", arts[0]["name"])

    def test_skips_already_transparent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            store = ChatStore(Path(raw))
            meta = store.create_chat()
            src = store.new_file(meta.id, "front.png")
            _rgba(src, 60)
            deps = ChatDeps(chat_id=meta.id, store=store)
            ctx = SimpleNamespace(deps=deps)
            with patch(
                "mesh_forge.tools.remove_background.cut_background",
                side_effect=AssertionError("should skip"),
            ):
                note = RemoveBackground().run(ctx, images=[src.name])
        self.assertIn("Already transparent", note)
        self.assertIn(src.name, note)


if __name__ == "__main__":
    unittest.main()
