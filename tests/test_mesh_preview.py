from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import trimesh
from PIL import Image

from mesh_forge.chat.store import ChatStore


class MeshPreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(root=Path(self.tmp.name))
        self.meta = self.store.create_chat("t")
        self.chat = self.meta.id

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_ensure_mesh_preview_writes_png_and_caches(self) -> None:
        mesh = trimesh.creation.box(extents=[1.0, 1.2, 0.8])
        dest = self.store.new_file(self.chat, "matched.stl")
        mesh.export(dest)
        preview = self.store.ensure_mesh_preview(self.chat, dest)
        self.assertTrue(preview.is_file())
        self.assertTrue(preview.name.endswith(".preview.png"))
        img = Image.open(preview)
        self.assertGreaterEqual(img.size[0], 64)
        self.assertGreaterEqual(img.size[1], 64)
        img.close()
        again = self.store.ensure_mesh_preview(self.chat, dest)
        self.assertEqual(again, preview)
        names = {art.name for art in self.store.list_files(self.chat)}
        self.assertIn(dest.name, names)
        self.assertNotIn(preview.name, names)


if __name__ == "__main__":
    unittest.main()
