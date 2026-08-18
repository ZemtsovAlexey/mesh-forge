from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mesh_forge.chat.store import ChatStore


def _stl(path: Path, name: str) -> Path:
    dest = path / name
    dest.write_bytes(b"solid empty\nendsolid empty\n")
    return dest


class MeshRestoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(root=Path(self.tmp.name))
        self.meta = self.store.create_chat("t")
        self.chat = self.meta.id
        self.files = self.store.files_dir(self.chat)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_source_edit_and_restore_previous(self) -> None:
        source = _stl(self.files, "aaaaaaa1_mesh.stl")
        self.store.set_current_mesh(self.chat, source, role="source")
        edited = _stl(self.files, "bbbbbbb2_smoothed.stl")
        self.store.set_current_mesh(self.chat, edited, role="edit")
        self.assertEqual(self.store.current_mesh(self.chat).name, edited.name)
        self.assertEqual(self.store.source_mesh(self.chat).name, source.name)
        self.assertEqual(self.store.previous_mesh(self.chat).name, source.name)

        restored = self.store.restore_mesh(self.chat, "previous")
        self.assertEqual(restored.name, source.name)
        self.assertEqual(self.store.current_mesh(self.chat).name, source.name)
        self.assertEqual(self.store.previous_mesh(self.chat).name, edited.name)

    def test_restore_source_skips_broken_edit(self) -> None:
        source = _stl(self.files, "aaaaaaa1_mesh.stl")
        self.store.set_current_mesh(self.chat, source, role="source")
        smoothed = _stl(self.files, "bbbbbbb2_smoothed.stl")
        self.store.set_current_mesh(self.chat, smoothed, role="edit")
        repaired = _stl(self.files, "ccccccc3_repaired.stl")
        self.store.set_current_mesh(self.chat, repaired, role="edit")
        self.assertEqual(self.store.previous_mesh(self.chat).name, smoothed.name)

        restored = self.store.restore_mesh(self.chat, "source")
        self.assertEqual(restored.name, source.name)
        self.assertEqual(self.store.current_mesh(self.chat).name, source.name)
        self.assertEqual(self.store.source_mesh(self.chat).name, source.name)

    def test_previous_walks_undo_stack(self) -> None:
        source = _stl(self.files, "aaaaaaa1_mesh.stl")
        self.store.set_current_mesh(self.chat, source, role="source")
        a = _stl(self.files, "bbbbbbb2_smoothed.stl")
        self.store.set_current_mesh(self.chat, a, role="edit")
        b = _stl(self.files, "ccccccc3_removed.stl")
        self.store.set_current_mesh(self.chat, b, role="edit")
        self.assertEqual(self.store.mesh_history(self.chat), [source.name, a.name])
        first = self.store.restore_mesh(self.chat, "previous")
        self.assertEqual(first.name, a.name)
        second = self.store.restore_mesh(self.chat, "previous")
        self.assertEqual(second.name, source.name)

    def test_first_mesh_without_role_becomes_source(self) -> None:
        mesh = _stl(self.files, "uploaded.stl")
        self.store.set_current_mesh(self.chat, mesh)
        self.assertEqual(self.store.source_mesh(self.chat).name, mesh.name)
        self.assertEqual(self.store.current_mesh(self.chat).name, mesh.name)

    def test_edit_clears_mesh_pick(self) -> None:
        source = _stl(self.files, "aaaaaaa1_mesh.stl")
        self.store.set_current_mesh(self.chat, source, role="source")
        self.store.set_mesh_pick(self.chat, 0.8, 0.6, 0.3)
        self.assertEqual(self.store.get_meta(self.chat).mesh_region, "right")
        edited = _stl(self.files, "bbbbbbb2_smoothed.stl")
        self.store.set_current_mesh(self.chat, edited, role="edit")
        meta = self.store.get_meta(self.chat)
        self.assertEqual(meta.mesh_pick, [])
        self.assertEqual(meta.mesh_region, "")


if __name__ == "__main__":
    unittest.main()
