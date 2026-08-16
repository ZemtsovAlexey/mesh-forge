from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mesh_forge.agent.prompt import SYSTEM_PROMPT
from mesh_forge.agent.workspace import build_workspace_brief
from mesh_forge.chat.store import ChatStore


class WorkspaceMeshEditTests(unittest.TestCase):
    def test_prompt_has_restore_not_regen_for_existing_mesh(self) -> None:
        self.assertIn("restore_mesh", SYSTEM_PROMPT)
        self.assertIn("carve_mesh", SYSTEM_PROMPT)
        self.assertIn("подлокотник", SYSTEM_PROMPT)
        self.assertIn("Меш уже есть", SYSTEM_PROMPT)
        self.assertIn("НЕ вызывай generate_image", SYSTEM_PROMPT)

    def test_brief_without_mesh_keeps_regen_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("t").id
            brief = build_workspace_brief(store, chat)
        self.assertIn("Current mesh: none", brief)
        self.assertIn("NEXT: regen", brief)
        self.assertNotIn("Mesh-edit mode", brief)

    def test_brief_with_mesh_is_edit_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("t").id
            files = store.files_dir(chat)
            source = files / "aaaaaaa1_mesh.stl"
            source.write_bytes(b"solid empty\nendsolid empty\n")
            store.set_current_mesh(chat, source, role="source")
            edited = files / "bbbbbbb2_smoothed.stl"
            edited.write_bytes(b"solid empty\nendsolid empty\n")
            store.set_current_mesh(chat, edited, role="edit")
            brief = build_workspace_brief(store, chat)
        self.assertIn("Mesh-edit mode", brief)
        self.assertIn("do NOT generate_image", brief)
        self.assertIn("restore_mesh", brief)
        self.assertIn(source.name, brief)
        self.assertIn(edited.name, brief)
        self.assertNotIn("NEXT: regen", brief)


if __name__ == "__main__":
    unittest.main()
