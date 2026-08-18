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
        self.assertIn("remove_extra", SYSTEM_PROMPT)
        self.assertIn("mask_mesh", SYSTEM_PROMPT)
        self.assertNotIn("select_mesh", SYSTEM_PROMPT)
        self.assertIn("Меш уже есть", SYSTEM_PROMPT)
        self.assertIn("generate_image", SYSTEM_PROMPT)
        self.assertIn("auto-first", SYSTEM_PROMPT)
        self.assertIn("proposal", SYSTEM_PROMPT)
        self.assertIn("needs click", SYSTEM_PROMPT)

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
        self.assertIn("restore_mesh", brief)
        self.assertIn(source.name, brief)
        self.assertIn(edited.name, brief)
        self.assertNotIn("NEXT: regen", brief)

    def test_brief_includes_mesh_click(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("t").id
            files = store.files_dir(chat)
            source = files / "aaaaaaa1_mesh.stl"
            source.write_bytes(b"solid empty\nendsolid empty\n")
            store.set_current_mesh(chat, source, role="source")
            store.set_mesh_pick(chat, 0.9, 0.75, 0.2)
            brief = build_workspace_brief(store, chat)
        self.assertIn("user click at", brief)
        self.assertIn("without region", brief)

    def test_brief_keeps_click_from_user_message_after_consume(self) -> None:
        from mesh_forge.chat.models import UiMessage

        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("t").id
            files = store.files_dir(chat)
            source = files / "aaaaaaa1_mesh.stl"
            source.write_bytes(b"solid empty\nendsolid empty\n")
            store.set_current_mesh(chat, source, role="source")
            store.set_mesh_pick(chat, 0.9, 0.75, 0.2)
            region, pick = store.active_mesh_target(chat)
            store.save_messages(
                chat,
                [
                    UiMessage(
                        id="u1",
                        role="user",
                        content="убери этот нарост",
                        mesh_region=region,
                        mesh_pick=pick,
                    )
                ],
            )
            store.clear_mesh_pick(chat)
            self.assertEqual(store.get_meta(chat).mesh_pick, [])
            brief = build_workspace_brief(store, chat)
        self.assertIn("user click at", brief)
        self.assertIn("without region", brief)

    def test_look_region_does_not_hide_message_click(self) -> None:
        from mesh_forge.chat.models import UiMessage

        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("t").id
            files = store.files_dir(chat)
            source = files / "aaaaaaa1_mesh.stl"
            source.write_bytes(b"solid empty\nendsolid empty\n")
            store.set_current_mesh(chat, source, role="source")
            store.set_mesh_pick(chat, 0.9, 0.75, 0.2)
            region, pick = store.active_mesh_target(chat)
            store.save_messages(
                chat,
                [
                    UiMessage(
                        id="u1",
                        role="user",
                        content="убери лепесток",
                        mesh_region=region,
                        mesh_pick=pick,
                    )
                ],
            )
            store.clear_mesh_pick(chat)
            store.set_mesh_region(chat, "seat")
            _, kept = store.active_mesh_target(chat)
        self.assertGreaterEqual(len(kept), 3)

    def test_brief_allows_second_look_after_one(self) -> None:
        from mesh_forge.chat.models import ToolCallRecord, UiMessage

        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("t").id
            files = store.files_dir(chat)
            source = files / "aaaaaaa1_mesh.stl"
            source.write_bytes(b"solid empty\nendsolid empty\n")
            store.set_current_mesh(chat, source, role="source")
            store.save_messages(
                chat,
                [
                    UiMessage(
                        id="a1",
                        role="assistant",
                        tools=[
                            ToolCallRecord(
                                id="t1",
                                name="look",
                                title="Смотрю",
                                status="ok",
                                summary="юбка справа",
                            )
                        ],
                    )
                ],
            )
            brief = build_workspace_brief(store, chat)
        self.assertNotIn("Do not call look", brief)
        self.assertIn("remove_extra", brief)

    def test_brief_allows_look_after_two(self) -> None:
        from mesh_forge.chat.models import ToolCallRecord, UiMessage

        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("t").id
            files = store.files_dir(chat)
            source = files / "aaaaaaa1_mesh.stl"
            source.write_bytes(b"solid empty\nendsolid empty\n")
            store.set_current_mesh(chat, source, role="source")
            store.save_messages(
                chat,
                [
                    UiMessage(
                        id="a1",
                        role="assistant",
                        tools=[
                            ToolCallRecord(
                                id="t1",
                                name="look",
                                title="Смотрю",
                                status="ok",
                            ),
                            ToolCallRecord(
                                id="t2",
                                name="look",
                                title="Смотрю",
                                status="ok",
                            ),
                        ],
                    )
                ],
            )
            brief = build_workspace_brief(store, chat)
        self.assertNotIn("Two looks already", brief)
        self.assertNotIn("Do not call look again", brief)

    def test_brief_does_not_cap_live_look_counter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("t").id
            files = store.files_dir(chat)
            source = files / "aaaaaaa1_mesh.stl"
            source.write_bytes(b"solid empty\nendsolid empty\n")
            store.set_current_mesh(chat, source, role="source")
            brief = build_workspace_brief(store, chat, looks_without_edit=2)
        self.assertNotIn("Two looks already", brief)
        self.assertNotIn("Do not call look again", brief)

    def test_brief_mask_mentions_auto_review_proposal(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("t").id
            files = store.files_dir(chat)
            source = files / "aaaaaaa1_mesh.stl"
            source.write_bytes(b"solid empty\nendsolid empty\n")
            store.set_current_mesh(chat, source, role="source")
            store.set_mesh_mask(chat, source.name, np.arange(40, dtype=np.int32))
            store.set_mask_state(
                chat,
                {"proposal_status": "ready", "review_verdict": "ok", "candidate_faces": 40},
            )
            store.set_look_view(chat, views="right")
            brief = build_workspace_brief(store, chat)
        self.assertIn("proposal=ready", brief)
        self.assertIn("multi-view detection", brief)
        self.assertIn("auto-review", brief)

    def test_brief_removal_proposal_mentions_remove_extra_apply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("t").id
            files = store.files_dir(chat)
            source = files / "aaaaaaa1_mesh.stl"
            source.write_bytes(b"solid empty\nendsolid empty\n")
            store.set_current_mesh(chat, source, role="source")
            store.set_removal_state(
                chat,
                {"strategy": "protrusion_cut", "proposal_status": "ready", "mesh": source.name},
            )
            brief = build_workspace_brief(store, chat)
        self.assertIn("Removal proposal", brief)
        self.assertIn("remove_extra(apply=True)", brief)

    def test_brief_rejection_blocks_apply_and_keeps_old_goal(self) -> None:
        from mesh_forge.chat.models import UiMessage

        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("t").id
            files = store.files_dir(chat)
            source = files / "aaaaaaa1_mesh.stl"
            source.write_bytes(b"solid empty\nendsolid empty\n")
            store.set_current_mesh(chat, source, role="source")
            store.set_removal_state(
                chat,
                {"strategy": "protrusion_cut", "proposal_status": "ready", "mesh": source.name},
            )
            store.save_messages(
                chat,
                [
                    UiMessage(id="u1", role="user", content="убери лепесток справа"),
                    UiMessage(id="u2", role="user", content="нет"),
                ],
            )
            brief = build_workspace_brief(store, chat)
        self.assertIn("Goal: убери лепесток справа", brief)
        self.assertNotIn("Goal: нет", brief)
        self.assertIn("rejects the current removal proposal", brief)


class ChatMetaStoreTests(unittest.TestCase):
    def test_get_meta_recovers_trailing_junk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("t").id
            path = store._meta_path(chat)
            path.write_text(path.read_text(encoding="utf-8") + '"zoom": 1.0\n}\n', encoding="utf-8")
            meta = store.get_meta(chat)
        self.assertEqual(meta.id, chat)
        self.assertEqual(meta.title, "t")


if __name__ == "__main__":
    unittest.main()
