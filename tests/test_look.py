from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mesh_forge.chat.store import ChatStore
from mesh_forge.tools.look import (
    _mask_next_camera,
    _mask_review_question,
    _prefer_named_images,
    default_mesh_look,
    parse_look_shots,
)


class LookShotsTests(unittest.TestCase):
    def test_default_is_overview(self) -> None:
        shots = parse_look_shots("")
        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].camera, "viewer")
        self.assertEqual(shots[0].zoom, 1.0)

    def test_orbit_is_four_sides(self) -> None:
        shots = parse_look_shots("orbit")
        self.assertEqual([s.camera for s in shots], ["front", "left", "back", "right"])

    def test_comma_list_and_aliases(self) -> None:
        shots = parse_look_shots("спереди, сверху")
        self.assertEqual([s.camera for s in shots], ["front", "top"])

    def test_region_raises_zoom(self) -> None:
        shots = parse_look_shots("overview", region="спинка")
        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].camera, "viewer")
        self.assertEqual(shots[0].region, "backrest")
        self.assertGreater(shots[0].zoom, 1.5)

    def test_detail_without_region_uses_few_closeups(self) -> None:
        shots = parse_look_shots("detail", zoom=2.0)
        self.assertGreaterEqual(len(shots), 2)
        self.assertLessEqual(len(shots), 4)
        self.assertTrue(all(s.zoom == 2.0 for s in shots))

    def test_caps_at_four(self) -> None:
        shots = parse_look_shots("orbit,top,viewer")
        self.assertEqual(len(shots), 4)

    def test_yaw_list_is_custom_orbit(self) -> None:
        shots = parse_look_shots("20, 90, -40", pitch=12.0, zoom=2.0)
        self.assertEqual(len(shots), 3)
        self.assertEqual([s.camera for s in shots], ["custom", "custom", "custom"])
        self.assertEqual([s.yaw for s in shots], [20.0, 90.0, -40.0])
        self.assertTrue(all(s.pitch == 12.0 for s in shots))

    def test_yaw_param_without_views(self) -> None:
        shots = parse_look_shots("", yaw=55.0, pitch=-10.0)
        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].camera, "custom")
        self.assertEqual(shots[0].yaw, 55.0)
        self.assertEqual(shots[0].pitch, -10.0)

    def test_named_views_with_yaw_stay_distinct(self) -> None:
        shots = parse_look_shots("viewer,left,right", yaw=45.0, pitch=-10.0, zoom=2.0)
        self.assertEqual(len(shots), 3)
        yaws = [s.yaw for s in shots]
        self.assertEqual(len(set(yaws)), 3)
        self.assertTrue(all(s.pitch == -10.0 for s in shots))

    def test_orbit_around_free_yaw_is_four_distinct(self) -> None:
        shots = parse_look_shots("orbit", yaw=45.0, pitch=-10.0)
        self.assertEqual(len(shots), 4)
        yaws = [s.yaw for s in shots]
        self.assertEqual(len(set(yaws)), 4)

    def test_default_mesh_look_free_camera_skips_side_pack(self) -> None:
        views, _ = default_mesh_look(free_camera=True)
        self.assertEqual(views, "")

    def test_region_without_views_keeps_overview(self) -> None:
        shots = parse_look_shots("", region="низ")
        self.assertEqual(len(shots), 2)
        self.assertEqual(shots[0].zoom, 1.0)
        self.assertEqual(shots[0].region, "")
        self.assertEqual(shots[1].region, "bottom")
        self.assertGreater(shots[1].zoom, 1.5)

    def test_default_mesh_look_compares_sides(self) -> None:
        views, question = default_mesh_look()
        self.assertIn("left", views)
        self.assertIn("right", views)
        self.assertEqual(question, "")

    def test_default_mesh_look_keeps_explicit_views(self) -> None:
        views, question = default_mesh_look("front", question="спинка?")
        self.assertEqual(views, "front")
        self.assertEqual(question, "спинка?")

    def test_default_mesh_look_mask_is_one_camera(self) -> None:
        views, _ = default_mesh_look(mask=True, start_view="right")
        self.assertEqual(views, "right")
        views, _ = default_mesh_look("front,left,right", mask=True)
        self.assertEqual(views, "front")
        views, _ = default_mesh_look("orbit", mask=True, start_view="left")
        self.assertEqual(views, "left")

    def test_mask_next_camera_skips_already_seen(self) -> None:
        self.assertEqual(_mask_next_camera("right", ["right"]), "left")
        self.assertEqual(_mask_next_camera("right", ["right", "left"]), "front")

    def test_mask_review_question_names_this_frame(self) -> None:
        text = _mask_review_question("лепесток", camera="right", seen=["right"])
        self.assertIn("кадр с маской: right", text)
        self.assertIn("без команд NEXT", text)
        text = _mask_review_question("", camera="front", seen=["right", "front"])
        self.assertIn("Уже смотрели: right", text)

    def test_default_mesh_look_with_pick_zooms_on_click(self) -> None:
        views, question = default_mesh_look(pick=True)
        self.assertIn("front", views)
        self.assertIn("оранжевая", question.lower())
        self.assertIn("face", question.lower())


class VisionPromptTests(unittest.TestCase):
    def test_mesh_prompt_does_not_ask_to_match_camera_label(self) -> None:
        from mesh_forge.backends.lmstudio import inspect_vision_prompt

        text = inspect_vision_prompt(
            kind="mesh",
            question="на юбке справа есть лепестковый отросток, удали его",
        )
        self.assertIn("Запрос пользователя", text)
        self.assertIn("лепестковый", text)
        self.assertIn("НЕ запрос", text)
        self.assertNotIn("NEXT:", text)

    def test_mask_prompt_is_overlay_review_only(self) -> None:
        from mesh_forge.backends.lmstudio import inspect_vision_prompt

        text = inspect_vision_prompt(kind="mask", question="отросток на юбке справа")
        self.assertIn("proposal", text)
        self.assertIn("без строки NEXT", text)
        self.assertIn("слишком много", text)
        self.assertNotIn("NEXT:", text)

    def test_photo_prompt_keeps_next_line(self) -> None:
        from mesh_forge.backends.lmstudio import inspect_vision_prompt

        text = inspect_vision_prompt(kind="photo")
        self.assertIn("NEXT: mesh", text)

    def test_parse_multi_view_mask_detection(self) -> None:
        from mesh_forge.backends.lmstudio import parse_multi_view_mask_detection

        data = parse_multi_view_mask_detection(
            """
            {
              "observations": [
                {"view":"right","visible":true,"confidence":0.9,"x0":0.7,"y0":0.5,"x1":0.9,"y1":0.8,"kind":"protrusion","touchesBody":true,"note":"виден"},
                {"view":"left","visible":false,"confidence":0.2,"kind":"unknown","touchesBody":false,"note":"не видно"}
              ]
            }
            """
        )
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["view"], "right")
        self.assertTrue(data[0]["visible"])
        self.assertEqual(data[1]["view"], "left")
        self.assertFalse(data[1]["visible"])

    def test_parse_multi_view_mask_detection_rejects_json_array(self) -> None:
        from mesh_forge.backends.lmstudio import parse_multi_view_mask_detection

        self.assertEqual(parse_multi_view_mask_detection('[{"view":"right"}]'), [])

    def test_parse_mask_review_accepts_strict_fail_labels(self) -> None:
        from mesh_forge.backends.lmstudio import parse_mask_review

        tiny = parse_mask_review('{"verdict":"tiny_spot","confidence":0.9,"note":"tiny","views":"right","x":0.5,"y":0.5}')
        part = parse_mask_review('{"verdict":"partial","confidence":0.8,"note":"part","views":"front","x":0.4,"y":0.6}')
        self.assertEqual(tiny["verdict"], "tiny_spot")
        self.assertEqual(part["verdict"], "partial")

    def test_parse_mask_review_rejects_json_array(self) -> None:
        from mesh_forge.backends.lmstudio import parse_mask_review

        self.assertEqual(parse_mask_review('[{"verdict":"ok"}]'), {})


class PhotoLookTests(unittest.TestCase):
    def test_prefer_named_images_keeps_all_when_unlabeled(self) -> None:
        items = [("1-127", Path("1-127.jpg"))]
        self.assertEqual(_prefer_named_images(items, "front"), items)

    def test_images_target_with_views_does_not_need_mesh(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch

        from mesh_forge.agent.deps import ChatDeps
        from mesh_forge.tools.look import Look

        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("t").id
            photo = store.files_dir(chat) / "1-127.jpg"
            photo.write_bytes(b"\xff\xd8\xff")
            art = store.artifact_from_path(chat, photo, label="front", view="front")
            deps = ChatDeps(chat_id=chat, store=store, attachments=[art])
            ctx = SimpleNamespace(deps=deps)
            with patch("mesh_forge.tools.look.LMStudioClient") as client_cls:
                client_cls.return_value.inspect_images.return_value = "NEXT: cutout"
                note = Look().run(ctx, target="images", views="front")
            self.assertEqual(note, "NEXT: cutout")
            kwargs = client_cls.return_value.inspect_images.call_args.kwargs
            self.assertEqual(kwargs.get("kind"), "auto")
            seen = client_cls.return_value.inspect_images.call_args.args[0]
            self.assertEqual(Path(seen[0][1]).name, "1-127.jpg")

    def test_auto_views_falls_back_to_photo_without_mesh(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch

        from mesh_forge.agent.deps import ChatDeps
        from mesh_forge.tools.look import Look

        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(root=Path(tmp))
            chat = store.create_chat("t").id
            photo = store.files_dir(chat) / "shot.jpg"
            photo.write_bytes(b"\xff\xd8\xff")
            art = store.artifact_from_path(chat, photo, label="photo")
            deps = ChatDeps(chat_id=chat, store=store, attachments=[art])
            ctx = SimpleNamespace(deps=deps)
            with patch("mesh_forge.tools.look.LMStudioClient") as client_cls:
                client_cls.return_value.inspect_images.return_value = "NEXT: mesh"
                note = Look().run(ctx, target="auto", views="front")
            self.assertEqual(note, "NEXT: mesh")
            self.assertEqual(
                client_cls.return_value.inspect_images.call_args.kwargs.get("kind"),
                "auto",
            )


if __name__ == "__main__":
    unittest.main()
