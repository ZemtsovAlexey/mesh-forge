from __future__ import annotations

import unittest
from types import SimpleNamespace

from mesh_forge.backends.lmstudio import completion_delta_parts


class VisionStreamTests(unittest.TestCase):
    def test_content_delta(self) -> None:
        delta = SimpleNamespace(content="Объект: кукла", model_extra={})
        self.assertEqual(completion_delta_parts(delta), [("text", "Объект: кукла")])

    def test_reasoning_then_content(self) -> None:
        delta = SimpleNamespace(
            content="срез",
            reasoning="сравниваю левый и правый кадр",
            model_extra={},
        )
        self.assertEqual(
            completion_delta_parts(delta),
            [
                ("thinking", "сравниваю левый и правый кадр"),
                ("text", "срез"),
            ],
        )

    def test_reasoning_content_object(self) -> None:
        delta = SimpleNamespace(
            content=None,
            reasoning_content={"text": "смотрю юбку"},
            model_extra={},
        )
        self.assertEqual(completion_delta_parts(delta), [("thinking", "смотрю юбку")])

    def test_reasoning_in_model_extra(self) -> None:
        delta = SimpleNamespace(content=None, model_extra={"reasoning": "точка справа"})
        self.assertEqual(completion_delta_parts(delta), [("thinking", "точка справа")])

    def test_dict_delta(self) -> None:
        self.assertEqual(
            completion_delta_parts({"reasoning": "a", "content": "b"}),
            [("thinking", "a"), ("text", "b")],
        )

    def test_empty(self) -> None:
        self.assertEqual(completion_delta_parts(None), [])
        self.assertEqual(completion_delta_parts(SimpleNamespace(content=None, model_extra={})), [])


if __name__ == "__main__":
    unittest.main()
