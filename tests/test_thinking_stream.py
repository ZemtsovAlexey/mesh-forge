from __future__ import annotations

import unittest

from pydantic_ai.messages import PartDeltaEvent, PartStartEvent, TextPart, ThinkingPart, ThinkingPartDelta

from mesh_forge.agent.gpu_model import reasoning_text
from mesh_forge.agent import runner as runner_mod
from mesh_forge.agent.runner import _append_thinking, _finalize_interrupted_turn, _map_agent_event
from mesh_forge.chat.models import ToolCallRecord, UiMessage


class ThinkingStreamTests(unittest.TestCase):
    def test_maps_thinking_start_and_delta(self) -> None:
        start = PartStartEvent(index=0, part=ThinkingPart(content="сначала "))
        self.assertEqual(_map_agent_event(start), {"type": "thinking_delta", "delta": "сначала "})
        delta = PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="посмотрю mesh"))
        self.assertEqual(_map_agent_event(delta), {"type": "thinking_delta", "delta": "посмотрю mesh"})

    def test_text_still_maps(self) -> None:
        start = PartStartEvent(index=1, part=TextPart(content="готово"))
        self.assertEqual(_map_agent_event(start), {"type": "text_delta", "delta": "готово"})

    def test_append_thinking_does_not_touch_content(self) -> None:
        msg = UiMessage(id="a", role="assistant")
        _append_thinking(msg, "думаю")
        _append_thinking(msg, " дальше")
        self.assertEqual(msg.content, "")
        self.assertEqual(len(msg.blocks), 1)
        self.assertEqual(msg.blocks[0].kind, "thinking")
        self.assertEqual(msg.blocks[0].text, "думаю дальше")

    def test_tool_deltas_append_to_running_look(self) -> None:
        from mesh_forge.agent.runner import _append_tool_delta

        msg = UiMessage(id="a", role="assistant")
        msg.tools.append(ToolCallRecord(id="t1", name="look", status="running"))
        _append_tool_delta(msg, "thinking", "сравниваю ")
        _append_tool_delta(msg, "thinking", "ракурсы")
        _append_tool_delta(msg, "text", "Объект: кукла")
        self.assertEqual(msg.tools[0].thinking, "сравниваю ракурсы")
        self.assertEqual(msg.tools[0].summary, "Объект: кукла")
        self.assertEqual(msg.content, "")

    def test_tool_deltas_skip_finished_tool(self) -> None:
        from mesh_forge.agent.runner import _append_tool_delta

        msg = UiMessage(id="a", role="assistant")
        msg.tools.append(ToolCallRecord(id="t1", name="look", status="ok", summary="старое"))
        self.assertIsNone(_append_tool_delta(msg, "text", "новое"))
        self.assertEqual(msg.tools[0].summary, "старое")

    def test_finalize_interrupted_turn_persists_before_sse(self) -> None:
        msg = UiMessage(id="a", role="assistant")
        msg.tools.append(ToolCallRecord(id="t1", name="mask_mesh", status="running"))
        msg.tools.append(ToolCallRecord(id="t2", name="look", status="ok", summary="готово"))
        order: list[str] = []

        def fake_persist() -> None:
            order.append("persist")

        orig_finish = runner_mod.prog.finish
        runner_mod.prog.finish = lambda *args, **kwargs: order.append("finish")
        try:
            events = _finalize_interrupted_turn("chat-1", msg, fake_persist)
        finally:
            runner_mod.prog.finish = orig_finish

        self.assertEqual(msg.tools[0].status, "error")
        self.assertEqual(msg.tools[0].summary, "Прервано")
        self.assertEqual(msg.tools[1].status, "ok")
        self.assertEqual(order, ["finish", "persist"])
        self.assertEqual(
            events,
            [{"type": "tool_end", "id": "t1", "ok": False, "summary": "Прервано"}],
        )

    def test_reasoning_text_from_object(self) -> None:
        self.assertEqual(reasoning_text("abc"), "abc")
        self.assertEqual(reasoning_text({"content": "xyz"}), "xyz")
        self.assertEqual(reasoning_text({"text": "q"}), "q")
        self.assertEqual(reasoning_text(None), "")

    def test_thinking_delta_uses_dict_profile(self) -> None:
        from mesh_forge.agent.gpu_model import GpuOpenAIStreamedResponse

        self.assertFalse(hasattr(GpuOpenAIStreamedResponse._map_thinking_delta, "__wrapped__"))
        source = GpuOpenAIStreamedResponse._map_thinking_delta.__doc__ or ""
        import inspect

        text = inspect.getsource(GpuOpenAIStreamedResponse._map_thinking_delta)
        self.assertNotIn("from_profile", text)
        self.assertIn("openai_chat_thinking_field", text)
