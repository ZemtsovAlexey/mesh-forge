from __future__ import annotations

import unittest

from mesh_forge import progress as prog


class ProgressCancelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pid = "chat-cancel-test"
        prog.clear(self.pid)

    def tearDown(self) -> None:
        prog.clear(self.pid)

    def test_stop_after_finish_does_not_block_next_start(self) -> None:
        prog.start(self.pid, "chat", "agent")
        prog.request_cancel(self.pid)
        self.assertTrue(prog.is_cancelled(self.pid))
        prog.finish(self.pid, ok=False, error="Остановлено")
        self.assertFalse(prog.is_cancelled(self.pid))
        prog.request_cancel(self.pid)
        self.assertFalse(prog.is_cancelled(self.pid))
        prog.start(self.pid, "chat", "agent")
        self.assertFalse(prog.is_cancelled(self.pid))
        state = prog.get(self.pid)
        self.assertIsNotNone(state)
        self.assertTrue(state.active)

    def test_stop_before_start_still_cancels_that_turn(self) -> None:
        prog.clear(self.pid)
        prog.request_cancel(self.pid)
        self.assertTrue(prog.is_cancelled(self.pid))
        prog.start(self.pid, "chat", "agent")
        self.assertTrue(prog.is_cancelled(self.pid))


if __name__ == "__main__":
    unittest.main()
