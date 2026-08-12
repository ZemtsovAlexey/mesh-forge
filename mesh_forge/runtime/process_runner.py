from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger("mesh_forge.process_runner")

LineHandler = Callable[[str], None]


@dataclass
class ProcessResult:
    returncode: int
    elapsed_sec: float
    lines: list[str] = field(default_factory=list)

    def tail(self, count: int = 80) -> str:
        return "\n".join(self.lines[-count:])


class ProcessRunner:
    def run(
        self,
        command: list[str],
        *,
        label: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int = 1800,
        line_handler: LineHandler | None = None,
    ) -> ProcessResult:
        started = time.perf_counter()
        run_env = dict(os.environ)
        if env:
            run_env.update(env)
        run_env.setdefault("PYTHONUNBUFFERED", "1")
        run_env.setdefault("PYTHONIOENCODING", "utf-8")
        logger.info("%s cmd: %s", label, " ".join(command))

        proc = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=run_env,
            bufsize=1,
        )
        assert proc.stdout is not None
        lines: list[str] = []

        def _reader() -> None:
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                lines.append(line)
                if len(lines) > 400:
                    del lines[:200]
                if line_handler is not None:
                    line_handler(line)

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            reader.join(timeout=2)
            raise RuntimeError(f"{label} timed out after {timeout_s}s") from None

        reader.join(timeout=5)
        elapsed = time.perf_counter() - started
        result = ProcessResult(returncode=rc, elapsed_sec=elapsed, lines=lines)
        if rc != 0:
            logger.error("%s failed rc=%s after %.1fs\n%s", label, rc, elapsed, result.tail())
            raise RuntimeError(f"{label} failed:\n{result.tail(60)}")
        logger.info("%s done in %.1fs", label, elapsed)
        return result
