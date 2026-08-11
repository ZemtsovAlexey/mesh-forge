from __future__ import annotations

import logging
import os
import sys


def setup_logging(level: str | None = None) -> None:
    """Configure console logging for MeshForge and uvicorn."""
    log_level = (level or os.environ.get("MESHFORGE_LOG_LEVEL", "INFO")).upper()
    numeric = getattr(logging, log_level, logging.INFO)

    fmt = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
    datefmt = "%H:%M:%S"

    logging.basicConfig(
        level=numeric,
        format=fmt,
        datefmt=datefmt,
        stream=sys.stderr,
        force=True,
    )

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        logger.addHandler(handler)
        logger.setLevel(numeric)
        logger.propagate = False

    logging.getLogger("mesh_forge").setLevel(numeric)
    logging.getLogger("api").setLevel(numeric)
