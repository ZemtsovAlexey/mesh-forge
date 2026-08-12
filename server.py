"""MeshForge FastAPI server."""

from __future__ import annotations

import logging
import os

import uvicorn

from mesh_forge.config import load_config
from mesh_forge.logging_config import setup_logging

logger = logging.getLogger("mesh_forge.server")


def main() -> None:
    setup_logging()
    cfg = load_config()
    cfg.projects_dir.mkdir(parents=True, exist_ok=True)
    log_level = os.environ.get("MESHFORGE_LOG_LEVEL", "info").lower()
    logger.info(
        "Starting MeshForge on %s:%s (projects=%s)",
        os.environ.get("MESHFORGE_HOST", cfg.server.host),
        int(os.environ.get("MESHFORGE_PORT", cfg.server.port)),
        cfg.projects_dir,
    )
    uvicorn.run(
        "api.main:app",
        host=os.environ.get("MESHFORGE_HOST", cfg.server.host),
        port=int(os.environ.get("MESHFORGE_PORT", cfg.server.port)),
        reload=False,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
