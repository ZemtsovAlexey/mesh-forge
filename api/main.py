from __future__ import annotations

import logging

from fastapi import FastAPI

from api.frontend import attach_frontend
from api.logging_middleware import register_logging
from api.routes import chats, system
from mesh_forge.config import load_config
from mesh_forge.logging_config import setup_logging

logger = logging.getLogger("api.main")


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title="MeshForge", version="2.0.0")
    register_logging(app)
    app.include_router(system.router)
    app.include_router(chats.router)
    attach_frontend(app)
    return app


app = create_app()


@app.on_event("startup")
def _startup() -> None:
    cfg = load_config()
    cfg.projects_dir.mkdir(parents=True, exist_ok=True)
    logger.info("MeshForge API ready — chats dir: %s", cfg.projects_dir)
