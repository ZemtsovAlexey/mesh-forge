from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.logging_middleware import register_logging
from api.routes import projects, system
from mesh_forge.config import load_config
from mesh_forge.logging_config import setup_logging

logger = logging.getLogger("api.main")
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title="MeshForge", version="1.0.0")
    register_logging(app)
    app.include_router(system.router)
    app.include_router(projects.router)

    if WEB_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=WEB_DIR / "assets"), name="assets")

        @app.get("/")
        def index():
            return FileResponse(WEB_DIR / "index.html")

    return app


app = create_app()


@app.on_event("startup")
def _startup() -> None:
    cfg = load_config()
    cfg.projects_dir.mkdir(parents=True, exist_ok=True)
    logger.info("MeshForge API ready — projects dir: %s", cfg.projects_dir)
