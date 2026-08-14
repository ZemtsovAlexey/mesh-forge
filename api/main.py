from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.logging_middleware import register_logging
from api.routes import chats, system
from mesh_forge.config import load_config
from mesh_forge.logging_config import setup_logging

logger = logging.getLogger("api.main")
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
DIST_DIR = WEB_DIR / "dist"


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title="MeshForge", version="2.0.0")
    register_logging(app)
    app.include_router(system.router)
    app.include_router(chats.router)

    assets = DIST_DIR / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/")
    def index():
        index_file = DIST_DIR / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)
        fallback = WEB_DIR / "index.html"
        if fallback.is_file():
            return FileResponse(fallback)
        return {"detail": "UI not built. Run npm install && npm run build in web/"}

    return app


app = create_app()


@app.on_event("startup")
def _startup() -> None:
    cfg = load_config()
    cfg.projects_dir.mkdir(parents=True, exist_ok=True)
    logger.info("MeshForge API ready — chats dir: %s", cfg.projects_dir)
