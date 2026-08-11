from __future__ import annotations

import logging
import time
import traceback
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger("api.http")


def register_logging(app: FastAPI) -> None:
    @app.middleware("http")
    async def log_requests(request: Request, call_next: Callable) -> Response:
        if request.url.path.startswith("/assets/"):
            return await call_next(request)

        started = time.perf_counter()
        client = request.client.host if request.client else "?"
        logger.info("→ %s %s from %s", request.method, request.url.path, client)
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "✗ %s %s failed after %.0fms",
                request.method,
                request.url.path,
                elapsed_ms,
            )
            raise

        elapsed_ms = (time.perf_counter() - started) * 1000
        level = logging.INFO if response.status_code < 400 else logging.WARNING
        if response.status_code >= 500:
            level = logging.ERROR
        logger.log(
            level,
            "← %s %s %s %.0fms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled %s on %s %s\n%s",
            type(exc).__name__,
            request.method,
            request.url.path,
            traceback.format_exc(),
        )
        return JSONResponse(status_code=500, content={"detail": str(exc)})
