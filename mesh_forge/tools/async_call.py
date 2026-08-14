from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import wraps
from typing import Any


def in_thread(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Run a sync tool on a worker thread so SSE progress can keep flowing."""

    @wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(fn, *args, **kwargs)

    return wrapper

