from __future__ import annotations

from functools import lru_cache

from mesh_forge.agent.runner import ChatRunner
from mesh_forge.chat.store import ChatStore


@lru_cache(maxsize=1)
def get_store() -> ChatStore:
    return ChatStore()


@lru_cache(maxsize=1)
def get_runner() -> ChatRunner:
    return ChatRunner(get_store())
