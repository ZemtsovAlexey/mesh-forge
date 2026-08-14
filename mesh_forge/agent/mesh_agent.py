from __future__ import annotations

from pydantic_ai import Agent

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.agent.gpu_model import build_chat_model
from mesh_forge.agent.prompt import SYSTEM_PROMPT
from mesh_forge.config import load_config
from mesh_forge.tools import registered_tools


def build_agent() -> Agent[ChatDeps, str]:
    config = load_config()
    return Agent(
        build_chat_model(config),
        deps_type=ChatDeps,
        instructions=SYSTEM_PROMPT,
        tools=registered_tools(),
        retries=1,
        tool_timeout=1800,
    )
