from __future__ import annotations

from typing import Literal, cast

from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.models.openai import OpenAIChatModelSettings
from pydantic_ai.tools import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.agent.gpu_model import build_chat_model
from mesh_forge.agent.prompt import SYSTEM_PROMPT
from mesh_forge.agent.workspace import compact_history, workspace_instructions
from mesh_forge.config import load_config, normalize_reasoning_effort
from mesh_forge.tools import registered_tools


def _instructions(ctx: RunContext[ChatDeps]) -> str:
    """One system message: Qwen's chat template rejects a second `system` role."""
    brief = workspace_instructions(ctx)
    if brief:
        return f"{SYSTEM_PROMPT.rstrip()}\n\n{brief}"
    return SYSTEM_PROMPT


def build_agent() -> Agent[ChatDeps, str]:
    config = load_config()
    effort = normalize_reasoning_effort(config.llm.reasoning_effort)
    settings: OpenAIChatModelSettings = {
        "openai_reasoning_effort": cast(Literal["low", "medium", "high", "xhigh"], effort),
    }
    return Agent(
        build_chat_model(config),
        deps_type=ChatDeps,
        instructions=_instructions,
        tools=registered_tools(),
        retries=1,
        tool_timeout=1800,
        model_settings=settings,
        capabilities=[ProcessHistory(compact_history)],
    )
