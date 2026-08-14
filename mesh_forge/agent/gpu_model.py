from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from mesh_forge import progress as prog
from mesh_forge.config import AppConfig
from mesh_forge.runtime import get_gpu_scheduler


class GpuOpenAIChatModel(OpenAIChatModel):
    """OpenAI-compatible LM Studio model that takes the LLM GPU slot per request."""

    async def request(self, messages, model_settings, model_request_parameters):
        project_id = prog.current_project_id()
        prog.raise_if_cancelled(project_id)
        with get_gpu_scheduler().acquire("LM Studio", kind="llm", project_id=project_id):
            prog.raise_if_cancelled(project_id)
            return await super().request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages,
        model_settings,
        model_request_parameters,
        run_context=None,
    ) -> AsyncGenerator[Any, None]:
        project_id = prog.current_project_id()
        prog.raise_if_cancelled(project_id)
        with get_gpu_scheduler().acquire("LM Studio", kind="llm", project_id=project_id):
            prog.raise_if_cancelled(project_id)
            async with super().request_stream(
                messages,
                model_settings,
                model_request_parameters,
                run_context,
            ) as stream:
                yield stream


def build_chat_model(config: AppConfig) -> GpuOpenAIChatModel:
    return GpuOpenAIChatModel(
        config.llm.planner_model,
        provider=OpenAIProvider(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key or "lm-studio",
        ),
    )
