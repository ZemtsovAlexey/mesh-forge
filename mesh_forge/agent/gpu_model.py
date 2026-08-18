from __future__ import annotations

from collections.abc import AsyncGenerator, Iterable
from contextlib import asynccontextmanager
from typing import Any

from pydantic_ai.models.openai import OpenAIChatModel, OpenAIStreamedResponse
from pydantic_ai.providers.openai import OpenAIProvider

from mesh_forge import progress as prog
from mesh_forge.config import AppConfig, llm_http_timeout, llm_uses_gpu
from mesh_forge.runtime import acquire_llm


def reasoning_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "reasoning"):
            raw = value.get(key)
            if isinstance(raw, str) and raw:
                return raw
    return ""


class GpuOpenAIStreamedResponse(OpenAIStreamedResponse):
    """LM Studio may put CoT on reasoning / reasoning_content, sometimes as an object."""

    def _map_thinking_delta(self, choice) -> Iterable[Any]:
        delta = choice.delta
        extra = getattr(delta, "model_extra", None) or {}
        profile = self._model_profile
        custom = profile.get("openai_chat_thinking_field") if isinstance(profile, dict) else None
        for field_name in (custom, "reasoning", "reasoning_content"):
            if not field_name:
                continue
            raw = getattr(delta, field_name, None)
            if raw is None and isinstance(extra, dict):
                raw = extra.get(field_name)
            text = reasoning_text(raw)
            if not text:
                continue
            yield from self._parts_manager.handle_thinking_delta(
                vendor_part_id=field_name,
                id=field_name,
                content=text,
                provider_name=self.provider_name,
            )
            return


class GpuOpenAIChatModel(OpenAIChatModel):
    """OpenAI-compatible chat model; local LM Studio also takes the LLM GPU slot."""

    @property
    def _streamed_response_cls(self) -> type[OpenAIStreamedResponse]:
        return GpuOpenAIStreamedResponse

    async def request(self, messages, model_settings, model_request_parameters):
        project_id = prog.current_project_id()
        prog.raise_if_cancelled(project_id)
        with acquire_llm(project_id=project_id):
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
        with acquire_llm(project_id=project_id):
            prog.raise_if_cancelled(project_id)
            async with super().request_stream(
                messages,
                model_settings,
                model_request_parameters,
                run_context,
            ) as stream:
                yield stream


def build_chat_model(config: AppConfig) -> GpuOpenAIChatModel:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=config.llm.base_url,
        api_key=config.llm.api_key or "lm-studio",
        timeout=llm_http_timeout(config),
    )
    local = llm_uses_gpu(config)
    return GpuOpenAIChatModel(
        config.llm.planner_model,
        provider=OpenAIProvider(openai_client=client),
        profile={
            "openai_chat_supports_multiple_system_messages": not local,
        },
    )
