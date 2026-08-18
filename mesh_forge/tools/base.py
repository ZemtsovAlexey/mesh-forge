from __future__ import annotations

import importlib
import pkgutil
from functools import wraps
from pathlib import Path
from typing import Any, ClassVar

from pydantic_ai import RunContext, Tool

from mesh_forge.tools.async_call import in_thread

VIEW_STAGES: dict[str, str] = {
    "front": "спереди",
    "left": "слева",
    "right": "справа",
    "back": "сзади",
    "views": "виды",
    "mesh": "mesh",
    "image": "изображение",
    "cutout": "фон",
    "concept": "концепт",
    "guided": "правка",
}

_REGISTRY: dict[str, type["MeshTool"]] = {}


class MeshTool:
    """Base class for chat tools. Subclass in its own module; it registers itself."""

    name: ClassVar[str] = ""
    title: ClassVar[str] = ""
    heavy: ClassVar[bool] = False
    expose: ClassVar[bool] = True
    stages: ClassVar[dict[str, str]] = VIEW_STAGES

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls is MeshTool or getattr(cls, "run", None) is MeshTool.run:
            return
        name = cls.name or cls.__module__.rsplit(".", 1)[-1]
        cls.name = name
        if not cls.title:
            cls.title = name.replace("_", " ")
        _REGISTRY[name] = cls

    def run(self, ctx: RunContext, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError

    def to_pydantic_tool(self) -> Tool:
        run = self.run
        name = self.name

        @wraps(run)
        def tracked(ctx: RunContext, *args: Any, **kwargs: Any) -> Any:
            deps = getattr(ctx, "deps", None)
            if deps is not None and hasattr(deps, "note_tool"):
                deps.note_tool(name)
            return run(ctx, *args, **kwargs)

        fn = tracked
        if self.heavy:
            fn = in_thread(fn)
        return Tool(
            fn,
            name=self.name,
            takes_ctx=True,
            metadata={"title": self.title, "stages": dict(self.stages)},
        )


def discover_tools() -> None:
    skip = {"base", "common", "knobs", "async_call"}
    package = Path(__file__).resolve().parent
    for mod in pkgutil.iter_modules([str(package)]):
        if mod.name in skip or mod.name.startswith("_"):
            continue
        importlib.import_module(f"mesh_forge.tools.{mod.name}")


def registered_tools() -> list[Tool]:
    discover_tools()
    return [cls().to_pydantic_tool() for cls in _REGISTRY.values() if cls.expose]


def tool_title(name: str) -> str:
    cls = _REGISTRY.get(name)
    return cls.title if cls else name.replace("_", " ")


def tool_stage_label(name: str, stage: str) -> str:
    if not stage:
        return ""
    cls = _REGISTRY.get(name)
    mapping = cls.stages if cls is not None else VIEW_STAGES
    return mapping.get(stage, VIEW_STAGES.get(stage, stage))
