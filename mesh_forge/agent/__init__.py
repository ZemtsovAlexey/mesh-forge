from mesh_forge.agent.deps import ChatDeps

__all__ = ["ChatDeps", "ChatRunner", "build_agent"]


def __getattr__(name: str):
    if name == "build_agent":
        from mesh_forge.agent.mesh_agent import build_agent

        return build_agent
    if name == "ChatRunner":
        from mesh_forge.agent.runner import ChatRunner

        return ChatRunner
    raise AttributeError(name)
