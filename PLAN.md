# MeshForge — 3D-агент в чате

Локальный чат: пользователь пишет и прикрепляет файлы, агент вызывает тулы (картинки, Hunyuan mesh, ремонт, масштаб). UI как Cursor/Claude Code — лента, карточки тулов, картинки и интерактивный 3D.

Стек: pydantic-ai + FastAPI SSE + Vite/React, ComfyUI и trimesh как реализации тулов. Шаговый pipeline / notebook / отдельный workspace больше не используются.

Подробности реализации — в коде `mesh_forge/agent`, `mesh_forge/tools`, `mesh_forge/chat`, `web/src`.
