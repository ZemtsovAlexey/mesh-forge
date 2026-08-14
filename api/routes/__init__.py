from __future__ import annotations

from fastapi import APIRouter

from api.routes import chats, system

router = APIRouter()
router.include_router(system.router)
router.include_router(chats.router)
