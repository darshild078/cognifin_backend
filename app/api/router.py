from fastapi import APIRouter

from app.api.routes.auth import router as auth_router
from app.api.routes.chat import router as chat_router
from app.api.routes.retrieval import router as retrieval_router
from app.api.routes.upload import router as upload_router
from app.api.routes.conversations import router as conversations_router
from app.api.routes.health import router as health_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(auth_router)
api_router.include_router(chat_router)
api_router.include_router(retrieval_router)
api_router.include_router(upload_router)
api_router.include_router(conversations_router)
