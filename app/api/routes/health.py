from fastapi import APIRouter
from app.schemas.common import ApiResponse, success_response
from app.schemas.health import HealthResponseData
from app.services.rag_service import rag_service

router = APIRouter(tags=["System"])


@router.get("/health", response_model=ApiResponse[HealthResponseData])
def health_check():
    indexed = rag_service.corpus_manager is not None and rag_service.corpus_manager.is_indexed
    num_chunks = rag_service.corpus_manager.num_chunks if indexed else 0
    generation_ready = rag_service.llm_client is not None and rag_service.llm_client.is_configured

    data = HealthResponseData(
        status="ok",
        indexed=indexed,
        num_chunks=num_chunks,
        generation_ready=generation_ready,
    )
    return success_response(message="Service is healthy.", data=data)


@router.get("/")
def root():
    return success_response(
        message="Welcome to CogniFin AI API",
        data={
            "docs": "/docs",
            "health": "/health",
        },
    )
