from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form

from app.schemas.common import ApiResponse, success_response
from app.schemas.upload import UploadResponseData
from app.services.upload_service import process_pdf_upload
from app.services.rag_service import rag_service

router = APIRouter(tags=["Upload"])


@router.post("/upload", response_model=ApiResponse[UploadResponseData])
def upload(
    file: UploadFile = File(..., description="PDF document to upload"),
    company_name: str = Form(..., description="Company name identifier"),
    year: Optional[str] = Form(default=None, description="Filing year"),
    document_type: Optional[str] = Form(default=None, description="Filing type"),
):
    result = process_pdf_upload(
        file=file,
        company_name=company_name,
        year=year,
        document_type=document_type,
        corpus_router=rag_service.corpus_router,
    )
    return success_response(message="Document indexed for session.", data=result)
