import os
import shutil
import tempfile
import logging
from uuid import uuid4
from typing import Optional
from fastapi import UploadFile

from app.core.config import settings
from app.core.exceptions import BadRequestException, DependencyUnavailableException
from app.core.constants import ErrorCode
from app.schemas.upload import UploadResponseData
from app.rag.retriever_pipeline import RetrieverPipeline
from app.rag.corpus_manager import CorpusManager

logger = logging.getLogger("cognifin.service.upload")


def process_pdf_upload(
    file: UploadFile,
    company_name: str,
    year: Optional[str] = None,
    document_type: Optional[str] = None,
    corpus_router = None,
) -> UploadResponseData:
    if corpus_router is None:
        raise DependencyUnavailableException(
            message="Server retrieval subsystem is initializing. Please try again in a few seconds.",
            error_code=ErrorCode.CORPUS_NOT_READY,
        )

    year = year or settings.DEFAULT_YEAR
    document_type = document_type or "Annual_Report"

    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix="cognifin_upload_")
        tmp_path = os.path.join(tmp_dir, file.filename or "upload.pdf")

        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        session_pipeline = RetrieverPipeline()
        session_corpus = CorpusManager(session_pipeline)

        num_chunks = session_corpus.add_document(
            pdf_path=tmp_path,
            company=company_name,
            document_type=document_type,
            year=year,
        )

        session_id = str(uuid4())
        corpus_router.register_session(session_id, session_corpus)
        logger.info(f"action=upload_pdf session_id={session_id} company='{company_name}' chunks={num_chunks}")

        return UploadResponseData(
            session_id=session_id,
            chunks=num_chunks,
            company=company_name,
            year=year,
            document_type=document_type,
        )
    except Exception as e:
        logger.error(f"action=upload_pdf_failed error='{e}'", exc_info=True)
        raise BadRequestException(message=f"Failed to process PDF file: {str(e)}")
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
