import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.middleware.sessions import SessionMiddleware

from app.core.config import settings
from app.core.logging import setup_logging, logger
from app.core.middleware import RequestIdAndLoggingMiddleware
from app.core.database import ensure_indexes
from app.core.exceptions import AppException
from app.core.handlers import (
    app_exception_handler,
    validation_exception_handler,
    unhandled_exception_handler,
)
from app.services.rag_service import rag_service
from app.api.router import api_router

# Setup structured logging
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing CogniFin AI application...")
    ensure_indexes()
    try:
        rag_service.initialize()
    except Exception as e:
        logger.error(f"RAG initialization warning/error: {e}", exc_info=True)
    logger.info("CogniFin AI application ready.")
    yield
    logger.info("CogniFin AI application shutdown.")


app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.ENABLE_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_DOCS else None,
    openapi_url="/openapi.json" if settings.ENABLE_DOCS else None,
)

# Request ID & Logging Middleware
app.add_middleware(RequestIdAndLoggingMiddleware)

# Session Middleware (Required by Authlib for Google OAuth state)
app.add_middleware(SessionMiddleware, secret_key=settings.JWT_SECRET)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handlers
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# Static PDF Viewer Mount
pdf_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
if os.path.isdir(pdf_dir):
    app.mount("/pdfs", StaticFiles(directory=pdf_dir), name="pdfs")

# Include all API routes (No v1 prefix as requested)
app.include_router(api_router)
