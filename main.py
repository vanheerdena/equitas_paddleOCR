"""FastAPI application entrypoint."""

# CRITICAL: Import paddle_config FIRST to set environment variables
# before any PaddlePaddle imports occur
import app.paddle_config  # noqa: F401 - side effects only

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.ocr_service import get_ocr_service
from app.routes import router as ocr_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifespan events.

    This context manager initializes OCR engines at startup to ensure
    model weights are downloaded and loaded before handling requests.
    This prevents slow first-request response times and repeated downloads.

    Args:
        app: The FastAPI application instance.

    Yields:
        None after startup initialization is complete.
    """
    # Startup: Initialize OCR engines (downloads models if needed)
    logger.info("Application startup: Initializing OCR service...")
    service = get_ocr_service()
    service.initialize_engines()
    logger.info("Application startup complete: OCR service ready")

    yield

    # Shutdown: Cleanup if needed
    logger.info("Application shutdown")


app = FastAPI(
    title="PaddleOCR Service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint providing basic service information.

    Returns:
        A dictionary with a status message.
    """
    return {"message": "PaddleOCR wrapper is running."}


app.include_router(ocr_router)