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


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint to verify deployment and dependencies.

    Returns:
        A dictionary with health status and dependency info.
    """
    import sys
    
    health_status: dict = {
        "status": "healthy",
        "python_version": sys.version,
        "dependencies": {},
        "ocr_engines": {
            "text_engine": False,
            "table_engine": False,
        },
    }
    
    # Check PaddlePaddle
    try:
        import paddle
        health_status["dependencies"]["paddlepaddle"] = paddle.__version__
    except ImportError as e:
        health_status["dependencies"]["paddlepaddle"] = f"NOT INSTALLED: {e}"
        health_status["status"] = "degraded"
    
    # Check PaddleOCR
    try:
        import paddleocr
        health_status["dependencies"]["paddleocr"] = paddleocr.__version__
    except ImportError as e:
        health_status["dependencies"]["paddleocr"] = f"NOT INSTALLED: {e}"
        health_status["status"] = "degraded"
    
    # Check NumPy
    try:
        import numpy
        health_status["dependencies"]["numpy"] = numpy.__version__
    except ImportError as e:
        health_status["dependencies"]["numpy"] = f"NOT INSTALLED: {e}"
    
    # Check OpenCV
    try:
        import cv2
        health_status["dependencies"]["opencv"] = cv2.__version__
    except ImportError as e:
        health_status["dependencies"]["opencv"] = f"NOT INSTALLED: {e}"
    
    # Check if OCR engines are initialized
    try:
        service = get_ocr_service()
        health_status["ocr_engines"]["text_engine"] = service._text_engine is not None
        health_status["ocr_engines"]["table_engine"] = service._table_engine is not None
    except Exception as e:
        health_status["ocr_engines"]["error"] = str(e)
    
    return health_status


app.include_router(ocr_router)