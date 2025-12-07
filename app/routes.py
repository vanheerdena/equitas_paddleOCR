"""API routes for OCR operations."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, Depends, File, HTTPException, UploadFile, status

from .config import Settings, get_settings
from .image_io import load_image
from .ocr_service import OcrService, get_ocr_service
from .schemas import ImageUrlPayload, TableOcrResponse, TextOcrResponse
from .security import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocr", tags=["ocr"])


def _get_service(settings: Settings = Depends(get_settings)) -> OcrService:
    """Provide a cached OCR service instance with configured settings."""

    return get_ocr_service(settings)


@router.post(
    "/text",
    response_model=TextOcrResponse,
    dependencies=[Depends(verify_api_key)],
)
async def ocr_text(
    file: UploadFile | None = File(default=None),
    payload: ImageUrlPayload | None = Body(default=None),
    settings: Settings = Depends(get_settings),
    service: OcrService = Depends(_get_service),
) -> TextOcrResponse:
    """Perform text OCR on an uploaded image or a URL-hosted image."""

    try:
        image = await load_image(file=file, url_payload=payload, settings=settings)
        return service.run_text(image)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in text OCR endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(e)}",
        ) from e


@router.post(
    "/table",
    response_model=TableOcrResponse,
    dependencies=[Depends(verify_api_key)],
)
async def ocr_table(
    file: UploadFile | None = File(default=None),
    payload: ImageUrlPayload | None = Body(default=None),
    settings: Settings = Depends(get_settings),
    service: OcrService = Depends(_get_service),
) -> TableOcrResponse:
    """Perform table extraction on an uploaded image or a URL-hosted image."""

    try:
        image = await load_image(file=file, url_payload=payload, settings=settings)
        return service.run_table(image)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in table OCR endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(e)}",
        ) from e

