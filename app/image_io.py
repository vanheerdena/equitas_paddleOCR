"""Utilities for loading and validating images from uploads or URLs."""

from __future__ import annotations

import cv2
import httpx
import numpy as np
from fastapi import HTTPException, UploadFile, status

from .config import Settings
from .schemas import ImageUrlPayload

SUPPORTED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}


def _ensure_size_within_limit(data: bytes, max_bytes: int) -> None:
    """Raise if the provided bytes exceed configured size."""

    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image payload exceeds allowed size of {max_bytes} bytes.",
        )


def _validate_content_type(content_type: str | None) -> None:
    """Ensure MIME type is one of the supported image formats."""

    if content_type and content_type.lower() in SUPPORTED_MIME_TYPES:
        return
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Unsupported content type. Please submit a standard image format.",
    )


def _decode_image(data: bytes) -> np.ndarray:
    """Decode raw bytes into an OpenCV image array."""

    image_array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to decode image content.",
        )
    return image


async def load_image_from_upload(file: UploadFile, settings: Settings) -> np.ndarray:
    """Load an image from an uploaded file."""

    _validate_content_type(file.content_type)
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    _ensure_size_within_limit(content, settings.max_image_bytes)
    return _decode_image(content)


async def download_image(url_payload: ImageUrlPayload, settings: Settings) -> np.ndarray:
    """Download an image from a remote URL and decode it."""

    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        response = await client.get(str(url_payload.url), follow_redirects=True)
    if response.status_code >= 400:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to download image from URL.",
        )
    _validate_content_type(response.headers.get("content-type"))
    content_length = response.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > settings.max_image_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Remote image exceeds allowed size.",
                )
        except ValueError:
            pass
    _ensure_size_within_limit(response.content, settings.max_image_bytes)
    return _decode_image(response.content)


async def load_image(
    file: UploadFile | None,
    url_payload: ImageUrlPayload | None,
    settings: Settings,
) -> np.ndarray:
    """Load an image from either an upload or a URL payload."""

    if file and url_payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either an uploaded file or a URL, not both.",
        )
    if not file and not url_payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing image input. Send a file upload or a URL payload.",
        )
    if file:
        return await load_image_from_upload(file, settings)
    return await download_image(url_payload, settings)

