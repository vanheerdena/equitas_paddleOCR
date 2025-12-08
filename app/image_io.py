"""Utilities for loading and validating images and PDFs from uploads or URLs."""

from __future__ import annotations

from typing import List

import cv2
import httpx
import numpy as np
from fastapi import HTTPException, UploadFile, status

from .config import Settings
from .pdf_utils import is_pdf, pdf_to_images
from .schemas import ImageUrlPayload

SUPPORTED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

SUPPORTED_PDF_TYPES = {
    "application/pdf",
}


def _ensure_size_within_limit(data: bytes, max_bytes: int) -> None:
    """Raise if the provided bytes exceed configured size."""

    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image payload exceeds allowed size of {max_bytes} bytes.",
        )


def _validate_content_type(content_type: str | None, filename: str | None = None) -> str:
    """Validate content type and return the file type category.

    Args:
        content_type: MIME type of the file.
        filename: Original filename for fallback detection.

    Returns:
        "image" or "pdf" based on the detected type.

    Raises:
        HTTPException: If the content type is not supported.
    """
    ct_lower = content_type.lower() if content_type else ""

    # Check for PDF
    if ct_lower in SUPPORTED_PDF_TYPES or is_pdf(content_type, filename):
        return "pdf"

    # Check for image
    if ct_lower in SUPPORTED_IMAGE_TYPES:
        return "image"

    # Fallback: check filename extension
    if filename:
        fname_lower = filename.lower()
        if fname_lower.endswith(".pdf"):
            return "pdf"
        if any(fname_lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]):
            return "image"

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Unsupported content type. Please submit an image (JPEG, PNG, etc.) or PDF.",
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


async def load_images_from_upload(file: UploadFile, settings: Settings) -> List[np.ndarray]:
    """Load images from an uploaded file (image or PDF).

    Args:
        file: The uploaded file.
        settings: Application settings.

    Returns:
        List of numpy arrays. Single item for images, multiple for PDFs (one per page).
    """
    file_type = _validate_content_type(file.content_type, file.filename)
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    _ensure_size_within_limit(content, settings.max_image_bytes)

    if file_type == "pdf":
        return pdf_to_images(content)
    else:
        return [_decode_image(content)]


async def download_images(url_payload: ImageUrlPayload, settings: Settings) -> List[np.ndarray]:
    """Download an image or PDF from a remote URL and decode it.

    Args:
        url_payload: The URL payload containing the file URL.
        settings: Application settings.

    Returns:
        List of numpy arrays. Single item for images, multiple for PDFs (one per page).
    """
    url_str = str(url_payload.url)

    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        response = await client.get(url_str, follow_redirects=True)
    if response.status_code >= 400:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to download file from URL.",
        )

    # Determine file type from content-type header or URL
    content_type = response.headers.get("content-type")
    filename = url_str.split("/")[-1].split("?")[0]  # Extract filename from URL
    file_type = _validate_content_type(content_type, filename)

    content_length = response.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > settings.max_image_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Remote file exceeds allowed size.",
                )
        except ValueError:
            pass
    _ensure_size_within_limit(response.content, settings.max_image_bytes)

    if file_type == "pdf":
        return pdf_to_images(response.content)
    else:
        return [_decode_image(response.content)]


async def load_images(
    file: UploadFile | None,
    url_payload: ImageUrlPayload | None,
    settings: Settings,
) -> List[np.ndarray]:
    """Load images from either an upload or a URL payload.

    Supports both images and PDFs. For PDFs, returns one image per page.

    Args:
        file: Optional uploaded file.
        url_payload: Optional URL payload.
        settings: Application settings.

    Returns:
        List of numpy arrays (one per page/image).
    """
    if file and url_payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either an uploaded file or a URL, not both.",
        )
    if not file and not url_payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing input. Send a file upload or a URL payload.",
        )
    if file:
        return await load_images_from_upload(file, settings)
    return await download_images(url_payload, settings)

