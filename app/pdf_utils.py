"""PDF processing utilities for converting PDFs to images."""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import List, Optional

import numpy as np
from pdf2image import convert_from_bytes
from PIL import Image

logger = logging.getLogger(__name__)


def _get_poppler_path() -> Optional[str]:
    """Get the path to Poppler binaries.

    On Windows, looks for poppler executables in bin/ directory.
    On Linux/Docker, returns None (uses system poppler).

    Returns:
        Path to poppler bin directory, or None to use system PATH.
    """
    if platform.system() != "Windows":
        return None  # Linux/Docker uses system-installed poppler

    # Look for bin/ directory relative to this file's location
    # app/pdf_utils.py -> project_root/bin
    project_root = Path(__file__).parent.parent
    bin_path = project_root / "bin"

    # Check if poppler executables are directly in bin/
    if bin_path.exists() and (bin_path / "pdftoppm.exe").exists():
        logger.debug(f"Using local Poppler at: {bin_path}")
        return str(bin_path)

    # Also check bin/poppler/ subdirectory
    poppler_subdir = bin_path / "poppler"
    if poppler_subdir.exists() and (poppler_subdir / "pdftoppm.exe").exists():
        logger.debug(f"Using local Poppler at: {poppler_subdir}")
        return str(poppler_subdir)

    # Check Library/bin for some Windows Poppler installs (conda-forge layout)
    poppler_lib_path = poppler_subdir / "Library" / "bin"
    if poppler_lib_path.exists() and (poppler_lib_path / "pdftoppm.exe").exists():
        logger.debug(f"Using local Poppler at: {poppler_lib_path}")
        return str(poppler_lib_path)

    logger.warning(
        "Poppler not found in bin/ directory. "
        "PDF support may not work. Download from: "
        "https://github.com/osborn/poppler-windows/releases"
    )
    return None


def pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> List[np.ndarray]:
    """Convert a PDF file to a list of images (one per page).

    Args:
        pdf_bytes: Raw PDF file bytes.
        dpi: Resolution for rendering PDF pages. Higher = better quality but slower.
             Default 200 is good balance for OCR.

    Returns:
        List of numpy arrays (BGR format) suitable for PaddleOCR, one per page.

    Raises:
        ValueError: If PDF conversion fails.
    """
    try:
        # Get poppler path (Windows only)
        poppler_path = _get_poppler_path()

        # Convert PDF to PIL Images
        pil_images: List[Image.Image] = convert_from_bytes(
            pdf_bytes, 
            dpi=dpi,
            poppler_path=poppler_path
        )
        logger.info(f"Converted PDF to {len(pil_images)} page(s)")

        # Convert PIL Images to numpy arrays (RGB -> BGR for OpenCV/PaddleOCR)
        np_images: List[np.ndarray] = []
        for i, pil_img in enumerate(pil_images):
            # Convert to RGB if necessary
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            # Convert to numpy array
            np_img = np.array(pil_img)

            # Convert RGB to BGR (OpenCV/PaddleOCR format)
            np_img = np_img[:, :, ::-1].copy()

            np_images.append(np_img)
            logger.debug(f"Page {i + 1}: shape={np_img.shape}")

        return np_images

    except Exception as e:
        logger.error(f"Failed to convert PDF: {e}")
        raise ValueError(f"Failed to convert PDF to images: {e}") from e


def is_pdf(content_type: str | None, filename: str | None) -> bool:
    """Check if a file is a PDF based on content type or filename.

    Args:
        content_type: MIME type of the file.
        filename: Original filename.

    Returns:
        True if the file appears to be a PDF.
    """
    if content_type and "pdf" in content_type.lower():
        return True
    if filename and filename.lower().endswith(".pdf"):
        return True
    return False
