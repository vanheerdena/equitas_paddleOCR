"""API routes for OCR operations."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import numpy as np
from fastapi import APIRouter, Body, Depends, File, HTTPException, UploadFile, status

from .config import Settings, get_settings
from .image_io import load_images
from .ocr_service import OcrService, get_ocr_service
from .schemas import (
    ImageUrlPayload,
    InsertsOcrResponse,
    LayoutOcrResponse,
    PageInsertResult,
    PageLayoutResult,
    PageTableResult,
    PageTextResult,
    TableOcrResponse,
    TableResult,
    TextOcrResponse,
)

# Cache filenames
INSERTS_CACHE_FILE = "memo_inserts_response.json"
from .security import verify_api_key

logger = logging.getLogger(__name__)

# Required keywords for valid Q&A tables
REQUIRED_TABLE_KEYWORDS = ["Question", "Answer", "Marks"]


def is_valid_qa_table(table: TableResult) -> bool:
    """Check if a table contains the required Q&A keywords in its HTML.

    A valid Q&A table must contain "Question", "Answer", and "Marks" in its
    HTML content. Tables without these keywords are filtered out.

    Args:
        table: The TableResult to check.

    Returns:
        True if the table contains all required keywords, False otherwise.
    """
    if not table.html:
        return False
    
    html_content = table.html
    return all(keyword in html_content for keyword in REQUIRED_TABLE_KEYWORDS)


def filter_qa_tables(tables: List[TableResult]) -> List[TableResult]:
    """Filter a list of tables to only include valid Q&A tables.

    Args:
        tables: List of TableResult objects to filter.

    Returns:
        Filtered list containing only tables with Question/Answer/Marks headers.
    """
    return [table for table in tables if is_valid_qa_table(table)]

router = APIRouter(prefix="/ocr", tags=["ocr"])

# Cache directory for mock responses
CACHE_DIR = Path(__file__).parent.parent / "cache"


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
    """Perform text OCR on an uploaded image/PDF or a URL-hosted image/PDF.

    For PDFs, each page is processed separately and results are collated.
    """
    try:
        images: List[np.ndarray] = await load_images(
            file=file, url_payload=payload, settings=settings
        )

        pages: List[PageTextResult] = []
        for page_num, image in enumerate(images, start=1):
            page_result = service.run_text_single(image)
            pages.append(PageTextResult(page=page_num, blocks=page_result.blocks))

        return TextOcrResponse(pages=pages, total_pages=len(pages))

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
    """Perform table extraction on an uploaded image/PDF or a URL-hosted image/PDF.

    For PDFs, each page is processed separately and results are collated.
    Tables are filtered to only include those containing Question/Answer/Marks headers.
    """
    try:
        images: List[np.ndarray] = await load_images(
            file=file, url_payload=payload, settings=settings
        )

        pages: List[PageTableResult] = []
        total_tables_before = 0
        total_tables_after = 0
        
        for page_num, image in enumerate(images, start=1):
            page_result = service.run_table_single(image)
            total_tables_before += len(page_result.tables)
            
            # Filter tables to only include valid Q&A tables
            filtered_tables = filter_qa_tables(page_result.tables)
            total_tables_after += len(filtered_tables)
            
            pages.append(PageTableResult(page=page_num, tables=filtered_tables))
        
        logger.info(
            f"Table filtering: {total_tables_before} tables found, "
            f"{total_tables_after} tables kept (containing Question/Answer/Marks)"
        )

        return TableOcrResponse(pages=pages, total_pages=len(pages))

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in table OCR endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(e)}",
        ) from e


@router.post(
    "/layout",
    response_model=LayoutOcrResponse,
    dependencies=[Depends(verify_api_key)],
)
async def ocr_layout(
    file: UploadFile | None = File(default=None),
    payload: ImageUrlPayload | None = Body(default=None),
    settings: Settings = Depends(get_settings),
    service: OcrService = Depends(_get_service),
) -> LayoutOcrResponse:
    """Perform layout analysis to detect ALL region types including figures/images.

    This endpoint returns bounding boxes for:
    - "figure" - images, photos, diagrams
    - "text" - text paragraphs
    - "title" - titles/headings
    - "table" - tables (with HTML)
    - "equation" - formulas
    - "header"/"footer" - page headers/footers

    Use this to find image bounding boxes in documents.
    For PDFs, each page is processed separately.
    """
    try:
        images: List[np.ndarray] = await load_images(
            file=file, url_payload=payload, settings=settings
        )

        pages: List[PageLayoutResult] = []
        for page_num, image in enumerate(images, start=1):
            page_result = service.run_layout_single(image)
            pages.append(PageLayoutResult(page=page_num, regions=page_result.regions))

        return LayoutOcrResponse(pages=pages, total_pages=len(pages))

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in layout OCR endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(e)}",
        ) from e


@router.post(
    "/inserts",
    response_model=InsertsOcrResponse,
    dependencies=[Depends(verify_api_key)],
)
async def ocr_inserts(
    file: UploadFile | None = File(default=None),
    payload: ImageUrlPayload | None = Body(default=None),
    settings: Settings = Depends(get_settings),
    service: OcrService = Depends(_get_service),
) -> InsertsOcrResponse:
    """Detect text at ALL orientations and extract figure images.

    This endpoint is designed for documents with rotated content (like inserts,
    photos with captions at various angles). It:

    1. Rotates each page 0°, 90°, 180°, 270°
    2. Runs OCR on each rotation to catch text at any angle
    3. Deduplicates and combines all detected text
    4. Extracts any figure/image regions as base64 PNGs

    Returns a simplified response with just text and images per page.
    """
    try:
        images: List[np.ndarray] = await load_images(
            file=file, url_payload=payload, settings=settings
        )

        pages: List[PageInsertResult] = []
        for page_num, image in enumerate(images, start=1):
            page_result = service.run_inserts_single(image, page_num)
            pages.append(page_result)

        return InsertsOcrResponse(pages=pages, total_pages=len(pages))

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in inserts OCR endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(e)}",
        ) from e


# =============================================================================
# CACHE ENDPOINTS - Return pre-saved responses for fast testing
# =============================================================================


@router.post(
    "/text/cache",
    response_model=TextOcrResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["cache"],
)
async def ocr_text_cached(
    file: UploadFile | None = File(default=None),
    payload: ImageUrlPayload | None = Body(default=None),
) -> TextOcrResponse:
    """Return cached text OCR response (for testing - ignores uploaded file).

    Returns the pre-saved response from cache/memo_text_response.json.
    Use this for fast testing without running actual OCR.
    """
    cache_file = CACHE_DIR / "memo_text_response.json"

    if not cache_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cache file not found: {cache_file}. Run test_pdf.py first to generate.",
        )

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return TextOcrResponse(**data)
    except Exception as e:
        logger.exception("Error loading cached text response")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load cache: {str(e)}",
        ) from e


@router.post(
    "/table/cache",
    response_model=TableOcrResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["cache"],
)
async def ocr_table_cached(
    file: UploadFile | None = File(default=None),
    payload: ImageUrlPayload | None = Body(default=None),
) -> TableOcrResponse:
    """Return cached table OCR response (for testing - ignores uploaded file).

    Returns the pre-saved response from cache/memo_table_response.json.
    Tables are filtered to only include those containing Question/Answer/Marks headers.
    Use this for fast testing without running actual OCR.
    """
    cache_file = CACHE_DIR / "memo_table_response.json"

    if not cache_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cache file not found: {cache_file}. Run test_pdf.py first to generate.",
        )

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Apply Q&A table filtering to cached data
        response = TableOcrResponse(**data)
        total_before = sum(len(page.tables) for page in response.pages)
        
        for page in response.pages:
            page.tables = filter_qa_tables(page.tables)
        
        total_after = sum(len(page.tables) for page in response.pages)
        logger.info(
            f"Cached table filtering: {total_before} tables found, "
            f"{total_after} tables kept (containing Question/Answer/Marks)"
        )
        
        return response
    except Exception as e:
        logger.exception("Error loading cached table response")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load cache: {str(e)}",
        ) from e


@router.post(
    "/inserts/cache",
    response_model=InsertsOcrResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["cache"],
)
async def ocr_inserts_cached(
    file: UploadFile | None = File(default=None),
    payload: ImageUrlPayload | None = Body(default=None),
) -> InsertsOcrResponse:
    """Return cached inserts OCR response (for testing - ignores uploaded file).

    Returns the pre-saved response from cache/memo_inserts_response.json.
    Use this for fast testing without running actual OCR.
    """
    cache_file = CACHE_DIR / INSERTS_CACHE_FILE

    if not cache_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cache file not found: {cache_file}. Run test_insert.py first to generate.",
        )

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return InsertsOcrResponse(**data)
    except Exception as e:
        logger.exception("Error loading cached inserts response")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load cache: {str(e)}",
        ) from e

