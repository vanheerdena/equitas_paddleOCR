"""Pydantic schemas for request and response models."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl, validator


class ImageUrlPayload(BaseModel):
    """JSON payload representing an image hosted at a URL."""

    url: HttpUrl


class TextBlock(BaseModel):
    """Recognized text block with bounding box and confidence."""

    text: str = Field(..., description="Recognized text content.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score.")
    box: List[List[float]] = Field(
        ...,
        description="Quadrilateral bounding box, list of four [x,y] points.",
    )

    @validator("box")
    def validate_box(cls, v: List[List[float]]) -> List[List[float]]:
        """Validate that box has exactly 4 points, each with 2 coordinates.

        Args:
            v: The box value to validate.

        Returns:
            The validated box value.

        Raises:
            ValueError: If box format is invalid.
        """
        if len(v) != 4:
            raise ValueError(f"box must have exactly 4 points, got {len(v)}")
        for i, point in enumerate(v):
            if len(point) != 2:
                raise ValueError(f"point {i} must have 2 coordinates [x, y], got {len(point)}")
        return v


class PageTextResult(BaseModel):
    """Text OCR result for a single page."""

    page: int = Field(..., description="Page number (1-indexed).")
    blocks: List[TextBlock] = Field(default_factory=list)


class SinglePageTextResult(BaseModel):
    """Internal result type for single page text OCR (used by service)."""

    blocks: List[TextBlock] = Field(default_factory=list)


class TextOcrResponse(BaseModel):
    """Response containing recognized text blocks.

    For single images, returns one page. For PDFs, returns multiple pages.
    """

    pages: List[PageTextResult] = Field(default_factory=list)
    total_pages: int = Field(default=0, description="Total number of pages processed.")


class TableCell(BaseModel):
    """Single table cell with bounding box and optional text."""

    bbox: List[float] = Field(
        ...,
        description="Bounding box for the cell in [x1, y1, x2, y2] format.",
        min_items=4,
        max_items=4,
    )
    text: Optional[str] = Field(
        default=None, description="Text content recognized within the cell."
    )


class TableResult(BaseModel):
    """Detected table result, optionally including HTML and cells."""

    html: Optional[str] = Field(
        default=None, description="HTML representation of the detected table."
    )
    cells: List[TableCell] = Field(default_factory=list)


class PageTableResult(BaseModel):
    """Table OCR result for a single page."""

    page: int = Field(..., description="Page number (1-indexed).")
    tables: List[TableResult] = Field(default_factory=list)


class SinglePageTableResult(BaseModel):
    """Internal result type for single page table OCR (used by service)."""

    tables: List[TableResult] = Field(default_factory=list)


class TableOcrResponse(BaseModel):
    """Response containing detected tables.

    For single images, returns one page. For PDFs, returns multiple pages.
    """

    pages: List[PageTableResult] = Field(default_factory=list)
    total_pages: int = Field(default=0, description="Total number of pages processed.")


# =============================================================================
# LAYOUT ANALYSIS SCHEMAS - For detecting regions including figures/images
# =============================================================================


class LayoutRegion(BaseModel):
    """A detected region from layout analysis."""

    type: str = Field(
        ...,
        description="Region type: 'text', 'title', 'figure', 'table', 'equation', 'header', 'footer'",
    )
    bbox: List[float] = Field(
        ...,
        description="Bounding box [x1, y1, x2, y2] for the region.",
        min_items=4,
        max_items=4,
    )
    confidence: Optional[float] = Field(
        default=None, description="Detection confidence score."
    )
    text: Optional[str] = Field(
        default=None, description="Extracted text (for text/title regions)."
    )
    html: Optional[str] = Field(
        default=None, description="HTML content (for table regions)."
    )


class PageLayoutResult(BaseModel):
    """Layout analysis result for a single page."""

    page: int = Field(..., description="Page number (1-indexed).")
    regions: List[LayoutRegion] = Field(default_factory=list)


class SinglePageLayoutResult(BaseModel):
    """Internal result type for single page layout analysis."""

    regions: List[LayoutRegion] = Field(default_factory=list)


class LayoutOcrResponse(BaseModel):
    """Response containing all detected layout regions.

    This includes figures, text blocks, titles, tables, etc.
    Use this to find image bounding boxes in documents.
    """

    pages: List[PageLayoutResult] = Field(default_factory=list)
    total_pages: int = Field(default=0, description="Total number of pages processed.")


# =============================================================================
# INSERTS SCHEMA - For detecting rotated text and extracting images
# =============================================================================


class DetectedImage(BaseModel):
    """A detected image/figure extracted from the document."""

    bbox: List[float] = Field(
        ...,
        description="Bounding box [x1, y1, x2, y2] where image was found.",
        min_items=4,
        max_items=4,
    )
    base64: str = Field(..., description="Base64 encoded image (PNG format).")
    rotation: int = Field(
        default=0,
        description="Rotation angle (0, 90, 180, 270) where the image was detected.",
    )


class PageInsertResult(BaseModel):
    """Insert detection result for a single page."""

    page: int = Field(..., description="Page number (1-indexed).")
    text: str = Field(
        default="",
        description="All detected text from all rotations, deduplicated and combined.",
    )
    images: List[DetectedImage] = Field(
        default_factory=list,
        description="Extracted images/figures as base64 encoded PNGs.",
    )
    rotations_with_text: List[int] = Field(
        default_factory=list,
        description="Which rotations (0, 90, 180, 270) found text.",
    )


class InsertsOcrResponse(BaseModel):
    """Response for insert detection - text from all orientations + extracted images.

    This endpoint rotates each page 0°, 90°, 180°, 270° to detect text at any
    orientation, then extracts any detected figure/image regions.
    """

    pages: List[PageInsertResult] = Field(default_factory=list)
    total_pages: int = Field(default=0, description="Total number of pages processed.")

