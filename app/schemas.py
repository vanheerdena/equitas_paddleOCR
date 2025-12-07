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


class TextOcrResponse(BaseModel):
    """Response containing recognized text blocks."""

    blocks: List[TextBlock] = Field(default_factory=list)


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


class TableOcrResponse(BaseModel):
    """Response containing detected tables."""

    tables: List[TableResult] = Field(default_factory=list)

