"""OCR engine management and result parsing."""

from __future__ import annotations

# CRITICAL: Import paddle_config FIRST to set environment variables
# before any PaddlePaddle imports occur
from . import paddle_config

import logging
from typing import Iterable, List, Optional

import numpy as np

# Configure paddle runtime settings after import
paddle_config.configure_paddle()

from paddleocr import PPStructure, PaddleOCR

from .config import Settings, get_settings
import base64
import cv2

from .schemas import (
    DetectedImage,
    LayoutRegion,
    PageInsertResult,
    SinglePageLayoutResult,
    SinglePageTableResult,
    SinglePageTextResult,
    TableCell,
    TableResult,
    TextBlock,
)

logger = logging.getLogger(__name__)


def _normalize_bbox(raw_bbox: Iterable[float]) -> Optional[List[float]]:
    """Normalize a raw bounding box to four float coordinates.

    Args:
        raw_bbox: An iterable of coordinate values.

    Returns:
        A list of four float coordinates, or None if insufficient values.
    """
    bbox = list(raw_bbox)
    if len(bbox) < 4:
        return None
    normalized = [float(coord) for coord in bbox[:4]]
    return normalized


class OcrService:
    """Wrapper around PaddleOCR engines for text and table extraction.

    This service manages PaddleOCR and PPStructure engines, providing
    lazy initialization and caching to avoid repeated model downloads.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the service with configuration.

        Args:
            settings: Application settings containing OCR configuration.
        """
        self.settings = settings
        self._text_engine: PaddleOCR | None = None
        self._table_engine: PPStructure | None = None

    def initialize_engines(self) -> None:
        """Pre-initialize all OCR engines at startup.

        This method should be called during application startup (lifespan)
        to ensure models are downloaded and loaded before handling requests.
        This prevents model downloads from happening during request handling.
        """
        logger.info("Initializing OCR engines (this may download models on first run)...")
        _ = self._get_text_engine()
        logger.info("Text OCR engine initialized")
        _ = self._get_table_engine()
        logger.info("Table OCR engine initialized")
        logger.info("All OCR engines ready")

    def _get_text_engine(self) -> PaddleOCR:
        """Lazily instantiate the text OCR engine.

        Returns:
            The PaddleOCR text engine instance.
        """
        if self._text_engine is None:
            self._text_engine = PaddleOCR(
                use_angle_cls=True,
                lang=self.settings.ocr_lang,
                enable_mkldnn=False,
                use_gpu=False,
            )
        return self._text_engine

    def _get_table_engine(self) -> PPStructure:
        """Lazily instantiate the table OCR engine.

        Returns:
            The PPStructure table engine instance.
        """
        if self._table_engine is None:
            self._table_engine = PPStructure(
                show_log=False,
                lang=self.settings.ocr_lang,
                enable_mkldnn=False,
                use_gpu=False,
            )
        return self._table_engine

    def run_text_single(self, image: np.ndarray) -> SinglePageTextResult:
        """Execute text OCR on a single image/page.

        Args:
            image: The image as a numpy array (BGR format).

        Returns:
            SinglePageTextResult containing recognized text blocks.
        """
        ocr_result = self._get_text_engine().ocr(image, cls=True)
        blocks: List[TextBlock] = []
        if not ocr_result or not ocr_result[0]:
            return SinglePageTextResult(blocks=blocks)
        for line in ocr_result[0]:
            box, info = line
            text, score = info
            normalized_box = [
                [float(point[0]), float(point[1])] for point in box  # type: ignore[arg-type]
            ]
            blocks.append(
                TextBlock(text=str(text), confidence=float(score), box=normalized_box)
            )
        return SinglePageTextResult(blocks=blocks)

    def run_table_single(self, image: np.ndarray) -> SinglePageTableResult:
        """Execute table extraction on a single image/page.

        Args:
            image: The image as a numpy array (BGR format).

        Returns:
            SinglePageTableResult containing detected tables.
        """
        raw_result = self._get_table_engine()(image) or []
        if not isinstance(raw_result, list):
            raw_result = [raw_result]
        tables: List[TableResult] = []
        for entry in raw_result:
            html: str | None = None
            cells: List[TableCell] = []

            if isinstance(entry, dict):
                html_candidate = entry.get("html")
                if isinstance(html_candidate, str):
                    html = html_candidate
                res = entry.get("res")
                if isinstance(res, dict) and isinstance(res.get("html"), str) and html is None:
                    html = res["html"]  # type: ignore[index]
                cell_candidates = entry.get("cell_bbox") or []
                if isinstance(res, dict) and res.get("cell_bbox"):
                    cell_candidates = res["cell_bbox"]  # type: ignore[assignment]
                for cell in cell_candidates:
                    bbox: Optional[List[float]] = None
                    text_value: str | None = None
                    if isinstance(cell, dict):
                        bbox = _normalize_bbox(cell.get("bbox") or cell.get("box") or [])
                        text_value = cell.get("text")
                    elif isinstance(cell, (list, tuple)):
                        bbox = _normalize_bbox(cell)
                    if bbox:
                        cells.append(TableCell(bbox=bbox, text=text_value))
            tables.append(TableResult(html=html, cells=cells))
        return SinglePageTableResult(tables=tables)

    def run_layout_single(self, image: np.ndarray) -> SinglePageLayoutResult:
        """Execute layout analysis on a single image/page.

        Returns ALL detected regions including figures, text, titles, tables.
        This is useful for finding image/figure bounding boxes.

        Args:
            image: The image as a numpy array (BGR format).

        Returns:
            SinglePageLayoutResult containing all detected regions with types.
        """
        raw_result = self._get_table_engine()(image) or []
        if not isinstance(raw_result, list):
            raw_result = [raw_result]

        regions: List[LayoutRegion] = []
        for entry in raw_result:
            if not isinstance(entry, dict):
                continue

            region_type = entry.get("type", "unknown")
            raw_bbox = entry.get("bbox", [])

            # Normalize bbox to [x1, y1, x2, y2]
            bbox = _normalize_bbox(raw_bbox)
            if not bbox:
                continue

            # Extract confidence if available
            confidence = entry.get("score") or entry.get("confidence")
            if confidence is not None:
                confidence = float(confidence)

            # Extract text/html based on region type
            text_content: str | None = None
            html_content: str | None = None
            res = entry.get("res")

            if region_type == "table":
                # Tables have HTML in res
                if isinstance(res, dict):
                    html_content = res.get("html")
                elif isinstance(entry.get("html"), str):
                    html_content = entry.get("html")
            elif region_type in ("text", "title", "header", "footer"):
                # Text regions have text content
                if isinstance(res, list):
                    # res is list of text lines
                    text_parts = []
                    for line in res:
                        if isinstance(line, dict) and "text" in line:
                            text_parts.append(line["text"])
                        elif isinstance(line, (list, tuple)) and len(line) >= 2:
                            # Format: [bbox, (text, conf)]
                            if isinstance(line[1], (list, tuple)):
                                text_parts.append(str(line[1][0]))
                    text_content = " ".join(text_parts) if text_parts else None

            regions.append(
                LayoutRegion(
                    type=region_type,
                    bbox=bbox,
                    confidence=confidence,
                    text=text_content,
                    html=html_content,
                )
            )

        return SinglePageLayoutResult(regions=regions)

    def run_inserts_single(self, image: np.ndarray, page_num: int = 1) -> PageInsertResult:
        """Detect text at all rotations and extract figure images.

        Rotates the image 0°, 90°, 180°, 270° and runs OCR on each to catch
        text at any orientation. Also detects figure regions and extracts
        them as base64 encoded images.

        Args:
            image: The image as a numpy array (BGR format).
            page_num: Page number for the result.

        Returns:
            PageInsertResult with combined text and extracted images.
        """
        rotations = [0, 90, 180, 270]
        all_texts: List[str] = []
        rotations_with_text: List[int] = []
        detected_images: List[DetectedImage] = []
        seen_texts: set[str] = set()  # For deduplication

        for rotation in rotations:
            # Rotate image
            if rotation == 0:
                rotated = image
            elif rotation == 90:
                rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                rotated = cv2.rotate(image, cv2.ROTATE_180)
            else:  # 270
                rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Run text OCR
            text_result = self.run_text_single(rotated)
            page_texts: List[str] = []
            for block in text_result.blocks:
                text = block.text.strip()
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    page_texts.append(text)
                    all_texts.append(text)

            if page_texts:
                rotations_with_text.append(rotation)

            # Run layout detection to find figures (only on 0° rotation)
            if rotation == 0:
                layout_result = self.run_layout_single(image)
                for region in layout_result.regions:
                    if region.type == "figure":
                        # Extract the figure region
                        bbox = region.bbox
                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        # Ensure valid bounds
                        h, w = image.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            cropped = image[y1:y2, x1:x2]
                            # Encode as base64 PNG
                            success, buffer = cv2.imencode(".png", cropped)
                            if success and buffer is not None:
                                b64_string = base64.b64encode(buffer).decode("utf-8")
                                detected_images.append(
                                    DetectedImage(
                                        bbox=bbox,
                                        base64=b64_string,
                                        rotation=0,
                                    )
                                )
                            else:
                                logger.warning(
                                    f"Failed to encode figure region at bbox {bbox}"
                                )

        # Combine all text with newlines
        combined_text = "\n".join(all_texts)

        return PageInsertResult(
            page=page_num,
            text=combined_text,
            images=detected_images,
            rotations_with_text=rotations_with_text,
        )


_ocr_service_cache: OcrService | None = None


def get_ocr_service(settings: Settings | None = None) -> OcrService:
    """Return a cached OCR service instance."""

    global _ocr_service_cache
    
    # For simplicity, cache a single instance since settings don't change at runtime
    # If settings are explicitly provided and differ, create a new instance
    if _ocr_service_cache is None:
        resolved_settings = settings or get_settings()
        _ocr_service_cache = OcrService(resolved_settings)
    
    return _ocr_service_cache

