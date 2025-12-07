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
from .schemas import TableCell, TableResult, TextBlock, TextOcrResponse, TableOcrResponse

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

    def run_text(self, image: np.ndarray) -> TextOcrResponse:
        """Execute text OCR on the provided image."""

        ocr_result = self._get_text_engine().ocr(image, cls=True)
        blocks: List[TextBlock] = []
        if not ocr_result or not ocr_result[0]:
            return TextOcrResponse(blocks=blocks)
        for line in ocr_result[0]:
            box, info = line
            text, score = info
            normalized_box = [
                [float(point[0]), float(point[1])] for point in box  # type: ignore[arg-type]
            ]
            blocks.append(
                TextBlock(text=str(text), confidence=float(score), box=normalized_box)
            )
        return TextOcrResponse(blocks=blocks)

    def run_table(self, image: np.ndarray) -> TableOcrResponse:
        """Execute table extraction on the provided image."""

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
        return TableOcrResponse(tables=tables)


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

