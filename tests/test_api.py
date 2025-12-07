"""Tests for OCR FastAPI endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from app import routes
from app.config import Settings, get_settings
from app.routes import _get_service
from app.schemas import (
    ImageUrlPayload,
    TableCell,
    TableOcrResponse,
    TableResult,
    TextBlock,
    TextOcrResponse,
)
from main import app

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class FakeService:
    """Mock OCR service returning deterministic results for testing."""

    def run_text(self, image: np.ndarray) -> TextOcrResponse:
        """Return a single text block."""

        block = TextBlock(
            text="hello",
            confidence=0.99,
            box=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        )
        return TextOcrResponse(blocks=[block])

    def run_table(self, image: np.ndarray) -> TableOcrResponse:
        """Return a single table with one cell."""

        cell = TableCell(bbox=[0.0, 0.0, 10.0, 10.0], text="value")
        table = TableResult(html="<table></table>", cells=[cell])
        return TableOcrResponse(tables=[table])


@pytest.fixture()
def client(monkeypatch: "MonkeyPatch") -> TestClient:
    """Return a TestClient with dependencies overridden for isolation."""

    settings = Settings(api_key="test-key")
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[_get_service] = lambda: FakeService()

    async def fake_load_image(
        file: UploadFile | None,
        url_payload: ImageUrlPayload | None,
        settings: Settings,
    ) -> np.ndarray:
        """Return a dummy image array or raise when input is absent."""

        if not file and not url_payload:
            raise HTTPException(
                status_code=400,
                detail="Missing image input. Send a file upload or a URL payload.",
            )
        return np.zeros((5, 5, 3), dtype=np.uint8)

    monkeypatch.setattr(routes, "load_image", fake_load_image)
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


def test_auth_missing_returns_401(client: TestClient) -> None:
    """Requests without API key should be rejected."""

    response = client.post("/ocr/text", json={"url": "https://example.com/image.png"})
    assert response.status_code == 401


def test_text_url_success(client: TestClient) -> None:
    """Successful text OCR returns mocked block structure."""

    response = client.post(
        "/ocr/text",
        json={"url": "https://example.com/image.png"},
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["blocks"][0]["text"] == "hello"
    assert payload["blocks"][0]["confidence"] == pytest.approx(0.99)


def test_table_missing_input_returns_400(client: TestClient) -> None:
    """Missing payload should be rejected with 400."""

    response = client.post("/ocr/table", headers={"X-API-Key": "test-key"})
    assert response.status_code == 400


def test_table_success(client: TestClient) -> None:
    """Successful table OCR returns mocked table data."""

    response = client.post(
        "/ocr/table",
        json={"url": "https://example.com/table.png"},
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["tables"][0]["html"] == "<table></table>"
    assert data["tables"][0]["cells"][0]["text"] == "value"

