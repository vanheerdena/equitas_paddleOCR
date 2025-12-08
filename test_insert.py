"""Test script to send PDFs to OCR inserts endpoint and display results.

Supports caching responses to JSON files for faster subsequent testing.

Usage:
    python test_insert.py          # Run OCR and save response to cache
    python test_insert.py --cached # Use cached response (skip OCR)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("API_KEY", "test-key")
BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
PDF_PATH = Path("Final Insert.pdf")

# Cache file paths
CACHE_DIR = Path("cache")
INSERTS_CACHE = CACHE_DIR / "memo_inserts_response.json"


def print_inserts_response(response: httpx.Response) -> None:
    """Pretty print inserts OCR response for PDF.

    Args:
        response: The HTTP response from the inserts endpoint.
    """
    print(f"\n{'='*80}")
    print("INSERTS OCR RESULTS (Rotated Text + Images)")
    print(f"{'='*80}")
    print(f"Status Code: {response.status_code}\n")

    if response.status_code != 200:
        print(f"❌ Error: {response.text}")
        return

    data = response.json()
    total_pages = data.get("total_pages", 0)
    pages = data.get("pages", [])

    total_images = sum(len(p.get("images", [])) for p in pages)
    print(f"📄 Total Pages: {total_pages}")
    print(f"🖼️  Total Images Found: {total_images}\n")

    for page_data in pages:
        page_num = page_data.get("page", "?")
        text = page_data.get("text", "")
        images = page_data.get("images", [])
        rotations_with_text = page_data.get("rotations_with_text", [])

        print(f"\n{'─'*80}")
        print(f"📖 Page {page_num}")
        print(f"{'─'*80}")

        # Show which rotations found text
        if rotations_with_text:
            print(f"   🔄 Text found at rotations: {rotations_with_text}°")
        else:
            print("   🔄 No text found at any rotation")

        # Show text summary
        text_lines = text.strip().split("\n") if text.strip() else []
        print(f"   📝 Text lines: {len(text_lines)}")

        if text_lines:
            print(f"\n   First 5 text lines:")
            for i, line in enumerate(text_lines[:5], 1):
                preview = line[:80] + "..." if len(line) > 80 else line
                print(f"     {i}. {preview}")
            if len(text_lines) > 5:
                print(f"     ... and {len(text_lines) - 5} more lines")

        # Show full text preview
        if text:
            print(f"\n   📄 Full Text Preview ({len(text)} chars):")
            preview = text[:500].replace("\n", " ↵ ")
            print(f"   {preview}{'...' if len(text) > 500 else ''}")

        # Show images
        print(f"\n   🖼️  Images detected: {len(images)}")
        for i, img in enumerate(images, 1):
            bbox = img.get("bbox", [])
            b64 = img.get("base64", "")
            rotation = img.get("rotation", 0)
            # Calculate approximate size from base64 length
            approx_size_kb = len(b64) * 3 / 4 / 1024
            print(f"     Image {i}: bbox={bbox}, rotation={rotation}°, ~{approx_size_kb:.1f} KB")


def print_inserts_from_data(data: dict) -> None:
    """Pretty print inserts OCR response from dict data.

    Args:
        data: The response data dictionary.
    """

    class MockResponse:
        """Mock response object for printing cached data."""

        status_code = 200

        def json(self) -> dict:
            """Return the cached data."""
            return data

    print_inserts_response(MockResponse())


def save_cache(cache_path: Path, data: dict) -> None:
    """Save response data to cache file.

    Args:
        cache_path: Path to save the cache file.
        data: Response data to cache.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"   💾 Saved to cache: {cache_path}")


def load_cache(cache_path: Path) -> dict | None:
    """Load response data from cache file.

    Args:
        cache_path: Path to the cache file.

    Returns:
        Cached data or None if not found.
    """
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def run_live_test() -> None:
    """Run live OCR against the API and save response to cache."""
    if not PDF_PATH.exists():
        print(f"❌ Error: PDF file '{PDF_PATH}' not found!")
        return

    file_size_mb = PDF_PATH.stat().st_size / (1024 * 1024)
    print(f"\n{'='*80}")
    print(f"🔍 Testing PaddleOCR INSERTS with PDF: {PDF_PATH}")
    print(f"{'='*80}")
    print(f"📁 File Size: {file_size_mb:.2f} MB")
    print(f"🌐 API Base URL: {BASE_URL}")
    print(f"🔑 Using API Key: {API_KEY[:10]}..." if len(API_KEY) > 10 else f"🔑 Using API Key: {API_KEY}")
    print("\n⚠️  This will run OCR at 4 rotations per page - expect 4x normal time!")

    headers = {"X-API-Key": API_KEY}

    # Very long timeout - 4x rotations takes a while
    with httpx.Client(timeout=1200.0) as client:
        print(f"\n\n{'='*80}")
        print("[1] Testing /ocr/inserts endpoint (this may take 10+ minutes)...")
        print(f"{'='*80}")
        try:
            with open(PDF_PATH, "rb") as f:
                files = {"file": (PDF_PATH.name, f, "application/pdf")}
                response = client.post(
                    f"{BASE_URL}/ocr/inserts",
                    headers=headers,
                    files=files,
                )
            if response.status_code == 200:
                save_cache(INSERTS_CACHE, response.json())
            print_inserts_response(response)
        except Exception as e:
            print(f"❌ Error calling inserts endpoint: {e}")

    print(f"\n{'='*80}")
    print("✅ Inserts Testing Complete - Response cached!")
    print(f"{'='*80}\n")


def run_cached_test() -> None:
    """Display results from cached response (no API calls)."""
    print(f"\n{'='*80}")
    print("📦 Using CACHED response (no API calls)")
    print(f"{'='*80}")

    print(f"\n\n{'='*80}")
    print("[1] Cached /ocr/inserts response")
    print(f"{'='*80}")
    data = load_cache(INSERTS_CACHE)
    if data:
        print_inserts_from_data(data)
    else:
        print(f"❌ Cache not found: {INSERTS_CACHE}")
        print("   Run without --cached first to generate cache.")

    print(f"\n{'='*80}")
    print("✅ Cached Test Display Complete")
    print(f"{'='*80}\n")


def main() -> None:
    """Send test PDF to inserts OCR endpoint."""
    use_cache = "--cached" in sys.argv or "-c" in sys.argv

    if use_cache:
        run_cached_test()
    else:
        run_live_test()


if __name__ == "__main__":
    main()
