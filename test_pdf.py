"""Test script to send PDFs to OCR endpoints and display results.

Supports caching responses to JSON files for faster subsequent testing.

Usage:
    python test_pdf.py          # Run OCR and save responses to cache
    python test_pdf.py --cached # Use cached responses (skip OCR)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("API_KEY", "test-key")
BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
PDF_PATH = Path("Memo.pdf")

# Cache file paths
CACHE_DIR = Path("cache")
TEXT_CACHE = CACHE_DIR / "memo_text_response.json"
TABLE_CACHE = CACHE_DIR / "memo_table_response.json"

# Required keywords for valid Q&A tables
REQUIRED_TABLE_KEYWORDS = ["Question", "Answer", "Marks"]


def is_valid_qa_table(table: dict[str, Any]) -> bool:
    """Check if a table contains the required Q&A keywords in its HTML.

    A valid Q&A table must contain "Question", "Answer", and "Marks" in its
    HTML content. Tables without these keywords are filtered out.

    Args:
        table: The table dict to check (must have 'html' key).

    Returns:
        True if the table contains all required keywords, False otherwise.
    """
    html = table.get("html")
    if not html:
        return False
    return all(keyword in html for keyword in REQUIRED_TABLE_KEYWORDS)


def filter_qa_tables_in_response(data: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    """Filter tables in a response to only include valid Q&A tables.

    Args:
        data: The response data dict containing 'pages' with 'tables'.

    Returns:
        Tuple of (filtered_data, total_before, total_after).
    """
    total_before = 0
    total_after = 0
    
    for page in data.get("pages", []):
        tables = page.get("tables", [])
        total_before += len(tables)
        filtered_tables = [t for t in tables if is_valid_qa_table(t)]
        total_after += len(filtered_tables)
        page["tables"] = filtered_tables
    
    return data, total_before, total_after


def print_text_response(response: httpx.Response) -> None:
    """Pretty print text OCR response for PDF."""
    print(f"\n{'='*80}")
    print("TEXT OCR RESULTS")
    print(f"{'='*80}")
    print(f"Status Code: {response.status_code}\n")

    if response.status_code != 200:
        print(f"❌ Error: {response.text}")
        return

    data = response.json()
    total_pages = data.get("total_pages", 0)
    pages = data.get("pages", [])

    print(f"📄 Total Pages: {total_pages}")
    print(f"📊 Total Text Blocks Across All Pages: {sum(len(p.get('blocks', [])) for p in pages)}\n")

    for page_data in pages:
        page_num = page_data.get("page", "?")
        blocks = page_data.get("blocks", [])

        print(f"\n{'─'*80}")
        print(f"📖 Page {page_num} - {len(blocks)} text blocks")
        print(f"{'─'*80}")

        if not blocks:
            print("   (No text found)")
            continue

        # Show first 3 blocks in detail
        for i, block in enumerate(blocks[:3], 1):
            text = block.get("text", "")
            confidence = block.get("confidence", 0)
            print(f"\n   Block {i}:")
            print(f"   Text: {text[:100]}{'...' if len(text) > 100 else ''}")
            print(f"   Confidence: {confidence:.2%}")

        if len(blocks) > 3:
            print(f"\n   ... and {len(blocks) - 3} more text blocks")

        # Show all text concatenated
        all_text = " ".join(block.get("text", "") for block in blocks)
        print(f"\n   📝 Full Page Text ({len(all_text)} chars):")
        print(f"   {all_text[:300]}{'...' if len(all_text) > 300 else ''}")


def print_table_response(
    response: httpx.Response,
    tables_before: int | None = None,
    tables_after: int | None = None,
) -> None:
    """Pretty print table OCR response for PDF.

    Args:
        response: The HTTP response from the table endpoint.
        tables_before: Optional count of tables before filtering.
        tables_after: Optional count of tables after filtering.
    """
    print(f"\n{'='*80}")
    print("TABLE OCR RESULTS")
    print(f"{'='*80}")
    print(f"Status Code: {response.status_code}\n")

    if response.status_code != 200:
        print(f"❌ Error: {response.text}")
        return

    data = response.json()
    total_pages = data.get("total_pages", 0)
    pages = data.get("pages", [])

    total_tables = sum(len(p.get("tables", [])) for p in pages)
    print(f"📄 Total Pages: {total_pages}")
    
    # Show filtering stats if available
    if tables_before is not None and tables_after is not None:
        print(f"📊 Tables Found: {tables_before} → Kept: {tables_after} (filtered for Question/Answer/Marks)")
    else:
        print(f"📊 Total Tables Found: {total_tables}\n")

    for page_data in pages:
        page_num = page_data.get("page", "?")
        tables = page_data.get("tables", [])

        print(f"\n{'─'*80}")
        print(f"📖 Page {page_num} - {len(tables)} table(s)")
        print(f"{'─'*80}")

        if not tables:
            print("   (No tables found)")
            continue

        for table_idx, table in enumerate(tables, 1):
            html = table.get("html")
            cells = table.get("cells", [])

            print(f"\n   Table {table_idx}:")
            print(f"   - Cells: {len(cells)}")

            if html:
                # Extract table dimensions from HTML
                row_count = html.count("<tr>")
                col_count = html.count("<td>") // max(row_count, 1) if row_count else 0
                print(f"   - Structure: ~{row_count} rows × ~{col_count} cols")
                print(f"   - HTML Length: {len(html)} chars")

                # Show HTML preview
                html_preview = html[:200].replace("\n", " ")
                print(f"   - HTML Preview: {html_preview}...")

            # Show first few cells with text
            cells_with_text = [c for c in cells if c.get("text")]
            if cells_with_text:
                print(f"\n   Sample Cell Contents:")
                for cell in cells_with_text[:5]:
                    text = cell.get("text", "N/A")
                    bbox = cell.get("bbox", [])
                    print(f"     • {text[:50]}{'...' if len(text) > 50 else ''}")
                if len(cells_with_text) > 5:
                    print(f"     ... and {len(cells_with_text) - 5} more cells with text")


def save_cache(cache_path: Path, data: dict) -> None:
    """Save response data to cache file."""
    CACHE_DIR.mkdir(exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"   💾 Saved to cache: {cache_path}")


def load_cache(cache_path: Path) -> dict | None:
    """Load response data from cache file."""
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def run_live_test() -> None:
    """Run live OCR against the API and save responses to cache."""
    if not PDF_PATH.exists():
        print(f"❌ Error: PDF file '{PDF_PATH}' not found!")
        return

    file_size_mb = PDF_PATH.stat().st_size / (1024 * 1024)
    print(f"\n{'='*80}")
    print(f"🔍 Testing PaddleOCR with PDF: {PDF_PATH}")
    print(f"{'='*80}")
    print(f"📁 File Size: {file_size_mb:.2f} MB")
    print(f"🌐 API Base URL: {BASE_URL}")
    print(f"🔑 Using API Key: {API_KEY[:10]}..." if len(API_KEY) > 10 else f"🔑 Using API Key: {API_KEY}")

    headers = {"X-API-Key": API_KEY}

    # Very long timeout for PDFs - they take time to process
    with httpx.Client(timeout=600.0) as client:
        # Test text endpoint
        print(f"\n\n{'='*80}")
        print("[1] Testing /ocr/text endpoint (this may take 5+ minutes)...")
        print(f"{'='*80}")
        try:
            with open(PDF_PATH, "rb") as f:
                files = {"file": (PDF_PATH.name, f, "application/pdf")}
                response = client.post(
                    f"{BASE_URL}/ocr/text",
                    headers=headers,
                    files=files,
                )
            if response.status_code == 200:
                save_cache(TEXT_CACHE, response.json())
            print_text_response(response)
        except Exception as e:
            print(f"❌ Error calling text endpoint: {e}")

        # Test table endpoint
        print(f"\n\n{'='*80}")
        print("[2] Testing /ocr/table endpoint (this may take 5+ minutes)...")
        print(f"{'='*80}")
        try:
            with open(PDF_PATH, "rb") as f:
                files = {"file": (PDF_PATH.name, f, "application/pdf")}
                response = client.post(
                    f"{BASE_URL}/ocr/table",
                    headers=headers,
                    files=files,
                )
            if response.status_code == 200:
                save_cache(TABLE_CACHE, response.json())
            print_table_response(response)
        except Exception as e:
            print(f"❌ Error calling table endpoint: {e}")

    print(f"\n{'='*80}")
    print("✅ PDF Testing Complete - Responses cached!")
    print(f"{'='*80}\n")


def run_cached_test() -> None:
    """Display results from cached responses (no API calls).

    Applies Q&A table filtering to cached table responses.
    """
    print(f"\n{'='*80}")
    print("📦 Using CACHED responses (no API calls)")
    print(f"{'='*80}")

    # Text response
    print(f"\n\n{'='*80}")
    print("[1] Cached /ocr/text response")
    print(f"{'='*80}")
    text_data = load_cache(TEXT_CACHE)
    if text_data:
        # Create a mock response-like object
        class MockResponse:
            status_code = 200
            def json(self) -> dict:
                return text_data
        print_text_response(MockResponse())
    else:
        print(f"❌ Cache not found: {TEXT_CACHE}")
        print("   Run without --cached first to generate cache.")

    # Table response
    print(f"\n\n{'='*80}")
    print("[2] Cached /ocr/table response (filtered)")
    print(f"{'='*80}")
    table_data = load_cache(TABLE_CACHE)
    if table_data:
        # Apply Q&A table filtering
        filtered_data, tables_before, tables_after = filter_qa_tables_in_response(table_data)
        
        class MockResponse:
            status_code = 200
            def json(self) -> dict:
                return filtered_data
        print_table_response(MockResponse(), tables_before, tables_after)
    else:
        print(f"❌ Cache not found: {TABLE_CACHE}")
        print("   Run without --cached first to generate cache.")

    print(f"\n{'='*80}")
    print("✅ Cached Test Display Complete")
    print(f"{'='*80}\n")


def main() -> None:
    """Send test PDF to both OCR endpoints."""
    use_cache = "--cached" in sys.argv or "-c" in sys.argv

    if use_cache:
        run_cached_test()
    else:
        run_live_test()


if __name__ == "__main__":
    main()
