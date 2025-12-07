"""Test script to send images to OCR endpoints and display results."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("API_KEY", "test-key")
BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
IMAGE_PATH = Path("pg1.png")


def print_response(title: str, response: httpx.Response) -> None:
    """Pretty print API response."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nResponse JSON:\n{json.dumps(data, indent=2)}")
        
        # Extract and display text for text endpoint
        if "blocks" in data:
            print(f"\nExtracted Text Blocks ({len(data['blocks'])}):")
            print("-" * 60)
            for i, block in enumerate(data["blocks"], 1):
                print(f"{i}. Text: {block['text']}")
                print(f"   Confidence: {block['confidence']:.4f}")
                print(f"   Box: {block['box']}")
        
        # Extract and display tables for table endpoint
        if "tables" in data:
            print(f"\nDetected Tables ({len(data['tables'])}):")
            print("-" * 60)
            for i, table in enumerate(data["tables"], 1):
                print(f"\nTable {i}:")
                if table.get("html"):
                    print(f"  HTML: {table['html'][:200]}..." if len(table['html']) > 200 else f"  HTML: {table['html']}")
                print(f"  Cells: {len(table.get('cells', []))}")
                for j, cell in enumerate(table.get("cells", [])[:5], 1):  # Show first 5 cells
                    print(f"    Cell {j}: {cell.get('text', 'N/A')} at {cell.get('bbox')}")
                if len(table.get("cells", [])) > 5:
                    print(f"    ... and {len(table['cells']) - 5} more cells")
    else:
        print(f"Error: {response.text}")


def main() -> None:
    """Send test image to both OCR endpoints."""
    
    if not IMAGE_PATH.exists():
        print(f"Error: Image file '{IMAGE_PATH}' not found!")
        return
    
    headers = {"X-API-Key": API_KEY}
    
    print(f"Testing OCR endpoints with image: {IMAGE_PATH}")
    print(f"API Base URL: {BASE_URL}")
    print(f"Using API Key: {API_KEY[:10]}..." if len(API_KEY) > 10 else f"Using API Key: {API_KEY}")
    
    with httpx.Client(timeout=60.0) as client:
        # Test text endpoint
        print("\n\n[1] Testing /ocr/text endpoint...")
        try:
            with open(IMAGE_PATH, "rb") as f:
                files = {"file": (IMAGE_PATH.name, f, "image/png")}
                response = client.post(
                    f"{BASE_URL}/ocr/text",
                    headers=headers,
                    files=files,
                )
            print_response("Text OCR Response", response)
        except Exception as e:
            print(f"Error calling text endpoint: {e}")
        
        # Test table endpoint
        print("\n\n[2] Testing /ocr/table endpoint...")
        try:
            with open(IMAGE_PATH, "rb") as f:
                files = {"file": (IMAGE_PATH.name, f, "image/png")}
                response = client.post(
                    f"{BASE_URL}/ocr/table",
                    headers=headers,
                    files=files,
                )
            print_response("Table OCR Response", response)
        except Exception as e:
            print(f"Error calling table endpoint: {e}")


if __name__ == "__main__":
    main()

