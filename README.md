---
title: FastAPI
description: A FastAPI server
tags:
  - fastapi
  - python
---

# PaddleOCR FastAPI Service

FastAPI wrapper around PaddleOCR for English text and table extraction. Supports file uploads and URL-based images with simple API-key auth.

## Quickstart

1. Install dependencies
   - `pip install -r requirements.txt`
2. Configure environment
   - Copy `.env` and set required variables (see below).
3. Run locally
   - `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
4. Open docs at `http://localhost:8000/docs`.

## Environment

Set as local `.env` or Railway project variables:

```
API_KEY=your-long-api-key
OCR_LANG=en
MAX_IMAGE_BYTES=10485760
REQUEST_TIMEOUT_SECONDS=10
```

`API_KEY` is required. Others are optional and default to the values shown.

## Endpoints

All endpoints require header `X-API-Key: <API_KEY>`.

- `POST /ocr/text` — returns recognized text blocks (`text`, `confidence`, `box`).
- `POST /ocr/table` — returns detected tables (HTML plus cell bboxes/text when available).

### Input formats

- Multipart upload: `file` field containing the image.
- JSON: `{"url": "https://your-r2-bucket/path/image.png"}`.

Only one of `file` or `url` should be provided per request.

### Example (Next.js / fetch)

```ts
// JSON with URL
await fetch("/ocr/text", {
  method: "POST",
  headers: { "X-API-Key": process.env.NEXT_PUBLIC_OCR_KEY!, "Content-Type": "application/json" },
  body: JSON.stringify({ url: "https://r2-bucket/file.png" }),
});
```

```ts
// Multipart upload
const form = new FormData();
form.append("file", fileInput.files[0]);
await fetch("/ocr/table", {
  method: "POST",
  headers: { "X-API-Key": process.env.NEXT_PUBLIC_OCR_KEY! },
  body: form,
});
```

## Railway notes

- Add `API_KEY` (and optional tuning vars) in the Railway dashboard environment tab.
- The provided `Procfile` runs `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- If you pre-download PaddleOCR models, mount/cache them and point to paths via env vars.

## Testing

`pytest` uses dependency overrides and does not load real Paddle models:

```
pytest
```