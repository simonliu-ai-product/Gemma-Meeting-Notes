"""
Backend API — runs on port 8000
Handles file uploads, audio chunking, calls model_service, returns results.

POST /api/transcribe
  Form: file=<audio file>, language=zh|en
  Returns: { "text": str, "srt": str, "chunks_count": int }

GET /api/health
"""

import os
import tempfile
import logging
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from transcriber import build_output, load_audio, split_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://localhost:8001")
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}

app = FastAPI(title="Gemma 4 Audio Transcription API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend from /  (index.html lives in ../frontend/)
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            r = await client.get(f"{MODEL_SERVICE_URL}/health")
            model_status = r.json()
        except Exception:
            model_status = {"status": "unreachable"}
    return {"api": "ok", "model_service": model_status}


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("zh"),
):
    # ── Validate file extension ──────────────────────────────────────────────
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    # ── Read file ────────────────────────────────────────────────────────────
    file_bytes = await file.read()
    logger.info(f"Received file: {file.filename} ({len(file_bytes) / 1024:.1f} KB), lang={language}")

    # ── Load & chunk audio ───────────────────────────────────────────────────
    try:
        audio = load_audio(file_bytes, suffix)
        chunks = split_audio(audio)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Audio processing error: {e}")

    logger.info(f"Split into {len(chunks)} chunk(s)")

    # ── Call model service for each chunk ────────────────────────────────────
    texts: list[str] = []
    async with httpx.AsyncClient(timeout=120) as client:
        for i, chunk in enumerate(chunks):
            logger.info(f"Transcribing chunk {i + 1}/{len(chunks)} ...")
            try:
                r = await client.post(
                    f"{MODEL_SERVICE_URL}/transcribe",
                    json={"audio_b64": chunk["b64"], "language": language},
                )
                r.raise_for_status()
                text = r.json().get("text", "")
            except httpx.HTTPStatusError as e:
                raise HTTPException(
                    status_code=502,
                    detail=f"Model service error on chunk {i + 1}: {e.response.text}",
                )
            except Exception as e:
                raise HTTPException(
                    status_code=502,
                    detail=f"Model service unreachable: {e}",
                )
            texts.append(text)

    # ── Merge results ────────────────────────────────────────────────────────
    output = build_output(chunks, texts)
    output["chunks_count"] = len(chunks)

    return JSONResponse(content=output)
