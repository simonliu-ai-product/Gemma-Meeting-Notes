"""
Backend API — runs on port 8000
Handles file uploads, audio chunking, calls model_service, streams results.

POST /api/transcribe
  Form: file=<audio file>, language=zh|en
  Returns: text/event-stream SSE
    data: {"type":"start",  "total": <n>}
    data: {"type":"chunk",  "index": <i>, "total": <n>, "text": "<partial>"}
    data: {"type":"result", "text": "<full>", "srt": "<srt>", "chunks_count": <n>}
    data: {"type":"error",  "detail": "<msg>"}

GET /api/health
"""

import json
import os
import logging
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from transcriber import build_output, load_audio, split_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://localhost:8001")
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}

app = FastAPI(title="Gemma 4 Audio Transcription API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        return JSONResponse({"error": "frontend not found"}, status_code=404)
    return FileResponse(str(index))


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
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    logger.info(f"Received: {file.filename} ({len(file_bytes)/1024:.1f} KB), lang={language}")

    try:
        audio = load_audio(file_bytes, suffix)
        chunks = split_audio(audio)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Audio processing error: {e}")

    logger.info(f"Split into {len(chunks)} chunk(s)")

    async def event_stream():
        def sse(data: dict) -> str:
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        yield sse({"type": "start", "total": len(chunks)})

        texts: list[str] = []
        # read=None：模型推論無固定上限，避免長音訊推論超時
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=30, read=None, write=60, pool=30)) as client:
            for i, chunk in enumerate(chunks):
                logger.info(f"Transcribing chunk {i+1}/{len(chunks)} ...")
                try:
                    r = await client.post(
                        f"{MODEL_SERVICE_URL}/transcribe",
                        json={"audio_b64": chunk["b64"], "language": language},
                    )
                    r.raise_for_status()
                    text = r.json().get("text", "")
                except httpx.HTTPStatusError as e:
                    yield sse({"type": "error", "detail": f"Model service error on chunk {i+1}: {e.response.text}"})
                    return
                except Exception as e:
                    yield sse({"type": "error", "detail": f"Model service unreachable: {e}"})
                    return

                texts.append(text)
                yield sse({"type": "chunk", "index": i + 1, "total": len(chunks), "text": text})

        output = build_output(chunks, texts)
        yield sse({"type": "result", "text": output["text"], "srt": output["srt"], "chunks_count": len(chunks)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
