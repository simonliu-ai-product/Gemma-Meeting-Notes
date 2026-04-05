"""
Backend API — runs on port 8000

POST /api/transcribe
  Form: file=<audio>, language=zh|en
  Returns: { "task_id": str }

GET /api/progress/{task_id}
  Returns: { "status": "processing"|"done"|"error",
             "chunk": int, "total": int, "detail": str }

GET /api/result/{task_id}
  Returns: { "text": str, "srt": str, "chunks_count": int }  (when done)
           404 if still processing, 500 if error

GET /api/health
"""

import asyncio
import os
import logging
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

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

# In-memory task store: task_id → state dict
tasks: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Background transcription task
# ---------------------------------------------------------------------------
async def run_transcription(task_id: str, file_bytes: bytes, suffix: str, language: str):
    state = tasks[task_id]
    try:
        audio  = load_audio(file_bytes, suffix)
        chunks = split_audio(audio)
        state["total"] = len(chunks)
        logger.info(f"[{task_id}] {len(chunks)} chunk(s)")

        texts: list[str] = []
        timeout = httpx.Timeout(connect=30, read=None, write=60, pool=30)
        async with httpx.AsyncClient(timeout=timeout) as client:
            for i, chunk in enumerate(chunks):
                state["chunk"] = i      # i = 已完成數，推論前先更新
                state["processing"] = i + 1
                logger.info(f"[{task_id}] chunk {i+1}/{len(chunks)}")
                r = await client.post(
                    f"{MODEL_SERVICE_URL}/transcribe",
                    json={"audio_b64": chunk["b64"], "language": language},
                )
                r.raise_for_status()
                texts.append(r.json().get("text", ""))
                state["chunk"] = i + 1  # 完成後更新已完成數

        output = build_output(chunks, texts)
        state["result"] = {**output, "chunks_count": len(chunks)}
        state["status"] = "done"
        logger.info(f"[{task_id}] done")

    except Exception as e:
        logger.exception(f"[{task_id}] failed")
        state["status"] = "error"
        state["detail"] = str(e)


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
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = Form("zh"),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'.")

    file_bytes = await file.read()
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "chunk": 0, "total": 1, "detail": ""}
    background_tasks.add_task(run_transcription, task_id, file_bytes, suffix, language)
    logger.info(f"[{task_id}] started — {file.filename} ({len(file_bytes)//1024} KB)")
    return {"task_id": task_id}


@app.get("/api/progress/{task_id}")
async def progress(task_id: str):
    state = tasks.get(task_id)
    if not state:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "status":     state["status"],
        "chunk":      state["chunk"],       # 已完成片段數
        "processing": state.get("processing", 0),  # 正在處理的片段號
        "total":      state["total"],
        "detail":     state.get("detail", ""),
    }


@app.get("/api/result/{task_id}")
async def result(task_id: str):
    state = tasks.get(task_id)
    if not state:
        raise HTTPException(status_code=404, detail="Task not found")
    if state["status"] == "processing":
        raise HTTPException(status_code=202, detail="Still processing")
    if state["status"] == "error":
        raise HTTPException(status_code=500, detail=state.get("detail", "Unknown error"))
    return JSONResponse(content=state["result"])
