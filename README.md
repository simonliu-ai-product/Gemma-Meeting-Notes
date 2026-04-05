# Gemma-Meeting-Notes

> **Audio transcription service powered by Gemma 4 E4B / E2B** — upload a meeting recording and get a full transcript with SRT subtitles.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simonliu-ai-product/Gemma-Meeting-Notes/blob/main/colab_demo.ipynb)

---

## Overview

| | |
|---|---|
| **Model** | Gemma 4 E4B (4.5 B effective params) · switchable to E2B |
| **Modality** | Audio → Text (ASR) |
| **Max chunk** | 25 s per inference call (Gemma 4 audio limit: 30 s) |
| **Output** | Plain text + SRT subtitles with timestamps |
| **UI language** | Traditional Chinese / English (toggle) |
| **Demo** | Google Colab (T4 GPU) |

---

## Architecture

```
┌─────────────────────────────┐
│   Frontend  (index.html)    │  Bilingual UI — file upload, progress, result
└────────────┬────────────────┘
             │ POST /api/transcribe
┌────────────▼────────────────┐
│   Backend  (FastAPI :8000)  │  Audio chunking, orchestration
└────────────┬────────────────┘
             │ POST /transcribe  (per chunk)
┌────────────▼────────────────┐
│  Model Service (Flask :8001)│  Gemma 4 E4B via HuggingFace Transformers
└─────────────────────────────┘
```

The frontend is served as a static file directly from the FastAPI backend — no separate server needed.

---

## File Structure

```
Gemma-Meeting-Notes/
├── backend/
│   ├── model_service.py   # Loads Gemma 4 E4B, exposes /transcribe
│   ├── main.py            # FastAPI — upload, chunk, merge
│   ├── transcriber.py     # Audio split (pydub) + SRT builder
│   └── requirements.txt
├── frontend/
│   └── index.html         # Single-file bilingual UI
└── colab_demo.ipynb       # One-click Colab demo
```

---

## Quick Start (Google Colab)

1. Click **Open in Colab** above
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run each cell in order:

| Cell | Action |
|------|--------|
| 0 | Verify GPU |
| 1 | Install ffmpeg |
| 2 | Clone this repo |
| 3 | Install Python dependencies |
| 4 | HuggingFace login (required for gated model) |
| 5 | Start model service — **loads Gemma 4 E4B (~9 GB, takes a few minutes)** |
| 6 | Start backend API |
| 7 | Open demo window |
| 8 | Stop all services (run when done) |

---

## Local Development

```bash
# 1. Install ffmpeg
brew install ffmpeg   # macOS
# or: apt-get install ffmpeg

# 2. Install Python dependencies
pip install -r backend/requirements.txt

# 3. Start model service
GEMMA_MODEL_ID=google/gemma-4-e4b-it python backend/model_service.py

# 4. Start backend (new terminal)
MODEL_SERVICE_URL=http://localhost:8001 \
  uvicorn main:app --host 0.0.0.0 --port 8000 --app-dir backend

# 5. Open http://localhost:8000
```

---

## Switching Models

Edit `GEMMA_MODEL_ID` in Cell 5 of the notebook (or the env var locally):

| Variant | Model ID | VRAM |
|---------|----------|------|
| E4B (default) | `google/gemma-4-e4b-it` | ~9 GB |
| E2B | `google/gemma-4-e2b-it` | ~5 GB |

---

## Supported Audio Formats

`mp3` · `wav` · `m4a` · `ogg` · `flac` · `webm`

Long recordings are automatically split into ≤ 25 s chunks and merged after transcription.

---

## License

- **Project code** — Apache 2.0, see [LICENSE](LICENSE)
- **Gemma 4 model weights** — Apache 2.0 ([Gemma 4 release note](https://ai.google.dev/gemma))
