"""
Model Service — runs on port 8001
Loads Gemma 4 E4B (or E2B) locally and exposes a /transcribe endpoint.

POST /transcribe
  Body: { "audio_b64": "<base64 WAV>", "language": "zh" | "en" }
  Returns: { "text": "<transcription>" }

GET /health
  Returns: { "status": "ok", "model": "<model_id>" }
"""

import base64
import io
import os
import logging

import torch
import torchaudio
from flask import Flask, jsonify, request
from transformers import AutoProcessor, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Gemma 4 E4B (multimodal: text + image + audio).
# Change to "google/gemma-4-E2B-it" for the smaller 2B variant.
MODEL_ID = os.getenv("GEMMA_MODEL_ID", "google/gemma-4-E4B-it")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model loading (happens once at startup)
# ---------------------------------------------------------------------------
logger.info(f"Loading model {MODEL_ID} on {DEVICE} ...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=DTYPE,
    device_map="auto",
)
model.eval()
logger.info("Model loaded successfully.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PROMPTS = {
    "zh": "請將以下語音片段轉錄為繁體中文文字，只輸出轉錄結果，不要加任何說明：",
    "en": "Please transcribe the following speech segment into English text. Output only the transcription, no explanations:",
}


def decode_audio(b64_str: str) -> tuple[torch.Tensor, int]:
    """Decode base64 WAV → (waveform tensor [1, T], sample_rate)."""
    wav_bytes = base64.b64decode(b64_str)
    buf = io.BytesIO(wav_bytes)
    waveform, sr = torchaudio.load(buf)
    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sr


def transcribe_chunk(audio_b64: str, language: str = "zh") -> str:
    prompt_text = PROMPTS.get(language, PROMPTS["zh"])

    waveform, sr = decode_audio(audio_b64)

    # Build conversation with audio input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": waveform, "sampling_rate": sr},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=512)

    response = processor.decode(output_ids[0][input_len:], skip_special_tokens=False)
    result = processor.parse_response(response)
    return result.strip()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_ID, "device": DEVICE})


@app.post("/transcribe")
def transcribe():
    data = request.get_json(force=True)
    audio_b64 = data.get("audio_b64", "")
    language = data.get("language", "zh")

    if not audio_b64:
        return jsonify({"error": "audio_b64 is required"}), 400

    try:
        text = transcribe_chunk(audio_b64, language)
        return jsonify({"text": text})
    except Exception as e:
        logger.exception("Transcription failed")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("MODEL_SERVICE_PORT", 8001))
    app.run(host="0.0.0.0", port=port, debug=False)
