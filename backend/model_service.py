"""
Model Service — runs on port 8001
Loads Gemma 4 E2B/E4B locally and exposes a /transcribe endpoint.

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
import tempfile

import torch
from flask import Flask, jsonify, request
from transformers import AutoProcessor, AutoModelForMultimodalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID = os.getenv("GEMMA_MODEL_ID", "google/gemma-4-E2B-it")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model loading (happens once at startup)
# ---------------------------------------------------------------------------
logger.info(f"Loading model {MODEL_ID} on {DEVICE} ...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
)
model.eval()
logger.info("Model loaded successfully.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PROMPTS = {
    "zh": "請將以下語音片段轉錄為繁體中文文字。只輸出轉錄結果，不要加任何說明，數字請直接寫數字。",
    "en": "Transcribe the following speech segment in its original language. Only output the transcription, with no newlines. When transcribing numbers, write the digits.",
}


def transcribe_chunk(audio_b64: str, language: str = "zh") -> str:
    prompt_text = PROMPTS.get(language, PROMPTS["zh"])

    # 存成暫存 WAV 檔（processor 需要檔案路徑）
    wav_bytes = base64.b64decode(audio_b64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name

    try:
        # 音訊放在文字前面（官方建議順序）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "url": tmp_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=512)

        response = processor.decode(output_ids[0][input_len:], skip_special_tokens=False)
        result = processor.parse_response(response)

        if isinstance(result, dict):
            text = result.get("text") or result.get("content") or str(result)
        else:
            text = result

        return text.strip()

    finally:
        os.unlink(tmp_path)


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
