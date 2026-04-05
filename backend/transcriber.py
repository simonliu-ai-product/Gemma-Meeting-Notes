"""
Audio chunking utilities for Gemma 4 transcription.
Gemma 4 E4B/E2B supports max 30s audio input — we use 25s chunks with 1s overlap.
"""

import base64
import io
import math
from pydub import AudioSegment


CHUNK_SECONDS = 25
OVERLAP_SECONDS = 1


def load_audio(file_bytes: bytes, file_ext: str) -> AudioSegment:
    """Load audio from raw bytes, normalise to mono 16kHz."""
    audio = AudioSegment.from_file(io.BytesIO(file_bytes), format=file_ext.lstrip("."))
    audio = audio.set_channels(1).set_frame_rate(16000)
    return audio


def split_audio(audio: AudioSegment) -> list[dict]:
    """
    Split audio into chunks of CHUNK_SECONDS with slight overlap.

    Returns list of:
        {
          "index": int,
          "start_ms": int,
          "end_ms": int,
          "b64": str,        # base64-encoded WAV bytes
        }
    """
    chunk_ms = CHUNK_SECONDS * 1000
    overlap_ms = OVERLAP_SECONDS * 1000
    total_ms = len(audio)

    chunks = []
    idx = 0
    pos = 0

    while pos < total_ms:
        end = min(pos + chunk_ms, total_ms)
        segment = audio[pos:end]

        buf = io.BytesIO()
        segment.export(buf, format="wav")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        chunks.append({
            "index": idx,
            "start_ms": pos,
            "end_ms": end,
            "b64": b64,
        })

        idx += 1
        if end >= total_ms:
            break
        pos = end - overlap_ms  # slight overlap to avoid cutting words

    return chunks


def ms_to_srt_time(ms: int) -> str:
    """Convert milliseconds to SRT timestamp format HH:MM:SS,mmm."""
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    seconds = ms // 1000
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def build_output(chunks: list[dict], texts: list[str]) -> dict:
    """
    Merge chunk transcriptions into plain text and SRT format.

    Args:
        chunks: list of chunk metadata (from split_audio)
        texts:  list of transcription strings, one per chunk

    Returns:
        { "text": str, "srt": str }
    """
    plain_parts = []
    srt_parts = []

    for i, (chunk, text) in enumerate(zip(chunks, texts)):
        text = text.strip()
        if not text:
            continue

        plain_parts.append(text)

        srt_parts.append(
            f"{i + 1}\n"
            f"{ms_to_srt_time(chunk['start_ms'])} --> {ms_to_srt_time(chunk['end_ms'])}\n"
            f"{text}\n"
        )

    return {
        "text": "\n".join(plain_parts),
        "srt": "\n".join(srt_parts),
    }
