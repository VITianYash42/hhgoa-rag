"""
Speech-to-text via Sarvam Saaras. Sign up free at https://dashboard.sarvam.ai
(no card needed, ~1000 free credits) and set SARVAM_API_KEY as an env var.

Docs: https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe
"""
import os
import time
import requests

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


class STTError(Exception):
    pass


def transcribe(audio_path: str, language_code: str = "en-IN", timeout: float = 8.0) -> dict:
    """
    Sends a local audio file to Sarvam STT. Returns {"text": ..., "latency_ms": ...}.
    Raises STTError on failure (caller's harness handles retry).
    """
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        raise STTError("SARVAM_API_KEY not set. Get a free key at dashboard.sarvam.ai")

    start = time.perf_counter()
    try:
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
            data = {"language_code": language_code, "model": "saaras:v2"}
            headers = {"api-subscription-key": api_key}
            resp = requests.post(SARVAM_STT_URL, headers=headers, files=files, data=data, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as e:
        raise STTError(f"Sarvam STT request failed: {e}") from e

    latency_ms = (time.perf_counter() - start) * 1000
    text = payload.get("transcript", "").strip()
    if not text:
        raise STTError("Sarvam STT returned empty transcript.")
    return {"text": text, "latency_ms": latency_ms}
