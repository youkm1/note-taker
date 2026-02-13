import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from faster_whisper import WhisperModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("stt")

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
# Supports:
#   - Standard faster-whisper models: "small", "medium", "large-v3"
#   - HuggingFace model IDs for fine-tuned CTranslate2 models
#     e.g. "seastar105/whisper-large-v3-turbo-ksponspeech-ct2"
#   - Local path to a CTranslate2-converted model directory
# ---------------------------------------------------------------------------
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3-turbo")
MODEL_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
MODEL_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# Optional: HuggingFace fine-tuned model for Korean (KsponSpeech)
# Set WHISPER_HF_MODEL to a HF repo ID to use a fine-tuned model instead.
# Example: "seastar105/whisper-large-v3-turbo-ksponspeech-ct2"
HF_MODEL = os.getenv("WHISPER_HF_MODEL", "")

model_identifier = HF_MODEL if HF_MODEL else MODEL_SIZE

logger.info(
    "Loading Faster-Whisper model=%s device=%s compute_type=%s",
    model_identifier,
    MODEL_DEVICE,
    MODEL_COMPUTE_TYPE,
)
model = WhisperModel(
    model_identifier,
    device=MODEL_DEVICE,
    compute_type=MODEL_COMPUTE_TYPE,
)
logger.info("Model loaded successfully: %s", model_identifier)

app = FastAPI(title="Korean STT Service (Whisper Large V3 Turbo)", version="0.2.0")


@app.get("/health")
async def health():
    return {"status": "ok", "model": model_identifier}


@app.post("/stt")
async def transcribe(audio: UploadFile = File(...), lang: Optional[str] = None):
    """Transcribe audio to text.

    - Default language is Korean ("ko") when using a KsponSpeech fine-tuned model.
    - Pass lang="auto" to enable automatic language detection.
    """
    if not audio.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is required.",
        )

    # Default to Korean when using a KsponSpeech fine-tuned model
    if lang is None and HF_MODEL and "ksponspeech" in HF_MODEL.lower():
        lang = "ko"

    temp_path = None
    try:
        suffix = Path(audio.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            data = await audio.read()
            if not data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded file is empty.",
                )
            tmp.write(data)
            temp_path = tmp.name

        effective_lang = None if lang == "auto" else lang
        logger.info("Transcribing file=%s language=%s", audio.filename, lang or "auto")

        segments, info = model.transcribe(
            temp_path,
            language=effective_lang,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=300,
            ),
        )
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        logger.info(
            "Transcription finished language=%s duration=%.2fs",
            info.language,
            info.duration,
        )
        return {
            "text": transcript,
            "language": info.language,
            "duration": round(info.duration, 2),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Transcription failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Transcription failed.",
        ) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning("Temporary file cleanup failed path=%s", temp_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("stt.app:app", host="0.0.0.0", port=8000, reload=False)
