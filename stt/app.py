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

MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
MODEL_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
MODEL_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

logger.info(
    "Loading Faster-Whisper model size=%s device=%s compute_type=%s",
    MODEL_SIZE,
    MODEL_DEVICE,
    MODEL_COMPUTE_TYPE,
)
model = WhisperModel(
    MODEL_SIZE,
    device=MODEL_DEVICE,
    compute_type=MODEL_COMPUTE_TYPE,
)

app = FastAPI(title="Local STT Service", version="0.1.0")


@app.post("/stt")
async def transcribe(audio: UploadFile = File(...), lang: Optional[str] = None):
    if not audio.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is required.",
        )

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

        logger.info("Transcribing file=%s language=%s", audio.filename, lang or "auto")
        segments, info = model.transcribe(
            temp_path,
            language=lang if lang else None,
            beam_size=2,
        )
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        logger.info(
            "Transcription finished language=%s duration=%.2fs",
            info.language,
            info.duration,
        )
        return {"text": transcript}
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
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
