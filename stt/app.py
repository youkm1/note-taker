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

# --- 모델 설정 ---
# HuggingFace 파인튜닝 모델 (예: seastar105/whisper-large-v3-turbo-ksponspeech-ct2)
HF_MODEL = os.getenv("WHISPER_HF_MODEL", "")
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3-turbo")
MODEL_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
MODEL_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# KsponSpeech 모델 사용 시 한국어를 기본 언어로 설정
_is_ksponspeech = "ksponspeech" in HF_MODEL.lower()
DEFAULT_LANGUAGE: Optional[str] = "ko" if _is_ksponspeech else None

model_id = HF_MODEL if HF_MODEL else MODEL_SIZE

logger.info(
    "Loading Faster-Whisper model=%s device=%s compute_type=%s (ksponspeech=%s)",
    model_id,
    MODEL_DEVICE,
    MODEL_COMPUTE_TYPE,
    _is_ksponspeech,
)
model = WhisperModel(
    model_id,
    device=MODEL_DEVICE,
    compute_type=MODEL_COMPUTE_TYPE,
)

app = FastAPI(title="Local STT Service", version="0.2.0")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": model_id,
        "device": MODEL_DEVICE,
        "compute_type": MODEL_COMPUTE_TYPE,
        "default_language": DEFAULT_LANGUAGE,
    }


@app.post("/stt")
async def transcribe(audio: UploadFile = File(...), lang: Optional[str] = None):
    if not audio.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is required.",
        )

    # 언어 결정: 요청 파라미터 > KsponSpeech 기본값(ko) > 자동 감지
    language = lang if lang else DEFAULT_LANGUAGE

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

        logger.info("Transcribing file=%s language=%s", audio.filename, language or "auto")
        segments, info = model.transcribe(
            temp_path,
            language=language,
            beam_size=5,
            vad_filter=True,
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
