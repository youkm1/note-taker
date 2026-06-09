import io
import json
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlencode

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydub import AudioSegment
import requests

if os.getenv("STT_PROVIDER", "whisper").lower() not in ("ibm", "rtzr"):
    from faster_whisper import WhisperModel
else:
    WhisperModel = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("stt")

# --- 모델 설정 ---
STT_PROVIDER = os.getenv("STT_PROVIDER", "whisper").lower()

# HuggingFace 파인튜닝 모델 (예: seastar105/whisper-large-v3-turbo-ksponspeech-ct2)
HF_MODEL = os.getenv("WHISPER_HF_MODEL", "")
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3-turbo")
MODEL_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
MODEL_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

IBM_STT_APIKEY = os.getenv("IBM_STT_APIKEY", "")
IBM_STT_URL = os.getenv("IBM_STT_URL", "").rstrip("/")
IBM_STT_MODEL = os.getenv("IBM_STT_MODEL", "ko-KR_Multimedia")
IBM_STT_TIMEOUT = int(os.getenv("IBM_STT_TIMEOUT", "900"))
IBM_STT_LEARNING_OPT_OUT = os.getenv("IBM_STT_LEARNING_OPT_OUT", "true").lower() == "true"
IBM_STT_CUSTOMER_ID = os.getenv("IBM_STT_CUSTOMER_ID", "notetaker")

IBM_MODEL_BY_LANG = {
    "ko": "ko-KR_Multimedia",
    "en": "en-US_Multimedia",
    "ja": "ja-JP_Multimedia",
}

# --- RTZR (리턴제로) 설정 ---
RTZR_CLIENT_ID = os.getenv("RTZR_CLIENT_ID", "")
RTZR_CLIENT_SECRET = os.getenv("RTZR_CLIENT_SECRET", "")
RTZR_BASE_URL = "https://openapi.vito.ai"
RTZR_POLL_INTERVAL = int(os.getenv("RTZR_POLL_INTERVAL", "5"))
RTZR_TIMEOUT = int(os.getenv("RTZR_TIMEOUT", "900"))

# KsponSpeech 모델 사용 시 한국어를 기본 언어로 설정
_is_ksponspeech = "ksponspeech" in HF_MODEL.lower()
DEFAULT_LANGUAGE: Optional[str] = "ko" if _is_ksponspeech else None

model_id = HF_MODEL if HF_MODEL else MODEL_SIZE

if STT_PROVIDER == "ibm":
    logger.info("Using IBM Watson Speech to Text provider model=%s", IBM_STT_MODEL)
    model = None
elif STT_PROVIDER == "rtzr":
    logger.info("Using RTZR (Return Zero) STT provider")
    model = None
else:
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
    payload = {
        "status": "ok",
        "provider": STT_PROVIDER,
        "model": model_id,
        "device": MODEL_DEVICE,
        "compute_type": MODEL_COMPUTE_TYPE,
        "default_language": DEFAULT_LANGUAGE,
    }
    if STT_PROVIDER == "ibm":
        payload.update({
            "model": IBM_STT_MODEL,
            "url_configured": bool(IBM_STT_URL),
            "apikey_configured": bool(IBM_STT_APIKEY),
        })
    elif STT_PROVIDER == "rtzr":
        payload.update({
            "model": "sommers",
            "client_id_configured": bool(RTZR_CLIENT_ID),
            "client_secret_configured": bool(RTZR_CLIENT_SECRET),
        })
    return payload


def _audio_bytes_to_wav(audio_bytes: bytes) -> bytes:
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_channels(1).set_frame_rate(16000)
    output = io.BytesIO()
    audio.export(output, format="wav")
    return output.getvalue()


def _ibm_model_for_language(language: Optional[str]) -> str:
    if not language:
        return IBM_STT_MODEL
    return IBM_MODEL_BY_LANG.get(language.lower(), IBM_STT_MODEL)


def _transcribe_ibm(audio_bytes: bytes, language: Optional[str]) -> str:
    if not IBM_STT_APIKEY or not IBM_STT_URL:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="IBM STT credentials are not configured.",
        )

    wav_bytes = _audio_bytes_to_wav(audio_bytes)
    params = {
        "model": _ibm_model_for_language(language),
        "timestamps": "true",
        "smart_formatting": "true",
    }
    headers = {
        "Content-Type": "audio/wav",
        "X-Watson-Metadata": f"customer_id={IBM_STT_CUSTOMER_ID}",
    }
    if IBM_STT_LEARNING_OPT_OUT:
        headers["X-Watson-Learning-Opt-Out"] = "true"

    url = f"{IBM_STT_URL}/v1/recognize?{urlencode(params)}"
    logger.info("Calling IBM STT model=%s wav_bytes=%d", params["model"], len(wav_bytes))
    response = requests.post(
        url,
        auth=("apikey", IBM_STT_APIKEY),
        headers=headers,
        data=wav_bytes,
        timeout=IBM_STT_TIMEOUT,
    )
    if response.status_code >= 400:
        logger.error("IBM STT failed status=%s body=%s", response.status_code, response.text[:1000])
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="IBM STT request failed.",
        )

    payload = response.json()
    transcripts = []
    for result in payload.get("results", []):
        alternatives = result.get("alternatives") or []
        if alternatives:
            transcripts.append(alternatives[0].get("transcript", "").strip())
    return " ".join(part for part in transcripts if part).strip()


# --- RTZR (리턴제로) 토큰 캐시 및 함수 ---
_rtzr_token_cache: dict = {"token": None, "expire_at": 0}


def _rtzr_get_token() -> str:
    """JWT 토큰 발급 (만료 30분 전 자동 갱신)."""
    if _rtzr_token_cache["token"] and _rtzr_token_cache["expire_at"] > time.time() + 1800:
        return _rtzr_token_cache["token"]
    resp = requests.post(
        f"{RTZR_BASE_URL}/v1/authenticate",
        data={"client_id": RTZR_CLIENT_ID, "client_secret": RTZR_CLIENT_SECRET},
    )
    resp.raise_for_status()
    payload = resp.json()
    _rtzr_token_cache["token"] = payload["access_token"]
    _rtzr_token_cache["expire_at"] = payload.get("expire_at", time.time() + 3600)
    logger.info("RTZR token issued/renewed")
    return _rtzr_token_cache["token"]


def _transcribe_rtzr(audio_bytes: bytes, language: Optional[str]) -> str:
    """리턴제로 Batch STT API (POST 요청 → 폴링으로 결과 수신)."""
    if not RTZR_CLIENT_ID or not RTZR_CLIENT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RTZR credentials are not configured.",
        )

    token = _rtzr_get_token()
    headers = {"Authorization": f"Bearer {token}"}

    lang = (language or "ko").lower()
    config = {
        "model_name": "sommers",
        "language": lang if lang in ("ko", "ja") else "ko",
        "use_diarization": False,
        "use_itn": True,
        "use_disfluency_filter": True,
        "use_paragraph_splitter": True,
        "paragraph_splitter": {"max": 50},
        "domain": "GENERAL",
    }

    # 1) 전사 요청
    logger.info("RTZR transcribe request lang=%s bytes=%d", lang, len(audio_bytes))
    resp = requests.post(
        f"{RTZR_BASE_URL}/v1/transcribe",
        headers=headers,
        data={"config": json.dumps(config)},
        files={"file": ("audio", audio_bytes)},
        timeout=60,
    )
    if resp.status_code >= 400:
        logger.error("RTZR submit failed status=%s body=%s", resp.status_code, resp.text[:500])
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"RTZR transcribe request failed: {resp.text[:200]}",
        )
    transcribe_id = resp.json()["id"]
    logger.info("RTZR transcribe_id=%s, polling every %ds", transcribe_id, RTZR_POLL_INTERVAL)

    # 2) 폴링
    deadline = time.time() + RTZR_TIMEOUT
    while time.time() < deadline:
        time.sleep(RTZR_POLL_INTERVAL)
        result = requests.get(
            f"{RTZR_BASE_URL}/v1/transcribe/{transcribe_id}",
            headers={"Authorization": f"Bearer {_rtzr_get_token()}"},
            timeout=30,
        )
        result.raise_for_status()
        data = result.json()

        if data["status"] == "completed":
            utterances = data.get("results", {}).get("utterances", [])
            transcript = " ".join(u["msg"] for u in utterances).strip()
            logger.info("RTZR transcription completed utterances=%d", len(utterances))
            return transcript
        elif data["status"] == "failed":
            error = data.get("error", {})
            logger.error("RTZR transcription failed: %s", error)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"RTZR transcription failed: {error}",
            )

    raise HTTPException(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        detail="RTZR transcription timed out",
    )


def split_audio_with_overlap(
    audio_bytes: bytes,
    chunk_length_ms: int = 30000,
    overlap_ms: int = 2000,
) -> List[AudioSegment]:
    """
    Split audio into chunks with overlap.

    Args:
        audio_bytes: Original audio binary data
        chunk_length_ms: Chunk length in milliseconds (default: 30 seconds)
        overlap_ms: Overlap length in milliseconds (default: 2 seconds)

    Returns:
        List of AudioSegment chunks
    """
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    total_length = len(audio)

    if total_length <= chunk_length_ms:
        # Short audio, no need to split
        return [audio]

    chunks = []
    start = 0

    while start < total_length:
        end = min(start + chunk_length_ms, total_length)
        chunk = audio[start:end]
        chunks.append(chunk)

        # Move to next chunk start (with overlap)
        start += chunk_length_ms - overlap_ms

    logger.info(
        "Audio split into %d chunks (chunk=%dms, overlap=%dms)",
        len(chunks),
        chunk_length_ms,
        overlap_ms,
    )
    return chunks


def _transcribe_chunk(chunk: AudioSegment, language: Optional[str]) -> dict:
    """
    Transcribe a single audio chunk.

    Args:
        chunk: AudioSegment chunk
        language: Language code

    Returns:
        Dict with text, duration, and language
    """
    temp_path = None
    try:
        # Export AudioSegment to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            chunk.export(tmp.name, format="wav")
            temp_path = tmp.name

        # Transcribe with Whisper
        segments, info = model.transcribe(
            temp_path,
            language=language,
            beam_size=5,
            vad_filter=True,
        )

        text = " ".join(segment.text.strip() for segment in segments).strip()

        return {
            "text": text,
            "duration": info.duration,
            "language": info.language,
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning("Failed to remove temp file path=%s", temp_path)


def parallel_transcribe(
    chunks: List[AudioSegment],
    language: Optional[str],
    max_workers: int = 4,
) -> str:
    """
    Transcribe multiple chunks in parallel.

    Args:
        chunks: List of AudioSegment chunks
        language: Language code
        max_workers: Maximum number of parallel workers

    Returns:
        Merged transcript text
    """
    results = [None] * len(chunks)  # Preserve order

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks with their indices
        future_to_index = {
            executor.submit(_transcribe_chunk, chunk, language): i
            for i, chunk in enumerate(chunks)
        }

        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result["text"]
                logger.info("Chunk %d/%d completed", index + 1, len(chunks))
            except Exception as exc:  # pragma: no cover
                logger.error("Chunk %d failed: %s", index, exc)
                results[index] = ""  # Empty string for failed chunks

    # Merge results
    full_text = " ".join(r for r in results if r).strip()
    return full_text


@app.post("/stt")
async def transcribe(
    audio: UploadFile = File(...),
    lang: Optional[str] = None,
    enable_chunking: bool = True,
):
    """
    Transcribe audio file to text.

    Args:
        audio: Audio file upload
        lang: Language code (ko, en, etc.)
        enable_chunking: Enable audio segmentation for long files

    Returns:
        {"text": "transcribed text"}
    """
    if not audio.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is required.",
        )

    # Determine language: request param > KsponSpeech default(ko) > auto-detect
    language = lang if lang else DEFAULT_LANGUAGE

    try:
        # Read audio data
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        logger.info(
            "Transcribing file=%s language=%s chunking=%s",
            audio.filename,
            language or "auto",
            enable_chunking,
        )

        if STT_PROVIDER == "rtzr":
            transcript = _transcribe_rtzr(audio_bytes, language)
            logger.info("RTZR transcription finished")
            return {"text": transcript, "provider": "rtzr"}

        if STT_PROVIDER == "ibm":
            transcript = _transcribe_ibm(audio_bytes, language)
            logger.info("IBM transcription finished")
            return {"text": transcript, "provider": "ibm"}

        # Use chunking if enabled
        if enable_chunking:
            chunks = split_audio_with_overlap(audio_bytes)

            if len(chunks) > 1:
                # Multiple chunks - parallel processing
                transcript = parallel_transcribe(chunks, language, max_workers=4)
                logger.info("Parallel processing completed (%d chunks)", len(chunks))
            else:
                # Short audio - single processing
                result = _transcribe_chunk(chunks[0], language)
                transcript = result["text"]
        else:
            # Legacy mode - no chunking
            temp_path = None
            try:
                suffix = Path(audio.filename).suffix or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_bytes)
                    temp_path = tmp.name

                segments, info = model.transcribe(
                    temp_path,
                    language=language,
                    beam_size=5,
                    vad_filter=True,
                )
                transcript = " ".join(segment.text.strip() for segment in segments).strip()
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        logger.warning("Temporary file cleanup failed path=%s", temp_path)

        logger.info("Transcription finished")
        return {"text": transcript}

    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Transcription failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Transcription failed.",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("stt.app:app", host="0.0.0.0", port=8000, reload=False)
