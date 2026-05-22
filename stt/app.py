import io
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from faster_whisper import WhisperModel
from pydub import AudioSegment


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
