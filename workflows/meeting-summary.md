# Meeting Ingestion Flow

오디오 업로드 → STT 전사 → Semantic Chunking → 임베딩 → Qdrant 저장.

## Flow

1. **Trigger**: webhook 또는 수동 트리거로 오디오 파일(`audio`)과 옵션 `lang` 수신.

2. **STT 전사**:
   ```
   POST http://stt:8000/stt
   (multipart form: audio 파일, lang 파라미터)
   ```
   - Whisper large-v3-turbo + VAD 필터 적용
   - KsponSpeech 모델 사용 시 한국어 자동 감지
   - 응답: `{"text": "전사된 텍스트"}`

3. **RAG Ingest** (Semantic Chunking → 임베딩 → 저장):
   ```
   POST http://rag:8001/ingest
   {"text": "<전사 텍스트>", "title": "회의 제목", "metadata": {"speaker": "..."}}
   ```
   - 임베딩 유사도 기반 Semantic Chunking으로 의미 단위 분할
   - nomic-embed-text로 각 청크 임베딩 생성
   - Qdrant에 벡터 + 메타데이터 저장
   - 응답: `{"chunks_stored": 5, "ids": ["uuid1", ...]}`

4. **(선택) 요약 생성**:
   ```
   POST http://ollama:11434/api/chat
   {"model": "llama3", "messages": [{"role": "user", "content": "다음 회의록을 요약해줘:\n<전사 텍스트>"}]}
   ```

## n8n 워크플로우 구성

- **Trigger Node**: Webhook (POST, multipart form)
- **HTTP Request Node (STT)**: `POST http://stt:8000/stt` → transcript 추출
- **HTTP Request Node (Ingest)**: `POST http://rag:8001/ingest` → 청킹 및 저장
- **HTTP Request Node (Summarize)**: (선택) `POST http://ollama:11434/api/chat` → 요약 생성
