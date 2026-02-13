# Local AI Note Taker 아키텍처 정리 (v2)

Docker Compose로 STT(Whisper Large V3 Turbo), Agentic RAG(LangGraph), LLM/임베딩(Ollama), 벡터 DB(Qdrant), 평가(RAGAS), 오케스트레이션(n8n) 서비스를 로컬에서 묶어 self-hosted 메모/요약/RAG 환경을 제공합니다.

## 전체 구조

```
┌──────────────────────────────────────────────────────────────────┐
│                     Docker Compose Network                       │
│                                                                  │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐      │
│  │   STT   │   │  Ollama  │   │  Qdrant  │   │   n8n    │      │
│  │  :8000  │   │  :11434  │   │  :6333   │   │  :5678   │      │
│  │ Whisper │   │  llama3  │   │ Vector DB│   │ Workflow │      │
│  │ LV3-T   │   │  nomic-  │   │          │   │          │      │
│  └─────────┘   │  embed   │   └────┬─────┘   └──────────┘      │
│                └────┬─────┘        │                             │
│                     │              │                             │
│            ┌────────┴──────────────┴────────┐                   │
│            │        RAG Service :8001        │                   │
│            │  ┌───────────────────────────┐  │                   │
│            │  │     LangGraph Pipeline    │  │                   │
│            │  │  ┌─────┐ ┌────┐ ┌──────┐ │  │                   │
│            │  │  │Query│→│Retr│→│Gener │ │  │                   │
│            │  │  │Anal.│ │ieve│ │ate   │ │  │                   │
│            │  │  └─────┘ └────┘ └──┬───┘ │  │                   │
│            │  │                    ↓      │  │                   │
│            │  │  ┌──────┐    ┌────────┐  │  │                   │
│            │  │  │Reform│←───│Reflect │  │  │                   │
│            │  │  │ulate │    │(self-  │  │  │                   │
│            │  │  └──────┘    │ eval)  │  │  │                   │
│            │  │              └────────┘  │  │                   │
│            │  └───────────────────────────┘  │                   │
│            │  Semantic Chunking              │                   │
│            │  Hybrid Search (BM25+Vector)    │                   │
│            │  Cross-Encoder Reranker         │                   │
│            └────────────────────────────────┘                   │
│                          │                                       │
│            ┌─────────────┴──────────────┐                       │
│            │    Eval Service :8002       │                       │
│            │    RAGAS Metrics            │                       │
│            │    - Faithfulness           │                       │
│            │    - Answer Relevance       │                       │
│            │    - Context Precision      │                       │
│            │    - Context Recall         │                       │
│            └────────────────────────────┘                       │
└──────────────────────────────────────────────────────────────────┘
```

- 네트워크: 단일 도커 브리지 네트워크에서 서비스끼리 컨테이너 이름으로 통신.
- 포트 매핑: STT 8000, RAG 8001, Eval 8002, Ollama 11434, Qdrant 6333, n8n 5678.
- 스토리지: `qdrant_data/`, `ollama_data/`, `n8n_data/`로 영속 볼륨 마운트.

## 기술 스택과 채택 이유

### STT: FastAPI + Faster-Whisper (Large V3 Turbo)
- **모델**: `whisper-large-v3-turbo` (기본) 또는 KsponSpeech Fine-tuned CTranslate2 모델
- **이유**: Large V3 Turbo는 Large V3 대비 속도가 빠르면서 정확도 유지. KsponSpeech 파인튜닝으로 한국어 인식률 향상.
- **설정**: `WHISPER_HF_MODEL` 환경변수로 HuggingFace의 파인튜닝 모델 지정 가능.
- **최적화**: VAD(Voice Activity Detection) 필터로 무음 구간 스킵, beam_size=5로 디코딩 품질 향상.

### RAG Pipeline: LangGraph + Hybrid Search
- **Semantic Chunking**: 임베딩 유사도 기반 문장 분할. 고정 길이 대비 의미적 일관성 유지.
- **Hybrid Search**: BM25(스파스) + 벡터(덴스) 검색 결합 → Reciprocal Rank Fusion(RRF).
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder로 최종 정밀 랭킹.
- **LangGraph Agentic RAG**:
  - 쿼리 분석: 라우팅(검색 필요 여부) + 쿼리 분해(복합 질문 → 서브쿼리).
  - 조건 분기: 단순 질문은 직접 답변, 지식 기반 질문은 검색 파이프라인으로 라우팅.
  - 반성 루프: 생성된 답변의 관련성/충실성을 자체 평가, 기준 미달 시 쿼리 재구성 후 재검색 (최대 2회).

### LLM & 임베딩: Ollama
- `llama3`: 채팅/요약/답변 생성.
- `nomic-embed-text`: 텍스트 임베딩 (768차원).

### 벡터 스토어: Qdrant
- Cosine Distance 기반 유사도 검색.
- 메타데이터(제목/요약/원문/chunk_index)와 벡터를 함께 저장.

### 평가: RAGAS
- **Faithfulness**: 답변이 검색된 컨텍스트에 근거하는지 측정.
- **Answer Relevance**: 답변이 질문에 적절히 대응하는지 측정.
- **Context Precision**: 검색된 문서가 질문에 관련 있는지 측정.
- **Context Recall**: 검색된 컨텍스트가 정답을 커버하는지 측정.
- 단일 평가(`/evaluate`) 및 배치 평가(`/evaluate/batch`) 지원.

### 오케스트레이션: n8n
- 시각적 워크플로우 편집기로 STT → RAG 인제스트 → 쿼리 파이프라인 연결.

## 동작 흐름

### 문서 인제스트 (`workflows/meeting-summary.md`)
1) 오디오 업로드 트리거.
2) STT 서비스(`POST /stt`)로 전사 → Whisper Large V3 Turbo (한국어 최적화).
3) RAG 서비스(`POST /ingest`)로 인제스트:
   - Semantic Chunking (임베딩 유사도 기반 분할).
   - 각 청크 자동 임베딩 (nomic-embed-text).
   - Qdrant에 저장.

### Agentic RAG 질의 (`workflows/rag-query.md`)
1) 질문 수신 (`POST /query`).
2) **쿼리 분석**: 라우팅 결정 + 서브쿼리 분해.
3) **Hybrid 검색**: 벡터 + BM25 → RRF → Cross-Encoder Reranking.
4) **답변 생성**: 컨텍스트 기반 답변.
5) **반성 루프**: 관련성/충실성 자체 평가 → 미달 시 재구성 후 재검색.
6) 최종 답변 + 출처 + 메타데이터 반환.

### RAG 품질 평가
1) 평가 데이터셋 준비 (질문, 답변, 컨텍스트, 정답).
2) `POST /evaluate/batch`로 RAGAS 메트릭 자동 측정.
3) Faithfulness, Answer Relevance 등 점수 확인.

## 주요 설정 포인트

### Whisper
- `WHISPER_MODEL_SIZE`: 기본 `large-v3-turbo`. 리소스 제약 시 `small`/`medium` 사용.
- `WHISPER_HF_MODEL`: KsponSpeech 파인튜닝 모델 HF ID (예: `seastar105/whisper-large-v3-turbo-ksponspeech-ct2`).
- `WHISPER_DEVICE`: `cpu` 또는 `cuda`.
- `WHISPER_COMPUTE_TYPE`: `int8`(CPU), `float16`(GPU).

### RAG
- `RERANKER_MODEL`: Cross-encoder 모델. 기본 `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- `EMBEDDING_DIM`: 임베딩 차원. 기본 768 (nomic-embed-text).
- `CHUNK_BREAKPOINT_THRESHOLD`: Semantic Chunking 유사도 임계값 (0.0~1.0). 낮을수록 더 세밀한 분할.

### Ollama
- entrypoint에서 `llama3`, `nomic-embed-text` 자동 pull.
- 추가 모델은 `docker exec notetaker-ollama ollama pull <model>`.

## 인터페이스 예시

### STT
```bash
curl -X POST "http://localhost:8000/stt" \
     -F "audio=@meeting.wav" \
     -F "lang=ko"
```

### 문서 인제스트
```bash
curl -X POST "http://localhost:8001/ingest" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "회의 전사 텍스트...",
       "title": "2024년 1분기 전략 회의",
       "source": "meeting"
     }'
```

### RAG 질의
```bash
curl -X POST "http://localhost:8001/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "1분기 전략 회의에서 결정된 사항이 뭐야?"}'
```

### RAGAS 평가
```bash
curl -X POST "http://localhost:8002/evaluate" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "1분기 전략은?",
       "answer": "생성된 답변...",
       "contexts": ["검색된 컨텍스트1...", "컨텍스트2..."],
       "ground_truth": "정답 텍스트 (선택)"
     }'
```

### 배치 평가
```bash
curl -X POST "http://localhost:8002/evaluate/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "samples": [
         {"question": "질문1"},
         {"question": "질문2", "ground_truth": "정답2"}
       ]
     }'
```

## 서비스 포트 정리

| 서비스 | 포트 | 역할 |
|--------|------|------|
| STT | 8000 | Whisper Large V3 Turbo 한국어 전사 |
| RAG | 8001 | Agentic RAG (인제스트 + 질의) |
| Eval | 8002 | RAGAS 품질 평가 |
| Ollama | 11434 | LLM 채팅 + 임베딩 |
| Qdrant | 6333 | 벡터 DB |
| n8n | 5678 | 워크플로우 오케스트레이션 |

## 이력서 포인트
- Whisper Large V3 Turbo 한국어 Fine-tuning(KsponSpeech) 및 FastAPI STT, Ollama, Qdrant를 Docker Compose로 구성한 End-to-End RAG 파이프라인 설계.
- Semantic Chunking 및 Hybrid Search(BM25 + 벡터) + Cross-Encoder Reranker 적용으로 RAG 검색 품질 고도화.
- LangGraph 기반 Agentic RAG 구현 (쿼리 분해, 조건 분기 재검색, 반성 루프).
- RAGAS 기반 RAG 품질 평가 파이프라인 구축 (Faithfulness, Answer Relevance 메트릭 자동 측정).
