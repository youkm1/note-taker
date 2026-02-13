# Local AI Note Taker

오디오를 로컬에서 전사하고, Semantic Chunking + Hybrid Search + LangGraph Agentic RAG로 질의응답하는 End-to-End RAG 파이프라인입니다.

## What it does

- **STT**: Whisper large-v3-turbo (KsponSpeech 파인튜닝 지원) + VAD 필터로 음성 전사.
- **RAG**: Semantic Chunking → Hybrid Search (BM25 + 벡터) → RRF → Cross-Encoder Reranker → LangGraph Agentic RAG (쿼리 분해, 반성 루프).
- **Evaluation**: RAGAS 기반 Faithfulness, Answer Relevance, Context Precision, Context Recall 자동 측정.
- **Storage**: Qdrant 벡터 DB에 임베딩과 메타데이터 저장.
- **LLM**: Ollama (llama3 + nomic-embed-text) 완전 로컬 실행.
- **Orchestration**: n8n 시각적 워크플로우 편집기.

## Requirements

- Docker and docker-compose

## Quick Start

```bash
cp .env.example .env   # 필요시 값 조정
docker-compose up -d
```

## Services

| 서비스 | 포트 | 설명 |
|--------|------|------|
| STT | 8000 | Whisper large-v3-turbo 음성 전사 |
| RAG | 8001 | Semantic Chunking + Hybrid Search + LangGraph Agentic RAG |
| Eval | 8002 | RAGAS 기반 RAG 품질 평가 |
| Qdrant | 6333 | 벡터 데이터베이스 |
| Ollama | 11434 | LLM (llama3) + Embedding (nomic-embed-text) |
| n8n | 5678 | 워크플로우 오케스트레이션 UI |

## API Endpoints

### STT (`http://localhost:8000`)
- `POST /stt` — 오디오 파일 전사 (multipart `audio`, 옵션 `lang`)
- `GET /health` — 서비스 상태 확인

### RAG (`http://localhost:8001`)
- `POST /ingest` — 텍스트 인제스트 (Semantic Chunking → 임베딩 → Qdrant 저장)
- `POST /query` — LangGraph Agentic RAG 질의
- `GET /health` — 서비스 상태 확인

### Eval (`http://localhost:8002`)
- `POST /evaluate` — 단일 QA 쌍 RAGAS 평가
- `POST /evaluate/batch` — 배치 QA 쌍 RAGAS 평가
- `GET /health` — 서비스 상태 확인

## Environment Variables

| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| `WHISPER_MODEL_SIZE` | `large-v3-turbo` | Whisper 모델 크기 |
| `WHISPER_HF_MODEL` | — | HuggingFace 파인튜닝 모델 (예: KsponSpeech) |
| `WHISPER_DEVICE` | `cpu` | 디바이스 (`cpu`/`cuda`) |
| `WHISPER_COMPUTE_TYPE` | `int8` | 연산 정밀도 |
| `EMBED_MODEL` | `nomic-embed-text` | 임베딩 모델 |
| `LLM_MODEL` | `llama3` | LLM 모델 |
| `QDRANT_COLLECTION` | `notes` | Qdrant 컬렉션명 |
| `VECTOR_SIZE` | `768` | 임베딩 벡터 차원 |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker 모델 |

## Example Workflows

- Meeting ingestion: `workflows/meeting-summary.md`
- RAG Q&A: `workflows/rag-query.md`

## Data Persistence

- Qdrant data: `qdrant_data/`
- Ollama models: `ollama_data/`
- n8n state: `n8n_data/`
