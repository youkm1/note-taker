# Eval 서비스 RAGAS 호환성 수정 — 의사결정 & 트러블슈팅 로그

## 1. 문제 요약

eval 서비스(`eval/app.py`)가 RAGAS 라이브러리의 API 변경(v0.1.x → v0.4.x)으로 인해 컨테이너 기동 시 런타임 에러가 발생하는 상태였다.

### 발견된 이슈 3건

| # | 이슈 | 심각도 |
|---|------|--------|
| 1 | 메트릭 import 방식: 소문자 인스턴스(`faithfulness`) → 클래스(`Faithfulness`) 변경 | 치명적 |
| 2 | 임베딩 래퍼: `langchain_google_genai` → `ragas.embeddings.GoogleEmbeddings` 전환 필요 | 치명적 |
| 3 | `evaluate()` 호출: `llm=`/`embeddings=`를 함수에 전달 → 메트릭 초기화 시 주입으로 변경 | 치명적 |

### 부수 이슈

| # | 이슈 | 심각도 |
|---|------|--------|
| 4 | `.env`의 `WHISPER_MODEL_SIZE=base`가 문서/docker-compose 기본값(`large-v3-turbo`)과 불일치 | 중간 |
| 5 | `eval/requirements.txt`에 `ragas` 버전 미고정 → 최신 설치 시 API 깨짐 | 높음 |

---

## 2. 수정 내역

### 2.1 eval/app.py — import 변경

**Before:**
```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
```

**After:**
```python
from ragas.embeddings import GoogleEmbeddings
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
```

**근거:**
- RAGAS v0.2.x부터 메트릭이 인스턴스 변수(소문자)에서 클래스(PascalCase)로 변경됨
- 공식 마이그레이션 가이드(docs.ragas.io/en/stable/howtos/migrations/migrate_from_v01_to_v02/) 참조
- `langchain-google-genai`는 불필요한 의존성 트리(langchain-core, langchain-community 등)를 끌고 오므로, RAGAS 네이티브 `GoogleEmbeddings`로 교체

### 2.2 eval/app.py — 임베딩 초기화 변경

**Before:**
```python
_evaluator_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY,
)
```

**After:**
```python
_evaluator_embeddings = GoogleEmbeddings(
    client=_gemini_client,
    model="gemini-embedding-001",
)
```

**근거:**
- `ragas.embeddings.GoogleEmbeddings`는 `google.genai.Client`를 직접 래핑
- 이미 생성된 `_gemini_client` 인스턴스를 재사용하므로 API 키 중복 전달 불필요
- `models/` prefix 불필요 (langchain 래퍼만 필요했던 것)
- 공식 Gemini 통합 가이드(docs.ragas.io/en/stable/howtos/integrations/gemini/) 패턴 적용

### 2.3 eval/app.py — 메트릭 초기화 + evaluate() 호출 변경

**Before:**
```python
metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
result = evaluate(dataset, metrics=metrics, llm=_evaluator_llm, embeddings=_evaluator_embeddings)
```

**After:**
```python
metrics = [
    Faithfulness(llm=_evaluator_llm),
    AnswerRelevancy(llm=_evaluator_llm, embeddings=_evaluator_embeddings),
    ContextPrecision(llm=_evaluator_llm),
    ContextRecall(llm=_evaluator_llm),
]
result = evaluate(dataset, metrics=metrics)
```

**근거:**
- RAGAS v0.4.x에서 메트릭은 클래스 인스턴스화 시 `llm`/`embeddings`를 주입받는 패턴
- `evaluate()` 함수에 `llm=`/`embeddings=`를 직접 전달하는 방식도 아직 동작하나, 공식 권장 패턴은 메트릭 레벨 주입
- `AnswerRelevancy`는 LLM과 임베딩 모두 필요 (질문 변형 생성 → 임베딩 유사도 비교)
- 나머지 3개 메트릭은 LLM만 필요

### 2.4 eval/requirements.txt

**Before:**
```
ragas
...
langchain-google-genai
```

**After:**
```
ragas>=0.4.0,<0.5.0
langchain-community>=0.3.0,<0.4.0
...
(langchain-google-genai 삭제)
```

**근거:**
- `ragas>=0.4.0,<0.5.0` — 클래스 기반 메트릭 API 보장 + 상한 고정으로 미래 breaking change 방어
- `langchain-community>=0.3.0,<0.4.0` — ragas 0.4.3이 내부적으로 `langchain_community.chat_models.vertexai.ChatVertexAI`를 import하는데, `langchain-community 0.4.x`에서 해당 모듈이 `langchain-google-vertexai` 패키지로 분리됨. 0.3.x에는 아직 존재하므로 하위 버전 고정 (아래 5.7 참조)
- `langchain-google-genai` 제거 — `ragas.embeddings.GoogleEmbeddings`로 대체했으므로 더 이상 불필요. Docker 이미지 크기 감소 + 의존성 충돌 방지

### 2.6 Dockerfile 수정 (rag/Dockerfile, eval/Dockerfile)

**Before:**
```dockerfile
RUN pip install --no-cache-dir -r /app/requirements.txt
```

**After:**
```dockerfile
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /app/requirements.txt
```

**근거:**
- `sentence-transformers` (RAG) 및 `ragas` (Eval)가 PyTorch에 의존
- 기본 pip install은 CUDA 포함 PyTorch를 설치 → NVIDIA 라이브러리만 ~300MB+
- CPU 전용 index로 먼저 설치하면 CUDA 라이브러리 다운로드를 방지
- 이 프로젝트는 `WHISPER_DEVICE=cpu`로 GPU를 사용하지 않으므로 CUDA 불필요

### 2.5 .env

**Before:** `WHISPER_MODEL_SIZE=base`
**After:** `WHISPER_MODEL_SIZE=large-v3-turbo`

**근거:**
- `docker-compose.yml` 기본값이 `large-v3-turbo`
- `ARCHITECTURE.md`, `README.md` 모두 `large-v3-turbo` 기준으로 작성
- `base`는 개발 편의상 임시로 설정된 것으로 판단

---

## 3. 핵심 의사결정: AnswerRelevancy vs AnswerCorrectness

### 배경
공식 RAGAS Gemini 통합 가이드에서는 `AnswerCorrectness` 메트릭을 예시로 사용한다. 기존 코드는 `answer_relevancy` 필드를 사용한다.

### 분석
이 둘은 **서로 다른 메트릭**이다:

| 메트릭 | 측정 대상 | 방식 |
|--------|----------|------|
| `AnswerRelevancy` | 답변이 질문에 얼마나 관련되는가 | LLM으로 답변에서 질문 변형 생성 → 원본 질문과 임베딩 유사도 비교 |
| `AnswerCorrectness` | 답변이 정답(ground truth)과 얼마나 일치하는가 | F1 기반 사실 비교 + 임베딩 유사도 |

### 결정
**`AnswerRelevancy` 유지.**

이유:
1. 기존 코드의 의도가 "답변 관련성" 측정이었음
2. `AnswerRelevancy` 클래스의 `name` 속성이 `"answer_relevancy"` → DataFrame 컬럼명이 기존과 동일
3. `EvalScores` 응답 모델의 `answer_relevancy` 필드명과 정확히 일치
4. n8n 워크플로우(`RAG-QA.json`)의 Format Response 노드가 이 필드명을 참조할 수 있음

`AnswerCorrectness`를 사용하면 컬럼명이 `answer_correctness`로 바뀌어 응답 구조가 변경되므로, 하위 호환성이 깨진다.

---

## 4. 변경하지 않은 것 (+ 이유)

| 항목 | 이유 |
|------|------|
| `_fill_from_rag()` 함수 | RAG 서비스 연동 로직은 RAGAS 버전과 무관 |
| `_scores_from_row()` 함수 | 메트릭 `name` 속성이 기존 컬럼명과 동일하므로 수정 불필요 |
| `EvalScores` 응답 모델 | 필드명 변경 없음 |
| FastAPI 엔드포인트 | API 인터페이스 변경 없음 |
| `Dockerfile` | requirements.txt만 변경되면 pip install이 새 버전을 가져옴 |
| `llm_factory()` 호출 | RAGAS v0.4.x에서도 동일 시그니처 유지 확인 |
| Dataset 필드명 (`ground_truth`) | RAGAS v0.2+에서 `ground_truth`(단수)가 올바른 형식 |

---

## 5. 트러블슈팅 가이드

### ImportError: cannot import name 'answer_relevancy' from 'ragas.metrics'
**원인:** RAGAS v0.4.x에서 소문자 인스턴스 제거됨
**해결:** `from ragas.metrics import AnswerRelevancy` (PascalCase 클래스) 사용

### ImportError: cannot import name 'GoogleGenerativeAIEmbeddings'
**원인:** `langchain-google-genai` 패키지가 requirements.txt에서 제거됨
**해결:** 정상. `ragas.embeddings.GoogleEmbeddings` 사용으로 전환 완료

### AnswerRelevancy 결과가 NaN
**원인:** `AnswerRelevancy` 초기화 시 `embeddings` 미전달
**해결:** `AnswerRelevancy(llm=llm, embeddings=embeddings)` — 반드시 임베딩 함께 전달

### Docker 빌드 시 캐시된 레이어에 langchain이 남아있음
**해결:** `docker-compose build --no-cache eval`로 캐시 없이 재빌드

### evaluate() deprecation 경고
**상태:** RAGAS v0.4.x에서 `@experiment` 데코레이터 패턴을 권장하지만, `evaluate()` 함수도 여전히 동작
**대응:** 현재는 무시. 향후 RAGAS가 `evaluate()` 완전 제거 시 `@experiment`로 전환 필요

### RAGAS 버전 업그레이드 후 컬럼명 변경됨
**확인 방법:** `result.to_pandas().columns`로 실제 컬럼명 확인
**현재 기대값:** `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`

### ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
**원인:** `ragas 0.4.3`의 `ragas/llms/base.py` line 12에서 `from langchain_community.chat_models.vertexai import ChatVertexAI`를 top-level import함. 이 모듈은 `langchain-community 0.3.x`까지 존재했으나, `0.4.x`에서 `langchain-google-vertexai` 별도 패키지로 분리됨.
**시도한 해결책:**
1. `langchain-google-vertexai` 설치 → 실패. import 경로가 `langchain_google_vertexai`로 다름 (`langchain_community.chat_models.vertexai` 경로를 복원하지 못함)
2. `langchain-community>=0.3.0,<0.4.0` 핀닝 → 성공. 0.3.x에 해당 모듈이 존재하므로 import 해결
**결론:** ragas 0.4.3의 패키징 문제. ragas가 `langchain-community 0.4.x`와의 호환성을 놓친 것으로 판단. 우리 측에서 `langchain-community` 버전을 0.3.x로 고정하여 해결.

### Docker 빌드 시 "No space left on device"
**원인:** `sentence-transformers` → `torch` 의존성이 기본적으로 CUDA 포함 버전을 설치하여 이미지 크기가 폭증
**해결:** Dockerfile에서 `pip install torch --index-url https://download.pytorch.org/whl/cpu`로 CPU 전용 PyTorch를 먼저 설치한 뒤 나머지 의존성 설치
**추가 조치:** `docker system prune -af --volumes`로 미사용 이미지/볼륨 정리

---

## 6. 참고 자료

- [RAGAS v0.1→v0.2 마이그레이션 가이드](https://docs.ragas.io/en/stable/howtos/migrations/migrate_from_v01_to_v02/)
- [RAGAS Gemini 통합 가이드](https://docs.ragas.io/en/stable/howtos/integrations/gemini/)
- [RAGAS evaluate() 레퍼런스](https://docs.ragas.io/en/stable/references/evaluate/)
- [RAGAS 커스텀 모델 설정](https://docs.ragas.io/en/stable/howtos/customizations/customize_models/)
