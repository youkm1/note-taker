# Troubleshooting Log — Note-Taker PoC (OpenShift + watsonx + RAGAS)

OpenShift Sandbox 배포 및 RAGAS 평가 과정에서 겪은 트러블슈팅 기록.

---

## 1. oc start-build 업로드 무한 대기 (FetchSourceFailed)

### 증상
```
Uploading directory "." as binary input for the build ...
.......................................................................
(30분 이상 점만 찍히다 FetchSourceFailed)
```

### 원인
`oc start-build --from-dir=.` 는 `.dockerignore`를 업로드 단계에서 **무시**한다.  
`.dockerignore`는 Docker 빌드 시 COPY 대상을 거르는 것이고, `oc`의 tar 압축·업로드 단계는 별개다.  
로컬에 `.venv`(수백 MB), `.git`, 음성 파일 등이 전부 업로드되어 타임아웃 발생.

### 해결
```bash
# 빌드에 필요한 폴더만 임시 디렉토리로 복사 후 업로드
rm -rf /tmp/ragbuild && mkdir -p /tmp/ragbuild
cp -r rag /tmp/ragbuild/rag
oc start-build notetaker-rag --from-dir=/tmp/ragbuild --follow

# eval도 동일하게
rm -rf /tmp/evalbuild && mkdir -p /tmp/evalbuild
cp -r eval /tmp/evalbuild/eval
oc start-build notetaker-eval --from-dir=/tmp/evalbuild --follow
```

### 교훈
`oc start-build --from-dir`에서는 `.dockerignore`가 업로드 필터로 작동하지 않는다.  
빌드 컨텍스트를 최소화하려면 **별도 디렉토리**를 만들어 필요한 파일만 복사해서 올려야 한다.

---

## 2. OpenShift Route 타임아웃으로 빈 curl 응답

### 증상
```
curl ... | python3 -m json.tool
Expecting value: line 1 column 1 (char 0)
```
RAGAS 처리 로그에는 완료가 찍히는데 curl 응답이 비어있음.

### 원인
OpenShift HAProxy Route 기본 타임아웃(~30~60초)이 RAGAS 처리 시간(60~120초)보다 짧아 연결이 강제 종료됨.

### 해결
```bash
oc annotate route notetaker-eval \
  haproxy.router.openshift.io/timeout=600s \
  --overwrite
```
또는 `08-routes.yaml`에 어노테이션 추가 후 `oc apply`:
```yaml
metadata:
  annotations:
    haproxy.router.openshift.io/timeout: 600s
```

### 교훈
장시간 처리 엔드포인트(RAGAS, LLM 추론 등)는 Route 타임아웃을 반드시 확인해야 한다.  
YAML 커밋만으로는 클러스터에 반영되지 않는다 — `oc apply`가 필요하다.

---

## 3. RAGAS 점수 전부 null (Gemini API quota 소진)

### 증상
```json
{
  "faithfulness": null,
  "answer_relevancy": null,
  "context_precision": 0.0,
  "context_recall": 0.0
}
```

### 원인
RAG 서비스의 Gemini API key quota 소진(429 RESOURCE_EXHAUSTED).  
RAG가 Gemini로 **임베딩(검색)과 생성** 양쪽을 처리하므로, 키가 죽으면 `contexts: []` + 에러 답변이 반환됨.  
RAGAS는 빈 context로 평가해 전부 null/0이 나옴.

```json
{
  "answer": "Gemini API 요청 한도에 걸렸습니다...",
  "contexts": []
}
```

### 해결
- 단기: 새 Gemini API key로 Secret 교체 후 `oc rollout restart`
- 장기: 벤더 의존성 제거

### 교훈
RAGAS 점수가 전부 null이면 **평가 LLM이 아니라 RAG 자체를 먼저 확인**해야 한다.  
`/query` 엔드포인트 직접 호출로 `contexts` 필드가 채워지는지 먼저 검증할 것.

---

## 4. RAGAS Evaluator 모델 교체 과정

### 4-1. Granite-3-8b-instruct 지원 안 됨
```
Model 'ibm/granite-3-8b-instruct' is not supported
```
→ `ibm/granite-4-h-small`로 변경

### 4-2. Granite-4-h-small JSON 파싱 실패
```
RagasOutputParserException: Could not parse output
```
RAGAS는 LLM에게 JSON 형식 출력을 요구하는데, 소형 모델이 지시를 따르지 못함.  
→ `meta-llama/llama-3-3-70b-instruct`로 변경

### 4-3. WATSONX_APIKEY 환경변수 인식 실패
`WatsonxLLM`은 파라미터 `apikey=`가 아니라 환경변수 `WATSONX_APIKEY`를 읽는다.  
Secret key 이름을 `WATSONX_API_KEY` → `WATSONX_APIKEY`로 수정하고 `oc apply` 재적용.

### 교훈
RAGAS LLM-as-judge는 JSON 출력 지시 준수 능력이 필요하다.  
소형 모델(8B 이하)은 실패 가능성이 높으므로 70B 이상 모델을 권장한다.

---

## 5. watsonx.ai 전체 마이그레이션 → Lite quota 소진으로 롤백

### 배경
Gemini quota 반복 소진으로 watsonx.ai 전면 전환 시도.
- 생성: `meta-llama/llama-3-3-70b-instruct`
- 임베딩: `intfloat/multilingual-e5-large` (1024차원, 한국어 multilingual)

### 시도한 것
- `rag/watsonx_client.py` 신규 작성 (chat + embed 공통 모듈)
- `graph.py`, `retriever.py`, `chunking.py`, `app.py` Gemini → watsonx 전체 교체
- Qdrant 컬렉션 삭제 후 1024차원으로 재생성·재인덱싱·인제스트까지 성공

### 실패 원인
```
Status code: 403, "code":"token_quota_reached"
```
watsonx.ai Lite 플랜 월 quota 소진.  
RAG 그래프는 질문당 LLM 4~6회 호출(analyse→generate→reflect→rewrite) + RAGAS evaluator 호출이 **같은 계정 quota를 공유**해 빠르게 소진됨.

### 결과
`git revert`로 watsonx 마이그레이션 롤백, Gemini 복귀.  
Qdrant 컬렉션 3072차원으로 재생성, 재인덱싱.

### 교훈
무료 LLM quota로 RAGAS baseline을 뽑는 것은 구조적으로 어렵다.  
RAG(4~6회/질문) + RAGAS evaluator가 같은 계정을 쓰면 10개 샘플에 수백 회 호출이 발생한다.  
유료 플랜 없이 운영하려면 RAG 그래프 호출 수를 최소화(reflect/rewrite 생략)해야 한다.

---

## 6. RAGAS Evaluator LLM 교체 (watsonx → Gemini)

### 원인
watsonx quota 소진 후 RAG는 Gemini로 복귀했으나,  
eval 서비스는 여전히 watsonx llama를 evaluator로 사용 → 평가 호출마다 403.

### 해결
`eval/app.py`에서 `WatsonxLLM` → `ChatGoogleGenerativeAI(gemini-2.5-flash)` 교체.  
임베딩은 이미 `GoogleEmbeddings(gemini-embedding-001)`이라 변경 없음.

```python
# Before
from langchain_ibm import WatsonxLLM
_granite_llm = WatsonxLLM(model_id="meta-llama/llama-3-3-70b-instruct", ...)
_evaluator_llm = LangchainLLMWrapper(_granite_llm)

# After
from langchain_google_genai import ChatGoogleGenerativeAI
_gemini_chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)
_evaluator_llm = LangchainLLMWrapper(_gemini_chat)
```

### 교훈
RAG와 Evaluator의 LLM을 다른 벤더로 섞으면 각각의 quota를 독립적으로 관리해야 한다.  
배포 후 로그에서 새 파드가 올바른 모델을 초기화하는지 반드시 확인할 것.

---

## 7. Qdrant 임베딩 차원 불일치

### 증상
벡터 검색 시 차원 불일치 에러 또는 검색 결과 없음.

### 원인
임베딩 모델을 바꾸면 벡터 차원이 달라진다.  
기존 컬렉션은 이전 차원으로 생성된 상태라 새 벡터와 호환되지 않는다.

| 모델 | 차원 |
|------|------|
| `gemini-embedding-001` | 3072 |
| `intfloat/multilingual-e5-large` (watsonx) | 1024 |

### 해결
임베딩 모델 교체 시 반드시:
1. Qdrant 기존 컬렉션 삭제
2. 새 차원으로 컬렉션 재생성 (app startup 시 자동)
3. 문서 재인제스트

```bash
WX=$(oc get secret notetaker-secrets -o jsonpath='{.data.QDRANT_API_KEY}' | base64 -d)
QURL=$(oc get cm notetaker-config -o jsonpath='{.data.QDRANT_URL}')
curl -X DELETE "$QURL/collections/notes" -H "api-key: $WX"
oc rollout restart deployment/notetaker-rag
```

### 교훈
임베딩 모델 교체 = 무조건 재인덱싱.  
`VECTOR_SIZE` 환경변수를 Deployment YAML에서 명시적으로 관리할 것.

---

## 8. RAGAS 점수 분석 — context 지표 낮은 원인

### 증상 (샘플 1개 결과)
```json
{
  "faithfulness": 1.0,
  "answer_relevancy": 0.99,
  "context_precision": 0.0,
  "context_recall": 0.5
}
```
생성 품질(faithfulness, answer_relevancy)은 높은데 검색 품질(context 지표)이 낮음.

### 원인 분석

**context_precision 0.0**: 검색된 context 중 관련 청크가 상위에 랭크되지 않음.  
현재 BM25가 한국어 어절 단위(`\w+` 정규식)로만 토크나이징 → 형태소 분리 없음.  
예: "정보통신과" → "정보", "통신"으로 분리 안 됨 → 키워드 매칭 부정확.

**context_recall 0.5**: ground_truth 정보의 절반만 context에 포함.  
청크 크기가 커서 관련 정보가 여러 청크에 분산됨.

### 개선 방향
| 문제 | 단기 | 장기 |
|------|------|------|
| BM25 한국어 품질 | - | Elasticsearch + nori 형태소 분석기 (Phase 2) |
| 청크 분산 | `max_chunk_sentences` 줄이기 | 청킹 전략 개선 |
| 재랭킹 | - | 한국어 CrossEncoder 모델 사용 |

---

## 요약 — 반복된 핵심 문제

| 문제 | 근본 원인 | 해결 |
|------|-----------|------|
| RAGAS null 점수 | RAG API quota 소진 → `contexts: []` | `/query` 먼저 확인 후 평가 |
| 빈 curl 응답 | Route 타임아웃 < RAGAS 처리 시간 | Route `timeout=600s` 어노테이션 |
| 빌드 무한 대기 | `oc start-build --from-dir`이 전체 업로드 | 필요한 폴더만 임시 디렉토리로 분리 |
| 임베딩 차원 불일치 | 모델 교체 시 Qdrant 미삭제 | 재인덱싱 필수 |
| LLM quota 소진 | RAG + Evaluator가 같은 계정 공유 | 벤더 분리 또는 유료 플랜 전환 |
