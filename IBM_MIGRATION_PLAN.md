# IBM Client Engineering Migration Plan

## Goal

Reframe this project as an IBM Client Engineering style AI and Automation PoC:

- Build a technical asset around Agentic AI, RAG, and automation.
- Show how a customer-facing PoC can move from a fast local prototype to an IBM enterprise architecture.
- Demonstrate hands-on experience with Python, Linux/Git, LLMs, data engineering, Kubernetes/OpenShift, and IBM AI technology.

## Current Architecture

```text
Audio upload
  -> Local Whisper STT service
  -> RAG ingest/query service
     -> Gemini LLM
     -> Gemini embedding
     -> Qdrant vector database
     -> LangGraph Agentic RAG
  -> RAGAS evaluation service
  -> n8n workflow automation
  -> Docker Compose deployment
```

This is a strong PoC baseline. The IBM migration should not replace everything at once. The goal is to keep the working architecture and introduce IBM services where they improve the enterprise story.

The migration should be benchmark-driven. IBM technology should be presented as an enterprise option, not as an automatic performance upgrade. For Korean meetings, Gemini and local Whisper can remain the quality and latency baselines if they outperform IBM alternatives.

## Recommended IBM Target Architecture

```text
Audio / meeting material upload
  -> IBM Watson Speech to Text or local Whisper
  -> RAG ingest/query service
     -> watsonx.ai Granite LLM
     -> watsonx.ai / Granite embedding model
     -> IBM Cloud Databases for Elasticsearch or Qdrant on OpenShift
     -> LangGraph Agentic RAG
  -> watsonx Orchestrate
     -> import/wrap LangGraph agent
     -> automate meeting summary, Q&A, evaluation, and follow-up actions
  -> OpenShift on IBM Cloud or IBM Kubernetes Service
     -> Container Registry
     -> Kubernetes Secret / IBM Secrets Manager
     -> PVC for stateful data
     -> Monitoring and logging
```

## Migration Priorities

### 1. LLM and Embedding: Gemini Baseline plus watsonx.ai Granite Option

**Current:** Gemini API powers answer generation, query analysis, reflection, and embeddings.

**IBM migration:** Add a provider abstraction so the project can run with either Gemini or watsonx.ai.

```text
LLM_PROVIDER=gemini | watsonx
EMBEDDING_PROVIDER=gemini | watsonx
```

Why this matters for the IBM role:

- Directly maps to Foundation Models and LLM hands-on experience.
- Shows that the PoC can be adapted to enterprise AI platform requirements.
- Creates a strong story: local Ollama for data privacy, Gemini for fast prototype validation, watsonx.ai Granite for IBM enterprise AI alignment.
- Shows that model selection is based on measurable trade-offs, not vendor branding.

Important trade-off:

- Gemini 2.5 Flash can remain the stronger performance baseline for Korean answer quality, speed, and general reasoning if benchmarks confirm it.
- Granite should be positioned for IBM alignment, enterprise controls, hybrid cloud options, governance, and customer preference, not guaranteed raw performance superiority.

Interview line:

> I initially evaluated a local Ollama setup for privacy, then migrated to Gemini for faster PoC iteration. For an IBM customer-facing architecture, I added a provider-agnostic LLM layer so Gemini can remain the performance baseline while watsonx.ai Granite can be evaluated for IBM enterprise requirements such as governance, deployment model, and customer platform preference.

### 2. Automation: n8n to watsonx Orchestrate

**Current:** n8n orchestrates meeting ingestion and RAG query workflows.

**IBM migration:** Keep the service APIs, but expose them as tools/actions that watsonx Orchestrate can call.

Candidate tools:

- `transcribe_audio`
- `ingest_meeting_transcript`
- `query_meeting_notes`
- `evaluate_rag_answer`
- `generate_meeting_summary`

Important design choice:

- Do not replace LangGraph immediately.
- Keep LangGraph as the code-level Agentic RAG engine.
- Use watsonx Orchestrate as the enterprise automation and agent operations layer.

Why this matters for the IBM role:

- Directly maps to Agentic AI and Automation pilot projects.
- Shows understanding that customer PoCs often need workflow integration, not just a chatbot.
- Enables a production story around hosting, authentication, monitoring, policy, and tool use.

Interview line:

> I kept LangGraph for fine-grained RAG reasoning, but planned watsonx Orchestrate as the enterprise automation layer. This lets the same agentic RAG capability become a customer-facing workflow asset rather than just a backend API.

### 3. STT: Local Whisper to IBM Watson Speech to Text

**Current:** Faster-Whisper runs locally in Docker. This maximizes data privacy but long Korean meeting files can be slow on CPU.

**IBM migration:** Add an optional STT provider.

```text
STT_PROVIDER=local_whisper | ibm_watson
```

Recommended behavior:

- Use local Whisper when raw audio must remain local.
- Use local Whisper when it gives better Korean transcription quality.
- Use IBM Watson Speech to Text when managed cloud STT, lower local compute cost, or IBM platform alignment are more important than local-only processing.
- Benchmark both with the same meeting audio and compare latency, Korean accuracy, and cost.

Why this matters for the IBM role:

- Shows structured trade-off analysis: privacy, latency, accuracy, cost, and managed service adoption.
- Demonstrates the ability to recommend options based on customer requirements.

Interview line:

> The local Whisper service protected raw audio and produced the best Korean transcription baseline in my tests, but it became the main CPU latency bottleneck. I designed IBM Watson Speech to Text as an optional provider so the customer can choose between local accuracy/privacy and managed IBM STT depending on security, performance, and platform requirements.

### 4. Vector Search: Qdrant to IBM-Managed Search

**Current:** Qdrant stores transcript chunks and vectors.

**IBM migration options:**

1. Keep Qdrant on OpenShift with PVC.
2. Replace Qdrant with IBM Cloud Databases for Elasticsearch if the PoC needs an IBM-managed search/vector-search story.

Recommended approach:

- For fastest migration, keep Qdrant on OpenShift.
- For stronger IBM positioning, create a second retriever implementation for Elasticsearch.

Why this matters for the IBM role:

- Shows basic data engineering knowledge for AI services.
- Enables a discussion about managed database trade-offs, persistence, scaling, and retrieval quality.

Interview line:

> I treated vector search as a replaceable retrieval layer. Qdrant is useful for fast local PoCs, while IBM-managed Elasticsearch can support a stronger enterprise deployment story.

### 5. Deployment: Docker Compose to OpenShift on IBM Cloud

**Current:** Docker Compose runs all services locally.

**IBM migration:** Create Kubernetes/OpenShift manifests.

Core resources:

- `Deployment` for `stt`, `rag`, and `eval`
- `StatefulSet` or `Deployment` plus `PVC` for Qdrant
- `Deployment` plus `PVC` for n8n or Orchestrate-adjacent demo workflow
- `Service` for internal communication
- `Route` or `Ingress` for external endpoints
- `Secret` for API keys
- `ConfigMap` for non-sensitive configuration

Recommended platform:

- Best IBM interview signal: Red Hat OpenShift on IBM Cloud.
- Acceptable alternative: IBM Kubernetes Service.
- Local practice path: minikube, kind, or Docker Desktop Kubernetes before IBM Cloud deployment.

Why this matters for the IBM role:

- Directly maps to Kubernetes and Cloud Skills.
- Shows that the project can move from a laptop PoC to an enterprise container platform.

Interview line:

> I containerized each service and planned the Docker Compose to OpenShift migration with Deployments, Services, Secrets, and PVCs. Stateful services such as Qdrant and workflow state require persistent storage, which is why OpenShift/Kubernetes is a better fit than a purely serverless deployment.

### 6. Secrets, Governance, and Operations

Add an enterprise operations layer:

- Move `.env` values into Kubernetes Secret or IBM Secrets Manager.
- Store audio files and transcripts in IBM Cloud Object Storage if raw file retention is needed.
- Track RAG evaluation metrics as governance evidence.
- Add monitoring and logs for latency, STT duration, retrieval quality, and failed workflows.

Why this matters for the IBM role:

- Shows that the PoC is not only a demo, but also has a path toward operational readiness.
- Connects with customer concerns around security, compliance, reliability, and observability.

Interview line:

> I separated prototype speed from enterprise readiness. The first version optimized for rapid validation, and the IBM migration plan adds secret management, persistence, monitoring, and governance so the PoC can be discussed with business and technical stakeholders.

## Fine-Grained IBM Model Candidates

Use these as benchmark candidates, not automatic replacements.

| Current component | Current implementation | IBM candidate | Recommendation |
|------------------|------------------------|---------------|----------------|
| Answer generation | Gemini 2.5 Flash | watsonx.ai Granite language model | Keep Gemini as baseline. Test Granite for IBM alignment, governance, and customer preference. |
| Query analysis / reflection | Gemini 2.5 Flash in LangGraph | watsonx.ai Granite language model | Optional. Accuracy and Korean instruction-following must be benchmarked before switching. |
| Text embedding | Gemini embedding-001 | Granite embedding multilingual models | Worth testing. This is one of the best IBM-specific swaps for a RAG project. |
| Semantic chunking embedding | Gemini embedding-001 | Granite embedding multilingual models | Worth testing with the same chunking threshold and retrieval metrics. |
| Cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | watsonx.ai text rerank API or Granite embedding reranker | Worth testing, but be careful with Korean. Some IBM reranker options are English-focused. |
| RAG faithfulness / groundedness check | RAGAS with Gemini evaluator | Granite Guardian or watsonx.governance-oriented evaluation | Good for governance story. Do not use it as the only quality metric for Korean answers. |
| STT | Local Whisper / optional providers | IBM Watson Speech to Text | Optional only. Keep local Whisper or RTZR if they perform better on Korean meetings. |
| Document parsing | Not implemented yet | IBM Docling / Granite Docling | High-value extension if meeting PDFs or slides are added to the RAG corpus. |

Reranker decision:

- The current reranker, `cross-encoder/ms-marco-MiniLM-L-6-v2`, is also not a Korean-specialized model.
- IBM has watsonx.ai reranking support and Granite embedding/reranker models, so this is a reasonable IBM experiment.
- The safest implementation is `RERANKER_PROVIDER=local | watsonx` instead of a hard replacement.
- If Korean retrieval quality drops, keep the best-performing non-IBM reranker and explain the IBM version as an enterprise-aligned option.

Interview line:

> I did not replace every model just to use IBM branding. I separated generation, embedding, reranking, evaluation, and STT into provider choices, then used benchmarks to decide which IBM components improved the enterprise story without hurting Korean quality.

## Recommended Implementation Order

### Phase 1: IBM AI Core

- Add `watsonx.ai` LLM provider.
- Add `watsonx.ai` embedding provider.
- Keep Gemini as the quality and latency baseline, not merely a fallback.
- Update README with model/provider comparison.
- Benchmark answer quality, Korean fluency, latency, and retrieval quality before choosing the default provider.

### Phase 2: IBM STT Option

- Add `STT_PROVIDER`.
- Implement IBM Watson Speech to Text provider.
- Benchmark against local Whisper on the same long Korean meeting audio.
- Document latency, accuracy, security, and cost trade-offs.
- Keep local Whisper as the default if it continues to outperform IBM STT on Korean meeting audio.

### Phase 3: OpenShift/Kubernetes Readiness

- Add Kubernetes manifests.
- Add Secret and ConfigMap templates.
- Add PVC templates for Qdrant and workflow state.
- Add a short runbook for local Kubernetes and IBM OpenShift deployment.

### Phase 4: Orchestrate Readiness

- Add OpenAPI descriptions for service endpoints.
- Define Orchestrate tools/actions around STT, ingest, query, summary, and evaluation.
- Keep LangGraph as the core agent and document how it can be imported or wrapped by watsonx Orchestrate.

### Phase 5: Enterprise PoC Story

- Add architecture diagram.
- Add interview-ready explanation.
- Add a benchmark table.
- Add a one-page customer workshop script.

## What Not To Overdo

- Do not replace LangGraph first. It is already a strong Agentic AI implementation.
- Do not migrate every component before the project is stable.
- Do not make the project depend on paid IBM services for every local run.
- Do not claim IBM STT is always more secure than local Whisper. The correct trade-off is local privacy versus managed enterprise service.
- Do not claim Granite is automatically better than Gemini. The correct trade-off is raw task performance versus IBM enterprise alignment, governance, and deployment options.
- Do not use Kubernetes only as a buzzword. Show Secrets, PVCs, Services, and debugging commands.

## Interview Narrative

Use this concise narrative:

> I built an AI meeting note-taker as a Client Engineering style PoC. It transcribes meeting audio, chunks the transcript semantically, stores it in a vector database, and uses Agentic RAG to answer questions with reflection and evaluation. The first version used local Whisper for Korean STT quality and privacy, and Gemini for fast LLM iteration. For IBM alignment, I designed optional providers for watsonx.ai Granite, IBM Watson Speech to Text, watsonx Orchestrate, and OpenShift on IBM Cloud. The key technical decision was to keep the architecture provider-agnostic so customer requirements around quality, security, latency, cost, governance, and deployment model can drive the final recommendation.

## Resume Bullets

- Built an Agentic AI meeting note-taker PoC with STT, semantic chunking, hybrid retrieval, LangGraph-based RAG, and RAGAS evaluation.
- Designed an IBM migration path from Gemini to watsonx.ai Granite and from n8n workflow automation to watsonx Orchestrate.
- Planned Docker Compose to OpenShift/Kubernetes migration using Deployments, Services, Secrets, and PVCs for stateful AI services.
- Evaluated local Whisper versus IBM Watson Speech to Text trade-offs across latency, data privacy, accuracy, and managed service adoption.
- Framed the project as a customer-facing AI use case validation asset for enterprise PoC and technical workshop scenarios.
