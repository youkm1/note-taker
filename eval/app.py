"""RAGAS 기반 RAG 품질 평가 서비스."""

import logging
import os
from typing import Optional

import requests
from datasets import Dataset
from fastapi import FastAPI, HTTPException, status
from google import genai
from pydantic import BaseModel
from ragas import evaluate
from ragas.embeddings import GoogleEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("eval")

RAG_URL = os.getenv("RAG_URL", "http://rag:8001")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

_gemini_client = genai.Client(api_key=GEMINI_API_KEY)
_evaluator_llm = llm_factory(
    "gemini-2.5-flash", provider="google", client=_gemini_client,
)
_evaluator_embeddings = GoogleEmbeddings(
    client=_gemini_client,
    model="gemini-embedding-001",
)

app = FastAPI(title="RAGAS Evaluation Service", version="0.1.0")


# ---------------------------------------------------------------------------
# 요청/응답 모델
# ---------------------------------------------------------------------------

class EvalSample(BaseModel):
    question: str
    ground_truth: str = ""
    answer: Optional[str] = None
    contexts: Optional[list[str]] = None


class EvalRequest(BaseModel):
    sample: EvalSample


class BatchEvalRequest(BaseModel):
    samples: list[EvalSample]


class EvalScores(BaseModel):
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None


class EvalResponse(BaseModel):
    scores: EvalScores
    detail: dict = {}


class BatchEvalResponse(BaseModel):
    average_scores: EvalScores
    per_sample: list[EvalScores]


# ---------------------------------------------------------------------------
# RAG 서비스 연동으로 answer/contexts 자동 조회
# ---------------------------------------------------------------------------

def _fill_from_rag(sample: EvalSample) -> EvalSample:
    """answer 또는 contexts가 없으면 RAG 서비스에서 조회."""
    if sample.answer and sample.contexts:
        return sample
    try:
        resp = requests.post(
            f"{RAG_URL}/query",
            json={"question": sample.question},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        if not sample.answer:
            sample.answer = data.get("answer", "")
        if not sample.contexts:
            sample.contexts = [c.get("text", "") for c in data.get("contexts", [])]
    except requests.RequestException as e:
        logger.warning("Failed to query RAG service: %s", e)
    return sample


# ---------------------------------------------------------------------------
# 평가 실행
# ---------------------------------------------------------------------------

def _run_evaluation(samples: list[EvalSample]) -> Dataset:
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    for s in samples:
        s = _fill_from_rag(s)
        data["question"].append(s.question)
        data["answer"].append(s.answer or "")
        data["contexts"].append(s.contexts or [])
        data["ground_truth"].append(s.ground_truth)

    dataset = Dataset.from_dict(data)

    metrics = [
        Faithfulness(llm=_evaluator_llm),
        AnswerRelevancy(llm=_evaluator_llm, embeddings=_evaluator_embeddings),
        ContextPrecision(llm=_evaluator_llm),
        ContextRecall(llm=_evaluator_llm),
    ]
    result = evaluate(dataset, metrics=metrics)
    return result


def _scores_from_row(row: dict) -> EvalScores:
    return EvalScores(
        faithfulness=row.get("faithfulness"),
        answer_relevancy=row.get("answer_relevancy"),
        context_precision=row.get("context_precision"),
        context_recall=row.get("context_recall"),
    )


# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvalResponse)
async def evaluate_single(req: EvalRequest):
    """단일 QA 쌍에 대한 RAGAS 평가."""
    try:
        result = _run_evaluation([req.sample])
        df = result.to_pandas()
        row = df.iloc[0].to_dict()
        return EvalResponse(scores=_scores_from_row(row), detail=row)
    except Exception as e:
        logger.exception("Evaluation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {e}",
        ) from e


@app.post("/evaluate/batch", response_model=BatchEvalResponse)
async def evaluate_batch(req: BatchEvalRequest):
    """배치 QA 쌍에 대한 RAGAS 평가."""
    if not req.samples:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No samples provided.")
    try:
        result = _run_evaluation(req.samples)
        df = result.to_pandas()

        per_sample = [_scores_from_row(df.iloc[i].to_dict()) for i in range(len(df))]
        avg = EvalScores(
            faithfulness=df["faithfulness"].mean() if "faithfulness" in df else None,
            answer_relevancy=df["answer_relevancy"].mean() if "answer_relevancy" in df else None,
            context_precision=df["context_precision"].mean() if "context_precision" in df else None,
            context_recall=df["context_recall"].mean() if "context_recall" in df else None,
        )
        return BatchEvalResponse(average_scores=avg, per_sample=per_sample)
    except Exception as e:
        logger.exception("Batch evaluation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch evaluation failed: {e}",
        ) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("eval.app:app", host="0.0.0.0", port=8002, reload=False)
