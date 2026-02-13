"""RAGAS Evaluation Service — FastAPI entry point.

Endpoints:
  POST /evaluate       — evaluate a single QA pair
  POST /evaluate/batch — evaluate a batch of QA pairs
  GET  /health         — service health check
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from eval.evaluate import evaluate_batch, evaluate_single

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("eval")

app = FastAPI(title="RAGAS Evaluation Service", version="0.1.0")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class EvalRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None


class EvalResponse(BaseModel):
    scores: dict


class BatchEvalRequest(BaseModel):
    samples: list[dict] = Field(
        ...,
        description=(
            "List of QA samples. Each must have 'question'. "
            "Optional: 'answer', 'contexts', 'ground_truth'. "
            "Missing answer/contexts will be fetched from RAG service."
        ),
    )


class BatchEvalResponse(BaseModel):
    aggregate: dict
    num_samples: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "eval"}


@app.post("/evaluate", response_model=EvalResponse)
async def eval_single(req: EvalRequest):
    """Evaluate a single QA pair with RAGAS metrics."""
    if not req.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question is required.",
        )

    scores = evaluate_single(
        question=req.question,
        answer=req.answer,
        contexts=req.contexts,
        ground_truth=req.ground_truth,
    )
    return EvalResponse(scores=scores)


@app.post("/evaluate/batch", response_model=BatchEvalResponse)
async def eval_batch(req: BatchEvalRequest):
    """Evaluate a batch of QA pairs with RAGAS metrics."""
    if not req.samples:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one sample is required.",
        )

    result = evaluate_batch(req.samples)
    return BatchEvalResponse(
        aggregate=result["aggregate"],
        num_samples=result["num_samples"],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("eval.app:app", host="0.0.0.0", port=8002, reload=False)
