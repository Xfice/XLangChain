"""FastAPI app exposing the LangGraph analysis workflow."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.agent import run_agent
from app.schemas import AnalyzeRequest, AnalyzeResponse

app = FastAPI(title="X Insights Tool API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    try:
        result = run_agent(
            keyword=payload.keyword,
            limit=payload.limit,
            sentiment_filter=payload.sentiment_filter,
            since_date=payload.since_date,
            source=payload.source,
        )
        return AnalyzeResponse(**result)
    except Exception as exc:  # pragma: no cover - covered by tests through expected failures
        raise HTTPException(status_code=400, detail=str(exc)) from exc
