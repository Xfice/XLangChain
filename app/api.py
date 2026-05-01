"""FastAPI app exposing the LangGraph analysis workflow."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

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


@app.post("/analyze-file", response_model=AnalyzeResponse)
async def analyze_file(
    file: UploadFile = File(...),
    keyword: str = Form(...),
    limit: int = Form(50),
    sentiment_filter: str | None = Form(None),
    since_date: str | None = Form(None),
) -> AnalyzeResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .csv")

    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(suffix=".csv", delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(await file.read())

        result = run_agent(
            keyword=keyword,
            limit=limit,
            sentiment_filter=sentiment_filter,
            since_date=since_date,
            source="dataset",
            dataset_path=temp_path,
        )
        return AnalyzeResponse(**result)
    except Exception as exc:  # pragma: no cover - covered by tests through expected failures
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
