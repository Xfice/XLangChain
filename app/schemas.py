"""Pydantic request/response models for API endpoints."""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    keyword: str = Field(..., min_length=1, description="Keyword or topic to search in posts")
    limit: int = Field(default=50, ge=1, le=500)
    sentiment_filter: str | None = Field(default=None, description="positive/negative/neutral")
    since_date: str | None = Field(default=None, description="ISO date string, e.g. 2024-01-01")
    source: Literal["dataset", "playwright"] = Field(
        default="dataset",
        description="Data source mode. Use 'playwright' only for optional live demo scraping.",
    )


class AnalyzeResponse(BaseModel):
    summary: str
    tool_output: dict[str, Any]
