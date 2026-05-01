"""Pydantic request/response models for API endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    keyword: str = Field(..., min_length=1, description="Keyword or topic to search in posts")
    limit: int = Field(default=50, ge=1, le=500)
    sentiment_filter: str | None = Field(default=None, description="positive/negative/neutral")
    since_date: str | None = Field(default=None, description="ISO date string, e.g. 2024-01-01")


class AnalyzeResponse(BaseModel):
    summary: str
    tool_output: dict[str, Any]
