"""LangGraph workflow that calls the reusable X insights tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.tool import TwitterDataTool


class AgentState(TypedDict, total=False):
    keyword: str
    limit: int
    sentiment_filter: str | None
    since_date: str | None
    source: str
    dataset_path: str
    tool_output: dict[str, Any]
    summary: str


def _fetch(state: AgentState) -> AgentState:
    dataset_path = state.get("dataset_path")
    tool = TwitterDataTool(dataset_path=dataset_path) if dataset_path else TwitterDataTool()
    output = tool.run(
        keyword=state["keyword"],
        limit=state.get("limit", 50),
        sentiment_filter=state.get("sentiment_filter"),
        since_date=state.get("since_date"),
        source=state.get("source", "dataset"),
    )
    return {"tool_output": output}


def _summarize(state: AgentState) -> AgentState:
    output = state["tool_output"]
    sentiment_distribution = output.get("sentiment_distribution", {})
    top_hashtags = output.get("top_hashtags", [])
    if sentiment_distribution:
        top_count = max(sentiment_distribution.values())
        leaders = sorted(
            [label for label, count in sentiment_distribution.items() if count == top_count]
        )
        dominant_sentiment = ", ".join(leaders)
    else:
        dominant_sentiment = "unknown"
    hashtags = (
        ", ".join([f"#{tag} ({count})" for tag, count in top_hashtags[:3]])
        if top_hashtags
        else "none"
    )
    summary = (
        f"Found {output['returned_count']} posts out of {output['total_matches']} matches for "
        f"'{output['keyword']}' using mode={output.get('mode', 'dataset')}. "
        f"Dominant sentiment: {dominant_sentiment}. "
        f"Top hashtags: {hashtags}."
    )
    return {"summary": summary}


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("fetch", _fetch)
    graph.add_node("summarize", _summarize)
    graph.add_edge(START, "fetch")
    graph.add_edge("fetch", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


def run_agent(
    keyword: str,
    limit: int = 50,
    sentiment_filter: str | None = None,
    since_date: str | None = None,
    source: str = "dataset",
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    app = build_agent()
    result = app.invoke(
        {
            "keyword": keyword,
            "limit": limit,
            "sentiment_filter": sentiment_filter,
            "since_date": since_date,
            "source": source,
            "dataset_path": str(dataset_path) if dataset_path else None,
        }
    )
    return {
        "summary": result["summary"],
        "tool_output": result["tool_output"],
    }
