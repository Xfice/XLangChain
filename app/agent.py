"""LangGraph workflow that calls the reusable X insights tool."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.tool import TwitterDataTool


class AgentState(TypedDict, total=False):
    keyword: str
    limit: int
    sentiment_filter: str | None
    since_date: str | None
    tool_output: dict[str, Any]
    summary: str


def _fetch(state: AgentState) -> AgentState:
    tool = TwitterDataTool()
    output = tool.run(
        keyword=state["keyword"],
        limit=state.get("limit", 50),
        sentiment_filter=state.get("sentiment_filter"),
        since_date=state.get("since_date"),
    )
    return {"tool_output": output}


def _summarize(state: AgentState) -> AgentState:
    output = state["tool_output"]
    sentiment_distribution = output.get("sentiment_distribution", {})
    top_hashtags = output.get("top_hashtags", [])
    dominant_sentiment = (
        max(sentiment_distribution, key=sentiment_distribution.get)
        if sentiment_distribution
        else "unknown"
    )
    hashtags = (
        ", ".join([f"#{tag} ({count})" for tag, count in top_hashtags[:3]])
        if top_hashtags
        else "none"
    )
    summary = (
        f"Found {output['returned_count']} posts out of {output['total_matches']} matches for "
        f"'{output['keyword']}'. Dominant sentiment: {dominant_sentiment}. "
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
) -> dict[str, Any]:
    app = build_agent()
    result = app.invoke(
        {
            "keyword": keyword,
            "limit": limit,
            "sentiment_filter": sentiment_filter,
            "since_date": since_date,
        }
    )
    return {
        "summary": result["summary"],
        "tool_output": result["tool_output"],
    }
