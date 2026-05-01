from app.agent import run_agent


def test_agent_returns_summary_and_tool_output():
    result = run_agent(keyword="ai", limit=3)
    assert "summary" in result
    assert "tool_output" in result
    assert result["tool_output"]["returned_count"] <= 3
