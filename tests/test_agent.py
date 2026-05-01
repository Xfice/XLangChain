from app.agent import run_agent


def test_agent_returns_summary_and_tool_output():
    result = run_agent(keyword="ai", limit=3)
    assert "summary" in result
    assert "tool_output" in result
    assert result["tool_output"]["returned_count"] <= 3


def test_agent_summary_lists_tied_sentiments(tmp_path):
    dataset = tmp_path / "sample.csv"
    dataset.write_text(
        "date,sentiment,text\n"
        "2024-01-01,4,ai agent improved workflow\n"
        "2024-01-02,0,ai rollout caused delay\n",
        encoding="utf-8",
    )
    result = run_agent(keyword="ai", limit=5, dataset_path=dataset, source="dataset")
    assert "Dominant sentiment: negative, positive." in result["summary"]
