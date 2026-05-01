from app.tool import TwitterDataTool


def test_tool_filters_keyword_and_limit():
    tool = TwitterDataTool(dataset_path="data/sample.csv")
    result = tool.run(keyword="ai", limit=2)
    assert result["returned_count"] == 2
    assert result["total_matches"] >= result["returned_count"]
    assert all("ai" in post["text"].lower() for post in result["posts"])


def test_tool_sentiment_filter():
    tool = TwitterDataTool(dataset_path="data/sample.csv")
    result = tool.run(keyword="ai", sentiment_filter="positive", limit=10)
    assert result["returned_count"] > 0
    assert all(post["sentiment"] == "positive" for post in result["posts"])


def test_tool_creates_csv_when_missing(tmp_path):
    missing = tmp_path / "missing.csv"
    tool = TwitterDataTool(dataset_path=missing)
    result = tool.run(keyword="ai", limit=2)
    assert missing.exists()
    assert result["returned_count"] >= 1
