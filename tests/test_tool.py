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


def test_tool_kaggle_source_forces_refresh(tmp_path, monkeypatch):
    dataset = tmp_path / "sample.csv"
    dataset.write_text("date,sentiment,text\n2024-01-01,4,old row\n", encoding="utf-8")

    def _fake_fetch_kaggle_dataset_to_csv(*, dataset, output_csv, selected_file, max_rows):
        output_csv.write_text(
            "date,sentiment,text\n2024-02-02,4,AI refreshed row #AI\n",
            encoding="utf-8",
        )
        return output_csv

    monkeypatch.setattr("app.tool.fetch_kaggle_dataset_to_csv", _fake_fetch_kaggle_dataset_to_csv)

    tool = TwitterDataTool(dataset_path=dataset)
    result = tool.run(keyword="ai", limit=5, source="kaggle")
    assert result["returned_count"] == 1
    assert "refreshed" in result["posts"][0]["text"].lower()
