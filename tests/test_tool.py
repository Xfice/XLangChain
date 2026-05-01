from app.tool import TwitterDataTool


def _write_fixture_csv(path):
    path.write_text(
        "date,sentiment,text\n"
        "2024-01-10,4,AI is changing developer workflows fast #AI\n"
        "2024-01-11,0,I am worried about misuse of AI content generation #AI\n"
        "2024-02-01,4,Python and LangChain are great for quick agent prototypes #Python #AI\n",
        encoding="utf-8",
    )


def test_tool_filters_keyword_and_limit(tmp_path):
    fixture = tmp_path / "sample.csv"
    _write_fixture_csv(fixture)
    tool = TwitterDataTool(dataset_path=fixture)
    result = tool.run(keyword="ai", limit=2)
    assert result["returned_count"] == 2
    assert result["total_matches"] >= result["returned_count"]
    assert all("ai" in post["text"].lower() for post in result["posts"])


def test_tool_sentiment_filter(tmp_path):
    fixture = tmp_path / "sample.csv"
    _write_fixture_csv(fixture)
    tool = TwitterDataTool(dataset_path=fixture)
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
    monkeypatch.setenv("KAGGLE_USERNAME", "test-user")
    monkeypatch.setenv("KAGGLE_KEY", "test-key")

    def _fake_fetch_kaggle_dataset_to_csv(
        *,
        dataset,
        output_csv,
        selected_file,
        max_rows,
        keyword_filter,
    ):
        output_csv.write_text(
            "date,sentiment,text\n2024-02-02,4,AI refreshed row #AI\n",
            encoding="utf-8",
        )
        assert keyword_filter == "ai"
        return output_csv

    monkeypatch.setattr("app.tool.fetch_kaggle_dataset_to_csv", _fake_fetch_kaggle_dataset_to_csv)

    tool = TwitterDataTool(dataset_path=dataset)
    result = tool.run(keyword="ai", limit=5, source="kaggle")
    assert result["returned_count"] == 1
    assert "refreshed" in result["posts"][0]["text"].lower()
    assert result["last_refetched_at"] is not None


def test_tool_kaggle_source_errors_when_credentials_missing(tmp_path, monkeypatch):
    dataset = tmp_path / "sample.csv"
    dataset.write_text("date,sentiment,text\n2024-01-01,4,old row\n", encoding="utf-8")
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    tool = TwitterDataTool(dataset_path=dataset)
    try:
        tool.run(keyword="ai", limit=5, source="kaggle")
        assert False, "expected kaggle source to fail without credentials"
    except ValueError as exc:
        assert "credentials are missing" in str(exc).lower()


def test_tool_dataset_source_uses_existing_dataset_without_refetch(tmp_path, monkeypatch):
    dataset = tmp_path / "sample.csv"
    dataset.write_text("date,sentiment,text\n2024-01-01,4,old row\n", encoding="utf-8")
    monkeypatch.setenv("KAGGLE_USERNAME", "test-user")
    monkeypatch.setenv("KAGGLE_KEY", "test-key")

    def _raise_if_called(**kwargs):
        raise AssertionError("dataset mode should not trigger kaggle refetch")

    monkeypatch.setattr("app.tool.fetch_kaggle_dataset_to_csv", _raise_if_called)

    tool = TwitterDataTool(dataset_path=dataset)
    result = tool.run(keyword="ai", limit=5, source="dataset")
    assert result["returned_count"] == 0


def test_tool_supports_sentiment140_raw_format(tmp_path):
    raw = tmp_path / "training.1600000.processed.noemoticon.csv"
    raw.write_text(
        '"4","111","Mon Apr 06 22:19:45 PDT 2009","NO_QUERY","user1","I love ai today #AI"\n'
        '"0","112","Mon Apr 06 22:20:00 PDT 2009","NO_QUERY","user2","I hate delays"\n',
        encoding="utf-8",
    )

    tool = TwitterDataTool(dataset_path=raw)
    result = tool.run(keyword="ai", limit=5, source="dataset")
    assert result["returned_count"] == 1
    assert result["posts"][0]["sentiment"] == "positive"
    assert result["posts"][0]["date"] is not None


def test_keyword_filter_uses_word_boundary(tmp_path):
    sample = tmp_path / "sample.csv"
    sample.write_text(
        "date,sentiment,text\n"
        "2024-01-01,4,I love ai agents\n"
        "2024-01-02,0,This is a paid feature\n",
        encoding="utf-8",
    )

    tool = TwitterDataTool(dataset_path=sample)
    result = tool.run(keyword="ai", limit=10, source="dataset")
    assert result["returned_count"] == 1
    assert "love ai agents" in result["posts"][0]["text"].lower()


def test_kaggle_filter_is_keyword_aware_for_any_term(tmp_path, monkeypatch):
    sample = tmp_path / "sample.csv"
    sample.write_text(
        "date,sentiment,text\n"
        "2024-01-01,4,This ML model improved support workflows\n"
        "2024-01-02,0,we edited html templates today\n",
        encoding="utf-8",
    )
    tool = TwitterDataTool(dataset_path=sample)

    # Keep existing file and avoid network by stubbing refresh behavior.
    def _fake_fetch_kaggle_dataset_to_csv(
        *, dataset, output_csv, selected_file, max_rows, keyword_filter
    ):
        assert keyword_filter == "ml"
        return output_csv

    from unittest.mock import patch

    monkeypatch.setenv("KAGGLE_USERNAME", "test-user")
    monkeypatch.setenv("KAGGLE_KEY", "test-key")
    with patch("app.tool.fetch_kaggle_dataset_to_csv", _fake_fetch_kaggle_dataset_to_csv):
        result = tool.run(keyword="ml", limit=10, source="kaggle")

    assert result["returned_count"] == 1
    assert "ml model improved" in result["posts"][0]["text"].lower()


def test_resolve_dataset_path_prefers_new_csv_when_config_missing(tmp_path):
    sample = tmp_path / "sample.csv"
    sample.write_text("date,sentiment,text\n2024-01-01,4,ai sample row\n", encoding="utf-8")
    newer = tmp_path / "training.1600000.processed.noemoticon.csv"
    newer.write_text("date,sentiment,text\n2024-01-02,4,ai new row\n", encoding="utf-8")

    tool = TwitterDataTool(dataset_path=tmp_path / "missing.csv")
    tool._resolve_dataset_path = lambda: newer  # keep test isolated from workspace data folder
    result = tool.run(keyword="ai", limit=10, source="dataset")
    assert result["returned_count"] == 1
    assert "new row" in result["posts"][0]["text"].lower()
