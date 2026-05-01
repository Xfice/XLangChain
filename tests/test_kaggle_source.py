from app.sources.kaggle_source import _normalize_to_app_schema


def test_normalize_outputs_balanced_sentiment_sample(tmp_path):
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "date,sentiment,text\n"
        "2024-01-01,0,negative one\n"
        "2024-01-02,0,negative two\n"
        "2024-01-03,0,negative three\n"
        "2024-01-04,4,positive one\n"
        "2024-01-05,4,positive two\n",
        encoding="utf-8",
    )

    output = tmp_path / "sample.csv"
    _normalize_to_app_schema(raw, output, max_rows=4)

    rows = output.read_text(encoding="utf-8").splitlines()
    sentiments = [line.split(",")[1] for line in rows[1:]]
    assert sentiments.count("0") == 2
    assert sentiments.count("4") == 2


def test_normalize_keyword_filter_keeps_only_keyword_rows(tmp_path):
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "date,sentiment,text\n"
        "2024-01-01,0,random chatter here\n"
        "2024-01-02,4,AI agents help automate work #AI\n"
        "2024-01-03,4,another random entry\n",
        encoding="utf-8",
    )

    output = tmp_path / "sample.csv"
    _normalize_to_app_schema(raw, output, max_rows=10, keyword_filter="ai")

    rows = output.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2  # header + 1 relevant row
    assert "AI agents help automate work #AI" in rows[1]
