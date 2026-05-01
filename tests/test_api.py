from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze():
    response = client.post(
        "/analyze",
        json={"keyword": "ai", "limit": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert "summary" in body
    assert "tool_output" in body


def test_analyze_playwright_disabled():
    response = client.post(
        "/analyze",
        json={"keyword": "ai", "limit": 2, "source": "playwright"},
    )
    assert response.status_code == 400
    assert "Playwright source is disabled" in response.json()["detail"]


def test_analyze_file_csv_upload():
    csv_content = (
        "date,sentiment,text\n"
        "2024-01-01,4,AI helps automate support #AI\n"
        "2024-01-02,0,unrelated topic here\n"
    )
    response = client.post(
        "/analyze-file",
        data={"keyword": "ai", "limit": "5"},
        files={"file": ("client_data.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["tool_output"]["returned_count"] == 1
    assert body["tool_output"]["posts"][0]["sentiment"] == "positive"


def test_analyze_file_rejects_non_csv():
    response = client.post(
        "/analyze-file",
        data={"keyword": "ai", "limit": "5"},
        files={"file": ("client_data.txt", "hello", "text/plain")},
    )
    assert response.status_code == 400
    assert "must be a .csv" in response.json()["detail"]
