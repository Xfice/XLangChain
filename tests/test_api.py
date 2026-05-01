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
