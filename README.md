# X Insights LangChain Tool

Standalone, reusable Python tool for fetching and processing public X/Twitter-like posts from a public dataset, wrapped in a LangGraph workflow and exposed via FastAPI.

## What this project demonstrates

- Reusable `run()` tool interface for repeated keyword analysis tasks
- LangGraph orchestration (`fetch -> summarize`)
- HTTP access via FastAPI for workflow tools like n8n
- CI/CD with linting, tests, and Docker build on push/PR
- Containerized deployment-ready app

## Architecture

1. `TwitterDataTool.run()` loads and filters a public dataset CSV
2. Processing normalizes text/sentiment and extracts hashtag trends
3. LangGraph agent summarizes trend outputs
4. FastAPI endpoint returns structured analysis

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .[dev]
uvicorn app.api:app --reload
```

API will run at `http://127.0.0.1:8000`.

## API usage

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"keyword":"ai","limit":5,"sentiment_filter":"positive"}'
```

Health check:

```bash
curl "http://127.0.0.1:8000/health"
```

## LangGraph usage in Python

```python
from app.agent import run_agent

result = run_agent(keyword="ai", limit=10, sentiment_filter="positive")
print(result["summary"])
```

## Tests and quality checks

```bash
ruff check .
black --check .
pytest -q
docker build -t x-insights-tool .
```

## CI/CD

GitHub Actions workflow runs:

- `ruff check .`
- `black --check .`
- `pytest -q`
- Docker image build

## Deployment (Render example)

1. Push this repo to GitHub
2. Create a new Render Web Service
3. Build command: `pip install -e .`
4. Start command: `uvicorn app.api:app --host 0.0.0.0 --port $PORT`
5. Add your live URL below

Live demo URL: `TODO_ADD_DEPLOYED_URL`

## Limitations and decisions

- Uses a local public dataset for deterministic and reliable behavior over live scraping
- Sentiment mapping is heuristic and based on common public dataset labels
- Summary is deterministic (no external LLM required), which simplifies CI and reproducibility
- Optional next step: add Playwright scraper for public profile pages with strict rate limiting and terms compliance

