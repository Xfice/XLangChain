# X Insights LangChain Tool

Standalone, reusable Python tool for fetching and processing public X/Twitter-like posts from a public dataset, wrapped in a LangGraph workflow and exposed via FastAPI.

## What this project demonstrates

- Reusable `run()` tool interface for repeated keyword analysis tasks
- LangGraph orchestration (`fetch -> summarize`)
- HTTP access via FastAPI for workflow tools like n8n
- Optional Playwright demo mode for public live-page scraping
- CI/CD with linting, tests, and Docker build on push/PR
- Containerized deployment-ready app

## Architecture

1. `TwitterDataTool.run()` loads and filters either:
   - `source=dataset` (default, reproducible)
   - `source=playwright` (optional, public-page live demo)
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

## Automated Kaggle dataset refresh

If you have a Kaggle token, you can auto-download CSV data instead of manually replacing files.

1. Configure credentials:
   - Option A: set env vars `KAGGLE_USERNAME` and `KAGGLE_KEY`
   - Option B: place `kaggle.json` at `%USERPROFILE%\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json`
2. Install the optional data dependency:

```bash
pip install -e .[data]
```

3. Download and normalize a dataset CSV into `data/sample.csv`:

```bash
python scripts/fetch_kaggle_data.py --dataset kazanova/sentiment140 --max-rows 100000
```

You can select a specific file when a dataset has multiple CSVs:

```bash
python scripts/fetch_kaggle_data.py --dataset <owner/dataset> --file <name.csv> --target-name sample.csv --max-rows 100000
```

The script normalizes CSV output to the app schema (`date,sentiment,text`) and supports
headerless Sentiment140 format.

## API usage

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"keyword":"ai","limit":5,"sentiment_filter":"positive","source":"dataset"}'
```

Source modes:
- `dataset`: tries Kaggle refresh first, then falls back to local CSV
- `kaggle`: force Kaggle refresh, then analyze
- `playwright`: optional public-page scraping

When `source="dataset"` and `data/sample.csv` is missing, `/analyze` tries to
auto-fetch from Kaggle at request time using:
- `KAGGLE_USERNAME`
- `KAGGLE_KEY`
- Optional: `KAGGLE_DATASET`, `KAGGLE_FILE`, `KAGGLE_MAX_ROWS`

To force Kaggle fetch on every call, use:

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"keyword":"ai","limit":5,"source":"kaggle"}'
```

Optional Playwright demo mode:

```bash
pip install -e .[scrape]
python -m playwright install chromium
export PLAYWRIGHT_DEMO_ENABLED=true  # PowerShell: $env:PLAYWRIGHT_DEMO_ENABLED="true"
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"keyword":"ai","limit":5,"source":"playwright"}'
```

Health check:

```bash
curl "http://127.0.0.1:8000/health"
```

## LangGraph usage in Python

```python
from app.agent import run_agent

result = run_agent(keyword="ai", limit=10, sentiment_filter="positive", source="dataset")
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
3. Set Render env vars:
   - `KAGGLE_USERNAME` (secret)
   - `KAGGLE_KEY` (secret)
   - Optional: `KAGGLE_DATASET` (default `kazanova/sentiment140`)
   - Optional: `KAGGLE_MAX_ROWS` (default `100000`)
4. Use `render.yaml` from this repo (recommended), or set build/start manually:
   - Build: `pip install -e .[data] && python scripts/fetch_kaggle_data.py --dataset ${KAGGLE_DATASET:-kazanova/sentiment140} --max-rows ${KAGGLE_MAX_ROWS:-100000} && pip install -e .`
   - Start: `uvicorn app.api:app --host 0.0.0.0 --port $PORT`
5. Add your live URL below

Live demo URL: https://xlangchain.onrender.com

## Limitations and decisions

- Uses a local public dataset for deterministic and reliable behavior over live scraping
- Kaggle data refresh can be automated with `scripts/fetch_kaggle_data.py`; missing Kaggle creds will break deploy-time fetch
- Playwright mode is intentionally minimal and opt-in (`PLAYWRIGHT_DEMO_ENABLED=true`) for demo use
- Sentiment mapping is heuristic and based on common public dataset labels
- Summary is deterministic (no external LLM required), which simplifies CI and reproducibility
- Playwright scraping can break with UI changes and should be treated as non-critical/fallback

