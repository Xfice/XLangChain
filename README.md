# X Insights LangChain Tool

Standalone Python service for fetching and processing public X/Twitter-like data, orchestrated by LangGraph, and exposed through FastAPI for direct use or workflow automation (for example n8n webhooks).

## What this delivers for the client

- Reusable analysis tool with a stable `run()` interface
- 3 flexible data sources for different client scenarios
- HTTP-first integration (`/analyze`, `/refetch-kaggle`, `/analyze-file`)
- CI/CD guardrails (lint, tests, Docker build)
- Deployable on free-tier services (Render)

## 3 Source Modes (Flexibility)

- `dataset`: analyze current local dataset (`data/sample.csv`) without refetching
- `kaggle`: refetch from Kaggle using the current request keyword, then analyze
- `playwright`: optional live public-page scrape mode for demo/experiments

This makes the system flexible for:
- stable repeated reporting (`dataset`)
- keyword-driven refreshes (`kaggle`)
- quick live-page proof-of-concept (`playwright`)

## API Flow

1. Client calls one of the endpoints
2. Source-specific fetch/load happens
3. Text cleanup + sentiment mapping + hashtag extraction run
4. LangGraph summarizes results
5. API returns `summary` and structured `tool_output`

## Endpoints

- `GET /health`: health check
- `POST /analyze`: analyze using selected `source`
- `POST /refetch-kaggle`: explicitly refresh Kaggle dataset using payload keyword, then analyze
- `POST /analyze-file`: analyze an uploaded CSV file (`multipart/form-data`)

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .[dev,data,scrape]
uvicorn app.api:app --reload
```

API: `http://127.0.0.1:8000`

## Usage Examples

### 1) Analyze existing dataset (no refetch)

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"keyword":"ai","limit":10,"source":"dataset"}'
```

### 2) Refetch from Kaggle using request keyword

```bash
curl -X POST "http://127.0.0.1:8000/refetch-kaggle" \
  -H "Content-Type: application/json" \
  -d '{"keyword":"ai","limit":10}'
```

Response includes `tool_output.last_refetched_at`.

### 3) Analyze uploaded client CSV

```bash
curl -X POST "http://127.0.0.1:8000/analyze-file" \
  -F "file=@data/sample.csv" \
  -F "keyword=ai" \
  -F "limit=10"
```

### 4) Playwright demo mode

```bash
export PLAYWRIGHT_DEMO_ENABLED=true  # PowerShell: $env:PLAYWRIGHT_DEMO_ENABLED="true"
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"keyword":"ai","limit":5,"source":"playwright"}'
```

## Kaggle Setup

Set credentials:
- `KAGGLE_USERNAME`
- `KAGGLE_KEY`

Optional:
- `KAGGLE_DATASET` (default `kazanova/sentiment140`)
- `KAGGLE_FILE` (specific CSV file name)
- `KAGGLE_MAX_ROWS` (default `1000`)
- `KAGGLE_MAX_ROWS_RUNTIME` (default `1000`, capped for memory safety)
- `KAGGLE_KEYWORD_FILTER` (optional build-time seed keyword)

Manual script:

```bash
python scripts/fetch_kaggle_data.py --dataset kazanova/sentiment140 --target-name sample.csv --max-rows 1000 --keyword-filter "<keyword>"
```

## Render Deployment

`render.yaml` is configured to:
- install dependencies
- seed data from Kaggle with bounded row count
- run FastAPI via uvicorn

Client should set secrets in Render:
- `KAGGLE_USERNAME`
- `KAGGLE_KEY`

## What I got stuck on (and workaround)

During implementation, large Kaggle CSV refreshes caused memory pressure/crashes on free-tier runtime.  
Workaround applied:
- strict row limits (`1000` default)
- explicit refetch endpoint (`/refetch-kaggle`) instead of always refetching in `/analyze`
- keyword-prioritized filtering to keep fetched data relevant

This keeps demo stability while still proving Kaggle integration.

## Production Recommendation (Kaggle + Playwright at scale)

For higher volume, a database-backed architecture is better than repeatedly processing CSV files:
- ingest job writes normalized rows into DB
- API queries DB indexes by keyword/date/sentiment
- scheduled refresh replaces request-time heavy fetches

Current implementation intentionally limits to `1000` rows for free-tier demonstration and reliability.

## CI/CD

GitHub Actions runs:
- `ruff check .`
- `black --check .`
- `pytest -q`
- Docker build

