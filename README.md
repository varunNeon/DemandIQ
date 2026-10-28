# DemandIQ

DemandIQ is a demand forecasting and inventory planning platform with a Next.js frontend, a FastAPI backend, and PostgreSQL for hosted deployments. Forecasting and inventory business logic lives in Python and is independent of the web interface.

## Current architecture

- **Frontend:** Next.js under `web/`.
- **API:** FastAPI under `src/demandiq/api/`, served by Uvicorn.
- **Forecasting and inventory logic:** Python modules under `src/demandiq/`.
- **Database:** PostgreSQL for deployment, configured with `DATABASE_URL`. SQLite remains available for local development and tests.

The Next.js frontend currently provides an overview, demand history, saved forecast runs, and saved inventory recommendations through the `/api/v1` read endpoints. Forecast generation and recommendation creation are available through the Python pipeline; interactive controls for those actions still need to be added to the Next.js/API stack.

## Run locally

Install Python dependencies from the project root:

```powershell
python -m pip install -e .
```

For local development, SQLite is used by default. To use PostgreSQL locally or in deployment, set `DATABASE_URL` to your PostgreSQL connection string before starting the API.

Load the demo data into the configured database (omit `--generate-demo` if the CSV demo files are already present):

```powershell
python scripts/run_pipeline.py --generate-demo --ingest
```

Start the API in one terminal:

```powershell
python -m uvicorn demandiq.api.main:app --reload
```

Start the Next.js frontend in a second terminal:

```powershell
cd web
npm install
npm run dev
```

Open <http://localhost:3000>. The API docs are at <http://localhost:8000/docs>.

To use a different API URL in the frontend, set `NEXT_PUBLIC_DEMANDIQ_API_URL` in `web/.env.local`. Set `DEMANDIQ_CORS_ORIGINS` to the deployed frontend origin(s), comma-separated, for cross-origin API access.

## Checks

```powershell
python -m pytest
cd web
npm run lint
npm run build
```
