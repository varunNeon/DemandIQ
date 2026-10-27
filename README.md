# DemandIQ

DemandIQ is a demand forecasting and inventory planning workspace. Its existing Streamlit interface and forecasting engine remain available while a new Next.js interface is added incrementally.

## Current architecture

- **Forecasting and data:** Python modules under `src/demandiq/`.
- **Database:** SQLite by default (`demandiq.db` in the project root). SQLAlchemy also supports PostgreSQL through `DATABASE_URL`.
- **Current interface:** Streamlit, launched with `python -m streamlit run src/demandiq/app/main.py`.
- **New interface:** Next.js under `web/`, using the read-only FastAPI endpoints under `/api/v1`.

The frontend currently provides an overview, demand history with the latest saved forecast, recent model runs and basic workspace health. Forecast generation and inventory controls remain in the Streamlit app while those workflows are migrated.

## Run locally

Install Python dependencies from the project root:

```powershell
python -m pip install -e .
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

To run the existing Streamlit app instead:

```powershell
python -m streamlit run src/demandiq/app/main.py
```

To use a different API URL in the frontend, set `NEXT_PUBLIC_DEMANDIQ_API_URL` in `web/.env.local`. Allowed browser origins can be set using the comma-separated `DEMANDIQ_CORS_ORIGINS` environment variable.

## Checks

```powershell
python -m pytest
cd web
npm run lint
npm run build
```
