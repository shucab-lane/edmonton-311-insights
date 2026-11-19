# Edmonton 311 Insights

[![Data pipeline](https://github.com/shucab-lane/edmonton-311-insights/actions/workflows/pipeline.yml/badge.svg)](https://github.com/shucab-lane/edmonton-311-insights/actions/workflows/pipeline.yml)

An automated **Python data pipeline** over the City of Edmonton's open 311 service-request data (3.5M+ requests since 2023). It extracts from the Socrata API, cleans with **pandas**, loads into a **SQLite** warehouse, analyzes with **SQL**, forecasts daily request volume with **scikit-learn**, and publishes an interactive **Plotly** dashboard, refreshed every week by **GitHub Actions**.

**Live dashboard:** https://shucab-lane.github.io/edmonton-311-insights/

## Results

- **3,553,728** requests analyzed (Jan 2023 to Nov 2026), averaging about **2,500 per day**
- The gradient-boosting forecast has **24% lower error** than a "same weekday last week" baseline on 8 weeks of held-out data (MAE 298 vs 394 requests/day)
- About **69%** of requests still arrive by phone. The channel-share chart shows how that is changing over time
- Median time-to-close per service area, neighbourhood hotspots on a map, and the most common request types

## Pipeline

```
Socrata API ──► extract.py ──► transform.py ──► load.py ──► analytics.py ──► forecast.py ──► dashboard.py
 (SoQL)         paging,         pandas:          SQLite:      SQL: CTEs,       scikit-learn     Plotly HTML
                retries,        clean text,      schema,      window           gradient         → GitHub Pages
                server-side     parse dates,     PKs, CHECKs, functions        boosting +
                GROUP BY        dedupe, derive   indexes                       backtest
```

| Stage | What it does |
| --- | --- |
| **Extract** | Pushes aggregation to the API: full history comes back as daily and monthly `GROUP BY` results (about 66K rows instead of 3.5M). Only the last 60 days are pulled at request level (about 125K rows) for resolution-time analysis. Paged with `$limit`/`$offset` and a stable sort; retries with exponential backoff on 429 and 5xx, and fails fast on other 4xx errors. |
| **Transform** | Normalizes inconsistent category text (`"Snow  and  Ice"` becomes `"Snow and Ice"`) and re-aggregates afterwards, parses timestamps, removes duplicate IDs, computes time-to-close and discards negative durations. Data-quality metrics are published on the dashboard. |
| **Load** | SQLite warehouse with primary keys, `CHECK` constraints and indexes ([`sql/schema.sql`](sql/schema.sql)). Each table is replaced in a single transaction, and every load is logged in `pipeline_runs`. |
| **Analyze** | SQL queries in [`analytics.py`](pipeline/analytics.py) using CTEs and window functions (`RANK() OVER`, `SUM() OVER (PARTITION BY month)`). Medians and percentiles are computed in pandas, since SQLite has no `MEDIAN()`. |
| **Forecast** | Gradient-boosting regressor on calendar features and lagged values (7–28 days), plus rolling means. Every feature is at least 7 days old, so nothing leaks from the future. Evaluated with a time-ordered split against a seasonal-naive baseline, then refit on all data to forecast the next 7 days. |
| **Publish** | A static HTML dashboard with 6 interactive charts, KPI cards and tables. GitHub Actions runs lint (ruff), tests (pytest), then the full pipeline every Monday and deploys to GitHub Pages. |

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pipeline                  # full run (about 2-3 minutes); writes data/warehouse.db and site/index.html
python -m pipeline --skip-extract   # rebuild analytics + dashboard from the existing warehouse
pytest                              # 16 tests: transforms, SQL queries, HTTP retry logic (mocked), forecasting
```

Optional: set `SOCRATA_APP_TOKEN` for higher API rate limits, and `RECENT_DAYS` to change the request-level window.

## Project structure

```
pipeline/
  config.py      settings (env-var overrides)
  extract.py     Socrata client: SoQL queries, pagination, retries
  transform.py   pandas cleaning + data-quality report
  load.py        SQLite warehouse loading
  analytics.py   SQL queries → DataFrames
  forecast.py    feature engineering, model, backtest
  dashboard.py   Plotly charts → static HTML
  __main__.py    CLI entry point
sql/schema.sql
tests/test_pipeline.py
.github/workflows/pipeline.yml
```

Data: [City of Edmonton Open Data, 311 Requests](https://data.edmonton.ca/City-Administration/311-Requests/q7ua-agfg) (Open Government Licence – City of Edmonton).
