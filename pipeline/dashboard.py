"""Build a static HTML dashboard from the warehouse (Plotly charts + KPI cards)."""

from __future__ import annotations

import html
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .forecast import ForecastResult

PALETTE = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2", "#db2777", "#65a30d"]
LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=40, r=20, t=50, b=40),
    font=dict(family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif", size=13),
    title=dict(x=0.01, font=dict(size=16)),
    legend=dict(orientation="h", yanchor="top", y=-0.12, x=0),
    colorway=PALETTE,
    height=380,
)


def _fig_html(fig: go.Figure, first: bool = False) -> str:
    title = fig.layout.title.text
    height = fig.layout.height
    margin = fig.layout.margin
    fig.update_layout(**LAYOUT)
    fig.update_layout(title_text=title)
    if height:
        fig.update_layout(height=height)
    if margin and margin.b == 0:
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False, config={"displaylogo": False, "responsive": True})


def volume_chart(fc: ForecastResult) -> go.Figure:
    hist = fc.history.rename("requests").reset_index().rename(columns={"index": "day"})
    hist["7-day average"] = hist["requests"].rolling(7).mean()
    fig = go.Figure()
    fig.add_scatter(x=hist["day"], y=hist["requests"], name="Daily requests", line=dict(width=1, color="#93c5fd"))
    fig.add_scatter(x=hist["day"], y=hist["7-day average"], name="7-day average", line=dict(width=2.5, color="#2563eb"))
    fig.add_scatter(x=fc.future.index, y=fc.future.values, name="Forecast (next 7 days)",
                    line=dict(width=2.5, color="#dc2626", dash="dot"), mode="lines+markers")
    fig.update_layout(title="Daily 311 requests with 7-day forecast", yaxis_title="Requests per day")
    return fig


def backtest_chart(fc: ForecastResult) -> go.Figure:
    bt = fc.backtest
    fig = go.Figure()
    fig.add_scatter(x=bt["date"], y=bt["actual"], name="Actual", line=dict(color="#111827", width=2))
    fig.add_scatter(x=bt["date"], y=bt["model"], name=f"Gradient boosting (MAE {fc.model_mae:.0f})", line=dict(color="#2563eb"))
    fig.add_scatter(x=bt["date"], y=bt["baseline"], name=f"Same weekday last week (MAE {fc.baseline_mae:.0f})",
                    line=dict(color="#9ca3af", dash="dash"))
    fig.update_layout(title="Backtest on the last 8 weeks (held-out data)", yaxis_title="Requests per day")
    return fig


def areas_chart(df: pd.DataFrame) -> go.Figure:
    fig = px.line(df, x="month", y="requests", color="service_area", markers=False)
    fig.update_layout(title="Monthly requests: 6 busiest service areas", yaxis_title="Requests", xaxis_title=None, legend_title=None)
    return fig


def channel_chart(df: pd.DataFrame) -> go.Figure:
    top = df.groupby("interaction_channel")["requests"].sum().nlargest(5).index
    d = df.assign(channel=df["interaction_channel"].where(df["interaction_channel"].isin(top), "Other"))
    d = d.groupby(["month", "channel"], as_index=False)["pct"].sum()
    fig = px.area(d, x="month", y="pct", color="channel")
    fig.update_layout(title="How residents contact 311 (share of requests by channel)", yaxis_title="% of requests",
                      xaxis_title=None, legend_title=None, yaxis_range=[0, 100])
    return fig


def resolution_chart(df: pd.DataFrame) -> go.Figure:
    d = df.head(15).sort_values("median_hours", ascending=False)
    d = d.assign(median_days=(d["median_hours"] / 24).round(1))
    fig = px.bar(d, x="median_days", y="service_area", orientation="h", text="median_days")
    fig.update_traces(marker_color="#16a34a", textposition="outside")
    fig.update_layout(title="Median time to close a request (recent window)", xaxis_title="Days", yaxis_title=None,
                      height=460)
    return fig


def map_chart(df: pd.DataFrame) -> go.Figure:
    d = df.head(250)
    fig = px.scatter_map(d, lat="lat", lon="lon", size="requests", color="requests", hover_name="neighbourhood",
                         hover_data={"ward": True, "requests": True, "lat": False, "lon": False},
                         color_continuous_scale="Blues", size_max=28, zoom=9.4, map_style="carto-positron")
    fig.update_layout(title="Requests by neighbourhood (recent window)", height=520, margin=dict(l=0, r=0, t=40, b=0))
    return fig


def _table(df: pd.DataFrame, columns: dict[str, str]) -> str:
    head = "".join(f"<th>{html.escape(v)}</th>" for v in columns.values())
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(f'{r[c]:,}' if isinstance(r[c], int) else str(r[c]))}</td>" for c in columns) + "</tr>"
        for _, r in df.iterrows()
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def build(path: Path, stats: dict, fc: ForecastResult, areas: pd.DataFrame, channels: pd.DataFrame,
          resolution: pd.DataFrame, hotspots: pd.DataFrame, top_types: pd.DataFrame, quality: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    updated = datetime.now(UTC).strftime("%B %d, %Y %H:%M UTC")
    charts = [
        _fig_html(volume_chart(fc), first=True),
        _fig_html(backtest_chart(fc)),
        _fig_html(areas_chart(areas)),
        _fig_html(channel_chart(channels)),
        _fig_html(resolution_chart(resolution)),
        _fig_html(map_chart(hotspots)),
    ]
    top_types = top_types.assign(requests=top_types["requests"].astype(int))
    kpis = [
        (f"{stats['total_requests']:,}", f"requests analyzed ({stats['first_day']} to {stats['last_day']})"),
        (f"{stats['avg_daily']:,.0f}", "requests per day on average"),
        (f"{fc.improvement_pct:.0f}%", "lower forecast error than the seasonal baseline"),
        (f"{stats['top_channel_pct']:.0f}%", f"of last month's requests came by {stats['top_channel']}"),
    ]
    kpi_html = "".join(f'<div class="kpi"><b>{v}</b><span>{html.escape(t)}</span></div>' for v, t in kpis)

    page = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Edmonton 311 Insights</title>
<meta name="description" content="Data pipeline and dashboard analyzing City of Edmonton 311 service requests, with a machine-learning volume forecast.">
<style>
  :root {{ --text:#111827; --muted:#6b7280; --border:#e5e7eb; --bg:#f9fafb; --accent:#2563eb; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; color:var(--text); background:var(--bg); }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:28px 16px 56px; }}
  h1 {{ margin:0; font-size:30px; letter-spacing:-.4px; }}
  .lead {{ color:var(--muted); max-width:780px; margin:6px 0 0; }}
  .meta {{ font-size:13px; color:var(--muted); margin-top:8px; }}
  a {{ color:var(--accent); }}
  .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:22px 0; }}
  .kpi {{ background:#fff; border:1px solid var(--border); border-radius:12px; padding:14px 16px; }}
  .kpi b {{ display:block; font-size:26px; }}
  .kpi span {{ color:var(--muted); font-size:13px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
  .card {{ background:#fff; border:1px solid var(--border); border-radius:12px; padding:8px; min-width:0; }}
  .wide {{ grid-column:1/-1; }}
  h2 {{ font-size:18px; margin:30px 0 10px; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; background:#fff; }}
  th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--border); }}
  th {{ color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; }}
  .notes {{ color:#374151; }}
  .notes li {{ margin-bottom:6px; }}
  code {{ background:#eef2ff; padding:1px 5px; border-radius:4px; font-size:13px; }}
  @media (max-width: 820px) {{ .kpis {{ grid-template-columns:1fr 1fr; }} .grid {{ grid-template-columns:1fr; }} }}
</style></head>
<body><div class="wrap">
  <h1>Edmonton 311 Insights</h1>
  <p class="lead">An automated Python data pipeline over the City of Edmonton's open 311 service-request data:
  extraction from the Socrata API, cleaning with pandas, a SQLite warehouse queried with SQL, and a
  scikit-learn model that forecasts daily request volume.</p>
  <div class="meta">Last refreshed {updated} by GitHub Actions ·
    <a href="https://github.com/shucab-lane/edmonton-311-insights">Source code</a> ·
    Data: <a href="https://data.edmonton.ca/City-Administration/311-Requests/q7ua-agfg">City of Edmonton Open Data</a></div>

  <div class="kpis">{kpi_html}</div>

  <div class="grid">
    <div class="card wide">{charts[0]}</div>
    <div class="card wide">{charts[1]}</div>
    <div class="card">{charts[2]}</div>
    <div class="card">{charts[3]}</div>
    <div class="card">{charts[4]}</div>
    <div class="card">{charts[5]}</div>
  </div>

  <h2>Most common request types (last {stats['recent_requests']:,} requests)</h2>
  {_table(top_types, {"service_description": "Request type", "service_area": "Service area", "requests": "Requests"})}

  <h2>Data quality checks (recent window)</h2>
  {_table(pd.DataFrame([quality]), {"rows": "Rows", "missing_close_date_pct": "% still open / no close date",
                                     "negative_durations": "Negative durations (dropped)", "unknown_ward_pct": "% unknown ward"})}

  <h2>Method</h2>
  <ul class="notes">
    <li><b>Extract:</b> 3+ years of history are aggregated on the server with SoQL <code>GROUP BY</code> queries
      (thousands of rows instead of millions); request-level rows are paged in only for the last {stats.get('recent_days', 60)} days.
      Requests retry with exponential backoff on rate limits and server errors.</li>
    <li><b>Transform:</b> pandas normalizes inconsistent category text, parses timestamps, computes time-to-close and
      drops duplicates and negative durations.</li>
    <li><b>Load + analyze:</b> data lands in a SQLite warehouse with primary keys, CHECK constraints and indexes;
      the charts come from SQL queries using CTEs and window functions (<code>RANK() OVER</code>, <code>SUM() OVER (PARTITION BY)</code>).</li>
    <li><b>Forecast:</b> a gradient-boosting regressor on calendar and lag features (each at least 7 days old, so nothing
      leaks from the future), evaluated on the last 8 weeks against a "same weekday last week" baseline:
      MAE {fc.model_mae:.0f} vs {fc.baseline_mae:.0f} requests/day ({fc.improvement_pct:.0f}% lower error).</li>
    <li><b>Automation:</b> a scheduled GitHub Actions workflow re-runs the pipeline and the pytest suite every week and redeploys this page.</li>
  </ul>
</div></body></html>"""
    (path / "index.html").write_text(page, encoding="utf-8")
    (path / ".nojekyll").write_text("", encoding="utf-8")
