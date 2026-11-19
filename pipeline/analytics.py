"""Analytics: SQL queries against the warehouse, returned as DataFrames."""

from __future__ import annotations

import sqlite3

import pandas as pd

QUERIES: dict[str, str] = {
    # Daily total volume across all service areas
    "daily_totals": """
        SELECT day, SUM(requests) AS requests
        FROM daily_by_area
        GROUP BY day
        ORDER BY day
    """,
    # Monthly volume for the 6 busiest service areas (CTE + window function ranking)
    "top_areas_monthly": """
        WITH ranked AS (
            SELECT service_area,
                   SUM(requests) AS total,
                   RANK() OVER (ORDER BY SUM(requests) DESC) AS rnk
            FROM daily_by_area
            GROUP BY service_area
        )
        SELECT substr(d.day, 1, 7) AS month, d.service_area, SUM(d.requests) AS requests
        FROM daily_by_area d
        JOIN ranked r ON r.service_area = d.service_area AND r.rnk <= 6
        GROUP BY month, d.service_area
        ORDER BY month, d.service_area
    """,
    # Share of requests by channel per month
    "channel_share": """
        SELECT month,
               interaction_channel,
               SUM(requests) AS requests,
               ROUND(100.0 * SUM(requests) / SUM(SUM(requests)) OVER (PARTITION BY month), 2) AS pct
        FROM monthly_by_channel
        GROUP BY month, interaction_channel
        ORDER BY month, requests DESC
    """,
    # Resolution time per service area from recent request-level data
    "resolution_by_area": """
        SELECT service_area,
               COUNT(*)                                   AS requests,
               SUM(is_closed)                             AS closed,
               ROUND(AVG(CASE WHEN is_closed THEN resolution_hours END), 1) AS avg_hours,
               ROUND(100.0 * SUM(is_closed) / COUNT(*), 1) AS closed_pct
        FROM requests
        GROUP BY service_area
        HAVING COUNT(*) >= 50
        ORDER BY requests DESC
    """,
    # Request counts by neighbourhood with coordinates, for the map
    "neighbourhood_hotspots": """
        SELECT neighbourhood, ward,
               COUNT(*) AS requests,
               AVG(nbhd_latitude) AS lat, AVG(nbhd_longitude) AS lon
        FROM requests
        WHERE nbhd_latitude IS NOT NULL AND neighbourhood <> 'Unknown'
        GROUP BY neighbourhood, ward
        ORDER BY requests DESC
    """,
    # Most common request types in the recent window
    "top_request_types": """
        SELECT service_description, service_area, COUNT(*) AS requests
        FROM requests
        GROUP BY service_description, service_area
        ORDER BY requests DESC
        LIMIT 12
    """,
}


def run(conn: sqlite3.Connection, name: str) -> pd.DataFrame:
    return pd.read_sql_query(QUERIES[name], conn)


def median_resolution_hours(conn: sqlite3.Connection) -> pd.DataFrame:
    """SQLite has no MEDIAN(), so medians are computed in pandas on the closed requests."""
    df = pd.read_sql_query(
        "SELECT service_area, resolution_hours FROM requests WHERE is_closed = 1", conn
    )
    if df.empty:
        return pd.DataFrame(columns=["service_area", "median_hours", "p90_hours", "closed"])
    g = df.groupby("service_area")["resolution_hours"]
    out = pd.DataFrame({
        "median_hours": g.median(),
        "p90_hours": g.quantile(0.9),
        "closed": g.size(),
    }).reset_index()
    return out[out["closed"] >= 50].sort_values("median_hours").reset_index(drop=True)


def headline_stats(conn: sqlite3.Connection) -> dict[str, float | int | str]:
    daily = run(conn, "daily_totals")
    recent = pd.read_sql_query("SELECT COUNT(*) n, SUM(is_closed) closed FROM requests", conn).iloc[0]
    channel = run(conn, "channel_share")
    last_month = channel["month"].max() if not channel.empty else None
    top_channel = (
        channel[channel["month"] == last_month].iloc[0] if last_month is not None else None
    )
    return {
        "total_requests": int(daily["requests"].sum()),
        "days_covered": int(len(daily)),
        "first_day": str(daily["day"].min()) if not daily.empty else "",
        "last_day": str(daily["day"].max()) if not daily.empty else "",
        "avg_daily": round(float(daily["requests"].mean()), 1) if not daily.empty else 0.0,
        "recent_requests": int(recent["n"] or 0),
        "recent_closed_pct": round(100 * float(recent["closed"] or 0) / max(int(recent["n"] or 0), 1), 1),
        "top_channel": str(top_channel["interaction_channel"]) if top_channel is not None else "",
        "top_channel_pct": float(top_channel["pct"]) if top_channel is not None else 0.0,
    }
