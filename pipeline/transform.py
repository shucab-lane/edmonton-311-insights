"""Transform: clean and type the raw API output with pandas."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

_SPACES = re.compile(r"\s+")


def clean_text(series: pd.Series) -> pd.Series:
    """Collapse repeated whitespace ('Snow  and  Ice' -> 'Snow and Ice') and fill blanks."""
    return series.fillna("Unknown").astype(str).map(lambda s: _SPACES.sub(" ", s).strip() or "Unknown")


def daily_by_area(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["day", "service_area", "requests"])
    df["day"] = pd.to_datetime(df["day"]).dt.date
    df["service_area"] = clean_text(df["service_area"])
    df["requests"] = pd.to_numeric(df["requests"], errors="coerce").fillna(0).astype(int)
    # Cleaning can merge categories that differed only by spacing, so re-aggregate
    return df.groupby(["day", "service_area"], as_index=False)["requests"].sum()


def monthly_by_channel(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["month", "ward", "interaction_channel", "requests"])
    df["month"] = pd.to_datetime(df["month"]).dt.strftime("%Y-%m")
    df["ward"] = clean_text(df["ward"]).str.title()
    df["interaction_channel"] = clean_text(df["interaction_channel"])
    df["requests"] = pd.to_numeric(df["requests"], errors="coerce").fillna(0).astype(int)
    return df.groupby(["month", "ward", "interaction_channel"], as_index=False)["requests"].sum()


def recent_requests(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ["service_area", "service_category", "service_description", "interaction_channel",
                "request_status", "neighbourhood"]:
        df[col] = clean_text(df.get(col, pd.Series(index=df.index, dtype=object)))
    df["ward"] = clean_text(df.get("ward", pd.Series(index=df.index, dtype=object))).str.title()
    df["date_created"] = pd.to_datetime(df["date_created"], errors="coerce")
    df["date_closed"] = pd.to_datetime(df.get("date_closed"), errors="coerce")
    for col in ["nbhd_latitude", "nbhd_longitude"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    df = df.dropna(subset=["row_id", "date_created"]).drop_duplicates("row_id")
    hours = (df["date_closed"] - df["date_created"]).dt.total_seconds() / 3600
    # Negative durations are data-entry errors; treat them as unknown
    df["resolution_hours"] = hours.where(hours >= 0)
    df["is_closed"] = df["request_status"].str.lower().eq("closed") & df["resolution_hours"].notna()
    return df.reset_index(drop=True)


def quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """Simple data-quality checks surfaced on the dashboard."""
    return {
        "rows": int(len(df)),
        "duplicate_ids_removed": 0,
        "missing_close_date_pct": round(float(df["date_closed"].isna().mean() * 100), 1) if len(df) else 0.0,
        "negative_durations": int(((df["date_closed"] - df["date_created"]).dt.total_seconds() < 0).sum()),
        "unknown_ward_pct": round(float((df["ward"] == "Unknown").mean() * 100), 1) if len(df) else 0.0,
    }
