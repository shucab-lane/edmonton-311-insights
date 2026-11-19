from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import requests

from pipeline import analytics, load, transform
from pipeline.config import Config
from pipeline.extract import SocrataClient
from pipeline.forecast import make_features, train_and_forecast

# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


def test_clean_text_collapses_whitespace_and_fills_blanks():
    s = pd.Series(["Snow  and  Ice", "  Parks ", None, ""])
    assert transform.clean_text(s).tolist() == ["Snow and Ice", "Parks", "Unknown", "Unknown"]


def test_daily_by_area_reaggregates_categories_that_differ_only_by_spacing():
    rows = [
        {"day": "2026-01-01T00:00:00.000", "service_area": "Roadway  Maintenance", "requests": "3"},
        {"day": "2026-01-01T00:00:00.000", "service_area": "Roadway Maintenance", "requests": "2"},
        {"day": "2026-01-02T00:00:00.000", "service_area": "Parks", "requests": "7"},
    ]
    df = transform.daily_by_area(rows)
    assert len(df) == 2
    assert df.loc[df["service_area"] == "Roadway Maintenance", "requests"].item() == 5


def _recent_rows():
    return [
        {"row_id": "1", "date_created": "2026-01-01T10:00:00", "date_closed": "2026-01-01T16:00:00",
         "request_status": "Closed", "service_area": "Parks", "ward": "papastew", "nbhd_latitude": "53.5"},
        {"row_id": "1", "date_created": "2026-01-01T10:00:00", "date_closed": "2026-01-01T16:00:00",
         "request_status": "Closed", "service_area": "Parks", "ward": "papastew"},  # duplicate
        {"row_id": "2", "date_created": "2026-01-02T10:00:00", "date_closed": "2026-01-01T10:00:00",
         "request_status": "Closed", "service_area": "Parks", "ward": "papastew"},  # closed before created
        {"row_id": "3", "date_created": "2026-01-03T10:00:00", "request_status": "Open", "service_area": "Waste"},
    ]


def test_recent_requests_computes_resolution_and_flags_bad_rows():
    df = transform.recent_requests(_recent_rows())
    assert df["row_id"].tolist() == ["1", "2", "3"]  # duplicate removed
    assert df.loc[0, "resolution_hours"] == pytest.approx(6.0)
    assert pd.isna(df.loc[1, "resolution_hours"])  # negative duration discarded
    assert df["is_closed"].tolist() == [True, False, False]
    assert df.loc[0, "ward"] == "Papastew"
    assert df.loc[2, "ward"] == "Unknown"


def test_quality_report_counts_problems():
    report = transform.quality_report(transform.recent_requests(_recent_rows()))
    assert report["rows"] == 3
    assert report["negative_durations"] == 1
    assert report["missing_close_date_pct"] == pytest.approx(33.3)


# ---------------------------------------------------------------------------
# Load + SQL analytics
# ---------------------------------------------------------------------------


@pytest.fixture
def conn():
    c = load.connect(":memory:")
    start = date(2026, 1, 1)
    daily = pd.DataFrame(
        [{"day": start + timedelta(days=d), "service_area": area, "requests": n}
         for d in range(10) for area, n in [("Parks", 10), ("Waste", 20), ("Transit", 5)]]
    )
    load.replace_table(c, "daily_by_area", daily)
    monthly = pd.DataFrame([
        {"month": "2026-01", "ward": "A", "interaction_channel": "Phone Call", "requests": 75},
        {"month": "2026-01", "ward": "B", "interaction_channel": "Mobile App", "requests": 25},
    ])
    load.replace_table(c, "monthly_by_channel", monthly)
    recent = transform.recent_requests(
        [{"row_id": str(i), "date_created": "2026-01-01T00:00:00",
          "date_closed": f"2026-01-01T{(i % 10) + 1:02d}:00:00", "request_status": "Closed",
          "service_area": "Parks", "service_description": "Litter", "neighbourhood": "Downtown",
          "ward": "O-day'min", "nbhd_latitude": "53.54", "nbhd_longitude": "-113.49"} for i in range(60)]
    )
    load.replace_table(c, "requests", recent[[col for col in recent.columns if col != "status_detail"]])
    yield c
    c.close()


def test_replace_table_is_idempotent(conn):
    df = pd.DataFrame([{"day": "2026-02-01", "service_area": "X", "requests": 1}])
    load.replace_table(conn, "daily_by_area", df)
    load.replace_table(conn, "daily_by_area", df)
    assert conn.execute("SELECT COUNT(*) FROM daily_by_area").fetchone()[0] == 1


def test_schema_rejects_negative_counts(conn):
    with pytest.raises(Exception):
        conn.execute("INSERT INTO daily_by_area VALUES ('2026-03-01', 'X', -1)")


def test_daily_totals_sum_across_areas(conn):
    df = analytics.run(conn, "daily_totals")
    assert len(df) == 10
    assert (df["requests"] == 35).all()


def test_top_areas_ranks_by_volume(conn):
    df = analytics.run(conn, "top_areas_monthly")
    assert set(df["service_area"]) == {"Parks", "Waste", "Transit"}
    assert df.loc[df["service_area"] == "Waste", "requests"].item() == 200


def test_channel_share_percentages_sum_to_100(conn):
    df = analytics.run(conn, "channel_share")
    assert df["pct"].sum() == pytest.approx(100)
    assert df.iloc[0]["interaction_channel"] == "Phone Call"


def test_median_resolution(conn):
    df = analytics.median_resolution_hours(conn)
    assert df.loc[0, "service_area"] == "Parks"
    assert df.loc[0, "median_hours"] == pytest.approx(5.5)


def test_headline_stats(conn):
    stats = analytics.headline_stats(conn)
    assert stats["total_requests"] == 350
    assert stats["top_channel"] == "Phone Call"
    assert stats["recent_closed_pct"] == 100.0


# ---------------------------------------------------------------------------
# Extract (HTTP mocked)
# ---------------------------------------------------------------------------


def _response(status: int, payload):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    r.raise_for_status.side_effect = None if status < 400 else requests.HTTPError(response=r)
    return r


def test_paginate_stops_on_short_page(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    session = MagicMock()
    session.headers = {}
    session.get.side_effect = [_response(200, [{"a": 1}] * 2), _response(200, [{"a": 1}])]
    client = SocrataClient(Config(page_size=2), session=session)
    pages = list(client.paginate({"$select": "a"}))
    assert [len(p) for p in pages] == [2, 1]
    assert session.get.call_args_list[1].kwargs["params"]["$offset"] == 2


def test_query_retries_on_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    session = MagicMock()
    session.headers = {}
    session.get.side_effect = [_response(429, None), _response(503, None), _response(200, [{"ok": True}])]
    client = SocrataClient(Config(), session=session)
    assert client.query({}) == [{"ok": True}]
    assert session.get.call_count == 3


def test_query_does_not_retry_client_errors(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    session = MagicMock()
    session.headers = {}
    session.get.return_value = _response(400, None)
    with pytest.raises(requests.HTTPError):
        SocrataClient(Config(), session=session).query({})
    assert session.get.call_count == 1


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------


def _synthetic_daily(days: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=days, freq="D")
    weekly = np.where(idx.dayofweek >= 5, 900, 2500)
    seasonal = 400 * np.sin(2 * np.pi * idx.dayofyear / 365)
    y = weekly + seasonal + rng.normal(0, 120, days)
    return pd.DataFrame({"day": idx.date, "requests": y.round()})


def test_features_never_use_data_from_the_last_week():
    s = pd.Series(np.arange(60, dtype=float), index=pd.date_range("2026-01-01", periods=60, freq="D"))
    X = make_features(s)
    # lag_7 on day t must equal the value from t-7
    assert X.loc["2026-02-10", "lag_7"] == s.loc["2026-02-03"]
    assert X.loc["2026-02-10", "roll_mean_7_lag7"] == s.loc["2026-01-28":"2026-02-03"].mean()


def test_model_beats_seasonal_naive_baseline():
    result = train_and_forecast(_synthetic_daily())
    assert result.model_mae < result.baseline_mae
    assert len(result.backtest) == 56
    assert len(result.future) == 7
    assert result.future.index[0] > result.history.index[-1]
