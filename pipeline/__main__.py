"""Run the pipeline end to end:  python -m pipeline"""

from __future__ import annotations

import argparse
import json
import logging
import time

import pandas as pd

from . import analytics, dashboard, extract, forecast, load, transform
from .config import Config

log = logging.getLogger("pipeline")


REQUEST_COLUMNS = [
    "row_id", "date_created", "date_closed", "request_status", "service_area", "service_category",
    "service_description", "interaction_channel", "ward", "neighbourhood", "nbhd_latitude",
    "nbhd_longitude", "resolution_hours", "is_closed",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Edmonton 311 data pipeline")
    parser.add_argument("--skip-extract", action="store_true",
                        help="rebuild analytics and the dashboard from the existing warehouse")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = Config()
    started = time.perf_counter()
    conn = load.connect(config.db_path)

    if args.skip_extract:
        recent = pd.read_sql_query("SELECT * FROM requests", conn, parse_dates=["date_created", "date_closed"])
        counts = {t: int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
                  for t in ("daily_by_area", "monthly_by_channel", "requests")}
    else:
        client = extract.SocrataClient(config)
        log.info("extract: daily counts by service area (server-side aggregation)")
        daily = transform.daily_by_area(extract.fetch_daily_by_area(client))
        log.info("extract: monthly counts by ward and channel")
        monthly = transform.monthly_by_channel(extract.fetch_monthly_by_channel(client))
        log.info("extract: request-level rows since %s", config.recent_start)
        recent = transform.recent_requests(extract.fetch_recent_requests(client))
        counts = {
            "daily_by_area": load.replace_table(conn, "daily_by_area", daily),
            "monthly_by_channel": load.replace_table(conn, "monthly_by_channel", monthly),
            "requests": load.replace_table(conn, "requests", recent[REQUEST_COLUMNS]),
        }
    log.info("load: %s", counts)

    stats = analytics.headline_stats(conn) | {"recent_days": config.recent_days}
    fc = forecast.train_and_forecast(analytics.run(conn, "daily_totals"))
    log.info("forecast: model MAE %.1f vs baseline %.1f (%.1f%% better)", fc.model_mae, fc.baseline_mae, fc.improvement_pct)

    dashboard.build(
        config.site_dir,
        stats=stats,
        fc=fc,
        areas=analytics.run(conn, "top_areas_monthly"),
        channels=analytics.run(conn, "channel_share"),
        resolution=analytics.median_resolution_hours(conn),
        hotspots=analytics.run(conn, "neighbourhood_hotspots"),
        top_types=analytics.run(conn, "top_request_types"),
        quality=transform.quality_report(recent),
    )
    summary = {**stats, **counts, "model_mae": round(fc.model_mae, 1), "baseline_mae": round(fc.baseline_mae, 1),
               "improvement_pct": round(fc.improvement_pct, 1), "seconds": round(time.perf_counter() - started, 1)}
    (config.site_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("done: %s", summary)


if __name__ == "__main__":
    main()
