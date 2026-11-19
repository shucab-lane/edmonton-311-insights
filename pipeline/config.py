"""Pipeline settings. Override with environment variables in CI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    # City of Edmonton Open Data (Socrata) — "311 Requests"
    base_url: str = "https://data.edmonton.ca/resource/q7ua-agfg.json"
    history_start: date = date(2023, 1, 1)
    # Raw rows are only pulled for a recent window; history is aggregated server-side
    recent_days: int = int(os.getenv("RECENT_DAYS", "60"))
    page_size: int = 50_000
    timeout_s: int = 120
    max_retries: int = 4
    db_path: Path = field(default_factory=lambda: Path(os.getenv("DB_PATH", ROOT / "data" / "warehouse.db")))
    site_dir: Path = field(default_factory=lambda: Path(os.getenv("SITE_DIR", ROOT / "site")))
    forecast_horizon_days: int = 14
    app_token: str | None = os.getenv("SOCRATA_APP_TOKEN")

    @property
    def recent_start(self) -> date:
        return date.today() - timedelta(days=self.recent_days)
