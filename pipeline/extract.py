"""Extract: pull 311 data from the Socrata API.

Long-range history is aggregated *server-side* with SoQL GROUP BY queries (a few thousand rows
instead of millions), and raw request-level rows are paged in only for a recent window.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any

import requests

from .config import Config

log = logging.getLogger(__name__)

RAW_COLUMNS = [
    "row_id",
    "date_created",
    "date_closed",
    "request_status",
    "service_area",
    "service_category",
    "service_description",
    "interaction_channel",
    "ward",
    "neighbourhood",
    "nbhd_latitude",
    "nbhd_longitude",
]


class SocrataClient:
    def __init__(self, config: Config, session: requests.Session | None = None):
        self.config = config
        self.session = session or requests.Session()
        if config.app_token:
            self.session.headers["X-App-Token"] = config.app_token

    def query(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Run one SoQL query with exponential-backoff retries on network errors and 429/5xx."""
        delay = 2.0
        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = self.session.get(self.config.base_url, params=params, timeout=self.config.timeout_s)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                return resp.json()
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                if attempt == self.config.max_retries:
                    raise
                log.warning("query failed (%s), retry %d in %.0fs", exc, attempt, delay)
                time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    def paginate(self, params: dict[str, Any]) -> Iterator[list[dict[str, Any]]]:
        """Yield pages using $limit/$offset with a stable $order so pages don't overlap."""
        offset = 0
        while True:
            page = self.query({**params, "$limit": self.config.page_size, "$offset": offset})
            if not page:
                return
            yield page
            if len(page) < self.config.page_size:
                return
            offset += self.config.page_size


def _since(day) -> str:
    return f"date_created >= '{day.isoformat()}T00:00:00'"


def fetch_daily_by_area(client: SocrataClient) -> list[dict[str, Any]]:
    """Daily request counts per service area for the full history (server-side GROUP BY)."""
    params = {
        "$select": "date_trunc_ymd(date_created) AS day, service_area, count(*) AS requests",
        "$where": _since(client.config.history_start),
        "$group": "day, service_area",
        "$order": "day, service_area",
    }
    return [row for page in client.paginate(params) for row in page]


def fetch_monthly_by_channel(client: SocrataClient) -> list[dict[str, Any]]:
    """Monthly counts per ward and interaction channel (phone, app, web...)."""
    params = {
        "$select": "date_trunc_ym(date_created) AS month, ward, interaction_channel, count(*) AS requests",
        "$where": _since(client.config.history_start),
        "$group": "month, ward, interaction_channel",
        "$order": "month, ward, interaction_channel",
    }
    return [row for page in client.paginate(params) for row in page]


def fetch_recent_requests(client: SocrataClient) -> list[dict[str, Any]]:
    """Request-level rows for the recent window (used for resolution-time analysis)."""
    params = {
        "$select": ", ".join(RAW_COLUMNS),
        "$where": _since(client.config.recent_start),
        "$order": "row_id",
    }
    rows: list[dict[str, Any]] = []
    for page in client.paginate(params):
        rows.extend(page)
        log.info("fetched %d recent rows", len(rows))
    return rows
