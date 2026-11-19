-- Warehouse schema (SQLite). Rebuilt on each pipeline run.

CREATE TABLE IF NOT EXISTS daily_by_area (
    day           TEXT    NOT NULL,          -- ISO date
    service_area  TEXT    NOT NULL,
    requests      INTEGER NOT NULL CHECK (requests >= 0),
    PRIMARY KEY (day, service_area)
);

CREATE TABLE IF NOT EXISTS monthly_by_channel (
    month                TEXT    NOT NULL,   -- YYYY-MM
    ward                 TEXT    NOT NULL,
    interaction_channel  TEXT    NOT NULL,
    requests             INTEGER NOT NULL CHECK (requests >= 0),
    PRIMARY KEY (month, ward, interaction_channel)
);

CREATE TABLE IF NOT EXISTS requests (
    row_id               TEXT PRIMARY KEY,
    date_created         TEXT NOT NULL,
    date_closed          TEXT,
    request_status       TEXT,
    service_area         TEXT,
    service_category     TEXT,
    service_description  TEXT,
    interaction_channel  TEXT,
    ward                 TEXT,
    neighbourhood        TEXT,
    nbhd_latitude        REAL,
    nbhd_longitude       REAL,
    resolution_hours     REAL,
    is_closed            INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_requests_area    ON requests (service_area);
CREATE INDEX IF NOT EXISTS idx_requests_created ON requests (date_created);
CREATE INDEX IF NOT EXISTS idx_daily_day        ON daily_by_area (day);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_at        TEXT NOT NULL,
    table_name    TEXT NOT NULL,
    rows_loaded   INTEGER NOT NULL
);
