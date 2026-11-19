"""Forecast daily 311 volume with scikit-learn, evaluated against a seasonal-naive baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

LAGS = (7, 14, 21, 28)


def make_features(series: pd.Series) -> pd.DataFrame:
    """Calendar + lag features. Every feature uses data at least 7 days old, so a
    model trained on them can forecast up to 7 days ahead without leaking the future."""
    s = series.asfreq("D").fillna(0)
    X = pd.DataFrame(index=s.index)
    X["dow"] = s.index.dayofweek
    X["month"] = s.index.month
    X["dayofyear"] = s.index.dayofyear
    X["is_weekend"] = (X["dow"] >= 5).astype(int)
    for lag in LAGS:
        X[f"lag_{lag}"] = s.shift(lag)
    X["roll_mean_7_lag7"] = s.shift(7).rolling(7).mean()
    X["roll_mean_28_lag7"] = s.shift(7).rolling(28).mean()
    X["same_dow_mean_4w"] = (s.shift(7) + s.shift(14) + s.shift(21) + s.shift(28)) / 4
    return X


@dataclass
class ForecastResult:
    history: pd.Series            # actual daily totals
    backtest: pd.DataFrame        # date, actual, model, baseline over the test window
    future: pd.Series             # next N days
    model_mae: float
    baseline_mae: float

    @property
    def improvement_pct(self) -> float:
        return 0.0 if self.baseline_mae == 0 else 100 * (1 - self.model_mae / self.baseline_mae)


def _model() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, random_state=42)


def train_and_forecast(daily: pd.DataFrame, test_days: int = 56, horizon: int = 7) -> ForecastResult:
    """Time-ordered split: train on everything before the last `test_days`, evaluate on them.
    The final model is refit on all data to forecast `horizon` days past the end."""
    s = daily.assign(day=pd.to_datetime(daily["day"])).set_index("day")["requests"].astype(float)
    s = s.asfreq("D").fillna(0)
    # The latest day is usually partial (data still arriving), so drop it
    s = s.iloc[:-1]
    horizon = min(horizon, 7)

    X = make_features(s)
    data = X.assign(y=s).dropna()
    train, test = data.iloc[:-test_days], data.iloc[-test_days:]
    features = [c for c in data.columns if c != "y"]

    model = _model().fit(train[features], train["y"])
    pred = model.predict(test[features])
    baseline = test["lag_7"].to_numpy()  # seasonal naive: same weekday last week

    backtest = pd.DataFrame({
        "date": test.index,
        "actual": test["y"].to_numpy(),
        "model": np.round(pred, 1),
        "baseline": baseline,
    })

    final = _model().fit(data[features], data["y"])
    future_index = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    extended = pd.concat([s, pd.Series(np.nan, index=future_index)])
    future_X = make_features(extended.fillna(0)).loc[future_index, features]
    future = pd.Series(np.round(final.predict(future_X), 0), index=future_index)

    return ForecastResult(
        history=s,
        backtest=backtest,
        future=future,
        model_mae=float(mean_absolute_error(backtest["actual"], backtest["model"])),
        baseline_mae=float(mean_absolute_error(backtest["actual"], backtest["baseline"])),
    )
