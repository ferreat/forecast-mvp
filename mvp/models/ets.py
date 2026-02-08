from __future__ import annotations
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def fit_predict(train: pd.DataFrame, horizon: int, freq: str, season_length: int):
    y = train["y"].astype(float).values
    seasonal = "add" if season_length >= 2 and len(y) >= (season_length * 2) else None
    model = ExponentialSmoothing(
        y,
        trend="add",
        seasonal=seasonal,
        seasonal_periods=season_length if seasonal else None,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)
    yhat = fit.forecast(horizon)
    future_ds = pd.date_range(start=train["ds"].iloc[-1], periods=horizon + 1, freq=freq)[1:]
    return pd.DataFrame({"ds": future_ds, "yhat": yhat})
