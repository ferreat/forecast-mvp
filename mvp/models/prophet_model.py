from __future__ import annotations
import pandas as pd
from prophet import Prophet

def fit_predict(train: pd.DataFrame, horizon: int, freq: str, season_length_weeks: int = 52):
    dfp = train[["ds", "y"]].copy()
    dfp["ds"] = pd.to_datetime(dfp["ds"])

    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
    )

    period_days = float(season_length_weeks) * 7.0
    m.add_seasonality(name="year_cycle", period=period_days, fourier_order=10)

    m.fit(dfp)
    future = m.make_future_dataframe(periods=horizon, freq=freq, include_history=False)
    fc = m.predict(future)
    return fc[["ds", "yhat"]].copy()
