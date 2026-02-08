from __future__ import annotations
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def fit_predict(train: pd.DataFrame, horizon: int, freq: str, order=(1,1,1)):
    y = train["y"].astype(float).values
    fit = ARIMA(y, order=order).fit()
    yhat = fit.forecast(steps=horizon)
    future_ds = pd.date_range(start=train["ds"].iloc[-1], periods=horizon + 1, freq=freq)[1:]
    return pd.DataFrame({"ds": future_ds, "yhat": yhat})
