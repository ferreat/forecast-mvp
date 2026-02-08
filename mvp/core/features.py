from __future__ import annotations
import pandas as pd

def make_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["weekofyear"] = out["ds"].dt.isocalendar().week.astype(int)
    out["year"] = out["ds"].dt.year.astype(int)
    return out

def make_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out["y"].shift(lag)
    return out

def split_train_test(df: pd.DataFrame, horizon: int):
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if len(df) <= horizon + 10:
        raise ValueError("Not enough history relative to evaluation horizon.")
    return df.iloc[:-horizon].copy(), df.iloc[-horizon:].copy()
