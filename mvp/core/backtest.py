from __future__ import annotations
import pandas as pd
from mvp.core.features import split_train_test
from mvp.core.metrics import all_metrics

def holdout_backtest(df: pd.DataFrame, horizon: int, model_fit_predict):
    train, test = split_train_test(df, horizon)
    freq = pd.infer_freq(df["ds"]) or "W-MON"
    fc = model_fit_predict(train, horizon, freq)
    merged = test[["ds", "y"]].merge(fc, on="ds", how="left")
    metrics = all_metrics(merged["y"].values, merged["yhat"].values)
    return merged, metrics
