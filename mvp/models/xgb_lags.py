from __future__ import annotations
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from mvp.core.features import make_weekly_features, make_lag_features

def fit_predict(train: pd.DataFrame, horizon: int, freq: str, season_lag: int = 52):
    lags = [1, 2, 3, 4, int(season_lag)]
    if int(season_lag) * 2 < len(train):
        lags.append(int(season_lag) * 2)

    hist = make_weekly_features(train)
    hist = make_lag_features(hist, lags).dropna().copy()

    if len(hist) < 40:
        return _flat(train, horizon, freq, float(train["y"].mean()))

    feature_cols = [c for c in hist.columns if c.startswith("lag_")] + ["weekofyear", "year"]
    X = hist[feature_cols].values
    y = hist["y"].values

    model = XGBRegressor(
        n_estimators=700,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    history = train.copy().reset_index(drop=True)
    future_ds = pd.date_range(start=history["ds"].iloc[-1], periods=horizon + 1, freq=freq)[1:]
    preds = []
    for ds in future_ds:
        temp = pd.concat([history, pd.DataFrame({"ds":[ds], "y":[np.nan]})], ignore_index=True)
        temp = make_weekly_features(temp)
        temp = make_lag_features(temp, lags)
        row = temp.iloc[-1]
        x = row[feature_cols].values.astype(float).reshape(1, -1)
        yhat = float(model.predict(x)[0])
        preds.append(yhat)
        history = pd.concat([history, pd.DataFrame({"ds":[ds], "y":[yhat]})], ignore_index=True)

    return pd.DataFrame({"ds": future_ds, "yhat": preds})

def _flat(train, horizon, freq, value):
    future_ds = pd.date_range(start=train["ds"].iloc[-1], periods=horizon + 1, freq=freq)[1:]
    return pd.DataFrame({"ds": future_ds, "yhat": [value]*horizon})
