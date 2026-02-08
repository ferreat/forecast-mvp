from __future__ import annotations
import numpy as np
import pandas as pd

def suggest_season_length(df: pd.DataFrame, candidates: list[int] | None = None) -> dict:
    if candidates is None:
        candidates = [4, 13, 26, 52]
    y = df["y"].astype(float).values
    y = y - np.mean(y)
    denom = np.sum(y**2) + 1e-12
    scores = {}
    for lag in candidates:
        if lag >= len(y) - 1:
            scores[lag] = float("-inf")
            continue
        scores[lag] = float(np.sum(y[lag:] * y[:-lag]) / denom)
    best = max(scores, key=scores.get)
    return {"best": int(best), "scores": {int(k): float(v) for k, v in scores.items()}}
