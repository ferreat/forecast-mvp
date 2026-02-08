from __future__ import annotations
import pandas as pd

def simple_fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["y"] = out["y"].astype(float)
    out["y"] = out["y"].interpolate(method="linear", limit_direction="both")
    return out

def enforce_weekly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().set_index("ds").sort_index()
    out = out.resample("W-MON").sum()
    return out.reset_index()
