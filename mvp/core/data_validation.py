from __future__ import annotations
import pandas as pd

class ValidationError(Exception):
    pass

def validate_and_normalize(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    if date_col not in df.columns:
        raise ValidationError(f"Date column '{date_col}' not found.")
    if value_col not in df.columns:
        raise ValidationError(f"Target column '{value_col}' not found.")
    out = df[[date_col, value_col]].copy()
    out.rename(columns={date_col: "ds", value_col: "y"}, inplace=True)
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
    if out["ds"].isna().any():
        raise ValidationError("Some dates could not be parsed.")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    if out["y"].isna().any():
        raise ValidationError("Some target values could not be parsed as numbers.")
    out = out.sort_values("ds").drop_duplicates(subset=["ds"]).reset_index(drop=True)
    if len(out) < 40:
        raise ValidationError("Not enough history. Provide at least 40 weekly rows.")
    return out
