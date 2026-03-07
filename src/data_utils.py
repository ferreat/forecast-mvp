from __future__ import annotations

from io import BytesIO
from typing import Tuple

import pandas as pd


EXPECTED_COLUMNS = ("date", "demand")


def generate_synthetic_weekly_data(
    periods: int = 156,
    start_date: str = "2023-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start_date, periods=periods, freq="W-SUN")
    t = np.arange(periods)
    trend = 120 + 0.8 * t
    season = 18 * np.sin(2 * np.pi * t / 52) + 8 * np.cos(2 * np.pi * t / 26)
    promo = np.where((t % 13) == 0, 20, 0)
    noise = rng.normal(0, 6, periods)
    demand = (trend + season + promo + noise).clip(min=20)
    return pd.DataFrame({"date": idx, "demand": demand.round(2)})


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    content = uploaded_file.getvalue()

    if name.endswith(".csv"):
        df = pd.read_csv(BytesIO(content))
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(BytesIO(content))
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")

    return standardize_dataframe(df)


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lowered = {col: col.strip().lower() for col in df.columns}
    df = df.rename(columns=lowered)

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Input data must contain 'date' and 'demand' columns. "
            f"Missing: {', '.join(missing)}"
        )

    df = df[["date", "demand"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    df = df.dropna().sort_values("date").drop_duplicates(subset=["date"])

    if len(df) < 30:
        raise ValueError("Please provide at least 30 weekly observations.")

    return infer_weekly_frequency(df)


def infer_weekly_frequency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date")
    df = df.set_index("date")
    df = df.asfreq("W-SUN")
    if df["demand"].isna().any():
        df["demand"] = df["demand"].interpolate(limit_direction="both")
    return df.reset_index()


def train_test_split_weekly(df: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least 1.")
    if len(df) <= horizon + 12:
        raise ValueError("Dataset is too short for the selected forecast horizon.")

    train = df.iloc[:-horizon].copy()
    test = df.iloc[-horizon:].copy()
    return train, test
