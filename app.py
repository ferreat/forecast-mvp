from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_utils import generate_synthetic_weekly_data, load_uploaded_file, train_test_split_weekly
from src.forecasting import ForecastingEngine

st.set_page_config(page_title="Forecasting MVP", page_icon="📈", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
COUNTRIES = ["UK", "USA", "France", "Peru"]

COUNTRY_SERIES_CONFIG = {
    "UK": {"seed": 7, "scale": 0.95, "offset": 8.0, "wave": 6.0, "phase": 0.8},
    "USA": {"seed": 11, "scale": 1.25, "offset": 20.0, "wave": 9.0, "phase": 0.3},
    "France": {"seed": 19, "scale": 1.05, "offset": 12.0, "wave": 7.5, "phase": 1.6},
    "Peru": {"seed": 29, "scale": 0.9, "offset": 6.0, "wave": 8.5, "phase": 2.1},
}

COUNTRY_MAP_CONFIG = {
    "UK": {"geojson_path": DATA_DIR / "maps" / "uk_adm1.geojson"},
    "USA": {"geojson_path": DATA_DIR / "maps" / "usa_adm1.geojson"},
    "France": {"geojson_path": DATA_DIR / "maps" / "france_adm1.geojson"},
    "Peru": {"geojson_path": DATA_DIR / "maps" / "peru_adm1.geojson"},
}


def sample_data_path(country: str) -> Path:
    slug = country.lower().replace(" ", "_")
    return DATA_DIR / f"sample_weekly_demand_{slug}.csv"


def ensure_sample_dataset(country: str) -> None:
    path = sample_data_path(country)
    if path.exists():
        return

    cfg = COUNTRY_SERIES_CONFIG[country]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample_df = generate_synthetic_weekly_data(seed=cfg["seed"])

    idx = np.arange(len(sample_df))
    season_shift = cfg["wave"] * np.sin((2 * np.pi * idx / 26) + cfg["phase"])
    sample_df["demand"] = (sample_df["demand"] * cfg["scale"] + cfg["offset"] + season_shift).clip(lower=20).round(2)
    sample_df.to_csv(path, index=False)


def load_sample_dataset(country: str) -> pd.DataFrame:
    ensure_sample_dataset(country)
    df = pd.read_csv(sample_data_path(country))
    df["date"] = pd.to_datetime(df["date"])
    return df


def plot_history(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["demand"], mode="lines+markers", name="Demand"))
    fig.update_layout(height=360, title=title, xaxis_title="Week", yaxis_title="Demand")
    return fig


@st.cache_data(show_spinner=False)
def load_geojson(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_country_map(country: str) -> go.Figure:
    geojson = load_geojson(str(COUNTRY_MAP_CONFIG[country]["geojson_path"]))
    features = geojson.get("features", [])
    if not features:
        raise ValueError("No region boundaries found for selected country.")

    id_field = "GID_1"
    label_field = "NAME_1"
    for feature in features:
        props = feature.get("properties", {})
        if id_field not in props or label_field not in props:
            raise ValueError(f"GeoJSON for {country} is missing required region properties.")

    locations = [feature["properties"][id_field] for feature in features]
    labels = [feature["properties"][label_field] for feature in features]
    values = [(i % 9) + 1 for i, _ in enumerate(locations)]

    fig = go.Figure(
        go.Choropleth(
            geojson=geojson,
            locations=locations,
            z=values,
            text=labels,
            featureidkey=f"properties.{id_field}",
            hovertemplate="%{text}<extra></extra>",
            marker_line_width=0.5,
            marker_line_color="#ffffff",
            colorscale="Tealgrn",
            showscale=False,
        )
    )
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        showcountries=False,
        showcoastlines=False,
        showframe=False,
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=36, b=0),
        title=f"{country} states / regions",
        annotations=[
            dict(
                x=0.5,
                y=0.0,
                xref="paper",
                yref="paper",
                text="Click-to-drilldown by region is planned for a future update.",
                showarrow=False,
                font=dict(size=10, color="#5f7285"),
            )
        ],
    )
    return fig


st.title("Demand Forecasting MVP")
st.caption("Weekly demand forecasting PoC using ETS, ARIMA, Prophet and XGBoost.")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "email" not in st.session_state:
    st.session_state.email = ""

if not st.session_state.authenticated:
    st.subheader("Authentication")
    email = st.text_input("Enter your email to continue", placeholder="name@company.com")
    if st.button("Continue", type="primary"):
        if "@" in email and "." in email:
            st.session_state.authenticated = True
            st.session_state.email = email
            st.rerun()
        else:
            st.error("Please enter a valid email address.")
    st.stop()

with st.sidebar:
    st.header("Configuration")
    horizon = st.slider("Forecast horizon (weeks)", min_value=1, max_value=12, value=6)
    cv_splits = st.slider("Walk-forward CV periods", min_value=2, max_value=8, value=4)
    st.markdown("---")
    st.write(f"Logged in as: **{st.session_state.email}**")

left, mid, right = st.columns([1.2, 2.2, 1.4])

with right:
    st.subheader("Region map")
    selected_country = st.selectbox("Country", COUNTRIES, index=COUNTRIES.index("USA"))
    st.plotly_chart(plot_country_map(selected_country), use_container_width=True)

with mid:
    st.subheader("Dataset")
    source = st.radio("Choose a source", ["Sample synthetic dataset", "Upload my own dataset"], horizontal=True)
    uploaded_file = None
    if source == "Upload my own dataset":
        uploaded_file = st.file_uploader("Upload CSV or Excel with 'date' and 'demand' columns", type=["csv", "xlsx", "xls"])

try:
    if source == "Sample synthetic dataset":
        st.caption(f"Active country dataset: **{selected_country}**")
        df = load_sample_dataset(selected_country)
    else:
        if uploaded_file is None:
            st.info("Upload a file to continue.")
            st.stop()
        df = load_uploaded_file(uploaded_file)

    with left:
        st.subheader("Data overview")
        st.metric("Rows", len(df))
        st.metric("Start", df["date"].min().date().isoformat())
        st.metric("End", df["date"].max().date().isoformat())
        st.metric("Average demand", f"{df['demand'].mean():.1f}")

    history_title = f"Weekly demand history - {selected_country}" if source == "Sample synthetic dataset" else "Weekly demand history"
    st.plotly_chart(plot_history(df, history_title), use_container_width=True)

    preview = st.expander("Preview data", expanded=False)
    preview.dataframe(df.tail(12), use_container_width=True)

    if st.button("Get forecasts", type="primary", use_container_width=True):
        train_df, test_df = train_test_split_weekly(df, horizon=horizon)
        progress_text = st.empty()
        progress_bar = st.progress(0, text="Preparing forecasting pipeline...")

        def update_progress(progress: float, message: str) -> None:
            progress_bar.progress(int(progress * 100), text=message)
            progress_text.caption(message)

        engine = ForecastingEngine(horizon=horizon, cv_splits=cv_splits)
        summary_df, results = engine.run(train_df, test_df, progress_callback=update_progress)
        progress_bar.progress(100, text="Forecasting completed.")
        progress_text.caption("Forecasting completed.")

        st.subheader("Model performance")
        st.dataframe(
            summary_df.style.format({"CV RMSE": "{:.2f}", "MAE": "{:.2f}", "MAPE": "{:.2f}", "RMSE": "{:.2f}"}),
            use_container_width=True,
        )

        best_row = summary_df.iloc[0]
        best_model = best_row["Model"]
        st.success(f"Best model: {best_model}")

        best_forecast = results[best_model].forecast.copy()
        actual_tail = df.tail(12).copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual_tail["date"], y=actual_tail["demand"], mode="lines+markers", name="Actuals"))
        fig.add_trace(
            go.Scatter(
                x=best_forecast["date"],
                y=best_forecast["forecast"],
                mode="lines+markers",
                name=f"{best_model} forecast",
            )
        )
        forecast_scope = selected_country if source == "Sample synthetic dataset" else "Uploaded dataset"
        fig.update_layout(
            height=420,
            title=f"Last 12 weeks actuals and {horizon}-week forecast ({best_model}) - {forecast_scope}",
            xaxis_title="Week",
            yaxis_title="Demand",
        )
        st.subheader("Best model forecast")
        st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Best model details")
            st.json({
                "scope": forecast_scope,
                "model": best_model,
                "cv_rmse": round(float(best_row["CV RMSE"]), 3),
                "test_metrics": {
                    "MAE": round(float(best_row["MAE"]), 3),
                    "MAPE": round(float(best_row["MAPE"]), 3),
                    "RMSE": round(float(best_row["RMSE"]), 3),
                },
                "best_parameters": results[best_model].best_params,
            })

except Exception as exc:
    st.error(str(exc))
    st.stop()
