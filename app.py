from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_utils import generate_synthetic_weekly_data, load_uploaded_file, train_test_split_weekly
from src.forecasting import ForecastingEngine

st.set_page_config(page_title="Forecasting MVP", page_icon="📈", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "sample_weekly_demand.csv"


def ensure_sample_dataset() -> None:
    if DATA_PATH.exists():
        return
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample_df = generate_synthetic_weekly_data()
    sample_df.to_csv(DATA_PATH, index=False)


def plot_history(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["demand"], mode="lines+markers", name="Demand"))
    fig.update_layout(height=360, title=title, xaxis_title="Week", yaxis_title="Demand")
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

with mid:
    st.subheader("Dataset")
    source = st.radio("Choose a source", ["Sample synthetic dataset", "Upload my own dataset"], horizontal=True)
    uploaded_file = None
    if source == "Upload my own dataset":
        uploaded_file = st.file_uploader("Upload CSV or Excel with 'date' and 'demand' columns", type=["csv", "xlsx", "xls"])

try:
    if source == "Sample synthetic dataset":
        ensure_sample_dataset()
        df = pd.read_csv(DATA_PATH)
        df["date"] = pd.to_datetime(df["date"])
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

    st.plotly_chart(plot_history(df, "Weekly demand history"), use_container_width=True)

    preview = st.expander("Preview data", expanded=False)
    preview.dataframe(df.tail(12), use_container_width=True)

    if st.button("Get forecasts", type="primary", use_container_width=True):
        train_df, test_df = train_test_split_weekly(df, horizon=horizon)
        with st.spinner("Running forecasting pipeline..."):
            engine = ForecastingEngine(horizon=horizon, cv_splits=cv_splits)
            summary_df, results = engine.run(train_df, test_df)

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
        fig.update_layout(
            height=420,
            title=f"Last 12 weeks actuals and {horizon}-week forecast ({best_model})",
            xaxis_title="Week",
            yaxis_title="Demand",
        )
        st.subheader("Best model forecast")
        st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Best model details")
            st.json({
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
