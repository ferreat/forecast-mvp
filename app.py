from __future__ import annotations

import json
import uuid
import datetime as dt
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from mvp.config import SETTINGS
from mvp.db import connect, init_db
from mvp.quota import can_run, consume_run
from mvp.storage import ensure_dirs, job_dir, save_upload, write_json, save_csv

from mvp.core.data_validation import validate_and_normalize
from mvp.core.preprocessing import simple_fill_missing, enforce_weekly
from mvp.core.seasonality import suggest_season_length
from mvp.core.backtest import holdout_backtest

from mvp.models.ets import fit_predict as ets_fit_predict
from mvp.models.arima import fit_predict as arima_fit_predict
from mvp.models.prophet_model import fit_predict as prophet_fit_predict
from mvp.models.xgb_lags import fit_predict as xgb_fit_predict

st.set_page_config(page_title=SETTINGS.app_name, layout="wide")

st.markdown("""
<style>
  .app-title {font-size: 1.8rem; font-weight: 780; margin-bottom: 0.25rem;}
  .subtle {color: rgba(49, 51, 63, 0.7);}
  .panel {border: 1px solid rgba(49, 51, 63, 0.12); padding: 14px; border-radius: 14px; background: white;}
  .pill {display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 0.85rem; border: 1px solid rgba(49, 51, 63, 0.15);}
</style>
""", unsafe_allow_html=True)

ensure_dirs()
conn = connect()
init_db(conn)

def get_or_create_user(email: str) -> int:
    now = dt.datetime.now().isoformat()
    norm = email.strip().lower()
    row = conn.execute("SELECT id FROM users WHERE email=?", (norm,)).fetchone()
    if row:
        return int(row[0])
    conn.execute("INSERT INTO users(email, created_at) VALUES(?,?)", (norm, now))
    conn.commit()
    return int(conn.execute("SELECT id FROM users WHERE email=?", (norm,)).fetchone()[0])

def list_jobs(user_id: int):
    return conn.execute(
        "SELECT id, created_at, status, best_model FROM jobs WHERE user_id=? ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()

def load_job(job_id: str):
    return conn.execute(
        "SELECT id, created_at, status, dataset_path, params_json, best_model, metrics_json, artifacts_json, error_message "
        "FROM jobs WHERE id=?",
        (job_id,),
    ).fetchone()

def update_job(job_id: str, **fields):
    keys = list(fields.keys())
    values = [fields[k] for k in keys]
    set_clause = ", ".join([f"{k}=?" for k in keys])
    conn.execute(f"UPDATE jobs SET {set_clause} WHERE id=?", (*values, job_id))
    conn.commit()

def create_job(user_id: int, dataset_path: str, params: dict) -> str:
    job_id = str(uuid.uuid4())
    now = dt.datetime.now().isoformat()
    conn.execute(
        "INSERT INTO jobs(id, user_id, created_at, started_at, status, dataset_path, params_json) VALUES(?,?,?,?,?,?,?)",
        (job_id, user_id, now, now, "RUNNING", dataset_path, json.dumps(params)),
    )
    conn.commit()
    return job_id

def choose_best(metrics_by_model: dict) -> str:
    # Lower is better
    def key(m):
        return (metrics_by_model[m]["MAPE"], metrics_by_model[m]["MAE"])
    return sorted(metrics_by_model.keys(), key=key)[0]

def plot_ts(df: pd.DataFrame, fc: pd.DataFrame | None = None, title: str = "Weekly sales", key: str | None = None):
    actual_color = "#2563eb"
    forecast_color = "#f97316"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], mode="lines", name="Actual", line=dict(color=actual_color, width=2)))
    if fc is not None:
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], mode="lines", name="Forecast", line=dict(color=forecast_color, width=3)))
        fig.add_vline(x=fc["ds"].min(), line_width=2, line_dash="dash", line_color=forecast_color)
    fig.update_layout(
        title=title,
        xaxis_title="Week",
        yaxis_title="Sales",
        hovermode="x unified",
        height=440,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

# Header
cA, cB = st.columns([3, 1])
with cA:
    st.markdown(f'<div class="app-title">{SETTINGS.app_name}</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Weekly sales forecasting • ETS / ARIMA / Prophet / XGBoost • Offline MVP</div>', unsafe_allow_html=True)
with cB:
    st.markdown('<div class="pill">Offline demo</div>', unsafe_allow_html=True)
st.write("")

with st.sidebar:
    st.header("Account")
    email = st.text_input("Email", value=st.session_state.get("email", ""))
    if st.button("Sign in", type="primary"):
        if not email or "@" not in email:
            st.error("Enter a valid email.")
        else:
            st.session_state["email"] = email.strip().lower()
            st.session_state["user_id"] = get_or_create_user(st.session_state["email"])
            st.success(f"Signed in as {st.session_state['email']}")

    user_id = st.session_state.get("user_id")
    if user_id:
        ok, used, limit_ = can_run(conn, user_id)
        st.caption("Usage")
        st.write(f"**{used}/{limit_}** runs this month")

    st.divider()
    st.header("Forecast settings")
    forecast_horizon = st.number_input("Forecast horizon (weeks)", 1, 52, SETTINGS.default_forecast_horizon, help="Default is T+6.")
    eval_horizon = st.number_input("Holdout window (weeks)", 8, 104, SETTINGS.default_eval_horizon)

    auto_season = st.checkbox("Auto-suggest seasonality", value=True, help="Autocorrelation over 4/13/26/52.")
    season_length = st.number_input("Season length (weeks)", 1, 104, SETTINGS.default_season_length, disabled=auto_season)
    season_lag = st.number_input("Seasonal lag for XGBoost", 4, 104, SETTINGS.default_season_length, disabled=auto_season)

    st.divider()
    st.header("Models")
    use_ets = st.checkbox("ETS", value=True)
    use_arima = st.checkbox("ARIMA", value=True)
    use_prophet = st.checkbox("Prophet", value=True)
    use_xgb = st.checkbox("XGBoost (lags)", value=True)

    st.caption("No optimization in this MVP. Defaults chosen to work well across many weekly series.")

if not user_id:
    st.info("Sign in from the sidebar to start.")
    st.stop()

tab_forecast, tab_runs = st.tabs(["Forecast", "Runs"])

with tab_forecast:
    left, right = st.columns([1.15, 0.85])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Data source")
        data_source = st.radio(
            "Choose input",
            ["Upload CSV", "Built-in sample (5y)", "Built-in sample (3y)"],
            label_visibility="collapsed",
        )
        st.caption("Upload your own CSV or use the built-in weekly sample dataset.")
        st.markdown("</div>", unsafe_allow_html=True)

        if data_source == "Upload CSV":
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Upload weekly sales data")
            upload = st.file_uploader("CSV upload", type=["csv"], label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)
            if upload is None:
                st.stop()

            dataset_bytes = upload.getvalue()
            size_mb = len(dataset_bytes) / (1024 * 1024)
            if size_mb > SETTINGS.max_upload_mb:
                st.error(f"File too large ({size_mb:.1f}MB). Max is {SETTINGS.max_upload_mb}MB.")
                st.stop()

            dataset_filename = upload.name
            df_raw = pd.read_csv(upload)
            series_label = "uploaded"
        else:
            sample_path = Path("sample_data/sample_weekly_sales_5y.csv")
            if not sample_path.exists():
                st.error("Built-in sample file is missing at sample_data/sample_weekly_sales_5y.csv.")
                st.stop()
            df_raw = pd.read_csv(sample_path)
            if data_source == "Built-in sample (3y)":
                df_raw = df_raw.tail(156).reset_index(drop=True)
                dataset_filename = "sample_weekly_sales_3y.csv"
                series_label = "sample (3y)"
            else:
                dataset_filename = "sample_weekly_sales_5y.csv"
                series_label = "sample (5y)"
            dataset_bytes = df_raw.to_csv(index=False).encode("utf-8")

        cols = list(df_raw.columns)
        if len(cols) < 2:
            st.error("CSV must contain at least two columns (date + sales).")
            st.stop()

        date_default = cols.index("week_start") if "week_start" in cols else 0
        value_default = cols.index("sales") if "sales" in cols else min(1, len(cols) - 1)
        if value_default == date_default and len(cols) > 1:
            value_default = 1 if date_default == 0 else 0

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Map columns")
        cc1, cc2 = st.columns(2)
        with cc1:
            date_col = st.selectbox("Date column", cols, index=date_default)
        with cc2:
            value_col = st.selectbox("Sales column", cols, index=value_default)
        st.caption("If data isn't perfectly weekly, we resample to W-MON by summing.")
        st.markdown("</div>", unsafe_allow_html=True)

        try:
            df_norm = validate_and_normalize(df_raw, date_col, value_col)
            df_norm = simple_fill_missing(df_norm)
            df_norm = enforce_weekly(df_norm)
        except Exception as e:
            st.error(f"Could not parse dataset: {e}")
            st.stop()

        suggested = suggest_season_length(df_norm)
        effective_season = suggested["best"] if auto_season else int(season_length)
        effective_season_lag = suggested["best"] if auto_season else int(season_lag)
        if auto_season:
            st.info(f"Auto-suggested seasonality: **{effective_season}**. Scores: {suggested['scores']}")

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader(f"Weekly sales ({series_label})")
        plot_ts(df_norm, title=f"Weekly sales ({series_label})", key=f"chart_{series_label.replace(' ', '_')}")
        with st.expander("Show last 30 rows"):
            st.dataframe(df_norm.tail(30), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Run forecast")
        ok, used, limit_ = can_run(conn, user_id)
        if not ok:
            st.error("Monthly run limit reached.")
            st.stop()

        selected = []
        if use_ets: selected.append("ETS")
        if use_arima: selected.append("ARIMA")
        if use_prophet: selected.append("Prophet")
        if use_xgb: selected.append("XGBoost")

        if not selected:
            st.warning("Select at least one model.")
            st.stop()

        st.write("Models: " + ", ".join(selected))
        st.write(f"Horizon: **T+{int(forecast_horizon)}** weeks")
        st.write(f"Holdout: **{int(eval_horizon)}** weeks")

        run_btn = st.button("Run Forecast", type="primary", use_container_width=True)
        st.caption("Fixed, sensible hyperparameters (no tuning).")
        st.markdown("</div>", unsafe_allow_html=True)

        if run_btn:
            t0 = time.time()
            st.toast("Forecast started…", icon="⏳")

            with st.status("Running forecast pipeline…", expanded=True) as status:
                progress = st.progress(0)
                status.write("Saving dataset snapshot + consuming quota…")
                consume_run(conn, user_id)

                params = {
                    "data_source": data_source,
                    "forecast_horizon": int(forecast_horizon),
                    "eval_horizon": int(eval_horizon),
                    "season_length_weeks": int(effective_season),
                    "xgb_season_lag": int(effective_season_lag),
                    "models": selected,
                    "mapping": {"date_col": date_col, "value_col": value_col},
                }

                pre_job_id = str(uuid.uuid4())
                dataset_path = save_upload(dataset_bytes, dataset_filename, pre_job_id)
                job_id = create_job(user_id, str(dataset_path), params)
                jp = job_dir(job_id)
                write_json(jp / "params.json", params)
                save_csv(jp / "data_normalized.csv", df_norm)
                progress.progress(10)

                eh = int(eval_horizon)
                if len(df_norm) <= eh + 10:
                    eh = max(8, min(26, max(8, len(df_norm)//5)))
                    status.write(f"Adjusted holdout window to {eh}.")

                metrics_by_model = {}
                backtests = {}

                def run_model(name: str, fn):
                    bt, met = holdout_backtest(df_norm, eh, lambda tr, h, f: fn(tr, h, f))
                    backtests[name] = bt
                    metrics_by_model[name] = met

                step = 10
                inc = int(70 / max(1, len(selected)))

                if "ETS" in selected:
                    status.write("• ETS…")
                    run_model("ETS", lambda tr, h, f: ets_fit_predict(tr, h, f, int(effective_season)))
                    step += inc; progress.progress(min(step, 80))

                if "ARIMA" in selected:
                    status.write("• ARIMA…")
                    run_model("ARIMA", lambda tr, h, f: arima_fit_predict(tr, h, f, order=(1,1,1)))
                    step += inc; progress.progress(min(step, 80))

                if "Prophet" in selected:
                    status.write("• Prophet…")
                    run_model("Prophet", lambda tr, h, f: prophet_fit_predict(tr, h, f, season_length_weeks=int(effective_season)))
                    step += inc; progress.progress(min(step, 80))

                if "XGBoost" in selected:
                    status.write("• XGBoost…")
                    run_model("XGBoost", lambda tr, h, f: xgb_fit_predict(tr, h, f, season_lag=int(effective_season_lag)))
                    step += inc; progress.progress(min(step, 80))

                status.write("Selecting best model + forecasting…")
                best = choose_best(metrics_by_model)
                freq = pd.infer_freq(df_norm["ds"]) or "W-MON"
                fh = int(forecast_horizon)

                if best == "ETS":
                    fc = ets_fit_predict(df_norm, fh, freq, int(effective_season))
                elif best == "ARIMA":
                    fc = arima_fit_predict(df_norm, fh, freq, order=(1,1,1))
                elif best == "Prophet":
                    fc = prophet_fit_predict(df_norm, fh, freq, season_length_weeks=int(effective_season))
                else:
                    fc = xgb_fit_predict(df_norm, fh, freq, season_lag=int(effective_season_lag))

                write_json(jp / "metrics.json", metrics_by_model)
                save_csv(jp / "forecast.csv", fc)
                save_csv(jp / f"backtest_{best}.csv", backtests[best])

                artifacts = {
                    "data_normalized_csv": str(jp / "data_normalized.csv"),
                    "metrics_json": str(jp / "metrics.json"),
                    "forecast_csv": str(jp / "forecast.csv"),
                    "best_backtest_csv": str(jp / f"backtest_{best}.csv"),
                }

                update_job(
                    job_id,
                    status="SUCCEEDED",
                    best_model=best,
                    metrics_json=json.dumps(metrics_by_model),
                    artifacts_json=json.dumps(artifacts),
                    finished_at=dt.datetime.now().isoformat(),
                    error_message=None,
                )

                progress.progress(100)
                status.update(label=f"Done in {time.time()-t0:.1f}s — Best model: {best}", state="complete")
                st.toast(f"Forecast complete (best: {best})", icon="✅")

            best_metrics = metrics_by_model[best]
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Results")
            met_df = pd.DataFrame(metrics_by_model).T.reset_index().rename(columns={"index":"Model"})
            st.dataframe(met_df, use_container_width=True, hide_index=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Best model", best)
            m2.metric("MAPE", f"{best_metrics['MAPE']:.2f}%")
            m3.metric("RMSE", f"{best_metrics['RMSE']:.2f}")

            plot_ts(df_norm, fc=fc, title=f"Best model forecast • {best} • MAPE {best_metrics['MAPE']:.2f}%", key=f"chart_result_{job_id}")
            st.download_button(
                "Download forecast.csv",
                data=Path(artifacts["forecast_csv"]).read_bytes(),
                file_name=f"forecast_{job_id}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

with tab_runs:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Run history")
    jobs = list_jobs(user_id)
    if not jobs:
        st.info("No runs yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    opts = [f"{j[0]}  |  {j[1]}  |  {j[2]}  | best={j[3] or '-'}" for j in jobs]
    idx = st.selectbox("Select a run", list(range(len(opts))), format_func=lambda i: opts[i])
    job_id = jobs[idx][0]
    row = load_job(job_id)

    job_id, created_at, status, dataset_path, params_json, best_model, metrics_json, artifacts_json, error_message = row
    st.write(f"**Status:** {status}  |  **Created:** {created_at}  |  **Best model:** {best_model or '-'}")
    if error_message:
        st.error(error_message)

    if status != "SUCCEEDED" or not artifacts_json:
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    artifacts = json.loads(artifacts_json)
    metrics = json.loads(metrics_json)

    st.subheader("Model metrics")
    st.dataframe(pd.DataFrame(metrics).T.reset_index().rename(columns={"index":"Model"}), use_container_width=True, hide_index=True)

    df_norm = pd.read_csv(artifacts["data_normalized_csv"])
    df_norm["ds"] = pd.to_datetime(df_norm["ds"])
    fc = pd.read_csv(artifacts["forecast_csv"])
    fc["ds"] = pd.to_datetime(fc["ds"])

    bm = best_model
    bm_mape = metrics[bm]["MAPE"]
    plot_ts(df_norm, fc=fc, title=f"Best model forecast • {bm} • MAPE {bm_mape:.2f}%", key=f"chart_history_{job_id}")
    st.markdown("</div>", unsafe_allow_html=True)
