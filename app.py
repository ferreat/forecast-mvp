from __future__ import annotations

import datetime as dt
import json
import time
import uuid
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mvp.config import SETTINGS
from mvp.core.backtest import holdout_backtest
from mvp.core.preprocessing import enforce_weekly, simple_fill_missing
from mvp.core.seasonality import suggest_season_length
from mvp.db import connect, init_db
from mvp.models.arima import fit_predict as arima_fit_predict
from mvp.models.ets import fit_predict as ets_fit_predict
from mvp.models.prophet_model import fit_predict as prophet_fit_predict
from mvp.models.xgb_lags import fit_predict as xgb_fit_predict
from mvp.quota import can_run, consume_run
from mvp.storage import ensure_dirs, job_dir, save_csv, save_upload, write_json

st.set_page_config(page_title=SETTINGS.app_name, layout="wide")

st.markdown(
    """
<style>
  .app-title {font-size: 1.8rem; font-weight: 780; margin-bottom: 0.25rem;}
  .subtle {color: rgba(49, 51, 63, 0.7);}
  .panel {border: 1px solid rgba(49, 51, 63, 0.12); padding: 14px; border-radius: 14px; background: white;}
  .pill {display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 0.85rem; border: 1px solid rgba(49, 51, 63, 0.15);}
</style>
""",
    unsafe_allow_html=True,
)

ensure_dirs()
conn = connect()
init_db(conn)


@st.cache_data
def load_peru_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sales_path = Path("sample_data/peru_weekly_sales_3y_by_region.csv")
    meta_path = Path("sample_data/peru_regions_meta.csv")
    shapes_path = Path("sample_data/peru_regions_boundaries.geojson")
    if not sales_path.exists() or not meta_path.exists() or not shapes_path.exists():
        raise FileNotFoundError(
            "Missing Peru sample data. Expected sample_data/peru_weekly_sales_3y_by_region.csv and "
            "sample_data/peru_regions_meta.csv and sample_data/peru_regions_boundaries.geojson"
        )

    sales = pd.read_csv(sales_path)
    meta = pd.read_csv(meta_path)
    shapes_geojson = json.loads(shapes_path.read_text(encoding="utf-8"))

    sales["week_start"] = pd.to_datetime(sales["week_start"])
    sales["sales"] = pd.to_numeric(sales["sales"], errors="coerce")
    sales = sales.dropna(subset=["region", "week_start", "sales"]).sort_values(["region", "week_start"])
    meta = meta.dropna(subset=["region", "lat", "lon"]).copy()
    meta["lat"] = pd.to_numeric(meta["lat"], errors="coerce")
    meta["lon"] = pd.to_numeric(meta["lon"], errors="coerce")
    meta = meta.dropna(subset=["lat", "lon"])

    valid_regions = set(meta["region"].astype(str))
    sales = sales[sales["region"].astype(str).isin(valid_regions)].copy()
    return sales, meta, shapes_geojson


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
    def key(model: str):
        return (metrics_by_model[model]["MAPE"], metrics_by_model[model]["MAE"])

    return sorted(metrics_by_model.keys(), key=key)[0]


def region_series(df_sales: pd.DataFrame, region: str) -> pd.DataFrame:
    df = df_sales[df_sales["region"] == region][["week_start", "sales"]].copy()
    df = df.rename(columns={"week_start": "ds", "sales": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).sort_values("ds")
    df = simple_fill_missing(df)
    df = enforce_weekly(df)
    return df


def plot_ts(df: pd.DataFrame, fc: pd.DataFrame | None = None, title: str = "Weekly sales", key: str | None = None):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["ds"],
            y=df["y"],
            mode="lines",
            name="Actual",
            line=dict(color="#2563eb", width=2),
        )
    )
    if fc is not None and not fc.empty:
        fig.add_trace(
            go.Scatter(
                x=fc["ds"],
                y=fc["yhat"],
                mode="lines",
                name="Forecast",
                line=dict(color="#f97316", width=3),
            )
        )
        fig.add_vline(x=fc["ds"].min(), line_width=2, line_dash="dash", line_color="#f97316")

    fig.update_layout(
        title=title,
        xaxis_title="Week",
        yaxis_title="Sales",
        hovermode="x unified",
        height=440,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return st.plotly_chart(fig, use_container_width=True, key=key)


def plot_peru_map(regions: list[str], shapes_geojson: dict, selected_region: str, key: str):
    map_df = pd.DataFrame({"region": regions})
    map_df["selected"] = (map_df["region"] == selected_region).astype(int)

    fig = go.Figure(
        go.Choropleth(
            geojson=shapes_geojson,
            locations=map_df["region"],
            z=map_df["selected"],
            featureidkey="properties.region",
            customdata=map_df["region"],
            colorscale=[[0.0, "#dbeafe"], [1.0, "#fb923c"]],
            showscale=False,
            marker_line_color="#475569",
            marker_line_width=1.3,
            hovertemplate="<b>%{customdata}</b><extra></extra>",
        )
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        projection_type="mercator",
        showland=True,
        landcolor="rgb(241,245,249)",
        showcountries=True,
        countrycolor="rgb(148,163,184)",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=380,
        title=f"Peru regions (selected: {selected_region})",
    )
    return st.plotly_chart(fig, use_container_width=True, key=key, on_select="rerun")


def extract_clicked_region(regions: list[str], map_event: object) -> str | None:
    if not isinstance(map_event, dict):
        return None
    selection = map_event.get("selection")
    if not isinstance(selection, dict):
        return None
    points = selection.get("points")
    if not isinstance(points, list) or not points:
        return None

    point = points[0]
    location = point.get("location")
    if isinstance(location, str) and location:
        return location

    custom = point.get("customdata")
    if isinstance(custom, str) and custom:
        return custom
    if isinstance(custom, list) and custom and isinstance(custom[0], str):
        return custom[0]

    idx = point.get("point_index")
    if idx is None:
        idx = point.get("pointNumber")
    if isinstance(idx, int) and 0 <= idx < len(regions):
        return str(regions[idx])
    return None


def trim_history_for_forecast(df_hist: pd.DataFrame, df_fc: pd.DataFrame, history_weeks: int = 26) -> pd.DataFrame:
    if df_fc.empty:
        return df_hist
    cutoff = df_fc["ds"].min() - pd.Timedelta(weeks=history_weeks)
    return df_hist[df_hist["ds"] >= cutoff]


def run_models_for_region(
    df_norm: pd.DataFrame,
    selected_models: list[str],
    eval_horizon: int,
    forecast_horizon: int,
    effective_season: int,
    effective_season_lag: int,
) -> tuple[dict, pd.DataFrame, str]:
    eh = int(eval_horizon)
    if len(df_norm) <= eh + 10:
        eh = max(8, min(26, max(8, len(df_norm) // 5)))

    metrics_by_model = {}

    def run_backtest(fn):
        _, met = holdout_backtest(df_norm, eh, lambda tr, h, f: fn(tr, h, f))
        return met

    if "ETS" in selected_models:
        metrics_by_model["ETS"] = run_backtest(lambda tr, h, f: ets_fit_predict(tr, h, f, int(effective_season)))
    if "ARIMA" in selected_models:
        metrics_by_model["ARIMA"] = run_backtest(lambda tr, h, f: arima_fit_predict(tr, h, f, order=(1, 1, 1)))
    if "Prophet" in selected_models:
        metrics_by_model["Prophet"] = run_backtest(
            lambda tr, h, f: prophet_fit_predict(tr, h, f, season_length_weeks=int(effective_season))
        )
    if "XGBoost" in selected_models:
        metrics_by_model["XGBoost"] = run_backtest(lambda tr, h, f: xgb_fit_predict(tr, h, f, season_lag=int(effective_season_lag)))

    best = choose_best(metrics_by_model)
    freq = pd.infer_freq(df_norm["ds"]) or "W-MON"
    fh = int(forecast_horizon)

    if best == "ETS":
        fc = ets_fit_predict(df_norm, fh, freq, int(effective_season))
    elif best == "ARIMA":
        fc = arima_fit_predict(df_norm, fh, freq, order=(1, 1, 1))
    elif best == "Prophet":
        fc = prophet_fit_predict(df_norm, fh, freq, season_length_weeks=int(effective_season))
    else:
        fc = xgb_fit_predict(df_norm, fh, freq, season_lag=int(effective_season_lag))

    return metrics_by_model, fc, best


# Header
cA, cB = st.columns([3, 1])
with cA:
    st.markdown(f'<div class="app-title">{SETTINGS.app_name}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtle">Peru regional weekly forecasting (3 years) • ETS / ARIMA / Prophet / XGBoost</div>',
        unsafe_allow_html=True,
    )
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
    forecast_horizon = st.number_input(
        "Forecast horizon (weeks)", 1, 52, SETTINGS.default_forecast_horizon, help="Default is T+6."
    )
    eval_horizon = st.number_input("Holdout window (weeks)", 8, 104, SETTINGS.default_eval_horizon)

    auto_season = st.checkbox("Auto-suggest seasonality", value=True, help="Autocorrelation over 4/13/26/52.")
    season_length = st.number_input(
        "Season length (weeks)", 1, 104, SETTINGS.default_season_length, disabled=auto_season
    )
    season_lag = st.number_input(
        "Seasonal lag for XGBoost", 4, 104, SETTINGS.default_season_length, disabled=auto_season
    )

    st.divider()
    st.header("Models")
    use_ets = st.checkbox("ETS", value=True)
    use_arima = st.checkbox("ARIMA", value=True)
    use_prophet = st.checkbox("Prophet", value=True)
    use_xgb = st.checkbox("XGBoost (lags)", value=True)

if not user_id:
    st.info("Sign in from the sidebar to start.")
    st.stop()

try:
    sales_df, meta_df, shapes_geojson = load_peru_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

regions = list(meta_df["region"].astype(str))
if not regions:
    st.error("No regions found in Peru metadata.")
    st.stop()

if "selected_region" not in st.session_state or st.session_state["selected_region"] not in regions:
    st.session_state["selected_region"] = "Lima" if "Lima" in regions else regions[0]
if "latest_forecast_state" not in st.session_state:
    st.session_state["latest_forecast_state"] = None

selected_models = []
if use_ets:
    selected_models.append("ETS")
if use_arima:
    selected_models.append("ARIMA")
if use_prophet:
    selected_models.append("Prophet")
if use_xgb:
    selected_models.append("XGBoost")

tab_forecast, tab_runs = st.tabs(["Forecast", "Runs"])

with tab_forecast:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Country-wide forecast run")
    ok, used, limit_ = can_run(conn, user_id)
    if not ok:
        st.error("Monthly run limit reached.")
        st.stop()
    if not selected_models:
        st.warning("Select at least one model in the sidebar.")
        st.stop()

    st.write("Models: " + ", ".join(selected_models))
    st.write(f"Regions: **{len(regions)}**")
    st.write(f"Horizon: **T+{int(forecast_horizon)}** weeks")
    st.write("Charts below update to clicked region. Forecast views show last 6 months of history.")
    run_btn = st.button("Run Forecast", type="primary", use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)

    if run_btn:
        t0 = time.time()
        st.toast("Country-wide forecast started…", icon="⏳")

        with st.status("Running forecasts across all regions…", expanded=True) as status:
            progress = st.progress(0)
            consume_run(conn, user_id)

            params = {
                "data_source": "peru_3y_regions",
                "forecast_horizon": int(forecast_horizon),
                "eval_horizon": int(eval_horizon),
                "season_length_weeks": int(season_length),
                "xgb_season_lag": int(season_lag),
                "auto_season": auto_season,
                "models": selected_models,
                "regions": regions,
            }

            dataset_bytes = sales_df.to_csv(index=False).encode("utf-8")
            pre_job_id = str(uuid.uuid4())
            dataset_path = save_upload(dataset_bytes, "peru_weekly_sales_3y_by_region.csv", pre_job_id)
            job_id = create_job(user_id, str(dataset_path), params)
            jp = job_dir(job_id)
            write_json(jp / "params.json", params)

            all_norm_parts: list[pd.DataFrame] = []
            all_fc_parts: list[pd.DataFrame] = []
            metrics_by_region: dict[str, dict] = {}
            best_model_by_region: dict[str, str] = {}

            total = len(regions)
            for idx, region in enumerate(regions, start=1):
                status.write(f"• {region} ({idx}/{total})")
                df_norm = region_series(sales_df, region)

                suggested = suggest_season_length(df_norm)
                effective_season = suggested["best"] if auto_season else int(season_length)
                effective_season_lag = suggested["best"] if auto_season else int(season_lag)

                metrics, fc, best = run_models_for_region(
                    df_norm,
                    selected_models,
                    int(eval_horizon),
                    int(forecast_horizon),
                    int(effective_season),
                    int(effective_season_lag),
                )

                df_norm = df_norm.copy()
                df_norm["region"] = region
                all_norm_parts.append(df_norm)

                fc = fc.copy()
                fc["region"] = region
                fc["best_model"] = best
                all_fc_parts.append(fc)

                metrics_by_region[region] = metrics
                best_model_by_region[region] = best
                progress.progress(int(100 * idx / total))

            all_norm = pd.concat(all_norm_parts, ignore_index=True)
            all_fc = pd.concat(all_fc_parts, ignore_index=True)

            save_csv(jp / "data_normalized.csv", all_norm)
            save_csv(jp / "forecast.csv", all_fc)
            write_json(jp / "metrics.json", metrics_by_region)
            write_json(jp / "best_models.json", best_model_by_region)

            artifacts = {
                "data_normalized_csv": str(jp / "data_normalized.csv"),
                "metrics_json": str(jp / "metrics.json"),
                "forecast_csv": str(jp / "forecast.csv"),
                "best_models_json": str(jp / "best_models.json"),
            }

            dominant_model = pd.Series(best_model_by_region).value_counts().idxmax()
            update_job(
                job_id,
                status="SUCCEEDED",
                best_model=dominant_model,
                metrics_json=json.dumps(metrics_by_region),
                artifacts_json=json.dumps(artifacts),
                finished_at=dt.datetime.now().isoformat(),
                error_message=None,
            )

            st.session_state["latest_forecast_state"] = {
                "job_id": job_id,
                "all_norm": all_norm,
                "all_fc": all_fc,
                "metrics_by_region": metrics_by_region,
                "best_model_by_region": best_model_by_region,
                "artifacts": artifacts,
            }

            status.update(
                label=f"Done in {time.time() - t0:.1f}s — Forecasts built for {len(regions)} regions",
                state="complete",
            )
            st.toast("Country-wide forecast complete", icon="✅")

    map_col, view_col = st.columns([1.0, 1.2], gap="small")

    with map_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Peru regions")
        st.caption("Click a region shape to highlight it and update plots.")
        map_event = plot_peru_map(regions, shapes_geojson, st.session_state["selected_region"], key="peru_map_forecast")
        clicked_region = extract_clicked_region(regions, map_event)
        if clicked_region and clicked_region in regions and clicked_region != st.session_state["selected_region"]:
            st.session_state["selected_region"] = clicked_region
            st.rerun()

        selected_region = st.selectbox(
            "Selected region",
            regions,
            index=regions.index(st.session_state["selected_region"]),
        )
        st.session_state["selected_region"] = selected_region
        st.markdown("</div>", unsafe_allow_html=True)

    with view_col:
        selected_region = st.session_state["selected_region"]
        latest = st.session_state.get("latest_forecast_state")
        region_hist = region_series(sales_df, selected_region)

        if latest and isinstance(latest, dict):
            all_norm = latest["all_norm"]
            all_fc = latest["all_fc"]
            metrics_by_region = latest["metrics_by_region"]
            best_model_by_region = latest["best_model_by_region"]
            artifacts = latest["artifacts"]
            job_id = latest["job_id"]

            hist = all_norm[all_norm["region"] == selected_region][["ds", "y"]].sort_values("ds")
            fc = all_fc[all_fc["region"] == selected_region][["ds", "yhat", "best_model"]].sort_values("ds")
            hist_trim = trim_history_for_forecast(hist, fc[["ds", "yhat"]], history_weeks=26)
            region_metrics = metrics_by_region[selected_region]
            region_best = best_model_by_region[selected_region]

            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader(f"{selected_region}: history + forecast")
            m1, m2, m3 = st.columns(3)
            m1.metric("Best model", region_best)
            m2.metric("MAPE", f"{region_metrics[region_best]['MAPE']:.2f}%")
            m3.metric("RMSE", f"{region_metrics[region_best]['RMSE']:.2f}")
            plot_ts(
                hist_trim,
                fc=fc[["ds", "yhat"]],
                title=f"{selected_region}: last 6 months history + forecast",
                key=f"chart_result_{job_id}_{selected_region}",
            )
            with st.expander("Model metrics"):
                st.dataframe(
                    pd.DataFrame(region_metrics).T.reset_index().rename(columns={"index": "Model"}),
                    use_container_width=True,
                    hide_index=True,
                )
            st.download_button(
                "Download country forecast.csv",
                data=Path(artifacts["forecast_csv"]).read_bytes(),
                file_name=f"forecast_peru_regions_{job_id}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader(f"Weekly sales history ({selected_region})")
            plot_ts(region_hist, title=f"Weekly sales history ({selected_region})", key=f"chart_hist_{selected_region}")
            with st.expander("Show last 30 rows"):
                st.dataframe(region_hist.tail(30), use_container_width=True)
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
    st.write(f"**Status:** {status}  |  **Created:** {created_at}  |  **Dominant model:** {best_model or '-'}")
    if error_message:
        st.error(error_message)

    if status != "SUCCEEDED" or not artifacts_json:
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    artifacts = json.loads(artifacts_json)
    metrics = json.loads(metrics_json)

    df_norm_all = pd.read_csv(artifacts["data_normalized_csv"])
    df_norm_all["ds"] = pd.to_datetime(df_norm_all["ds"])
    fc_all = pd.read_csv(artifacts["forecast_csv"])
    fc_all["ds"] = pd.to_datetime(fc_all["ds"])

    is_regional = "region" in df_norm_all.columns and "region" in fc_all.columns
    if not is_regional:
        st.info("This run predates regional Peru forecasts.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    hist_regions = sorted(df_norm_all["region"].dropna().astype(str).unique())
    run_region_key = f"run_region_{job_id}"
    if run_region_key not in st.session_state or st.session_state[run_region_key] not in hist_regions:
        st.session_state[run_region_key] = "Lima" if "Lima" in hist_regions else hist_regions[0]

    map_col, view_col = st.columns([1.0, 1.2], gap="small")
    with map_col:
        st.caption("Click a region shape to update the run plots.")
        map_event = plot_peru_map(
            hist_regions,
            shapes_geojson,
            st.session_state[run_region_key],
            key=f"peru_map_run_{job_id}",
        )
        clicked_region = extract_clicked_region(hist_regions, map_event)
        if clicked_region and clicked_region in hist_regions and clicked_region != st.session_state[run_region_key]:
            st.session_state[run_region_key] = clicked_region
            st.rerun()

        chosen = st.selectbox(
            "Region",
            hist_regions,
            index=hist_regions.index(st.session_state[run_region_key]),
            key=f"run_region_select_{job_id}",
        )
        st.session_state[run_region_key] = chosen

    with view_col:
        region_metrics = metrics.get(chosen, {})
        if region_metrics:
            st.subheader(f"Model metrics ({chosen})")
            st.dataframe(
                pd.DataFrame(region_metrics).T.reset_index().rename(columns={"index": "Model"}),
                use_container_width=True,
                hide_index=True,
            )

        hist = df_norm_all[df_norm_all["region"] == chosen][["ds", "y"]].sort_values("ds")
        fc = fc_all[fc_all["region"] == chosen].copy().sort_values("ds")
        fc_view = fc[["ds", "yhat"]]
        hist_trim = trim_history_for_forecast(hist, fc_view, history_weeks=26)

        best_for_region = "-"
        if "best_model" in fc.columns and not fc.empty:
            best_for_region = str(fc["best_model"].iloc[0])

        plot_ts(
            hist_trim,
            fc=fc_view,
            title=f"{chosen}: last 6 months history + forecast • best={best_for_region}",
            key=f"chart_history_{job_id}_{chosen}",
        )
    st.markdown("</div>", unsafe_allow_html=True)
