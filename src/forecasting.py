from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
except Exception:  # pragma: no cover
    Prophet = None

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


@dataclass
class ModelResult:
    name: str
    best_params: Dict[str, Any]
    metrics: Dict[str, float]
    forecast: pd.DataFrame
    cv_score: float
    fitted_model: Any = None


class ForecastingEngine:
    def __init__(self, horizon: int = 6, cv_splits: int = 4):
        self.horizon = horizon
        self.cv_splits = cv_splits
        self.lags = [1, 2, 3, 4, 5, 6, 52]

    def run(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, ModelResult]]:
        train_series = train_df.set_index("date")["demand"]
        test_series = test_df.set_index("date")["demand"]

        results: Dict[str, ModelResult] = {}
        for name, param_grid in self._model_grids().items():
            try:
                best_params, cv_score = self._select_best_params(name, train_series, param_grid)
                forecast_df, fitted_model = self._fit_and_forecast(name, train_series, test_series.index, best_params)
                metrics = calculate_metrics(test_series.values, forecast_df["forecast"].values)
                results[name] = ModelResult(
                    name=name,
                    best_params=best_params,
                    metrics=metrics,
                    forecast=forecast_df,
                    cv_score=cv_score,
                    fitted_model=fitted_model,
                )
            except Exception as exc:
                results[name] = ModelResult(
                    name=name,
                    best_params={"error": str(exc)},
                    metrics={"MAE": np.nan, "MAPE": np.nan, "RMSE": np.nan},
                    forecast=pd.DataFrame({"date": test_series.index, "forecast": np.nan}),
                    cv_score=np.nan,
                    fitted_model=None,
                )

        summary = pd.DataFrame(
            [
                {
                    "Model": res.name,
                    "CV RMSE": res.cv_score,
                    "MAE": res.metrics["MAE"],
                    "MAPE": res.metrics["MAPE"],
                    "RMSE": res.metrics["RMSE"],
                    "Best Parameters": ", ".join(f"{k}={v}" for k, v in res.best_params.items()),
                }
                for res in results.values()
            ]
        ).sort_values("RMSE", na_position="last")

        if not summary.empty:
            summary["Best Model"] = ""
            best_idx = summary["RMSE"].idxmin()
            if pd.notna(best_idx):
                summary.loc[best_idx, "Best Model"] = "★"

        return summary, results

    def _model_grids(self) -> Dict[str, List[Dict[str, Any]]]:
        grids: Dict[str, List[Dict[str, Any]]] = {
            "ETS": [
                {"trend": trend, "seasonal": seasonal, "seasonal_periods": 52 if seasonal else None}
                for trend, seasonal in itertools.product(["add", None], ["add", None])
                if not (trend is None and seasonal is None)
            ],
            "ARIMA": [
                {"order": order}
                for order in [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2)]
            ],
            "Prophet": [
                {"changepoint_prior_scale": x, "seasonality_mode": mode}
                for x, mode in itertools.product([0.05, 0.2, 0.5], ["additive", "multiplicative"])
            ]
            if Prophet is not None
            else [],
            "XGBoost": [
                {"n_estimators": n, "max_depth": d, "learning_rate": lr}
                for n, d, lr in itertools.product([200, 350], [3, 5], [0.03, 0.1])
            ]
            if XGBRegressor is not None
            else [],
        }
        return grids

    def _select_best_params(self, model_name: str, train_series: pd.Series, param_grid: List[Dict[str, Any]]):
        if not param_grid:
            raise ValueError(f"{model_name} dependency is not installed.")

        scores: List[Tuple[float, Dict[str, Any]]] = []
        for params in param_grid:
            score = self._walk_forward_cv(model_name, train_series, params)
            scores.append((score, params))

        scores = [(score, params) for score, params in scores if np.isfinite(score)]
        if not scores:
            raise ValueError(f"{model_name} failed during cross validation.")
        scores.sort(key=lambda x: x[0])
        return scores[0][1], float(scores[0][0])

    def _walk_forward_cv(self, model_name: str, series: pd.Series, params: Dict[str, Any]) -> float:
        total_needed = self.horizon * (self.cv_splits + 1)
        if len(series) <= total_needed:
            raise ValueError(
                f"Need more history for {self.cv_splits} CV splits and horizon {self.horizon}."
            )

        errors = []
        first_train_end = len(series) - self.horizon * self.cv_splits
        for split in range(self.cv_splits):
            train_end = first_train_end + split * self.horizon
            train_y = series.iloc[:train_end]
            val_y = series.iloc[train_end : train_end + self.horizon]
            pred = self._forecast_array(model_name, train_y, len(val_y), params)
            errors.append(np.sqrt(mean_squared_error(val_y.values, pred)))
        return float(np.mean(errors))

    def _fit_and_forecast(self, model_name: str, train_series: pd.Series, future_index: pd.Index, params: Dict[str, Any]):
        preds = self._forecast_array(model_name, train_series, len(future_index), params)
        forecast_df = pd.DataFrame({"date": future_index, "forecast": preds})
        return forecast_df, {"model": model_name, "params": params}

    def _forecast_array(self, model_name: str, train_series: pd.Series, horizon: int, params: Dict[str, Any]) -> np.ndarray:
        if model_name == "ETS":
            model = ExponentialSmoothing(
                train_series,
                trend=params.get("trend"),
                seasonal=params.get("seasonal"),
                seasonal_periods=params.get("seasonal_periods"),
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True)
            pred = fit.forecast(horizon)
            return np.asarray(pred)

        if model_name == "ARIMA":
            model = SARIMAX(
                train_series,
                order=params["order"],
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False)
            pred = fit.forecast(horizon)
            return np.asarray(pred)

        if model_name == "Prophet":
            prophet_df = pd.DataFrame({"ds": train_series.index, "y": train_series.values})
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=params["changepoint_prior_scale"],
                seasonality_mode=params["seasonality_mode"],
            )
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=horizon, freq="W")
            pred = model.predict(future).tail(horizon)["yhat"].to_numpy()
            return pred

        if model_name == "XGBoost":
            return self._xgb_recursive_forecast(train_series, horizon, params)

        raise ValueError(f"Unsupported model: {model_name}")

    def _build_supervised(self, series: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({"y": series.values}, index=series.index)
        for lag in self.lags:
            df[f"lag_{lag}"] = df["y"].shift(lag)
        df["weekofyear"] = df.index.isocalendar().week.astype(int)
        df["month"] = df.index.month
        df["trend_idx"] = np.arange(len(df))
        return df.dropna()

    def _xgb_recursive_forecast(self, train_series: pd.Series, horizon: int, params: Dict[str, Any]) -> np.ndarray:
        if XGBRegressor is None:
            raise ValueError("xgboost is not installed.")

        supervised = self._build_supervised(train_series)
        X = supervised.drop(columns=["y"])
        y = supervised["y"]

        model = XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            objective="reg:squarederror",
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        model.fit(X, y)

        history = train_series.copy()
        preds = []
        for step in range(horizon):
            next_date = history.index[-1] + pd.Timedelta(weeks=1)
            row = {}
            for lag in self.lags:
                row[f"lag_{lag}"] = history.iloc[-lag] if len(history) >= lag else history.iloc[0]
            row["weekofyear"] = int(next_date.isocalendar().week)
            row["month"] = next_date.month
            row["trend_idx"] = len(history)
            pred = float(model.predict(pd.DataFrame([row]))[0])
            preds.append(pred)
            history.loc[next_date] = pred
        return np.asarray(preds)



def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    eps = 1e-8
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), eps))) * 100
    return {"MAE": float(mae), "MAPE": float(mape), "RMSE": float(rmse)}
