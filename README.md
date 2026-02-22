# Forecast Studio (MVP) — SaaS-style weekly forecasting (Offline)

## Models (no hyperparameter optimization)
- ETS (Exponential Smoothing)
- ARIMA (fixed order)
- Prophet (fixed robust settings + optional custom seasonality)
- XGBoost (lag features; seasonal lag defaults to 52, but can be auto-suggested/overridden)

## Key behaviors
- Weekly sample data included (up to ~5 years).
- In-app data source selector: upload CSV, built-in 5y sample, or built-in 3y sample (weekly).
- Default forecast horizon is **T+6** weeks (configurable).
- Seasonality defaults to **52** but can be challenged via autocorrelation suggestion.
- UI shows per-model MAE/RMSE/MAPE, picks best model, and plots historic + best forecast with metric in title.

## Install & run (Conda recommended)
```bash
cd forecast_mvp_saas_prophet
conda env create -f environment.yml
conda activate forecast-mvp
streamlit run app.py
```

## Install & run (pip)
```bash
cd forecast_mvp_saas_prophet
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
> If Prophet fails with pip, use Conda.

## Sample dataset
Built-in sample (5y) and built-in sample (3y) are selectable directly in the app.

If you want to upload manually, use: `sample_data/sample_weekly_sales_5y.csv`
Map:
- Date column: `week_start`
- Target column: `sales`


### Fixes
- Added unique Streamlit keys to Plotly charts to avoid StreamlitDuplicateElementId.
