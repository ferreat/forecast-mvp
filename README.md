# Forecast Studio (MVP) — SaaS-style weekly forecasting (Offline)

## Models (no hyperparameter optimization)
- ETS (Exponential Smoothing)
- ARIMA (fixed order)
- Prophet (fixed robust settings + optional custom seasonality)
- XGBoost (lag features; seasonal lag defaults to 52, but can be auto-suggested/overridden)

## Key behaviors
- Built-in Peru dataset only: 3 years of weekly sales for all 25 Peruvian regions.
- Interactive Peru map in dashboard: click a region (for example, Lima) to inspect its series.
- When you run forecast, models are evaluated and forecasted for all regions.
- Region view updates dynamically: selecting another region shows that region's history + forecast.
- Forecast charts show only the previous 6 months of history plus the future horizon.

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
- Sales data: `sample_data/peru_weekly_sales_3y_by_region.csv`
- Region map points: `sample_data/peru_regions_meta.csv`
- Fields:
  - `region`
  - `week_start`
  - `sales`


### Fixes
- Added unique Streamlit keys to Plotly charts to avoid StreamlitDuplicateElementId.
