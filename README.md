# Forecasting MVP

A Streamlit-based demand forecasting MVP that follows the attached design brief. It includes:

- Email gate before showing the full dashboard
- Sample synthetic weekly dataset with 3 years of history
- Option to upload your own CSV/XLSX dataset with `date` and `demand` columns
- Forecasting models: ETS, ARIMA, Prophet, and XGBoost
- Walk-forward cross validation on the training set for model tuning
- Test-set evaluation using MAE, MAPE, and RMSE
- Professional dashboard layout with data preview, model comparison, and best-model chart

## Project structure

```text
forecasting_mvp/
├── app.py
├── data/
│   └── sample_weekly_demand.csv
├── src/
│   ├── data_utils.py
│   └── forecasting.py
├── .streamlit/
│   └── config.toml
├── environment.yml
└── requirements.txt
```

## Run locally

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate forecasting-mvp
streamlit run app.py
```

### Option 2: pip

```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
pip install -r requirements.txt
streamlit run app.py
```

## Input data format

Your upload should include:

| column | description |
|---|---|
| `date` | Weekly date column |
| `demand` | Weekly demand values |

The app will sort the data, infer a weekly frequency, and interpolate missing weekly points.

## Notes

- Default forecast horizon is 6 weeks, but the user can choose another value.
- The test set size matches the forecast horizon.
- Cross validation uses expanding-window walk-forward validation.
- Prophet and XGBoost run only if their dependencies install successfully.
