from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    app_name: str = "Forecast Studio (MVP)"
    data_dir: Path = Path("./data")
    db_path: Path = Path("./data/app.db")
    free_monthly_runs: int = 10
    default_forecast_horizon: int = 6
    default_eval_horizon: int = 26
    default_season_length: int = 52
    max_upload_mb: int = 50

SETTINGS = Settings()
