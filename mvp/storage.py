from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from mvp.config import SETTINGS

def ensure_dirs() -> None:
    SETTINGS.data_dir.mkdir(parents=True, exist_ok=True)
    (SETTINGS.data_dir / "jobs").mkdir(parents=True, exist_ok=True)
    (SETTINGS.data_dir / "uploads").mkdir(parents=True, exist_ok=True)

def job_dir(job_id: str) -> Path:
    d = SETTINGS.data_dir / "jobs" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_upload(csv_bytes: bytes, filename: str, job_id: str) -> Path:
    ensure_dirs()
    uploads = SETTINGS.data_dir / "uploads"
    path = uploads / f"{job_id}__{Path(filename).name}"
    path.write_bytes(csv_bytes)
    return path

def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

def save_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)
