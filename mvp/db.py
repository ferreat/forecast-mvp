import sqlite3
from pathlib import Path
from mvp.config import SETTINGS

SCHEMA = '''
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email TEXT UNIQUE NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  status TEXT NOT NULL,
  dataset_path TEXT NOT NULL,
  params_json TEXT NOT NULL,
  best_model TEXT,
  metrics_json TEXT,
  artifacts_json TEXT,
  error_message TEXT,
  FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS usage (
  user_id INTEGER NOT NULL,
  month_yyyymm TEXT NOT NULL,
  runs_used INTEGER NOT NULL,
  PRIMARY KEY(user_id, month_yyyymm),
  FOREIGN KEY(user_id) REFERENCES users(id)
);
'''

def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or SETTINGS.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()
