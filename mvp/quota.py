from __future__ import annotations
import datetime as dt
import sqlite3
from mvp.config import SETTINGS

def month_key(today: dt.date | None = None) -> str:
    d = today or dt.date.today()
    return f"{d.year:04d}{d.month:02d}"

def get_runs_used(conn: sqlite3.Connection, user_id: int, month: str) -> int:
    row = conn.execute(
        "SELECT runs_used FROM usage WHERE user_id=? AND month_yyyymm=?",
        (user_id, month),
    ).fetchone()
    return int(row[0]) if row else 0

def can_run(conn: sqlite3.Connection, user_id: int) -> tuple[bool, int, int]:
    mk = month_key()
    used = get_runs_used(conn, user_id, mk)
    limit_ = SETTINGS.free_monthly_runs
    return (used < limit_), used, limit_

def consume_run(conn: sqlite3.Connection, user_id: int) -> None:
    mk = month_key()
    used = get_runs_used(conn, user_id, mk)
    if used == 0:
        conn.execute(
            "INSERT OR REPLACE INTO usage(user_id, month_yyyymm, runs_used) VALUES(?,?,?)",
            (user_id, mk, 1),
        )
    else:
        conn.execute(
            "UPDATE usage SET runs_used=? WHERE user_id=? AND month_yyyymm=?",
            (used + 1, user_id, mk),
        )
    conn.commit()
