import sqlite3
from pathlib import Path


class IsaacDB:
    @staticmethod
    def query(db_path: Path, sql: str, params=()):
        with sqlite3.connect(db_path) as conn:
            return conn.execute(sql, params).fetchall()

    @staticmethod
    def execute(db_path: Path, sql: str, params=()):
        with sqlite3.connect(db_path) as conn:
            conn.execute(sql, params)
            conn.commit()
