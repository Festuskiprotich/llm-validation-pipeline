import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/metrics.db")
SCHEMA = """
CREATE TABLE IF NOT EXISTS validations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    keyword_score REAL NOT NULL,
    judge_score REAL NOT NULL,
    judge_reasoning TEXT,
    answer_latency REAL NOT NULL,
    judge_latency REAL NOT NULL,
    total_latency REAL NOT NULL
);
"""

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(SCHEMA)
        conn.commit()

def log_validation(question: str, answer: str, keyword_score: float,
                   judge_score: float, judge_reasoning: str,
                   answer_latency: float, judge_latency: float) -> int:
    init_db()
    timestamp = datetime.now(timezone.utc).isoformat()
    total_latency = answer_latency + judge_latency
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO validations
            (timestamp, question, answer, keyword_score, judge_score,
             judge_reasoning, answer_latency, judge_latency, total_latency)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, question, answer, keyword_score, judge_score,
             judge_reasoning, answer_latency, judge_latency, total_latency),
        )
        conn.commit()
        return cursor.lastrowid

def get_stats() -> dict:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("""
            SELECT COUNT(*), AVG(keyword_score), AVG(judge_score), 
                   AVG(total_latency), SUM(CASE WHEN judge_score < 0.5 THEN 1 ELSE 0 END)
            FROM validations
        """).fetchone()
    if not row or row[0] == 0:
        return {"total": 0}
    return {
        "total": row[0], "avg_keyword_score": round(row[1], 2),
        "avg_judge_score": round(row[2], 2), "avg_latency_seconds": round(row[3], 2),
        "hallucinations_caught": row[4],
    }
