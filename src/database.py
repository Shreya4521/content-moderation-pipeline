"""
database.py
Handles the SQLite audit log — every moderation decision is stored here
so the pipeline has a traceable compliance history (who/what/when/why flagged).
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "moderation_log.db")


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS moderation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            text_snippet TEXT,
            ml_prediction TEXT,
            ml_confidence REAL,
            rule_flagged INTEGER,
            profanity_flagged INTEGER,
            pii_flagged INTEGER,
            spam_flagged INTEGER,
            final_decision TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_entry(text, ml_prediction, ml_confidence, rule_result, final_decision):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO moderation_log (
            timestamp, text_snippet, ml_prediction, ml_confidence,
            rule_flagged, profanity_flagged, pii_flagged, spam_flagged, final_decision
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        text[:200],
        ml_prediction,
        ml_confidence,
        int(rule_result["rule_flagged"]),
        int(rule_result["profanity"]["flagged"]),
        int(rule_result["pii"]["flagged"]),
        int(rule_result["spam"]["flagged"]),
        final_decision,
    ))
    conn.commit()
    conn.close()


def fetch_all_logs():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM moderation_log ORDER BY id DESC")
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    conn.close()
    return cols, rows
