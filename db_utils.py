"""
db_utils.py
SQLite storage for CampusGPT:
- activity table      -> powers the dashboard (uploads, questions, quizzes, flashcards)
- chat_history table  -> keeps the chatbot conversation, so it survives app restarts
"""

import json
import sqlite3
from datetime import datetime

DB_PATH = "database.db"


def _conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = _conn()
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            detail TEXT,
            timestamp TEXT NOT NULL
        )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources TEXT,
            confidence REAL,
            abstained INTEGER DEFAULT 0,
            reason TEXT,
            timestamp TEXT NOT NULL
        )"""
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------- dashboard
def log_event(event_type, detail=""):
    conn = _conn()
    conn.execute(
        "INSERT INTO activity (event_type, detail, timestamp) VALUES (?, ?, ?)",
        (event_type, detail, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_stats():
    conn = _conn()
    c = conn.cursor()
    stats = {}
    for event_type in ("upload", "question", "quiz", "flashcard"):
        c.execute("SELECT COUNT(*) FROM activity WHERE event_type=?", (event_type,))
        stats[event_type] = c.fetchone()[0]
    conn.close()
    return stats


def get_recent_uploads(limit=5):
    conn = _conn()
    c = conn.cursor()
    c.execute(
        "SELECT detail, timestamp FROM activity WHERE event_type='upload' ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = c.fetchall()
    conn.close()
    return rows


# ------------------------------------------------------------- chat history
def save_chat(question, answer, sources=None, confidence=None, abstained=False, reason=""):
    conn = _conn()
    conn.execute(
        """INSERT INTO chat_history
           (question, answer, sources, confidence, abstained, reason, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            question,
            answer,
            json.dumps(sources or []),
            confidence,
            1 if abstained else 0,
            reason,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def load_chat(limit=50):
    """Return the most recent chat entries, oldest first."""
    conn = _conn()
    c = conn.cursor()
    c.execute(
        """SELECT question, answer, sources, confidence, abstained, reason
           FROM chat_history ORDER BY id DESC LIMIT ?""",
        (limit,),
    )
    rows = c.fetchall()
    conn.close()
    entries = []
    for q, a, s, conf, ab, reason in reversed(rows):
        entries.append(
            {
                "question": q,
                "answer": a,
                "sources": json.loads(s) if s else [],
                "confidence": conf,
                "abstained": bool(ab),
                "reason": reason or "",
                "sentences": [],
            }
        )
    return entries


def clear_chat():
    conn = _conn()
    conn.execute("DELETE FROM chat_history")
    conn.commit()
    conn.close()
