"""
SQLite database for user accounts and document metadata.
Vector data lives in Weaviate; only lightweight metadata lives here.
"""
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = "./pagemind.db"


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    title        TEXT,
    url          TEXT,
    source_type  TEXT NOT NULL DEFAULT 'webpage',
    content_type TEXT NOT NULL DEFAULT 'general',
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);
"""

# Migration: add content_type column to existing databases that lack it
_MIGRATIONS = [
    "ALTER TABLE documents ADD COLUMN content_type TEXT NOT NULL DEFAULT 'general'",
]


def init_db():
    with _conn() as conn:
        conn.executescript(_SCHEMA)
        # Apply migrations gracefully (ignore "duplicate column" errors)
        for sql in _MIGRATIONS:
            try:
                conn.execute(sql)
                conn.commit()
            except Exception:
                pass


# ── Connection helper ─────────────────────────────────────────────────────────

@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── User operations ───────────────────────────────────────────────────────────

def create_user(username: str, password_hash: str) -> str:
    user_id = str(uuid.uuid4())
    with _conn() as conn:
        conn.execute(
            "INSERT INTO users (id, username, password_hash, created_at) VALUES (?,?,?,?)",
            (user_id, username, password_hash, _now()),
        )
    return user_id


def get_user_by_username(username: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return dict(row) if row else None


# ── Document operations ───────────────────────────────────────────────────────

def upsert_document(
    doc_id: str, user_id: str, title: str, url: str,
    source_type: str, chunk_count: int, content_type: str = "general",
) -> None:
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO documents
                (doc_id, user_id, title, url, source_type, content_type, chunk_count, created_at)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(doc_id) DO UPDATE SET
                title        = excluded.title,
                content_type = excluded.content_type,
                chunk_count  = excluded.chunk_count,
                created_at   = excluded.created_at
            """,
            (doc_id, user_id, title, url, source_type, content_type, chunk_count, _now()),
        )


def get_user_documents(user_id: str) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_document(doc_id: str, user_id: str) -> bool:
    with _conn() as conn:
        cur = conn.execute(
            "DELETE FROM documents WHERE doc_id = ? AND user_id = ?",
            (doc_id, user_id),
        )
    return cur.rowcount > 0
