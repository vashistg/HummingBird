import io
import sqlite3
from pathlib import Path

import numpy as np

DB_PATH = Path("data/songs.db")


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS songs (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                title    TEXT    NOT NULL,
                artist   TEXT,
                filename TEXT    NOT NULL UNIQUE,
                embedding BLOB   NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                song_id        INTEGER NOT NULL REFERENCES songs(id),
                query_embedding BLOB   NOT NULL,
                created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def song_exists(filename: str) -> bool:
    with _conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM songs WHERE filename = ?", (filename,)
        ).fetchone()
    return row is not None


def insert_song(title: str, artist: str, filename: str, embedding: np.ndarray) -> int:
    buf = io.BytesIO()
    np.save(buf, embedding)
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO songs (title, artist, filename, embedding) VALUES (?, ?, ?, ?)
               ON CONFLICT(filename) DO UPDATE SET
                 title=excluded.title, artist=excluded.artist, embedding=excluded.embedding""",
            (title, artist, filename, buf.getvalue()),
        )
        conn.commit()
        return cur.lastrowid


def load_all_songs() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, title, artist, filename, embedding FROM songs ORDER BY id"
        ).fetchall()
    result = []
    for id_, title, artist, filename, blob in rows:
        embedding = np.load(io.BytesIO(blob))
        result.append(
            {"id": id_, "title": title, "artist": artist, "filename": filename, "embedding": embedding}
        )
    return result


def song_count() -> int:
    with _conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]


def add_feedback(song_id: int, query_embedding: np.ndarray) -> None:
    buf = io.BytesIO()
    np.save(buf, query_embedding)
    with _conn() as conn:
        conn.execute(
            "INSERT INTO feedback (song_id, query_embedding) VALUES (?, ?)",
            (song_id, buf.getvalue()),
        )
        conn.commit()


def load_feedback() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            """SELECT f.song_id, s.title, s.artist, f.query_embedding
               FROM feedback f JOIN songs s ON s.id = f.song_id"""
        ).fetchall()
    result = []
    for song_id, title, artist, blob in rows:
        result.append({
            "song_id": song_id,
            "title": title,
            "artist": artist,
            "embedding": np.load(io.BytesIO(blob)),
        })
    return result


def feedback_count() -> int:
    with _conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
