import os
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.db import add_feedback, feedback_count, init_db, load_all_songs, load_feedback, song_count
from app.embeddings import extract_embedding
from app.search import brute_force, dtw_search, faiss_search, hnswlib_search

# In-memory store of recent query embeddings keyed by UUID.
# Lets /feedback retrieve the embedding without re-uploading the file.
_pending: dict[str, np.ndarray] = {}
_MAX_PENDING = 50  # keep at most this many; drop oldest on overflow

app = FastAPI(title="HummingBird")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.on_event("startup")
async def startup() -> None:
    init_db()


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse(Path("app/static/index.html").read_text())


@app.get("/status")
async def status() -> dict:
    return {
        "songs_indexed": song_count(),
        "feedback_count": feedback_count(),
        "hnswlib_ready": hnswlib_search.index_exists(),
        "faiss_ready": faiss_search.indices_exist(),
    }


@app.get("/songs")
async def list_songs() -> list[dict]:
    songs = load_all_songs()
    return [{"id": s["id"], "title": s["title"], "artist": s["artist"]} for s in songs]


@app.post("/search")
async def search_hum(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        query_emb = extract_embedding(tmp_path)
        # Extract pitch sequence while file still exists (deleted in finally block)
        try:
            query_seq = dtw_search.extract_chroma_sequence(tmp_path)
        except Exception:
            query_seq = np.zeros((12, 0), dtype=np.float32)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {exc}") from exc
    finally:
        os.unlink(tmp_path)

    songs = load_all_songs()
    if not songs:
        raise HTTPException(
            status_code=503,
            detail="No songs indexed yet. Run: python scripts/ingest.py",
        )

    if not hnswlib_search.index_exists() or not faiss_search.indices_exist():
        raise HTTPException(
            status_code=503,
            detail="Indices not built. Run: python scripts/ingest.py",
        )

    # Store embedding so /feedback can retrieve it by ID without re-upload
    query_id = str(uuid.uuid4())
    _pending[query_id] = query_emb
    if len(_pending) > _MAX_PENDING:
        _pending.pop(next(iter(_pending)))

    feedback = load_feedback()

    return {
        "query_id": query_id,
        "n_songs": len(songs),
        "brute_force": brute_force.search(query_emb, songs, feedback=feedback),
        "hnswlib": hnswlib_search.search(query_emb),
        "faiss_flat": faiss_search.search(query_emb, variant="flat"),
        "faiss_ivf": faiss_search.search(query_emb, variant="ivf"),
        "faiss_hnsw": faiss_search.search(query_emb, variant="hnsw"),
        "dtw": dtw_search.search(query_seq) if (dtw_search.index_exists() and query_seq.shape[1] > 0) else None,
    }


class FeedbackIn(BaseModel):
    query_id: str
    song_id: int


@app.post("/feedback")
async def submit_feedback(body: FeedbackIn) -> dict:
    emb = _pending.pop(body.query_id, None)
    if emb is None:
        raise HTTPException(
            status_code=404,
            detail="Query not found — please search again before submitting feedback",
        )

    songs = load_all_songs()
    song = next((s for s in songs if s["id"] == body.song_id), None)
    if song is None:
        raise HTTPException(status_code=404, detail="Song not found")

    # Persist to DB (survives server restart — loaded into brute force on every search)
    add_feedback(body.song_id, emb)

    # Insert into live ANN indices so they benefit immediately without reindexing.
    # On restart the indices are reloaded from disk — which now includes this embedding.
    hnswlib_search.add_embedding(emb, song)
    faiss_search.add_embedding(emb, song)

    return {"ok": True, "feedback_total": feedback_count()}
