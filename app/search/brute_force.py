"""
Brute-force cosine similarity — O(n).

For every query we compute dot(query, song_i) for ALL n songs.
Time grows linearly with the library size — this is why ANN exists.

With L2-normalized vectors:
  cosine_similarity(a, b) = dot(a, b)
  The matrix multiply  E @ q  computes all n dot products at once,
  but the wall-clock time still scales as O(n * dim).
"""

import time

import numpy as np


def search(
    query_embedding: np.ndarray,
    songs: list[dict],
    top_k: int = 5,
    feedback: list[dict] | None = None,
) -> dict:
    if not songs:
        return {"results": [], "time_ms": 0.0, "method": "brute_force", "n_songs": 0}

    embeddings = np.stack([s["embedding"] for s in songs])

    start = time.perf_counter()
    # THE O(n) LINE: one dot product per song — no shortcuts
    scores = embeddings @ query_embedding   # shape: (n,)

    # Feedback boost: if a stored hum example scores higher than the song's
    # own embedding, use that score instead. This lets brute force improve
    # over time without retraining — ANN methods won't benefit until reindexed.
    if feedback:
        song_id_to_idx = {s["id"]: i for i, s in enumerate(songs)}
        for fb in feedback:
            idx = song_id_to_idx.get(fb["song_id"])
            if idx is None:
                continue
            fb_score = float(np.dot(fb["embedding"], query_embedding))
            if fb_score > scores[idx]:
                scores[idx] = fb_score

    top_indices = np.argsort(scores)[::-1][:top_k]
    elapsed_ms = (time.perf_counter() - start) * 1000

    results = [
        {
            "rank": rank + 1,
            "id": songs[i]["id"],
            "title": songs[i]["title"],
            "artist": songs[i]["artist"],
            "score": round(float(scores[i]), 6),
        }
        for rank, i in enumerate(top_indices)
    ]
    return {
        "results": results,
        "time_ms": round(elapsed_ms, 4),
        "method": "brute_force",
        "n_songs": len(songs),
    }
