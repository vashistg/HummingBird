"""
hnswlib — the reference HNSW implementation by the algorithm's original authors
(Malkov & Yashunin, 2018: "Efficient and Robust Approximate Nearest Neighbor Search
Using Hierarchical Navigable Small World Graphs").

How HNSW works:
  Build: each new vector is connected to M nearest neighbors at each layer.
         Lower layers are denser; the top layer has very few nodes.
         Think of it as a "skip list" in high-dimensional space.
  Query: start at a random node in the top (sparse) layer, greedily walk toward
         the query, then descend into denser layers for refinement.
         Explored neighbors at query time is controlled by ef (ef_search).

Why two HNSW implementations here?
  hnswlib  — standalone, ~300 LOC C++ core, easy to inspect and understand.
             Has online insertion (add items after build).
  FAISS HNSW — same algorithm, but integrated with FAISS's memory layout,
               SIMD kernels, and quantization options. Faster at large scale.

Key parameters:
  M               — connections per node per layer. Higher = better recall, more memory.
  ef_construction — neighbors explored during build. Higher = better graph, slower build.
  ef_search       — neighbors explored at query time. Runtime quality/speed trade-off.

Online insertion and feedback:
  hnswlib supports add_items() at any time after build. Feedback hums are inserted as
  new nodes pointing to the correct song. The graph grows: the new node connects to its
  M nearest existing neighbors, so future queries can reach the correct song via this
  new "bridge" node. Multiple entries can map to the same song_id — search deduplicates
  by song_id and keeps the highest-scoring entry.
"""

import time
from pathlib import Path

import hnswlib
import numpy as np

DIM = 104
M = 16
EF_CONSTRUCTION = 200
EF_SEARCH = 50
FEEDBACK_SLOTS = 500  # extra capacity reserved at build time for feedback insertions

INDEX_PATH = Path("data/indices/hnswlib.bin")
META_PATH = Path("data/indices/hnswlib_meta.npy")

_index: hnswlib.Index | None = None
_meta: list | None = None
_mtime: float = 0.0


def build_index(songs: list[dict]) -> None:
    global _index, _meta
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    n = len(songs)
    embeddings = np.stack([s["embedding"] for s in songs]).astype(np.float32)
    meta = [{"id": s["id"], "title": s["title"], "artist": s["artist"]} for s in songs]

    # space='cosine': hnswlib computes 1 - cosine_similarity
    # On L2-normalized vectors, cosine distance = (2 - 2 * dot) / 2 = 1 - dot
    index = hnswlib.Index(space="cosine", dim=DIM)
    # Reserve FEEDBACK_SLOTS extra capacity so add_embedding() never needs a resize
    # for normal usage. resize_index() is available if we ever exceed this.
    index.init_index(max_elements=n + FEEDBACK_SLOTS, ef_construction=EF_CONSTRUCTION, M=M)
    index.add_items(embeddings, list(range(n)))
    index.set_ef(EF_SEARCH)
    index.save_index(str(INDEX_PATH))

    np.save(str(META_PATH), meta)
    _index = index
    _meta = meta
    print(
        f"  hnswlib HNSW: M={M}, ef_construction={EF_CONSTRUCTION}, "
        f"ef_search={EF_SEARCH}, {n} songs → {INDEX_PATH}"
    )


def _load() -> tuple[hnswlib.Index, list]:
    global _index, _meta, _mtime
    current_mtime = META_PATH.stat().st_mtime if META_PATH.exists() else 0.0
    if _index is None or current_mtime != _mtime:
        _index = hnswlib.Index(space="cosine", dim=DIM)
        _index.load_index(str(INDEX_PATH), max_elements=_index.max_elements if _index else 0)
        _index.set_ef(EF_SEARCH)
        _meta = np.load(str(META_PATH), allow_pickle=True).tolist()
        _mtime = current_mtime
    return _index, _meta


def index_exists() -> bool:
    return INDEX_PATH.exists() and META_PATH.exists()


def add_embedding(embedding: np.ndarray, song_meta: dict) -> None:
    """
    Insert one feedback hum into the live index without rebuilding.

    How online insertion works in HNSW:
      The new node is assigned a random layer (usually layer 0, occasionally higher).
      At each layer it finds its M nearest existing neighbors and wires bidirectional
      edges to them. The graph grows by one node, and those neighbors now have a
      shortcut to this new region of the space.

    Effect on future queries:
      A query similar to this hum will reach this node during graph traversal and
      follow the edge to the correct song — even if the original song embedding
      was far from the query.
    """
    index, meta = _load()

    new_label = len(meta)
    if new_label >= index.max_elements:
        index.resize_index(index.max_elements + FEEDBACK_SLOTS)

    index.add_items(embedding.reshape(1, -1).astype(np.float32), [new_label])
    meta.append({"id": song_meta["id"], "title": song_meta["title"], "artist": song_meta["artist"]})

    index.save_index(str(INDEX_PATH))
    np.save(str(META_PATH), meta)

    # Update cached mtime so _load() doesn't reload from disk on next call
    global _mtime
    _mtime = META_PATH.stat().st_mtime


def search(query_embedding: np.ndarray, top_k: int = 5) -> dict:
    index, meta = _load()
    q = query_embedding.reshape(1, -1).astype(np.float32)

    # Fetch more candidates than top_k because multiple entries can map to the
    # same song_id (original embedding + feedback hums). We dedup below.
    k_fetch = min(top_k * 4, index.element_count)

    start = time.perf_counter()
    labels, distances = index.knn_query(q, k=k_fetch)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Deduplicate by song_id, keeping the highest-scoring entry per song
    seen: dict[int, dict] = {}
    for idx, dist in zip(labels[0], distances[0]):
        cos_sim = max(0.0, 1.0 - float(dist))
        song_id = meta[idx]["id"]
        if song_id not in seen or cos_sim > seen[song_id]["score"]:
            seen[song_id] = {
                "id": song_id,
                "title": meta[idx]["title"],
                "artist": meta[idx]["artist"],
                "score": round(cos_sim, 6),
            }

    results = sorted(seen.values(), key=lambda r: -r["score"])[:top_k]
    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    return {
        "results": results,
        "time_ms": round(elapsed_ms, 4),
        "method": "hnswlib",
        "params": {"M": M, "ef_search": EF_SEARCH},
    }
