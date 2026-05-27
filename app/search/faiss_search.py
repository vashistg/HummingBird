"""
FAISS — Facebook AI Similarity Search.

Three index types demonstrated here:

1. IndexFlatIP  (exact)
   Brute-force inner product. Same O(n) as our manual search but heavily SIMD-optimized.
   Baseline that proves the ANN indices give correct-ish results.

2. IndexIVFFlat  (ANN — inverted file)
   Splits the space into n_lists Voronoi cells (k-means clusters).
   At query time only nprobe cells are searched → sub-linear when nprobe << n_lists.
   Trade-off: lower recall if true neighbor is in an un-probed cell.

3. IndexHNSWFlat (ANN — hierarchical navigable small world graph)
   Builds a multi-layer graph where each node connects to M neighbors.
   Query = greedy graph traversal starting from a random entry point.
   Very fast queries, great recall, but higher memory and build time.

Online insertion and feedback:
  All three FAISS index types support add() after initial build.
  Flat: trivially appends to the vector list — still exact.
  IVF: assigns the new vector to its nearest Voronoi cell (no retraining needed).
       Recall may drop slightly if the new vector lands in a rarely-probed cell.
  HNSW: wires the new node into the graph just like hnswlib.
  Multiple entries can point to the same song_id — search deduplicates them.
"""

import time
from pathlib import Path

import faiss
import numpy as np

DIM = 104
FLAT_PATH = Path("data/indices/faiss_flat.index")
IVF_PATH = Path("data/indices/faiss_ivf.index")
HNSW_PATH = Path("data/indices/faiss_hnsw.index")
META_PATH = Path("data/indices/faiss_meta.npy")

_cache: dict = {}
_meta: list | None = None
_mtime: float = 0.0


def build_indices(songs: list[dict]) -> None:
    global _cache, _meta
    FLAT_PATH.parent.mkdir(parents=True, exist_ok=True)

    embeddings = np.stack([s["embedding"] for s in songs]).astype(np.float32)
    n = len(embeddings)
    meta = [{"id": s["id"], "title": s["title"], "artist": s["artist"]} for s in songs]

    # --- Flat (exact brute-force via FAISS) ---
    flat = faiss.IndexFlatIP(DIM)
    flat.add(embeddings)
    faiss.write_index(flat, str(FLAT_PATH))
    print(f"  FAISS Flat: indexed {n} songs → {FLAT_PATH}")

    # --- IVF (inverted file, approximate) ---
    # n_lists = Voronoi cells; FAISS needs ≥39 × n_lists training points
    # Rule of thumb: sqrt(n), but floor to n//39 for small datasets
    n_lists = max(1, min(int(np.sqrt(n)), n // 39 if n >= 39 else 1))
    quantizer = faiss.IndexFlatIP(DIM)
    ivf = faiss.IndexIVFFlat(quantizer, DIM, n_lists, faiss.METRIC_INNER_PRODUCT)
    ivf.train(embeddings)
    ivf.add(embeddings)
    ivf.nprobe = max(1, n_lists // 4)  # probe 25% of cells at query time
    faiss.write_index(ivf, str(IVF_PATH))
    print(f"  FAISS IVF: {n_lists} cells, nprobe={ivf.nprobe} → {IVF_PATH}")

    # --- HNSW (graph-based, approximate) ---
    # Using METRIC_INNER_PRODUCT so scores are cosine similarities (same as flat/ivf)
    hnsw = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    hnsw.add(embeddings)
    faiss.write_index(hnsw, str(HNSW_PATH))
    print(f"  FAISS HNSW: M=32, efConstruction=200 → {HNSW_PATH}")

    np.save(str(META_PATH), meta)
    _cache = {}
    _meta = None


def _load() -> tuple[dict, list]:
    global _cache, _meta, _mtime
    current_mtime = META_PATH.stat().st_mtime if META_PATH.exists() else 0.0
    if not _cache or current_mtime != _mtime:
        _cache["flat"] = faiss.read_index(str(FLAT_PATH))
        _cache["ivf"] = faiss.read_index(str(IVF_PATH))
        _cache["hnsw"] = faiss.read_index(str(HNSW_PATH))
        _meta = np.load(str(META_PATH), allow_pickle=True).tolist()
        _mtime = current_mtime
    return _cache, _meta


def indices_exist() -> bool:
    return all(p.exists() for p in [FLAT_PATH, IVF_PATH, HNSW_PATH, META_PATH])


def add_embedding(embedding: np.ndarray, song_meta: dict) -> None:
    """
    Insert one feedback hum into all three live FAISS indices without rebuilding.

    Flat: appends a new row to the vector matrix — still scans all rows at query time.
    IVF:  assigns the new vector to its nearest existing Voronoi cell (no retraining).
          If the cell is rarely probed (nprobe is low), this new entry might be missed —
          which is the IVF recall trade-off made visible by feedback.
    HNSW: wires the new node into the graph — same online insertion as hnswlib.
    """
    cache, meta = _load()
    e = embedding.reshape(1, -1).astype(np.float32)

    cache["flat"].add(e)
    cache["ivf"].add(e)
    cache["hnsw"].add(e)
    meta.append({"id": song_meta["id"], "title": song_meta["title"], "artist": song_meta["artist"]})

    faiss.write_index(cache["flat"], str(FLAT_PATH))
    faiss.write_index(cache["ivf"], str(IVF_PATH))
    faiss.write_index(cache["hnsw"], str(HNSW_PATH))
    np.save(str(META_PATH), meta)

    global _mtime
    _mtime = META_PATH.stat().st_mtime


def search(query_embedding: np.ndarray, variant: str = "ivf", top_k: int = 5) -> dict:
    """variant: 'flat' | 'ivf' | 'hnsw'"""
    indices, meta = _load()
    index = indices[variant]
    q = query_embedding.reshape(1, -1).astype(np.float32)

    # Fetch extra candidates to absorb duplicates from feedback insertions
    k_fetch = min(top_k * 4, index.ntotal)

    start = time.perf_counter()
    scores, idx_arr = index.search(q, k_fetch)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Deduplicate by song_id, keeping the highest-scoring entry per song
    seen: dict[int, dict] = {}
    for idx, score in zip(idx_arr[0], scores[0]):
        if idx < 0:
            continue
        song_id = meta[idx]["id"]
        s = round(float(score), 6)
        if song_id not in seen or s > seen[song_id]["score"]:
            seen[song_id] = {
                "id": song_id,
                "title": meta[idx]["title"],
                "artist": meta[idx]["artist"],
                "score": s,
            }

    results = sorted(seen.values(), key=lambda r: -r["score"])[:top_k]
    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    return {
        "results": results,
        "time_ms": round(elapsed_ms, 4),
        "method": f"faiss_{variant}",
    }
