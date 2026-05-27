"""
DTW (Dynamic Time Warping) search on chroma sequences.

Why chroma instead of pitch (pyin) sequences:

  pyin is a monophonic pitch tracker — it assumes one pitch per frame.
  A full song has multiple simultaneous pitches (chords, backing vocals,
  instruments). Even after HPSS, pyin gets confused and tracks random
  harmonic content rather than the melody → garbage sequences for songs.

  Chroma vectors are polyphonic-aware:
    A 12-bin chroma vector captures the energy distribution across all
    pitch classes (C, C#, D, …, B) in each frame, regardless of how many
    notes are sounding simultaneously. Multiple instruments playing the
    same chord contribute to the same chroma bins consistently.
    Chroma is also largely timbre-invariant — sitar and voice on the same
    note land in the same bin.

  DTW on chroma sequences:
    Each song and hum is represented as a (12, T) matrix — 12 chroma bins
    × T time frames. DTW finds the optimal elastic alignment between two
    such sequences without assuming they're the same length or tempo.

  Per-frame L2 normalization:
    We L2-normalize each chroma frame to unit length before comparison.
    This makes the comparison about *shape* (which pitch classes dominate)
    rather than *energy* (how loud). A soft hum and a loud song match if
    they emphasize the same pitch classes in the same order.

  Subsequence DTW (subseq=True):
    Your 10-second hum is matched against the best-fitting window within
    the 30-second song clip. Neither end needs to align — we find the most
    similar region.

  Score conversion:
    DTW returns a distance (lower = better). After L2 normalization, the
    max Euclidean distance between two unit chroma frames is sqrt(2) ≈ 1.41.
    We normalize the total cost by query length to get per-frame average cost,
    then convert: score = 1 / (1 + dist).
    DTW scores are NOT comparable to cosine similarity scores from other methods.
"""

import time
from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 22050
MAX_DURATION = 30.0

SEQ_PATH  = Path("data/indices/dtw_sequences.npz")
META_PATH = Path("data/indices/dtw_meta.npy")

_sequences: dict | None = None
_meta: list | None = None
_mtime: float = 0.0


def extract_chroma_sequence(audio_path: str) -> np.ndarray:
    """
    Return a (12, T) chroma CQT matrix, each frame L2-normalized.

    Steps:
      1. Load audio
      2. HPSS — keep harmonic component
      3. chroma_cqt — 12-bin pitch class energy per frame
      4. L2-normalize each frame so comparison is shape-based not energy-based
    """
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=MAX_DURATION)
    y_harmonic, _ = librosa.effects.hpss(y)

    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr).astype(np.float32)  # (12, T)

    # L2-normalize each frame (column) to unit length
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)          # (1, T)
    chroma = np.where(norms > 1e-8, chroma / norms, chroma)        # (12, T)

    return chroma


def build_index(songs: list[dict], sequences: list[np.ndarray]) -> None:
    SEQ_PATH.parent.mkdir(parents=True, exist_ok=True)

    npz_data = {f"seq_{s['id']}": seq for s, seq in zip(songs, sequences)}
    np.savez_compressed(str(SEQ_PATH), **npz_data)

    meta = [{"id": s["id"], "title": s["title"], "artist": s["artist"]} for s in songs]
    np.save(str(META_PATH), meta)
    print(f"  DTW: {len(songs)} chroma sequences → {SEQ_PATH}")


def _load() -> tuple[dict, list]:
    global _sequences, _meta, _mtime
    current_mtime = META_PATH.stat().st_mtime if META_PATH.exists() else 0.0
    if _sequences is None or current_mtime != _mtime:
        loaded = np.load(str(SEQ_PATH), allow_pickle=True)
        _sequences = {k: loaded[k] for k in loaded.files}
        _meta = np.load(str(META_PATH), allow_pickle=True).tolist()
        _mtime = current_mtime
    return _sequences, _meta


def index_exists() -> bool:
    return SEQ_PATH.exists() and META_PATH.exists()


def search(query_chroma: np.ndarray, top_k: int = 5) -> dict:
    """
    query_chroma: (12, T_hum) — L2-normalized chroma frames from the hum
    """
    if query_chroma is None or query_chroma.shape[1] < 5:
        return {
            "results": [],
            "time_ms": 0.0,
            "method": "dtw",
            "note": "hum too short",
        }

    sequences, meta = _load()

    start = time.perf_counter()
    scores = []

    for m in meta:
        seq = sequences.get(f"seq_{m['id']}")   # (12, T_song)

        if seq is None or seq.shape[1] < 5:
            scores.append((float("inf"), m))
            continue

        try:
            # DTW between two (12, T) chroma matrices.
            # librosa.sequence.dtw expects (dim, T) inputs.
            # subseq=True: hum is aligned to the best window in the song.
            D, _ = librosa.sequence.dtw(
                query_chroma,
                seq,
                subseq=True,
                metric="euclidean",
            )
            # D[-1].min() = minimum cumulative cost to reach the end of the hum
            # Divide by query length (frames) → per-frame average cost
            dist = float(D[-1].min()) / query_chroma.shape[1]
        except Exception:
            dist = float("inf")

        scores.append((dist, m))

    elapsed_ms = (time.perf_counter() - start) * 1000
    scores.sort(key=lambda x: x[0])

    results = []
    for rank, (dist, m) in enumerate(scores[:top_k], 1):
        # Per-frame dist on L2-normalized 12-dim vectors: 0 = perfect, ~1.41 = worst
        # 1/(1+dist) gives: dist=0→1.0, dist=0.5→0.67, dist=1.0→0.5
        sim = round(1.0 / (1.0 + dist), 6)
        results.append({
            "rank": rank,
            "id": m["id"],
            "title": m["title"],
            "artist": m["artist"],
            "score": sim,
        })

    return {
        "results": results,
        "time_ms": round(elapsed_ms, 4),
        "method": "dtw",
    }
