"""
Scan data/songs/ for audio files, extract MFCC embeddings,
store them in SQLite, then build Annoy and FAISS indices.

Run from the project root:
    python scripts/ingest.py

Supported formats: .wav .mp3 .flac .ogg .m4a
Filename convention (optional):  Artist - Title.wav
If the filename doesn't follow this pattern the filename becomes the title.
"""

import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from app import db, embeddings
from app.search import dtw_search, faiss_search, hnswlib_search

SONGS_DIR = Path("data/songs")
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def parse_filename(path: Path) -> tuple[str, str]:
    stem = path.stem
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
        return title.strip(), artist.strip()
    return stem, "Unknown"


def main() -> None:
    db.init_db()

    audio_files = sorted(p for p in SONGS_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if not audio_files:
        print(f"No audio files found in {SONGS_DIR}/")
        print("Tip: run  python scripts/generate_demo_songs.py  to create synthetic test songs.")
        sys.exit(1)

    new_count = 0
    pitch_sequences = {}   # filename -> np.ndarray, built in same pass
    for path in audio_files:
        exists = db.song_exists(path.name)
        action = "reindex" if exists else "index"
        title, artist = parse_filename(path)
        print(f"  {action} {path.name}  →  '{title}' by {artist} ...", end=" ", flush=True)
        try:
            emb = embeddings.extract_embedding(str(path))
            db.insert_song(title=title, artist=artist, filename=path.name, embedding=emb)
            # Extract pitch sequence for DTW (reuses HPSS+pyin pipeline)
            pitch_sequences[path.name] = dtw_search.extract_chroma_sequence(str(path))
            print("ok")
            new_count += 1
        except Exception as exc:
            print(f"FAILED: {exc}")

    total = db.song_count()
    print(f"\nDatabase: {total} songs total ({new_count} newly added)")

    if total == 0:
        print("Nothing to index.")
        sys.exit(1)

    songs = db.load_all_songs()

    print("\nBuilding hnswlib index...")
    hnswlib_search.build_index(songs)

    print("Building FAISS indices...")
    faiss_search.build_indices(songs)

    print("Building DTW pitch sequence index...")
    # Align sequences to song order from DB
    seqs = [pitch_sequences.get(s["filename"], np.zeros(0, dtype=np.float32)) for s in songs]
    dtw_search.build_index(songs, seqs)

    print(f"\nDone. {total} songs ready for search.")
    print("Start the server:  uvicorn app.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
