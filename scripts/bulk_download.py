"""
Bulk-download real song snippets from YouTube using only song names.
No URLs or start times needed — yt-dlp searches YouTube automatically,
and librosa picks the best 30-second clip by RMS energy (usually the chorus).

Usage:
    python scripts/bulk_download.py                  # download all songs in bulk_songs.txt
    python scripts/bulk_download.py --limit 20       # download first 20 only
    python scripts/bulk_download.py --limit 20 --offset 20  # next batch of 20

Then re-index:
    python scripts/ingest.py
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

SONGS_FILE      = Path("data/bulk_songs.txt")
SONGS_DIR       = Path("data/songs")
SNIPPET_DURATION = 30
TARGET_SR        = 22050


# ── helpers ──────────────────────────────────────────────────────────────────

def load_song_list() -> list[tuple[str, str]]:
    """Return list of (artist, title) from bulk_songs.txt."""
    songs = []
    for raw in SONGS_FILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if " - " in line:
            artist, title = line.split(" - ", 1)
            songs.append((artist.strip(), title.strip()))
        else:
            songs.append(("Unknown", line))
    return songs


def out_path(artist: str, title: str) -> Path:
    return SONGS_DIR / f"{artist} - {title}.wav"


def find_chorus_start(y: np.ndarray, sr: int, clip_duration: int) -> int:
    """
    Return the sample index of the 30-second window with highest RMS energy.
    The chorus is almost always the loudest sustained section of a song.
    We skip the first 30 seconds to avoid intros/silence.
    """
    clip_samples = sr * clip_duration
    min_start    = sr * 30                    # skip first 30 s (intro)
    step         = sr * 10                    # evaluate every 10 s

    if len(y) < min_start + clip_samples:
        return 0                              # short clip — just use start

    best_start  = min_start
    best_energy = -1.0

    for start in range(min_start, len(y) - clip_samples, step):
        segment = y[start : start + clip_samples]
        energy  = float(np.sqrt(np.mean(segment ** 2)))
        if energy > best_energy:
            best_energy = energy
            best_start  = start

    return best_start


def download_and_clip(artist: str, title: str, dest: Path) -> None:
    query = f"ytsearch1:{artist} {title} official audio"

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            "yt-dlp",
            "--quiet", "--no-warnings",
            "-x", "--audio-format", "wav", "--audio-quality", "0",
            "-o", str(Path(tmp) / "track.%(ext)s"),
            query,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "yt-dlp returned non-zero exit")

        wavs = list(Path(tmp).glob("*.wav"))
        if not wavs:
            raise RuntimeError("yt-dlp ran but produced no WAV file")

        # Load full audio first to find the chorus
        y_full, _ = librosa.load(str(wavs[0]), sr=TARGET_SR, mono=True)
        start_sample = find_chorus_start(y_full, TARGET_SR, SNIPPET_DURATION)
        start_sec    = start_sample / TARGET_SR

        y_clip = y_full[start_sample : start_sample + TARGET_SR * SNIPPET_DURATION]

    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), y_clip, TARGET_SR)
    return start_sec


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",  type=int, default=None, help="Max songs to download")
    parser.add_argument("--offset", type=int, default=0,    help="Skip first N songs in the list")
    args = parser.parse_args()

    all_songs = load_song_list()
    songs     = all_songs[args.offset:]
    if args.limit:
        songs = songs[:args.limit]

    pending = [(a, t) for a, t in songs if not out_path(a, t).exists()]
    skipped = len(songs) - len(pending)

    print(f"Song list: {len(all_songs)} total  |  this run: {len(songs)}  |  "
          f"already downloaded: {skipped}  |  to fetch: {len(pending)}\n")

    if not pending:
        print("Nothing to download.")
        return

    ok = failed = 0
    for i, (artist, title) in enumerate(pending, 1):
        dest = out_path(artist, title)
        print(f"  [{i}/{len(pending)}]  {artist} — {title} ...", end=" ", flush=True)
        try:
            start_sec = download_and_clip(artist, title, dest)
            dur = librosa.get_duration(path=str(dest))
            print(f"ok  (chorus at {start_sec:.0f}s, clipped {dur:.0f}s)")
            ok += 1
        except Exception as exc:
            print(f"FAILED  →  {exc}")
            failed += 1

    print(f"\nDone: {ok} downloaded, {skipped} skipped, {failed} failed.")
    if ok > 0:
        print("Re-index with:  python scripts/ingest.py")


if __name__ == "__main__":
    main()
