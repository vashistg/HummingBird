"""
Download real song snippets from YouTube using yt-dlp + ffmpeg.

Usage:
    python scripts/download_songs.py

Before running, fill in the "url" fields in data/song_sources.json.
Paste any YouTube URL for the song (official video, lyric video, etc.).

The script:
  1. Downloads the audio track (best quality, no video)
  2. Clips a 30-second window starting at start_sec (chorus/hook region)
  3. Saves as  data/songs/<Artist> - <Title>.wav
  4. Skips songs that already have a WAV file

Requirements (both must be on PATH):
  pip install yt-dlp
  brew install ffmpeg   # or apt install ffmpeg
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

SOURCES_FILE = Path("data/song_sources.json")
SONGS_DIR    = Path("data/songs")
SNIPPET_DURATION = 30   # seconds to keep
TARGET_SR        = 22050


def wav_path(artist: str, title: str) -> Path:
    return SONGS_DIR / f"{artist} - {title}.wav"


def download_and_clip(url: str, start_sec: int, out_path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_template = str(Path(tmp) / "track.%(ext)s")

        # yt-dlp: pull best audio, convert to wav via ffmpeg post-processor
        cmd = [
            "yt-dlp",
            "--quiet",
            "--no-warnings",
            "-x",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", tmp_template,
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed:\n{result.stderr.strip()}")

        tmp_wavs = list(Path(tmp).glob("*.wav"))
        if not tmp_wavs:
            raise RuntimeError("yt-dlp ran but produced no WAV file")

        # Load the 30-second clip with librosa (handles any sample rate)
        y, _ = librosa.load(
            str(tmp_wavs[0]),
            sr=TARGET_SR,
            mono=True,
            offset=float(start_sec),
            duration=float(SNIPPET_DURATION),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y, TARGET_SR)


def main() -> None:
    sources = json.loads(SOURCES_FILE.read_text())

    missing_urls = [s for s in sources if not s.get("url")]
    if missing_urls:
        print(f"⚠  {len(missing_urls)} songs have no URL in {SOURCES_FILE}:")
        for s in missing_urls:
            print(f"     {s['artist']} — {s['title']}")
        print()

    to_download = [s for s in sources if s.get("url")]
    if not to_download:
        print("Nothing to download. Add YouTube URLs to data/song_sources.json first.")
        sys.exit(1)

    print(f"Downloading {len(to_download)} songs → data/songs/\n")
    ok = skipped = failed = 0

    for song in to_download:
        artist = song["artist"]
        title  = song["title"]
        out    = wav_path(artist, title)

        if out.exists():
            print(f"  skip   {artist} — {title}")
            skipped += 1
            continue

        print(f"  fetch  {artist} — {title}  (from {song['start_sec']}s) ...", end=" ", flush=True)
        try:
            download_and_clip(song["url"], song["start_sec"], out)
            dur = librosa.get_duration(path=str(out))
            print(f"ok  ({dur:.1f}s)")
            ok += 1
        except Exception as exc:
            print(f"FAILED\n         {exc}")
            failed += 1

    print(f"\nDone: {ok} downloaded, {skipped} skipped, {failed} failed.")
    if ok > 0:
        print("Now re-run:  python scripts/ingest.py")


if __name__ == "__main__":
    main()
