"""
Generate synthetic demo songs as WAV files in data/songs/.

Each "song" is a short melody built from a sequence of sine-wave notes.
The tonal content is distinct enough that MFCC embeddings can differentiate them,
so you can immediately test the app without real music files.

Run from project root:
    python scripts/generate_demo_songs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf

SONGS_DIR = Path("data/songs")
SR = 22050
NOTE_DURATION = 0.4   # seconds per note
AMPLITUDE = 0.4


# MIDI note → frequency
def midi_to_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def make_note(freq: float, duration: float = NOTE_DURATION, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Combine fundamental + harmonics for a richer timbre
    wave = (
        np.sin(2 * np.pi * freq * t)
        + 0.5 * np.sin(2 * np.pi * 2 * freq * t)
        + 0.25 * np.sin(2 * np.pi * 3 * freq * t)
    )
    # Envelope: fade out last 10%
    fade_len = int(len(wave) * 0.1)
    wave[-fade_len:] *= np.linspace(1, 0, fade_len)
    return wave.astype(np.float32) * AMPLITUDE


def make_song(note_sequence: list[int], repeats: int = 3) -> np.ndarray:
    notes = [make_note(midi_to_hz(n)) for n in note_sequence]
    melody = np.concatenate(notes * repeats)
    # Pad to at least 5 s so librosa has enough frames
    min_samples = SR * 5
    if len(melody) < min_samples:
        melody = np.tile(melody, (min_samples // len(melody)) + 1)[:min_samples]
    return melody


# (filename_stem, note_sequence)
# Notes are MIDI numbers: 60=C4, 62=D4, 64=E4, 65=F4, 67=G4, 69=A4, 71=B4
DEMO_SONGS: list[tuple[str, str, list[int]]] = [
    # --- International ---
    ("The Beatles",    "Hey Jude",           [62, 64, 65, 67, 69, 67, 65, 64]),
    ("Queen",          "Bohemian Rhapsody",  [67, 65, 64, 62, 60, 62, 64, 65]),
    ("ABBA",           "Dancing Queen",      [69, 71, 72, 71, 69, 67, 69, 71]),
    ("Eagles",         "Hotel California",   [64, 62, 60, 59, 57, 59, 60, 62]),
    ("Nirvana",        "Smells Like Teen",   [55, 55, 58, 55, 55, 57, 55, 54]),
    ("Radiohead",      "Creep",              [60, 63, 67, 66, 60, 63, 67, 66]),
    ("Pink Floyd",     "Wish You Were Here", [71, 69, 67, 65, 64, 65, 67, 69]),
    ("Led Zeppelin",   "Stairway to Heaven", [57, 61, 64, 57, 60, 64, 57, 59]),
    ("Michael Jackson","Billie Jean",        [62, 62, 64, 62, 60, 59, 60, 62]),
    ("Adele",          "Rolling in Deep",    [65, 65, 67, 65, 64, 62, 60, 62]),
    ("Coldplay",       "Yellow",             [67, 67, 69, 67, 65, 64, 65, 67]),
    ("Bob Dylan",      "Blowin in Wind",     [60, 62, 64, 65, 64, 62, 60, 59]),

    # --- Bollywood ---
    # Chaiyya Chaiyya — AR Rahman (Dil Se, 1998): rising pentatonic riff in high register
    ("AR Rahman",      "Chaiyya Chaiyya",    [72, 74, 76, 74, 72, 69, 72, 74]),
    # Jai Ho — AR Rahman (Slumdog Millionaire, 2008): driving, mid-range rhythmic phrase
    ("AR Rahman",      "Jai Ho",             [64, 64, 67, 64, 62, 60, 62, 64]),
    # Tum Hi Ho — Mithoon (Aashiqui 2, 2013): slow romantic ascent
    ("Mithoon",        "Tum Hi Ho",          [62, 64, 66, 69, 71, 69, 66, 64]),
    # Kabhi Khushi Kabhie Gham — Jatin-Lalit (2001): grand, wide-range theme
    ("Jatin-Lalit",    "Kabhi Khushi Kabhie Gham", [60, 64, 67, 71, 72, 71, 67, 64]),
    # Dil Dhadakne Do — Shankar Ehsaan Loy (2015): bright, upbeat phrase
    ("Shankar Ehsaan Loy", "Dil Dhadakne Do", [67, 69, 71, 72, 71, 69, 67, 65]),
    # Tere Bina — AR Rahman (Guru, 2007): melancholic descending line
    ("AR Rahman",      "Tere Bina",          [71, 69, 67, 65, 64, 62, 60, 59]),
    # Rang De Basanti — AR Rahman (2006): folk-flavored, modal phrase
    ("AR Rahman",      "Rang De Basanti",    [60, 62, 63, 65, 67, 65, 63, 62]),
    # Dil Se Re — AR Rahman (Dil Se, 1998): intense, chromatic descent
    ("AR Rahman",      "Dil Se Re",          [72, 71, 69, 68, 67, 65, 64, 62]),
]


def main() -> None:
    SONGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(DEMO_SONGS)} demo songs in {SONGS_DIR}/")
    print("(12 international + 8 Bollywood)\n")

    for artist, title, notes in DEMO_SONGS:
        filename = f"{artist} - {title}.wav"
        path = SONGS_DIR / filename
        if path.exists():
            print(f"  skip  {filename}")
            continue
        audio = make_song(notes)
        sf.write(str(path), audio, SR)
        print(f"  wrote {filename}  ({len(audio)/SR:.1f}s)")

    print(f"\nDone. Now run:  python scripts/ingest.py")


if __name__ == "__main__":
    main()
