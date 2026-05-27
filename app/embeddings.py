import numpy as np
import librosa

# 40 MFCC means + 40 MFCC stds + 12 chroma CQT + 12 pitch histogram = 104 dimensions
N_MFCC = 40
DIM = 104
SAMPLE_RATE = 22050
MAX_DURATION = 30.0


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _pitch_histogram(y_harmonic: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract a 12-bin pitch class histogram using pyin (probabilistic YIN).

    pyin tracks the fundamental frequency (F0) frame-by-frame on a monophonic
    or near-monophonic signal. It returns:
      f0          — Hz per frame (NaN for unvoiced)
      voiced_flag — bool per frame (is this frame pitched?)

    We convert each voiced F0 to a MIDI note, take mod 12 to get the pitch
    class (0=C, 1=C#, …, 11=B), and accumulate a histogram.

    Why this beats chroma_stft for hum matching:
      chroma_stft on a full song: all chord tones fire simultaneously → dense
      chroma_stft on a hum: single melody note fires → sparse
      → profiles don't match even for the same song

      pyin on a full song (harmonic component): tracks ONE dominant pitch per frame
      pyin on a hum: tracks the pitch you're singing
      → both produce a sparse single-pitch profile → they match
    """
    f0, voiced_flag, _ = librosa.pyin(
        y_harmonic,
        fmin=librosa.note_to_hz("C2"),   # ~65 Hz — lowest reasonable hum
        fmax=librosa.note_to_hz("C7"),   # ~2093 Hz — highest reasonable voice
        sr=sr,
    )

    hist = np.zeros(12, dtype=np.float32)
    if f0 is not None:
        voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
        if len(voiced_f0) > 5:                         # need at least a few frames
            midi = librosa.hz_to_midi(voiced_f0)
            for m in midi:
                hist[int(round(m)) % 12] += 1
            hist /= hist.sum()

    return hist


def extract_embedding(audio_path: str) -> np.ndarray:
    """
    4-part audio fingerprint designed for hum-to-song matching.

    Each part is independently unit-normalized before concatenation so all four
    contribute equally to the final cosine similarity (1/4 each).

    Step 0 — Harmonic-Percussive Source Separation (HPSS):
      librosa.effects.hpss() splits audio into:
        harmonic  — sustained pitched sounds (melody, chords, vocals)
        percussive — transient sounds (drums, tabla, claps)
      We extract all features from the harmonic component only.
      Why: a hum is purely harmonic (no percussion). The song's tabla and rhythm
      section have completely different spectral characteristics from a hum and
      would otherwise pollute both MFCCs and chroma — pushing hum and song apart
      even when the melody matches.

    Part 1 — MFCC means (40 dims):
      Average spectral shape of the harmonic content.

    Part 2 — MFCC stds (40 dims):
      Temporal variation in spectral shape. More melody-informative than mean.

    Part 3 — Chroma CQT (12 dims):
      Pitch class energy using Constant-Q Transform. CQT has better frequency
      resolution at low pitches than STFT, making it more accurate for melody.
      Still captures chord tones (multiple notes per frame) — complementary to
      Part 4 which is single-note only.

    Part 4 — Pitch class histogram from pyin (12 dims):
      The most hum-friendly feature. Tracks one melody note per frame, converts
      to pitch class, builds a histogram of which notes appear. Ignores chord
      accompaniment entirely. Directly comparable between a hum (single voice)
      and the harmonic component of a song (dominant pitch tracked).
    """
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=MAX_DURATION)

    # Step 0: separate harmonic from percussive
    y_harmonic, _ = librosa.effects.hpss(y)

    # Part 1 + 2: MFCCs from harmonic
    mfcc = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=N_MFCC)    # (40, T)
    mfcc_mean = _unit(np.mean(mfcc, axis=1).astype(np.float32))          # (40,)
    mfcc_std  = _unit(np.std(mfcc,  axis=1).astype(np.float32))          # (40,)

    # Part 3: chroma CQT from harmonic
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)             # (12, T)
    chroma_mean = _unit(np.mean(chroma, axis=1).astype(np.float32))       # (12,)

    # Part 4: pitch histogram from pyin on harmonic
    pitch_hist = _unit(_pitch_histogram(y_harmonic, sr))                  # (12,)

    embedding = np.concatenate([mfcc_mean, mfcc_std, chroma_mean, pitch_hist])  # (104,)
    return _unit(embedding)
