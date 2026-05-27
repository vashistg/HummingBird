# Audio Embedding Design: Why We Use MFCCs + Deltas + Chroma

## The Goal

We want a fixed-size numeric vector ("embedding") for each song such that:
- Two clips of the **same song** have embeddings close to each other
- A **hum of a song** has an embedding close to the full recording
- Two **different songs** have embeddings far apart

"Close" and "far" are measured by cosine similarity — the angle between vectors in 92-dimensional space.

---

## What the Old Embedding Was Doing (and Why It Failed)

```python
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
embedding = np.mean(mfcc, axis=1)   # shape: (128,)
```

### What MFCCs are

MFCC stands for Mel-Frequency Cepstral Coefficients. The pipeline:

1. Take a 30-second audio clip
2. Split it into ~1300 short frames (each ~23ms long)
3. For each frame, compute the frequency spectrum and warp it onto the **Mel scale**
   — a scale that matches how human ears perceive pitch (logarithmic, not linear)
4. Apply a **cepstral transform** (DCT) to compress that spectrum into N numbers

`mfcc` comes out as shape `(128, 1300)` — 128 coefficients × 1300 time frames.

Then `np.mean(mfcc, axis=1)` collapses the time axis into a single 128-dim vector:
"on average, what does this audio sound like spectrally."

### Why averaging over time is a problem

Averaging destroys the melody. Two songs can have the same average spectral "feel"
while sounding completely different — especially when one is a full orchestral recording
and the other is you humming the same melody.

The embedding captured **timbre** (what the instruments sound like) but not **shape**
(how the melody moves).

With 12 songs this was fine — the songs were different enough genres that timbre alone
separated them. With 107 songs, many from the same artists/genre/key, the timbres
collapsed into a tight cluster. The similarity matrix showed mean pairwise similarity
of 0.96+ and many song pairs at 1.0000.

### Why 128 MFCC coefficients is too many for hum matching

- Coefficients 1–13: broad spectral shape — brightness, warmth, bass. Musically meaningful.
- Coefficients 13–40: finer spectral texture. Still useful.
- Coefficients 40–128: micro-texture differences between recording environments, microphone
  quality, compression artifacts.

That last group is completely different between a studio recording and a hum.
128 was the right number for song-to-song matching. For hum-to-song, it adds noise.

---

## The New Embedding: Three Lenses

```python
mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)   # (40, T)
delta  = librosa.feature.delta(mfcc)                     # (40, T)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)         # (12, T)

embedding = concat([mean(mfcc), mean(delta), mean(chroma)])  # (92,)
```

### Part 1: MFCCs — 40 dims

Same as before, but capped at 40 coefficients to drop the noise-heavy high coefficients.
Captures the average timbral/spectral character of the audio.

### Part 2: Delta MFCCs — 40 dims

```python
delta = librosa.feature.delta(mfcc)
```

`delta` is the **frame-to-frame difference** of the MFCC — a numerical first derivative over time.

If MFCC at frame t is `[c₁, c₂, ... c₄₀]`, then delta at frame t is:
```
delta[t] ≈ (mfcc[t+1] - mfcc[t-1]) / 2
```

What this captures:

| Feature | Says |
|---------|------|
| Static MFCC | "this audio is warm, mid-range, and bright **on average**" |
| Delta MFCC  | "this audio **starts** bright, **gets** darker, then bright again" |

The delta captures **how the spectrum moves over time** — the melodic contour.
When you hum a song, the instrument timbre is gone, but the *shape of the melody* —
the rises and falls — survives in the deltas.

This is the biggest practical gain for hum matching.

### Part 3: Chroma — 12 dims

```python
chroma = librosa.feature.chroma_stft(y=y, sr=sr)   # (12, T)
```

Chroma maps every frequency in the audio onto one of the **12 pitch classes**:
C, C#, D, D#, E, F, F#, G, G#, A, A#, B.

It completely ignores octave and timbre — a piano playing C4, a guitar playing C3,
and you humming C5 all contribute to the same "C" bin.

The 12-dim chroma vector tells you "which notes are being used and how much."
Songs in the same key or with similar chord progressions will have similar chroma profiles.

Chroma is **timbre-invariant by design** — it was built specifically for tasks like
cover song detection and hum matching where the instrument changes but the notes don't.

---

## Dimension Summary

| Component    | Dims | Captures |
|--------------|------|---------|
| MFCCs        | 40   | Average spectral/timbral character |
| Delta MFCCs  | 40   | Melodic contour (how spectrum changes over time) |
| Chroma       | 12   | Pitch class content (which notes, timbre-invariant) |
| **Total**    | **92** | |

---

## Why the DB and Indices Had to Be Wiped

The SQLite database stores embeddings as raw binary blobs (via `np.frombuffer`).
The FAISS and hnswlib index files store vectors in their own binary formats.

All of these were built assuming 128-dim float32 vectors (512 bytes per song).
Switching to 92-dim means each vector is 368 bytes. Mixing them would either crash
or silently return garbage results — the index would read 128 floats but only 92 were
written, interpreting random memory as the last 36 dimensions.

So: wipe DB, wipe indices, re-run `python scripts/ingest.py`.

---

## Why the Similarity Matrix Improved

Before: mean pairwise similarity 0.96+, many pairs at 1.0000 — all songs clustered.

After: mean 0.84, median 0.89, only 3 pairs above 0.99.

Two AR Rahman songs may have similar average timbre (similar raw MFCCs), but they
have different melodic contours (different deltas) and different harmonic content
(different chroma). The embedding now has three lenses on the audio instead of one.

---

## The Core Intuition

> Your hum carries **melody shape** (captured by delta-MFCCs) and **pitch classes**
> (captured by chroma) but almost no **timbre** (raw MFCCs).
>
> So the embedding should weight the things humming preserves, not the things it loses.

The full pipeline for every audio file — whether it's a studio recording or a phone hum:

```
audio file
    → librosa.load()           resample to 22050 Hz mono
    → split into 23ms frames
    → Mel-scale spectrum per frame
    → DCT → MFCCs (40 × T)
    → delta(MFCCs) (40 × T)    frame-to-frame derivative
    → chroma_stft (12 × T)     pitch class histogram per frame
    → mean over time axis      collapse T → single vector per feature
    → concatenate              [40] + [40] + [12] = [92]
    → L2 normalize             so cosine_sim = dot product
    → store in SQLite + ANN index
```

---

## Further Reading

- [MFCC on Wikipedia](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
- [Chroma features — Music Information Retrieval](https://musicinformationretrieval.com/chroma.html)
- librosa docs: `librosa.feature.mfcc`, `librosa.feature.delta`, `librosa.feature.chroma_stft`
