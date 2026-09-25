"""Prepare a folder of recordings (one speaker) for training.

    aris-prepare path/to/wavs data/myspeaker --sample-rate 16000

Writes ``audio/*.wav`` (mono, resampled), ``f0/*.txt`` (RMVPE, 5 ms),
``formants/*.npz`` (Praat Burg, 10 ms) and ``manifest.csv``. Items already
processed are skipped, so an interrupted run can be restarted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from aris.features import praat_formants, rmvpe_f0

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3"}


def split_of(content_hash: str, validation: float, test: float) -> str:
    """Deterministic split from the audio content, so duplicates share a split."""
    u = int(content_hash[:8], 16) / 16**8
    return "test" if u < test else "validation" if u < test + validation else "train"


def main(argv=None):
    p = argparse.ArgumentParser(prog="aris-prepare", description=__doc__.split("\n")[0])
    p.add_argument("source", type=Path, help="folder searched recursively for audio files")
    p.add_argument("output", type=Path)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--max-formant", type=float, default=5500.0,
                   help="Praat formant ceiling (Hz): ~5500 female, ~5000 male")
    p.add_argument("--validation", type=float, default=0.1, help="validation fraction")
    p.add_argument("--test", type=float, default=0.1, help="test fraction")
    p.add_argument("--device")
    a = p.parse_args(argv)

    files = sorted(f for f in a.source.rglob("*") if f.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        raise SystemExit(f"no audio files under {a.source}")
    for sub in ("audio", "f0", "formants"):
        (a.output / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    for n, path in enumerate(files, 1):
        relative = path.relative_to(a.source).as_posix()
        item = re.sub(r"[^A-Za-z0-9._-]", "-", path.stem) + "-" + hashlib.sha1(relative.encode()).hexdigest()[:8]
        wav, f0_path, fmt = (f"audio/{item}.wav", f"f0/{item}.txt", f"formants/{item}.npz")
        if all((a.output / x).exists() for x in (wav, f0_path, fmt)):
            audio, _ = sf.read(a.output / wav, dtype="float32")
        else:
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            if sr != a.sample_rate:
                audio = AF.resample(torch.from_numpy(audio), sr, a.sample_rate).numpy()
            f0 = rmvpe_f0(audio, a.sample_rate, a.device)
            formants = praat_formants(audio, a.sample_rate, max_formant_hz=a.max_formant)
            sf.write(a.output / wav, audio, a.sample_rate, subtype="FLOAT")
            np.savetxt(a.output / f0_path, f0, fmt="%.5f")
            np.savez_compressed(a.output / fmt, **formants)
        digest = hashlib.sha256(audio.tobytes()).hexdigest()
        rows.append(dict(id=item, split=split_of(digest, a.validation, a.test), audio_path=wav,
                         f0_path=f0_path, formant_path=fmt, samples=len(audio)))
        if n % 100 == 0 or n == len(files):
            print(f"{n}/{len(files)}", flush=True)

    with (a.output / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "validation", "test")}
    hours = sum(r["samples"] for r in rows) / a.sample_rate / 3600
    print(f"{a.output / 'manifest.csv'}: {len(rows)} items, {hours:.2f} h, {counts}")


if __name__ == "__main__":
    main()
