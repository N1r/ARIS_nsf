"""Manifest-backed training data.

A dataset directory holds ``manifest.csv`` with columns
``id, split, audio_path, f0_path, formant_path, samples``; paths are relative to
the manifest. F0 is a text track on a 5 ms grid (0 = unvoiced), formants an
``.npz`` with ``time_s, f1..f3, b1..b3`` on a 10 ms grid (see ``aris.prepare``).
"""

from __future__ import annotations

import csv
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import soundfile as sf
from torch.utils.data import DataLoader, Dataset

F0_HOP_SECONDS = 0.005
FORMANT_HOP_SECONDS = 0.010


def read_manifest(path: str | Path, split: str | None = None) -> list[dict]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if split is None or r["split"] == split]
    for row in rows:
        for key in ("audio_path", "f0_path", "formant_path"):
            row[key] = path.parent / row[key]
    return rows


def interpolate_f0(f0: np.ndarray, sample_rate: int, samples: int, offset: int = 0) -> np.ndarray:
    """Upsample a 5 ms F0 track to samples; any interval touching an unvoiced frame stays 0."""
    positions = np.arange(len(f0)) * (sample_rate * F0_HOP_SECONDS)
    target = np.arange(offset, offset + samples)
    unvoiced = np.interp(target, positions, (f0 <= 0).astype(float), right=float(f0[-1] <= 0)) > 0
    return np.where(unvoiced, 0, np.interp(target, positions, f0)).astype(np.float32)


def nearest_track(times, values, target_times, hop_seconds=FORMANT_HOP_SECONDS):
    """Nearest-frame lookup; targets further than half a hop from any frame get 0."""
    if not len(times):
        return np.zeros(len(target_times), dtype=np.float32)
    right = np.clip(np.searchsorted(times, target_times, side="left"), 0, len(times) - 1)
    left = np.clip(right - 1, 0, len(times) - 1)
    index = np.where(np.abs(target_times - times[left]) <= np.abs(times[right] - target_times),
                     left, right)
    valid = np.abs(times[index] - target_times) <= hop_seconds * 0.51
    return np.where(valid, values[index], 0).astype(np.float32)


class SegmentDataset(Dataset):
    """Fixed-length segments -> ``(audio, f0, f1, f2, f3, b1, b2, b3, voiced_mask)``.

    Audio and F0 are sample-rate arrays; formant targets are on the 10 ms grid.
    Segments start every ``duration - overlap`` s; short files are zero-padded, and
    audio after the last full segment of a long file is not used.
    """

    def __init__(self, manifest_path, split, duration=2.0, overlap=1.0):
        self.records = read_manifest(manifest_path, split)
        if not self.records:
            raise ValueError(f"No '{split}' records in {manifest_path}")
        self.sample_rate = sf.info(str(self.records[0]["audio_path"])).samplerate
        self.segment = int(duration * self.sample_rate)
        hop = int((duration - overlap) * self.sample_rate)
        if hop <= 0:
            raise ValueError("overlap must be smaller than duration")
        self.formant_hop = int(FORMANT_HOP_SECONDS * self.sample_rate)
        self.items = [(i, k * hop) for i, r in enumerate(self.records)
                      for k in range(1 + max(0, int(r["samples"]) - self.segment) // hop)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        record_index, offset = self.items[index]
        record = self.records[record_index]
        audio, sr = sf.read(str(record["audio_path"]), dtype="float32", always_2d=True)
        if sr != self.sample_rate:
            raise RuntimeError(f"{record['audio_path']}: {sr} Hz, expected {self.sample_rate} Hz")
        audio = audio.mean(axis=1)[offset: offset + self.segment]
        audio = np.pad(audio, (0, self.segment - len(audio)))
        f0 = interpolate_f0(np.loadtxt(record["f0_path"], dtype=np.float32, ndmin=1),
                            sr, self.segment, offset)
        frames = self.segment // self.formant_hop
        times = offset / sr + np.arange(frames, dtype=np.float64) * FORMANT_HOP_SECONDS
        with np.load(record["formant_path"]) as track:
            t = track["time_s"].astype(np.float64)
            f1, f2, f3, b1, b2, b3 = (nearest_track(t, track[k].astype(np.float32), times)
                                      for k in ("f1", "f2", "f3", "b1", "b2", "b3"))
        voiced = ((f0[:: self.formant_hop][:frames] > 0) & (f1 > 0) & (f2 > 0)).astype(np.float32)
        return audio.astype(np.float32), f0, f1, f2, f3, b1, b2, b3, voiced


class ManifestDataModule(pl.LightningDataModule):
    def __init__(self, manifest_path: str, batch_size: int = 16, duration: float = 2.0,
                 overlap: float = 1.0, num_workers: int = 4):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        h = self.hparams
        if stage in (None, "fit"):
            self.train_set = SegmentDataset(h.manifest_path, "train", h.duration, h.overlap)
        self.val_set = SegmentDataset(h.manifest_path, "validation", h.duration, h.overlap)

    def _loader(self, dataset, train):
        n = self.hparams.num_workers
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=train,
                          drop_last=train, num_workers=n, persistent_workers=n > 0,
                          pin_memory=True)

    def train_dataloader(self):
        return self._loader(self.train_set, True)

    def val_dataloader(self):
        return self._loader(self.val_set, False)
