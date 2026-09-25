"""Save copy-synthesis examples of fixed validation utterances during training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from lightning.pytorch.callbacks import Callback

from aris.audio_tensor import AudioTensor
from aris.data import interpolate_f0


class ValidationAudio(Callback):
    """Every ``every_n_validations`` validations, write ``<log_dir>/validation/step_N/*.wav``
    (and a spectrogram comparison ``.png`` if matplotlib is installed)."""

    def __init__(self, num_samples: int = 2, every_n_validations: int = 2, seed: int = 42):
        self.num_samples = num_samples
        self.every = every_n_validations
        self.seed = seed
        self.count = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        self.count += 1
        if self.count % self.every:
            return
        records = trainer.datamodule.val_set.records
        picks = np.random.default_rng(self.seed).choice(
            len(records), min(self.num_samples, len(records)), replace=False)
        out = Path(trainer.log_dir or trainer.default_root_dir) / "validation" / f"step_{trainer.global_step:06d}"
        out.mkdir(parents=True, exist_ok=True)
        for rank, index in enumerate(sorted(picks)):
            record = records[index]
            audio, sr = sf.read(str(record["audio_path"]), dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            f0 = interpolate_f0(np.loadtxt(record["f0_path"], ndmin=1), sr, len(audio))
            x = AudioTensor(torch.from_numpy(audio)[None].to(pl_module.device))
            f0_t = AudioTensor(torch.from_numpy(f0)[None].to(pl_module.device))
            devices = [x.device.index] if x.is_cuda else []
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                torch.manual_seed(self.seed + rank)
                y = pl_module(x, f0_t).as_tensor()[0].float().cpu().numpy()
            stem = out / f"{rank:02d}_{record['id']}"
            sf.write(f"{stem}_ref.wav", audio, sr)
            sf.write(f"{stem}_recon.wav", y / max(1.0, np.abs(y).max()), sr)
            _save_spectrograms(f"{stem}.png", audio, y, sr)


def _save_spectrograms(path, reference, reconstruction, sr):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, signal, title in zip(axes, (reference, reconstruction), ("reference", "ARIS")):
        ax.specgram(signal, NFFT=512, Fs=sr, noverlap=512 - sr // 100, cmap="magma", vmin=-120)
        ax.set_title(title)
        ax.set_ylabel("Hz")
    axes[-1].set_xlabel("s")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
