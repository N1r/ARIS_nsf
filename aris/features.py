"""Acoustic analysis used as model input (RMVPE F0) and training targets (Praat formants)."""

from __future__ import annotations

import os
import sys
import types
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

F0_HOP = 80  # RMVPE hop at its 16 kHz input rate: a 5 ms F0 grid


@lru_cache(maxsize=1)
def _rmvpe(root: str, checkpoint: str):
    # Import RMVPE's inference modules without its package __init__, which pulls in
    # training-only dependencies.
    package = types.ModuleType("_aris_rmvpe")
    package.__path__ = [str(Path(root) / "src")]
    sys.modules.setdefault("_aris_rmvpe", package)
    from _aris_rmvpe.constants import (
        MEL_FMAX,
        MEL_FMIN,
        N_MELS,
        SAMPLE_RATE,
        WINDOW_LENGTH,
    )
    from _aris_rmvpe.inference import RMVPE
    from _aris_rmvpe.model import E2E, E2E0
    from _aris_rmvpe.spec import MelSpectrogram

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    model = RMVPE.__new__(RMVPE)
    model.model = (E2E if any("tf.layers" in k for k in state) else E2E0)(4, 1, (2, 2))
    model.model.load_state_dict(state, strict=True)
    model.model.eval()
    model.hop_length, model.seg_length, model.resample_kernel = F0_HOP, 32 * F0_HOP, {}
    model.mel_extractor = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, F0_HOP, None,
                                         MEL_FMIN, MEL_FMAX)
    return model


def rmvpe_f0(audio: np.ndarray, sample_rate: int, device: str | None = None,
             root: str | None = None, checkpoint: str | None = None) -> np.ndarray:
    """F0 in Hz on a 5 ms grid (0 = unvoiced), from RMVPE (https://github.com/yxlllc/RMVPE).

    ``root`` is a clone of that repository and ``checkpoint`` its ``model.pt``; they default
    to ``$ARIS_RMVPE_ROOT`` and ``$ARIS_RMVPE_CHECKPOINT`` (or ``<root>/checkpoints/model.pt``).
    """
    root = root or os.environ.get("ARIS_RMVPE_ROOT")
    if not root:
        raise RuntimeError("Set ARIS_RMVPE_ROOT to a clone of https://github.com/yxlllc/RMVPE")
    checkpoint = checkpoint or os.environ.get("ARIS_RMVPE_CHECKPOINT",
                                              str(Path(root) / "checkpoints" / "model.pt"))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # thred: RMVPE's voicing threshold; frames below it are unvoiced (F0 = 0).
    f0 = _rmvpe(root, checkpoint).infer_from_audio(
        np.asarray(audio, dtype=np.float32), sample_rate=sample_rate, device=device,
        thred=0.03, use_viterbi=False)
    return np.asarray(f0, dtype=np.float32).reshape(-1)


def praat_formants(audio: np.ndarray, sample_rate: int, hop_seconds: float = 0.01,
                   max_formant_hz: float = 5500.0) -> dict[str, np.ndarray]:
    """Praat Burg F1-F3 and bandwidths every ``hop_seconds`` (0 where undefined).

    Praat settings: 5 formants, 25 ms window, pre-emphasis from 50 Hz.
    ``max_formant_hz`` is the formant ceiling (5500 Hz suits female voices; ~5000 Hz
    is the usual choice for male voices).
    """
    import parselmouth
    from parselmouth.praat import call

    sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sample_rate)
    formant = call(sound, "To Formant (burg)", hop_seconds, 5, max_formant_hz, 0.025, 50.0)
    times = np.asarray(formant.xs(), dtype=np.float64)
    out = {"time_s": times.astype(np.float32)}
    for n in (1, 2, 3):
        for key, query in ((f"f{n}", "Get value at time"), (f"b{n}", "Get bandwidth at time")):
            values = np.array([call(formant, query, n, float(t), "Hertz", "Linear") for t in times],
                              dtype=np.float64)
            out[key] = np.where(np.isfinite(values) & (values > 0), values, 0).astype(np.float32)
    return out
