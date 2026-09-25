"""Load a trained model, edit controls in acoustic units, and resynthesize.

Python::

    import aris, soundfile as sf
    model = aris.load("pretrained/f024.pt")
    audio, sr = sf.read("in.wav")
    y = aris.resynthesize(model, audio, f1=1.2, region=(0.3, 0.7))

Command line::

    aris-edit pretrained/f024.pt in.wav out.wav --f1 1.2 --f0 1.1
"""

from __future__ import annotations

import argparse
import math
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
import yaml

from aris.audio_tensor import AudioTensor
from aris.data import interpolate_f0
from aris.model import ARIS


def _instantiate(spec):
    """Build objects from nested ``{class_path, init_args}`` specifications."""
    if isinstance(spec, list):
        return [_instantiate(v) for v in spec]
    if not isinstance(spec, dict):
        return spec
    args = {k: _instantiate(v) for k, v in spec.get("init_args", spec).items() if k != "class_path"}
    if "class_path" not in spec:
        return args
    module, name = spec["class_path"].rsplit(".", 1)
    return getattr(import_module(module), name)(**args)


def build(config: dict) -> ARIS:
    """Model from the ``model:`` section of a training config (criterion omitted)."""
    args = _instantiate({k: v for k, v in config.get("init_args", config).items()
                         if k not in ("class_path", "criterion")})
    return ARIS(**args)


def load(path: str | Path, device: str | None = None) -> ARIS:
    """Load a released ``.pt`` (weights + config) or a Lightning ``.ckpt`` from ``aris.train``.

    A Lightning checkpoint is read with the ``config.yaml`` of its run
    (``<run>/checkpoints/x.ckpt`` -> ``<run>/config.yaml``).
    """
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt.get("config")
    if config is None:
        config = yaml.safe_load((path.parent.parent / "config.yaml").read_text())["model"]
    model = build(config)
    state = {k: v for k, v in ckpt["state_dict"].items()
             if k.startswith(("encoder.", "decoder."))}
    model.load_state_dict(state, strict=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval()


def edit(model: ARIS, controls: dict, *, f1: float = 1.0, f2: float = 1.0, f3: float = 1.0,
         rd: float = 1.0, noise_db: float = 0.0, mask: torch.Tensor | None = None) -> dict:
    """Return edited controls.

    ``f1``-``f3`` and ``rd`` are ratios (formant frequencies and the glottal R_d),
    ``noise_db`` a gain on the aperiodic branch.

    Formants are clamped to their ranges and bandwidths follow them. ``mask``
    (``[frames]`` in [0, 1]) confines the edit to a region by blending the control
    logits: ``(1 - mask) * original + mask * edited``.
    """
    out = dict(controls)
    if (f1, f2, f3) != (1.0, 1.0, 1.0):
        ctrl, *rest = controls["end_filter_params"]
        edited = model.decoder.end_filter.scale_formants(ctrl.as_tensor(), (f1, f2, f3))
        out["end_filter_params"] = (ctrl.new_tensor(edited), *rest)
    if rd != 1.0:
        # The R_d axis of the wavetable is log-spaced over [min, max], mapped to [0, 1].
        rd_values = model.decoder.harm_oscillator.R_d_values
        shift = math.log(rd) / math.log(float(rd_values[-1] / rd_values[0]))
        index, *rest = controls["harm_oscillator_params"]
        out["harm_oscillator_params"] = (torch.clamp(index + shift, 0.0, 1.0), *rest)
    if noise_db:
        log_mag, *rest = controls["noise_filter_params"]
        out["noise_filter_params"] = (log_mag + noise_db * math.log(10.0) / 20.0, *rest)
    if mask is not None:
        for key in out:
            if out[key] is controls[key]:
                continue
            blended = []
            for a, b in zip(controls[key], out[key]):
                m = mask[: a.shape[1]].to(a).view(1, -1, *[1] * (a.dim() - 2))
                blended.append(a.new_tensor(a.as_tensor() * (1 - m) + b.as_tensor() * m))
            out[key] = tuple(blended)
    return out


def _window(n: int, rate: float, region: tuple[float, float], fade: float) -> torch.Tensor:
    """1 inside ``region`` (seconds), 0 outside, with linear ramps of ``fade`` seconds."""
    t = torch.arange(n, dtype=torch.float32) / rate
    t0, t1 = region
    return torch.minimum(((t - (t0 - fade)) / fade).clamp(0, 1), (((t1 + fade) - t) / fade).clamp(0, 1))


@torch.inference_mode()
def resynthesize(model: ARIS, audio: np.ndarray, f0: np.ndarray | None = None, *,
                 f0_scale: float = 1.0, f1: float = 1.0, f2: float = 1.0, f3: float = 1.0,
                 rd: float = 1.0, noise_db: float = 0.0, region: tuple[float, float] | None = None,
                 fade: float = 0.02, seed: int | None = 0, components: bool = False):
    """Analyze ``audio``, edit, and resynthesize.

    ``audio`` must be mono at ``model.sample_rate`` (it is not resampled here).
    ``f0`` is a 5 ms track in Hz (0 = unvoiced); RMVPE is run if it is omitted.
    With all factors at 1 this is copy synthesis.

    ``region=(t0, t1)`` (s) confines all edits to that span, with ``fade`` s ramps;
    after a regional F0 edit, later glottal pulses are shifted in phase. ``seed``
    fixes the noise (``None`` draws new noise). ``components=True`` returns a dict
    with ``full``, ``harmonics``, ``noise``, ``harm_source`` and ``noise_source``.
    The output is not normalized.
    """
    sr = model.sample_rate
    audio = np.asarray(audio, dtype=np.float32)
    if f0 is None:
        from aris.features import rmvpe_f0
        f0 = rmvpe_f0(audio, sr)
    f0 = interpolate_f0(np.asarray(f0, dtype=np.float32), sr, len(audio))
    device = model.device
    x = AudioTensor(torch.from_numpy(audio)[None].to(device))
    f0_t = AudioTensor(torch.from_numpy(f0)[None].to(device))

    controls = model.analyze(x, f0_t)
    frame_mask = sample_mask = None
    if region is not None:
        hop = model.encoder.hop_length
        frame_mask = _window(controls["end_filter_params"][0].shape[1], sr / hop, region, fade).to(device)
        sample_mask = _window(len(audio), sr, region, fade).to(device)[None]
    controls = edit(model, controls, f1=f1, f2=f2, f3=f3, rd=rd, noise_db=noise_db, mask=frame_mask)
    if f0_scale != 1.0:
        f0_t = f0_t * (f0_scale if sample_mask is None else 1.0 + (f0_scale - 1.0) * sample_mask)

    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        if seed is not None:
            torch.manual_seed(seed)
        out = model.render(controls, f0_t, decompose=True)
    out = {k: v.as_tensor()[0].float().cpu().numpy() for k, v in out.items()}
    return out if components else out["full"]


def main(argv=None):
    import soundfile as sf

    p = argparse.ArgumentParser(prog="aris-edit", description=__doc__.split("\n")[0])
    p.add_argument("model", help="released .pt or Lightning .ckpt")
    p.add_argument("input", help="input wav (resampled to the model rate if needed)")
    p.add_argument("output", help="output wav")
    p.add_argument("--f0-track", help="5 ms F0 text file (default: run RMVPE)")
    p.add_argument("--f0", type=float, default=1.0, dest="f0_scale", help="F0 ratio")
    p.add_argument("--semitones", type=float, help="F0 shift in semitones (overrides --f0)")
    for k in ("f1", "f2", "f3", "rd"):
        p.add_argument(f"--{k}", type=float, default=1.0, help=f"{k.upper()} ratio")
    p.add_argument("--noise-db", type=float, default=0.0, help="aperiodic branch gain (dB)")
    p.add_argument("--region", type=float, nargs=2, metavar=("T0", "T1"),
                   help="edit only between T0 and T1 seconds")
    p.add_argument("--device")
    a = p.parse_args(argv)

    model = load(a.model, a.device)
    audio, sr = sf.read(a.input, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != model.sample_rate:
        import torchaudio.functional as AF
        audio = AF.resample(torch.from_numpy(audio), sr, model.sample_rate).numpy()
    f0 = np.loadtxt(a.f0_track, ndmin=1) if a.f0_track else None
    f0_scale = 2.0 ** (a.semitones / 12.0) if a.semitones is not None else a.f0_scale
    y = resynthesize(model, audio, f0, f0_scale=f0_scale, f1=a.f1, f2=a.f2, f3=a.f3, rd=a.rd,
                     noise_db=a.noise_db, region=tuple(a.region) if a.region else None)
    sf.write(a.output, y, model.sample_rate)
    print(f"wrote {a.output} ({len(y) / model.sample_rate:.2f} s, {model.sample_rate} Hz)")


if __name__ == "__main__":
    main()
