"""Analysis encoder: waveform (+ F0) -> frame-level decoder controls."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torchaudio.models import Conformer
from torchaudio.transforms import Spectrogram

from aris.audio_tensor import AudioTensor


def _projection(width: int, dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(width, dim), nn.LayerNorm(dim), nn.SiLU())


def _gather_bins(log_power: Tensor, bins: Tensor) -> Tensor:
    """Linear interpolation of ``log_power [B, T, F]`` at fractional ``bins [B, T, K]``."""
    bins = bins.clamp(0, log_power.shape[-1] - 1)
    lower = bins.floor().long()
    upper = (lower + 1).clamp_max(log_power.shape[-1] - 1)
    frac = bins - lower
    return (1 - frac) * log_power.gather(-1, lower) + frac * log_power.gather(-1, upper)


def _centered(log_power: Tensor) -> Tensor:
    """Per-frame mean removal on an absolute frequency grid (energy is a separate feature),
    scaled to roughly unit range."""
    return (log_power - log_power.mean(-1, keepdim=True)) / 4


class Encoder(nn.Module):
    """Three observation branches, a shared Conformer, and one output head per group.

    * ``tract``: long-window (``n_fft``, 64 ms at 16 kHz) log spectrum, resolving
      vocal-tract resonances.
    * ``source``: the same spectrum sampled at harmonics ``k F0`` and valleys
      ``(k - 1/2) F0``, plus their contrast.
    * ``noise``: short-window (``short_n_fft``, 16 ms at 16 kHz) spectrum with
      harmonic and noise envelopes.

    Every branch also receives F0 (octaves re 150 Hz), voicing and frame energy.

    Head ``g`` reads the Conformer context concatenated with the branch named
    ``head_inputs[g]`` and predicts the decoder channels ``head_indices[g]``.
    """

    def __init__(
        self,
        out_channels: int,
        head_indices: Sequence[Sequence[int]],
        head_inputs: Sequence[str] = ("source", "tract", "noise"),
        head_biases: Sequence[float | Sequence[float]] | None = None,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        short_n_fft: int = 256,
        hop_length: int = 160,
        num_harmonics: int = 100,
        model_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_mult: int = 3,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        flat = [i for group in head_indices for i in group]
        if sorted(flat) != list(range(out_channels)):
            raise ValueError("head_indices must partition every output channel exactly once")
        if len(head_inputs) != len(head_indices):
            raise ValueError("head_inputs needs one entry per head")
        if not 1 <= hop_length <= short_n_fft <= n_fft or n_fft % 2 or short_n_fft % 2:
            raise ValueError("Require 1 <= hop_length <= short_n_fft <= n_fft, even FFT sizes")
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.head_inputs = tuple(head_inputs)
        self.long_spectrum = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2,
                                         pad_mode="constant")
        self.short_spectrum = Spectrogram(n_fft=short_n_fft, hop_length=hop_length, power=2,
                                          pad_mode="constant")
        self.register_buffer("harmonic_numbers", torch.arange(1, num_harmonics + 1).float(),
                             persistent=False)
        self.register_buffer("valley_numbers", torch.arange(0.5, num_harmonics + 0.5).float(),
                             persistent=False)
        inverse = [0] * out_channels
        for position, channel in enumerate(flat):
            inverse[channel] = position
        self.register_buffer("output_inverse_permutation", torch.tensor(inverse), persistent=False)

        n_long, n_short = n_fft // 2 + 1, short_n_fft // 2 + 1
        self.observations = nn.ModuleDict({
            "tract": _projection(n_long + 3, model_dim),
            "noise": _projection(n_short + 4 * n_long + num_harmonics + 3, model_dim),
            "source": _projection(3 * num_harmonics + 3, model_dim),
        })
        self.fusion = _projection(3 * model_dim, model_dim)
        self.context = Conformer(input_dim=model_dim, num_heads=num_heads,
                                 ffn_dim=model_dim * ff_mult, num_layers=num_layers,
                                 depthwise_conv_kernel_size=conv_kernel_size, dropout=dropout,
                                 use_group_norm=True)
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * model_dim, model_dim), nn.SiLU(), nn.Dropout(dropout),
                          nn.Linear(model_dim, len(group)))
            for group in head_indices
        )
        biases = head_biases if head_biases is not None else [0.0] * len(head_indices)
        for head, bias in zip(self.heads, biases, strict=True):
            # Near-zero weights: logits start at the bias prior, yet gradients reach the branches.
            nn.init.normal_(head[-1].weight, std=1e-3)
            bias = [bias] * head[-1].out_features if isinstance(bias, (int, float)) else bias
            head[-1].bias.data.zero_()[: len(bias)] = torch.tensor(bias, dtype=torch.float32)

    def features(self, x: AudioTensor, f0: AudioTensor) -> dict[str, Tensor]:
        long_power = self.long_spectrum(x.as_tensor())
        short_power = self.short_spectrum(x.as_tensor())
        frames = min(long_power.shape[-1], short_power.shape[-1])
        long_power, short_power = long_power[..., :frames], short_power[..., :frames]
        log_long, log_short = long_power.add(1e-7).log().mT, short_power.add(1e-7).log().mT

        track = f0.reduce_hop_length().as_tensor()
        positions = (torch.arange(frames, device=x.device) * self.hop_length).clamp_max(track.shape[1] - 1)
        pitch = track[:, positions]
        voiced = (pitch > 0) & (pitch < self.sample_rate / 2)
        pitch_feature = torch.where(voiced, torch.log2(pitch.clamp_min(1) / 150) / 4, 0)
        energy = long_power.mean(dim=1).add(1e-7).log() / 8
        conditioning = torch.stack([pitch_feature, voiced.to(pitch.dtype), energy], dim=-1)

        # Harmonic peaks and inter-harmonic valleys; bins above Nyquist are masked.
        h_bins = pitch[..., None] * self.harmonic_numbers * self.n_fft / self.sample_rate
        h_valid = voiced[..., None] & (h_bins <= self.n_fft / 2)
        harmonic = _gather_bins(log_long, h_bins)
        center = (harmonic * h_valid).sum(-1, keepdim=True) / h_valid.sum(-1, keepdim=True).clamp_min(1)
        harmonic = torch.where(h_valid, (harmonic - center) / 4, 0)
        v_bins = pitch[..., None] * self.valley_numbers * self.n_fft / self.sample_rate
        v_valid = voiced[..., None] & (v_bins <= self.n_fft / 2)
        valley = torch.where(v_valid, (_gather_bins(log_long, v_bins) - center) / 4, 0)
        contrast = torch.where(h_valid & v_valid, harmonic - valley, 0.0)

        # GOLF-style harmonic / noise envelopes on the full-resolution grid.
        n_harm = self.harmonic_numbers.shape[0]
        interval = self.sample_rate / self.n_fft
        f0_full = torch.where(voiced, pitch, torch.tensor(self.sample_rate / 2 / (n_harm - 1),
                                                          device=pitch.device, dtype=pitch.dtype))
        pickup = f0_full.unsqueeze(-1) * torch.arange(0.0, n_harm + 1, 0.5, device=pitch.device)
        index = (pickup / interval).round().long().clamp(0, log_long.shape[-1] - 1)
        energies = long_power.mT.gather(-1, index)
        harm_energy = energies[..., ::2]
        noise_energy = torch.cat([energies[..., :1], energies[..., 1::2]], -1)

        freqs = torch.arange(0, self.n_fft // 2 + 1, device=pitch.device) * interval
        remap = freqs / f0_full.unsqueeze(-1)
        low = remap.floor().long().clamp(0, n_harm - 2)
        p = (remap - low).clamp(0, 1)
        harm_env = (1 - p) * harm_energy.gather(-1, low) + p * harm_energy.gather(-1, low + 1)
        remap = (freqs + f0_full.unsqueeze(-1) * 0.5) / f0_full.unsqueeze(-1)
        low = remap.floor().long().clamp(0, n_harm - 2)
        p = (remap - low).clamp(0, 1)
        p[low == 0] = (p[low == 0] - 0.5) * 2
        p = p.clamp(0, 1)
        noise_env = (1 - p) * noise_energy.gather(-1, low) + p * noise_energy.gather(-1, low + 1)
        noise_ratio = (noise_env / (harm_env + noise_env + 1e-12)).clamp(0, 1)
        noise_ratio = torch.where(voiced.unsqueeze(-1), noise_ratio, torch.ones_like(noise_ratio))

        long_feat = _centered(log_long)
        return {
            "tract": torch.cat([long_feat, conditioning], -1),
            "noise": torch.cat([_centered(log_short), long_feat,
                                _centered(harm_env.add(1e-7).log()),
                                _centered(noise_env.add(1e-7).log()),
                                noise_ratio, contrast, conditioning], -1),
            "source": torch.cat([harmonic, valley, contrast, conditioning], -1),
        }

    def forward(self, x: AudioTensor, f0: AudioTensor) -> AudioTensor:
        """``x``: [B, T] audio, ``f0``: [B, T] sample-rate F0 in Hz (0 = unvoiced).

        Returns raw control logits ``[B, frames, out_channels]`` (hop ``hop_length``).
        """
        views = {k: self.observations[k](v) for k, v in self.features(x, f0).items()}
        fused = self.fusion(torch.cat([views[k] for k in ("tract", "noise", "source")], -1))
        lengths = torch.full((fused.shape[0],), fused.shape[1], device=fused.device, dtype=torch.long)
        context, _ = self.context(fused, lengths)
        logits = torch.cat([head(torch.cat([context, views[name]], -1))
                            for head, name in zip(self.heads, self.head_inputs)], -1)
        return AudioTensor(logits[..., self.output_inverse_permutation], self.hop_length)
