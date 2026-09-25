"""Excitation: an LF glottal wavetable indexed by R_d, and Gaussian noise."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from kazane import Decimate
from torch import nn

from aris.audio_tensor import AudioTensor
from aris.dsp import lf_derivative_table


class Component(nn.Module):
    """A decoder stage that consumes ``split_size`` encoder channels.

    ``transform`` maps the raw encoder logits of those channels to the
    arguments of ``forward`` (after the signal input). ``control_partitions``
    names groups of those channels (e.g. ``formants``) so that encoder heads
    can address them as ``<component>.<part>``.
    """

    split_size: tuple[int, ...] = ()
    control_partitions: dict[str, tuple[int, ...]] = {}

    def transform(self, *logits):
        return logits


class GlottalSource(Component):
    """LF glottal-flow-derivative wavetable, indexed by a log-spaced R_d axis.

    Controls per frame, both in [0, 1]: ``rd``, the position on the log R_d axis
    (tense to lax), and ``tilt``, a high-frequency boost of up to ``1 + tilt_boost``
    at Nyquist. The pulses are generated at ``oversampling`` times the sample rate
    and decimated.
    """

    split_size = (1, 1)
    control_partitions = {"rd": (0,), "tilt": (1,)}

    def __init__(self, table_size: int = 100, min_R_d: float = 0.3, max_R_d: float = 2.7,
                 points: int = 2048, oversampling: int = 4, tilt_boost: float = 8.0):
        super().__init__()
        self.register_buffer(
            "R_d_values", torch.exp(torch.linspace(math.log(min_R_d), math.log(max_R_d), table_size))
        )
        table = lf_derivative_table(self.R_d_values, points)
        # Align the excitation instants (derivative minima) across the table so
        # that moving along R_d does not shift the pulse in time.
        peak = table.argmin(dim=1)
        target = peak.max().item()
        for i, p in enumerate(peak.tolist()):
            table[i] = torch.roll(table[i], target - p)
        table = table / table.norm(dim=1, keepdim=True) * math.sqrt(table.shape[1])
        self.register_buffer("table", table)
        self.register_buffer("tilt_param", torch.tensor(float(tilt_boost)), persistent=False)
        self.oversampling = oversampling
        if oversampling > 1:
            self.decimater = Decimate(oversampling)
            self.decimater.register_buffer("kernel", self.decimater.kernel, persistent=False)

    def transform(self, rd_logit, tilt_logit):
        return torch.sigmoid(rd_logit), torch.sigmoid(tilt_logit)

    @staticmethod
    def _lookup(wrapped_phase: torch.Tensor, tables: AudioTensor) -> AudioTensor:
        """Bilinear read of per-frame wavetables: along the period by the phase, and
        between neighbouring frames by the sample position."""
        batch, seq_len = wrapped_phase.shape
        hop_length = tables.hop_length
        tables = tables.as_tensor()
        blocks = (seq_len + hop_length - 1) // hop_length
        if tables.shape[1] < blocks + 1:
            tables = F.pad(tables, (0, 0, 0, blocks - tables.shape[1] + 1), "replicate")
        else:
            tables = tables[:, : blocks + 1]
        tables = torch.cat([tables, tables[:, :, :1]], dim=2)
        grid_x = wrapped_phase * 2 - 1
        grid_y = (torch.arange(seq_len, device=wrapped_phase.device, dtype=wrapped_phase.dtype)
                  .view(1, -1).broadcast_to(batch, -1) / (hop_length * blocks) * 2 - 1)
        grid = torch.stack([grid_x, grid_y], dim=2).unsqueeze(2)
        out = F.grid_sample(tables.unsqueeze(1), grid, mode="bilinear", align_corners=True)
        return AudioTensor(out.squeeze(-1).squeeze(1))

    def forward(self, phase: AudioTensor, rd_index: AudioTensor, tilt: AudioTensor) -> AudioTensor:
        """``phase``: F0 / sr per sample (cycles per sample); ``rd_index``, ``tilt``:
        [B, frames] in [0, 1]."""
        rd_index = rd_index.clamp(0.0, 1.0)
        n_tables, length = self.table.shape
        position = rd_index * (n_tables - 1)
        lower = torch.as_tensor(position).long().clip_(0, n_tables - 2)
        frac = (position - lower).unsqueeze(-1)
        shape = (lower.shape[0], lower.shape[1], length)
        tables = (self.table[lower.flatten()].view(shape) * (1 - frac)
                  + self.table[lower.flatten() + 1].view(shape) * frac)

        k = self.oversampling
        if k > 1:
            tables = AudioTensor(tables.as_tensor(), hop_length=tables.hop_length * k)
            phase = AudioTensor(phase.as_tensor() / k, hop_length=phase.hop_length * k)
        phase = phase.reduce_hop_length()
        if k > 1:
            # Decimation needs every oversample of the final sample interval.
            phase = AudioTensor(F.pad(phase.as_tensor(), (0, k - 1), mode="replicate"))
        with torch.autocast(device_type=phase.device.type, enabled=False):
            instant_phase = torch.cumsum(phase.float(), 1)
        y = self._lookup((instant_phase % 1).as_tensor(), tables)
        if k > 1:
            y = AudioTensor(self.decimater(y.as_tensor()))

        # Frame-varying first-difference boost: y + tau * (y[n] - y[n-1]) / 2, i.e. gain 1
        # at DC and 1 + tau at the Nyquist frequency.
        max_tilt = float(self.tilt_param)
        if max_tilt > 0.0:
            tau = tilt.reduce_hop_length().as_tensor() * max_tilt
            if tau.shape[-1] < y.shape[-1]:
                tau = F.pad(tau, (0, y.shape[-1] - tau.shape[-1]), mode="replicate")
            else:
                tau = tau[:, : y.shape[-1]]
            yt = y.as_tensor()
            y = AudioTensor(yt + tau * (0.5 * (yt - F.pad(yt[:, :-1], (1, 0)))))
        return y


class GaussianNoise(Component):
    """White Gaussian excitation for the aperiodic branch; takes no controls."""

    def forward(self, ref: AudioTensor) -> AudioTensor:
        return torch.randn_like(ref)
