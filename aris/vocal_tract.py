"""Vocal tract H(z) = prod_m B_m(z) / (prod_i A_Fi(z) * prod_j A_res_j(z)).

Explicit F1-F3 resonators, learned residual pole pairs and gated nasal zeros.
Each formant owns its own second-order section, so editing F_i changes only
that section of the transfer function.
"""

import math

import torch
from torchlpc import sample_wise_lpc

from aris.audio_tensor import AudioTensor, linear_upsample
from aris.dsp import biquads2lpc, conj_logits2biquads, fir_filt, resonator
from aris.source import Component


def _reference_bandwidth(f: torch.Tensor) -> torch.Tensor:
    """Reference bandwidth (Hz) of a formant at ``f`` Hz; B_i is learned as a factor of it.

    Note the step at 1500 Hz (225 -> 192.5 Hz).
    """
    return torch.where(f <= 1500.0, 60.0 + 0.11 * f, 50.0 + 0.05 * f + 3e-5 * f**2)


class VocalTract(Component):
    """All-pole vocal tract with explicit formants, followed by optional gated zeros.

    Controls per frame, in order::

        [log_gain, (F_i, B_i) x n_formants, (2 logits) x n_learned, (F_z, B_z, gate) x n_zeros]

    * F_i = sigmoid(logit) mapped to ``f_ranges[i]``.
    * B_i = reference bandwidth of F_i times a factor in [1/2.5, 2.5], clamped to
      ``b_ranges[i]``.
    * Residual pole pairs take up spectral detail beyond F1-F3.
    * Zero pairs (F_z, B_z), meant for nasals, are blended in by gates that start
      almost closed (``zero_gate_bias``).
    """

    def __init__(
        self,
        sr: int = 16000,
        f_ranges=((220, 1100), (700, 3400), (2200, 4800)),
        b_ranges=((40, 400), (50, 450), (80, 700)),
        n_learned: int = 8,
        n_zeros: int = 2,
        max_abs_pole: float = 0.99,
        max_abs_zero: float = 0.999,
        nasal_fz_range=(600, 4500),
        nasal_bz_range=(40, 600),
        zero_gate_bias: float = -3.5,
        normalize_dc: bool = True,
    ):
        super().__init__()
        if len(f_ranges) != len(b_ranges):
            raise ValueError("f_ranges and b_ranges need one entry per formant")
        self.sr = sr
        self.f_ranges = [tuple(r) for r in f_ranges]
        self.b_ranges = [tuple(r) for r in b_ranges]
        self.n_formants = len(self.f_ranges)
        self.n_learned = n_learned
        self.n_zeros = n_zeros
        self.max_abs_pole = max_abs_pole
        self.max_abs_zero = max_abs_zero
        self.nasal_fz_range = tuple(nasal_fz_range)
        self.nasal_bz_range = tuple(nasal_bz_range)
        self.zero_gate_bias = zero_gate_bias
        self.normalize_dc = normalize_dc
        self.use_tilt = False
        self.n_formant_ctrl = 1 + 2 * self.n_formants
        self.n_pole_ctrl = self.n_formant_ctrl + 2 * n_learned
        self.split_size = (self.n_pole_ctrl + 3 * n_zeros,)
        self.control_partitions = {
            "gain": (0,),
            "formants": tuple(range(1, self.n_formant_ctrl)),
            "residual": tuple(range(self.n_formant_ctrl, self.n_pole_ctrl)),
            "zeros": tuple(range(self.n_pole_ctrl, self.n_pole_ctrl + 3 * n_zeros)),
        }

    # ---- control decoding --------------------------------------------------------

    def get_formant_params(self, x: torch.Tensor) -> dict:
        """Raw controls -> ``log_gain``, ``f1..fN`` and ``b1..bN`` in Hz."""
        out = {"log_gain": x[..., 0]}
        for i, ((lo, hi), (b_lo, b_hi)) in enumerate(zip(self.f_ranges, self.b_ranges)):
            f = torch.sigmoid(x[..., 1 + 2 * i]) * (hi - lo) + lo
            delta = torch.tanh(x[..., 2 + 2 * i]) * math.log(2.5)
            out[f"f{i + 1}"] = f
            out[f"b{i + 1}"] = (_reference_bandwidth(f) * torch.exp(delta)).clamp(min=b_lo, max=b_hi)
        return out

    def scale_formants(self, x: torch.Tensor, scales) -> torch.Tensor:
        """Multiply each formant track by ``scales[i]``, clamped to its range.

        B_i keeps its factor relative to the reference bandwidth, so it follows F_i.
        """
        out = x.clone()
        p = self.get_formant_params(x)
        for i, scale in enumerate(scales):
            if scale == 1.0:
                continue
            lo, hi = self.f_ranges[i]
            f = (p[f"f{i + 1}"] * scale).clamp(lo + 1.0, hi - 1.0)
            out[..., 1 + 2 * i] = torch.log((f - lo) / (hi - f))
        return out

    def residual_regularization(self, x: torch.Tensor) -> torch.Tensor:
        """L2 on residual logits plus a penalty on sharp residual poles below 4 kHz.

        The penalty stops residual poles from taking over F1-F3.
        """
        residual = x[..., self.n_formant_ctrl:self.n_pole_ctrl]
        if not residual.numel():
            return x.sum() * 0
        pairs = residual.reshape(*residual.shape[:-1], self.n_learned, 2)
        r = torch.sigmoid(pairs[..., 0]) * self.max_abs_pole
        freq = torch.acos(torch.tanh(pairs[..., 1]).clamp(-0.9999, 0.9999)) * self.sr / (2 * math.pi)
        sharp_low = torch.relu(r - 0.65).square() * (freq < 4000.0).float()
        return residual.square().mean() + 10.0 * sharp_low.sum(dim=-1).mean()

    def _pole_sections(self, x: torch.Tensor) -> torch.Tensor:
        p = self.get_formant_params(x)
        formants = [resonator(p[f"f{i + 1}"], p[f"b{i + 1}"], self.sr, self.max_abs_pole)
                    for i in range(self.n_formants)]
        residual = x[..., self.n_formant_ctrl:self.n_pole_ctrl].reshape(*x.shape[:-1], self.n_learned, 2)
        return torch.cat([torch.stack(formants, dim=-2),
                          conj_logits2biquads(residual, self.max_abs_pole)], dim=-2)

    def _zero_sections(self, x: torch.Tensor) -> torch.Tensor:
        z = x[..., self.n_pole_ctrl:self.n_pole_ctrl + 3 * self.n_zeros]
        z = z.reshape(*x.shape[:-1], self.n_zeros, 3)
        (f_lo, f_hi), (b_lo, b_hi) = self.nasal_fz_range, self.nasal_bz_range
        fz = torch.sigmoid(z[..., 0]) * (f_hi - f_lo) + f_lo
        bz = torch.sigmoid(z[..., 1]) * (b_hi - b_lo) + b_lo
        gate = torch.sigmoid(z[..., 2] + self.zero_gate_bias)
        # (1 - g) + g * B(z): the gate g fades each zero pair toward identity.
        sections = resonator(fz, bz, self.sr, self.max_abs_zero)
        return torch.cat([sections[..., :1], sections[..., 1:] * gate[..., None]], dim=-1)

    # ---- filtering ---------------------------------------------------------------

    def forward(self, ex: AudioTensor, ctrl: AudioTensor) -> AudioTensor:
        x = ctrl.as_tensor()
        sections = self._pole_sections(x)
        gain = ctrl.new_tensor(torch.exp(x[..., 0]))
        if ctrl.hop_length > 1:
            # Interpolate per section (not the expanded polynomial) to keep every pole inside
            # the unit circle at every sample.
            B, T, S, _ = sections.shape
            flat = sections.reshape(B, T, S * 3).transpose(1, 2)
            sections = linear_upsample(flat, ctrl.hop_length).transpose(1, 2).reshape(B, -1, S, 3)
        a = biquads2lpc(sections.contiguous())
        ex = (ex * gain).as_tensor()[:, : a.shape[1]]
        y = sample_wise_lpc(ex, a[:, : ex.shape[1]]).to(ex.dtype)
        if not self.n_zeros:
            return AudioTensor(y)
        b = ctrl.new_tensor(biquads2lpc(self._zero_sections(x))).reduce_hop_length().as_tensor()
        fir = torch.cat([torch.ones_like(b[..., :1]), b], dim=-1)
        if self.normalize_dc:
            # Unit gain at DC, so opening a zero does not change the overall level.
            fir = fir / fir.sum(dim=-1, keepdim=True).clamp_min(1e-4)
        T = min(y.shape[1], fir.shape[1])
        return AudioTensor(fir_filt(y[:, :T], fir[:, :T]))
