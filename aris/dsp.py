"""Differentiable signal-processing primitives."""

import torch
import torch.nn.functional as F
from torch import Tensor


def lf_derivative_table(Rd: Tensor, points: int = 1024) -> Tensor:
    """Liljencrants-Fant (LF) glottal-flow derivative, one period per R_d value.

    R_d is mapped to the timing parameters R_a, R_k and R_g with Fant's (1995)
    regressions. Adapted from
    https://github.com/dsuedholt/vocal-tract-grad/blob/main/glottis.py.
    Returns ``[len(Rd), points]``, with time normalized to one period.
    """
    Rd = torch.as_tensor(Rd).view(-1, 1)
    Ra = -0.01 + 0.048 * Rd
    Rk = 0.224 + 0.118 * Rd
    Rg = (Rk / 4) * (0.5 + 1.2 * Rk) / (0.11 * Rd - Ra * (0.5 + 1.2 * Rk))

    Ta = Ra  # time constant of the return phase
    Tp = 1 / (2 * Rg)  # flow maximum
    Te = Tp + Tp * Rk  # derivative minimum (main excitation instant)

    epsilon = 1 / Ta
    shift = torch.exp(-epsilon * (1 - Te))
    delta = 1 - shift
    rhs_integral = ((1 / epsilon) * (shift - 1) + (1 - Te) * shift) / delta
    upper_integral = (Te - Tp) / 2 - rhs_integral

    omega = torch.pi / Tp
    s = torch.sin(omega * Te)
    alpha = torch.log(-torch.pi * s * upper_integral / (Tp * 2)) / (Tp / 2 - Te)
    E0 = -1 / (s * torch.exp(alpha * Te))

    t = torch.linspace(0, 1, points + 1)[None, :-1]
    before = E0 * torch.exp(alpha * t) * torch.sin(omega * t)
    after = (-torch.exp(-epsilon * (t - Te)) + shift) / delta
    # Open phase: growing sinusoid up to Te; return phase: exponential recovery to zero.
    return torch.where(t < Te, before, after).squeeze()


def window_fn(name: str = "hann"):
    return {"hann": torch.hann_window, "hanning": torch.hann_window,
            "hamming": torch.hamming_window, "blackman": torch.blackman_window}[name]


def fir_filt(x: Tensor, h: Tensor) -> Tensor:
    """Sample-wise time-varying FIR. ``x``: [B, T], ``h``: [B, T, taps]."""
    x = F.pad(x, (h.shape[-1] - 1, 0)).unfold(-1, h.shape[-1], 1)
    return torch.matmul(x.unsqueeze(-2), h.flip(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)


def _poly_product(polynomials: Tensor) -> Tensor:
    """Product of ``[n, batch, order+1]`` polynomials by recursive grouped convolution."""
    n = len(polynomials)
    if n == 1:
        return polynomials[0]
    c1 = _poly_product(polynomials[n // 2:])
    c2 = _poly_product(polynomials[: n // 2])
    if c1.shape[1] > c2.shape[1]:
        c1, c2 = c2, c1
    weight = c1.unsqueeze(1).flip(2)
    return F.conv1d(c2.unsqueeze(0), weight, padding=weight.shape[2] - 1,
                    groups=c2.shape[0]).squeeze(0)


def biquads2lpc(biquads: Tensor) -> Tensor:
    """Cascade ``[..., sections, 3]`` second-order sections into ``[..., 2*sections]``
    polynomial coefficients (leading 1 dropped)."""
    flat = biquads.view(-1, *biquads.shape[-2:]).transpose(0, 1)
    return _poly_product(flat).view(*biquads.shape[:-2], -1)[..., 1:]


def conj_logits2biquads(logits: Tensor, max_abs_pole: float) -> Tensor:
    """Two logits per section -> conjugate pole pair with ``|pole| < max_abs_pole``.

    The first logit sets the pole radius, the second the cosine of its angle.
    """
    mag = torch.sigmoid(logits[..., 0]) * max_abs_pole
    cos = torch.tanh(logits[..., 1])
    return torch.stack([torch.ones_like(mag), -2 * mag * cos, mag.square()], dim=-1)


def resonator(freq: Tensor, bandwidth: Tensor, sr: int, max_radius: float) -> Tensor:
    """Second-order section ``[1, -2r cos(theta), r^2]`` for a resonance (or anti-resonance).

    ``theta = 2 pi freq / sr`` and ``r = exp(-pi bandwidth / sr)``, with ``r`` capped at
    ``max_radius`` for stability.
    """
    r = torch.exp(-torch.pi * bandwidth / sr).clamp(max=max_radius)
    theta = 2 * torch.pi * freq / sr
    return torch.stack([torch.ones_like(r), -2 * r * torch.cos(theta), r * r], dim=-1)
