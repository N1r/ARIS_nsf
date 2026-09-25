"""Training objectives: multi-resolution STFT, band periodicity and formant supervision."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchaudio.transforms import Spectrogram

from aris.audio_tensor import AudioTensor
from aris.dsp import window_fn

# ---------------------------------------------------------------------------------
# Multi-resolution STFT loss
# ---------------------------------------------------------------------------------


class SpectralLoss(nn.Module):
    """``alpha * L1(log2 |S|) + linear_weight * L1(|S|)`` at one STFT resolution."""

    eps: float = 1e-5

    def __init__(self, n_fft: int, hop_length: int, alpha: float = 1.0,
                 linear_weight: float = 0.0, preemphasis: float = 0.0, window: str = "hann"):
        super().__init__()
        self.alpha, self.linear_weight, self.preemphasis = alpha, linear_weight, preemphasis
        self.spec = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1,
                                window_fn=window_fn(window))

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.preemphasis > 0.0:
            pred = pred - self.preemphasis * F.pad(pred[..., :-1], (1, 0))
            target = target - self.preemphasis * F.pad(target[..., :-1], (1, 0))
        s_pred, s_true = self.spec(pred), self.spec(target)
        loss = self.alpha * F.l1_loss((s_pred + self.eps).log2(), (s_true + self.eps).log2())
        if self.linear_weight > 0.0:
            loss = self.linear_weight * F.l1_loss(s_pred, s_true) + loss
        return loss


class MultiScaleSpectralLoss(nn.Module):
    """Sum of :class:`SpectralLoss` over several FFT sizes (hop = ``(1 - overlap) * n_fft``)."""

    def __init__(self, n_ffts=(253, 509, 1021, 2053), alpha: float = 1.0,
                 linear_weight: float = 1.0, overlap: float = 0.75, preemphasis: float = 0.85,
                 window: str = "hann"):
        super().__init__()
        self.losses = nn.ModuleList(
            SpectralLoss(n, int(n - n * overlap), alpha, linear_weight, preemphasis, window)
            for n in n_ffts
        )

    def forward(self, pred, target) -> Tensor:
        pred = pred.as_tensor() if isinstance(pred, AudioTensor) else pred
        target = target.as_tensor() if isinstance(target, AudioTensor) else target
        return sum(loss(pred, target) for loss in self.losses)


# ---------------------------------------------------------------------------------
# Band periodicity
# ---------------------------------------------------------------------------------


class AdaptiveBandPeriodicity(nn.Module):
    """Periodicity of speech in ERB bands, from harmonic peaks against inter-harmonic valleys.

    Per voiced frame and harmonic ``k``, the peak power ``P_k`` is tracked in an F0-adaptive
    window around ``k F0`` (with sub-bin parabolic refinement) and compared with the valley
    floor ``N_k`` sampled at ``(k +- 1/2) F0``; ``r_k = max(P_k - c N_k, 0) / P_k``.

    * Valleys are corrected for the window sidelobes of the neighbouring harmonics, so a
      harmonic far below its neighbour (e.g. near a vocal-tract zero) is not read as noise.
    * A harmonic below ``detect_margin`` x that leakage, or above Nyquist, cannot be judged;
      :meth:`analyze` returns per-band weights so the objective leaves such bands out.
    * Each ERB band averages the harmonics inside it; empty bands fall back to interpolation
      and are excluded below F0.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_bands: int = 64,
        win_length: int = 768,
        hop_length: int = 240,
        n_fft: int = 2048,
        f0_min: float = 50.0,
        f_min: float = 50.0,
        f_max: float = 12000.0,
        max_harmonics: int = 100,
        search_radius: float = 0.28,
        prominence_factor: float = 2.5,
        eps: float = 1e-10,
        detect_margin: float = 4.0,
    ) -> None:
        super().__init__()
        if not 0 < win_length <= n_fft or hop_length <= 0 or n_bands < 1:
            raise ValueError("Require 0 < win_length <= n_fft, positive hop_length and n_bands")
        if sample_rate <= 0 or f0_min <= 0:
            raise ValueError("sample_rate and f0_min must be positive")
        if detect_margin < 1.0:
            raise ValueError("detect_margin must be >= 1 (a harmonic must exceed the leakage it sits on)")

        self.sample_rate = sample_rate
        self.n_bands = n_bands
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.f0_min = f0_min
        self.f_min = f_min
        self.f_max = f_max
        self.max_harmonics = max_harmonics
        self.search_radius = search_radius
        self.prominence_factor = prominence_factor
        self.eps = eps
        self.detect_margin = float(detect_margin)
        self.df = sample_rate / n_fft

        win = torch.hann_window(win_length)
        self.register_buffer("window", win, persistent=False)
        # Leakage kernel: the window's power spectrum on the analysis grid, peak-normalised, so
        # W[d] is the fraction of a harmonic's peak power that appears d bins away.
        kernel = (torch.fft.rfft(win.double(), n_fft).abs().square())
        kernel = kernel / kernel[0]
        # Only sidelobe leakage is corrected. Inside the main lobe a valley is unresolved from
        # its neighbours anyway, and the kernel is so steep there that a half-bin position error
        # over-subtracts; the first null of the window marks the boundary.
        null = int(torch.nonzero(kernel[1:] < 1e-3)[0]) + 1 if bool((kernel[1:] < 1e-3).any()) else n_fft // 2
        kernel[:null] = 0.0
        self.main_lobe_bins = null
        self.register_buffer("leak_kernel", kernel.float(), persistent=False)

        # ERB scale (Glasberg & Moore 1990): band centres and edges
        erb_min = 21.4 * math.log10(1.0 + 0.00437 * f_min)
        erb_max = 21.4 * math.log10(1.0 + 0.00437 * f_max)
        erb_edges = torch.linspace(erb_min, erb_max, n_bands + 1)
        erb_centers = 0.5 * (erb_edges[:-1] + erb_edges[1:])
        to_hz = lambda e: (10.0 ** (e / 21.4) - 1.0) / 0.00437
        self.register_buffer("hz_centers", to_hz(erb_centers).float(), persistent=False)
        self.register_buffer("hz_edges", to_hz(erb_edges).float(), persistent=False)

        k_vec = torch.arange(1, max_harmonics + 1, dtype=torch.float32).view(1, 1, max_harmonics, 1)
        self.register_buffer("k_vec", k_vec, persistent=False)
        k_valley_vec = (torch.arange(max_harmonics + 1, dtype=torch.float32) + 0.5).view(1, 1, max_harmonics + 1, 1)
        self.register_buffer("k_valley_vec", k_valley_vec, persistent=False)
        self.register_buffer("valley_offsets", torch.tensor([-1, 0, 1]).view(1, 1, 1, 3), persistent=False)
        self.register_buffer("offsets", torch.arange(-14, 15).view(1, 1, 1, 29), persistent=False)

    # ------------------------------------------------------------------ helpers
    def _leak(self, power: Tensor, distance_bins: Tensor) -> Tensor:
        """Power leaking from a peak of ``power`` to a point ``distance_bins`` away."""
        d = distance_bins.abs().clamp(0, self.leak_kernel.numel() - 1)
        return power * self.leak_kernel[d]

    def _to_bands(self, r_k: Tensor, w_k: Tensor, hz_k: Tensor, safe_f0: Tensor) -> tuple[Tensor, Tensor]:
        """Map per-harmonic periodicity to ERB bands. Returns (r_band, weight_band)."""
        b_size, t_size, K = r_k.shape
        # interpolation at the band centre (fallback for bands without a harmonic)
        k_cont = (self.hz_centers.view(1, 1, -1) / safe_f0).clamp(1.0, float(self.max_harmonics))
        k_idx = k_cont.long()
        alpha_k = k_cont - k_idx.float()
        r_pad = torch.cat([r_k[:, :, :1], r_k, r_k[:, :, -1:]], dim=-1)
        w_pad = torch.cat([w_k[:, :, :1], w_k, w_k[:, :, -1:]], dim=-1)
        r_interp = (1.0 - alpha_k) * torch.gather(r_pad, 2, k_idx) + alpha_k * torch.gather(r_pad, 2, k_idx + 1)
        # an interpolated band is only as judgeable as the harmonics it interpolates between
        w_interp = (1.0 - alpha_k) * torch.gather(w_pad, 2, k_idx) + alpha_k * torch.gather(w_pad, 2, k_idx + 1)
        centre_ok = (self.hz_centers.view(1, 1, -1) >= safe_f0).float()
        # integration: weighted mean of the harmonics inside each band
        band = torch.bucketize(hz_k, self.hz_edges) - 1  # [B, T, K]; -1 / n_bands = outside
        inside = (band >= 0) & (band < self.n_bands)
        band = band.clamp(0, self.n_bands - 1)
        w_in = w_k * inside.float()
        r_sum = torch.zeros(b_size, t_size, self.n_bands, device=r_k.device, dtype=r_k.dtype)
        w_sum = torch.zeros_like(r_sum)
        r_sum.scatter_add_(2, band, r_k * w_in)
        w_sum.scatter_add_(2, band, w_in)
        has = w_sum > 0
        r_band = torch.where(has, r_sum / w_sum.clamp_min(self.eps), r_interp)
        # Bands without a harmonic keep the interpolated value; above F0 they stay in the loss
        # (this preserves the ERB emphasis on low frequencies), below F0 they are excluded
        # because they would only copy harmonic 1.
        weight = torch.where(has, torch.ones_like(centre_ok), centre_ok * w_interp)
        return r_band, weight

    # ------------------------------------------------------------------ analysis
    def analyze(self, audio: Tensor, f0_frames: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(r, weight)``, both ``[B, frames, n_bands]``; weight is 0 where a band holds
        no judgeable harmonic (below F0, above Nyquist, or buried in sidelobe leakage)."""
        if audio.dim() != 2 or f0_frames.dim() != 2:
            raise ValueError("audio must be [B, samples] and f0_frames [B, frames]")

        spectrum = torch.stft(
            audio.float(), n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window, center=True, pad_mode="constant", return_complex=True,
        )
        power = (spectrum.real.square() + spectrum.imag.square()).transpose(1, 2)  # [B, frames, bins]
        frames = min(power.shape[1], f0_frames.shape[1])
        power, f0_frames = power[:, :frames], f0_frames[:, :frames]
        b_size, t_size, bins = power.shape
        K = self.max_harmonics
        safe_f0 = f0_frames.clamp_min(self.f0_min).unsqueeze(-1)  # [B, frames, 1]
        safe_f0_4 = safe_f0.unsqueeze(-1)

        # 1. harmonic peak tracking with an F0-adaptive window and sub-bin refinement
        radius = torch.round(self.search_radius * safe_f0_4 / self.df).long().clamp(2, 14)
        valid_mask = self.offsets.abs() <= radius  # [B, frames, 1, 29]
        centers = torch.round(self.k_vec * safe_f0_4 / self.df).long()  # [B, frames, K, 1]
        indices = (centers + self.offsets).clamp(0, bins - 1)
        peaks_w = torch.gather(power.unsqueeze(2).expand(-1, -1, K, -1), 3, indices)  # [B, frames, K, 29]
        max_vals, max_pos = torch.where(valid_mask, peaks_w, torch.full_like(peaks_w, -1e9)).max(dim=-1)
        left_idx = (max_pos - 1).clamp(0, 28).unsqueeze(-1)
        right_idx = (max_pos + 1).clamp(0, 28).unsqueeze(-1)
        alpha = torch.log(peaks_w.gather(3, left_idx).squeeze(-1).clamp_min(1e-12)).detach()
        beta = torch.log(max_vals.clamp_min(1e-12)).detach()
        gamma = torch.log(peaks_w.gather(3, right_idx).squeeze(-1).clamp_min(1e-12)).detach()
        denom = alpha - 2.0 * beta + gamma
        denom_safe = torch.where(denom.abs() > 1e-6, denom, torch.ones_like(denom))
        delta = torch.where(denom.abs() > 1e-6, 0.5 * (alpha - gamma) / denom_safe, torch.zeros_like(denom)).clamp(-0.7, 0.7).detach()
        p_refined = max_vals * torch.exp(-0.25 * (alpha - gamma) * delta).detach()  # [B, frames, K]
        peak_bin = (centers.squeeze(-1) + (max_pos - 14)).clamp(0, bins - 1)  # [B, frames, K]

        # 2. inter-harmonic valleys (k + 0.5) F0, 3-bin mean
        valley_bin = torch.round(self.k_valley_vec * safe_f0_4 / self.df).long()  # [B, frames, K+1, 1]
        valley_idx = (valley_bin + self.valley_offsets).clamp(0, bins - 1)
        valleys = torch.gather(power.unsqueeze(2).expand(-1, -1, K + 1, -1), 3, valley_idx).mean(dim=-1)
        valley_bin = valley_bin.squeeze(-1)  # [B, frames, K+1]

        # 3. leakage correction: subtract the neighbouring harmonics' sidelobes from each valley,
        #    and flag harmonics that do not rise above the leakage they sit on
        zero = p_refined.new_zeros(b_size, t_size, 1)
        p_pad = torch.cat([zero, p_refined, zero], dim=-1)  # harmonic i at index i+1
        pb_pad = torch.cat([peak_bin[:, :, :1], peak_bin, peak_bin[:, :, -1:]], dim=-1)
        # valley v lies between harmonics v-1 (pad index v) and v (pad index v+1)
        leak_v = self._leak(p_pad[:, :, :-1], valley_bin - pb_pad[:, :, :-1]) + self._leak(p_pad[:, :, 1:], valley_bin - pb_pad[:, :, 1:])
        valleys = (valleys - leak_v).clamp_min(0.0)
        leak_p = self._leak(p_pad[:, :, :-2], peak_bin - pb_pad[:, :, :-2]) + self._leak(p_pad[:, :, 2:], peak_bin - pb_pad[:, :, 2:])
        # A harmonic is judgeable only if it rises above the leakage at its own position
        # *and* at the two valleys that define its noise floor; a corrected valley is the
        # difference of two similar numbers when leakage dominates, so it cannot be trusted.
        leak_ref = torch.maximum(leak_p, torch.maximum(leak_v[:, :, :-1], leak_v[:, :, 1:]))
        detectable = p_refined > self.detect_margin * leak_ref
        p_floor = 0.5 * (valleys[:, :, :-1] + valleys[:, :, 1:])  # [B, frames, K]

        # 4. harmonic prominence -> periodicity ratio, differentiable through peak and floor
        p_clean = torch.clamp_min(p_refined - self.prominence_factor * p_floor, 0.0)
        r_k = p_clean / (p_refined + self.eps)

        # 5. ERB bands: integrate the harmonics of each band; weights mark judgeable bands
        hz_k = self.k_vec.view(1, 1, K) * safe_f0  # [B, frames, K]
        w_k = (detectable & (hz_k < self.sample_rate / 2)).float()
        r_erb, w_erb = self._to_bands(r_k, w_k, hz_k, safe_f0)

        # 6. voicing gate
        voiced = (f0_frames > self.f0_min).unsqueeze(-1).float()
        return r_erb * voiced, w_erb * voiced

    def forward(self, audio: Tensor, f0_frames: Tensor) -> Tensor:
        """Per-frame band periodicity ``[B, frames, n_bands]``."""
        return self.analyze(audio, f0_frames)[0]


class AdaptivePeriodicityObjective(nn.Module):
    """Weighted L1 between adaptive ERB band periodicity of prediction and target on voiced frames.

    * Bands the *target* cannot judge (weight 0 from ``analyze``) are left out, so a harmonic
      buried in sidelobe leakage or a band below F0 never becomes a training signal.
    * ``boundary_margin`` frames on each side of a voicing transition are excluded: the F0
      track is least reliable there.
    * Temporal smoothing (``smooth_frames``) is a *masked* average, so voiced edge frames are
      not mixed with the zeros of gated unvoiced neighbours.
    """

    def __init__(
        self,
        periodicity: AdaptiveBandPeriodicity | None = None,
        smooth_frames: int = 3,
        boundary_margin: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.periodicity = periodicity if periodicity is not None else AdaptiveBandPeriodicity(**kwargs)
        if smooth_frames < 1 or boundary_margin < 0:
            raise ValueError("smooth_frames must be >= 1 and boundary_margin >= 0")
        self.smooth_frames = int(smooth_frames)
        self.boundary_margin = int(boundary_margin)

    def _masked_pool(self, values: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
        k = self.smooth_frames
        if k <= 1 or values.shape[1] < k:
            return values, weight
        pad = k // 2
        num = F.avg_pool1d((values * weight).transpose(1, 2), kernel_size=k, stride=1, padding=pad, count_include_pad=True)
        den = F.avg_pool1d(weight.transpose(1, 2), kernel_size=k, stride=1, padding=pad, count_include_pad=True)
        num, den = num.transpose(1, 2)[:, : values.shape[1]], den.transpose(1, 2)[:, : values.shape[1]]
        # pooled weight = coverage of the window, restricted to cells that were themselves
        # admissible, so boundary erosion is not undone by pooling
        return num / den.clamp_min(1e-6), den * (weight > 0).float()

    def forward(self, pred: Tensor, target: Tensor, f0_frames: Tensor) -> Tensor:
        samples = min(pred.shape[-1], target.shape[-1])
        r_pred, _ = self.periodicity.analyze(pred[..., :samples], f0_frames)
        with torch.no_grad():
            r_target, w_target = self.periodicity.analyze(target[..., :samples], f0_frames)
        frames = r_pred.shape[1]
        voiced = f0_frames[:, :frames] > self.periodicity.f0_min
        if self.boundary_margin > 0:
            m = self.boundary_margin
            eroded = voiced.clone()
            for shift in range(1, m + 1):
                eroded[:, shift:] &= voiced[:, :-shift]
                eroded[:, :-shift] &= voiced[:, shift:]
            voiced = eroded
        weight = w_target * voiced.unsqueeze(-1).float()
        r_pred, _ = self._masked_pool(r_pred, weight)
        r_target, weight = self._masked_pool(r_target, weight)
        if weight.sum() <= 0:
            return r_pred.sum() * 0
        return (weight * (r_pred - r_target).abs()).sum() / weight.sum()


# ---------------------------------------------------------------------------------
# Formant supervision
# ---------------------------------------------------------------------------------


def _weighted_smooth_l1(pred, target, mask, weight):
    # In kHz, so beta = 0.05 is a 50 Hz transition from quadratic to linear.
    loss = F.smooth_l1_loss(pred / 1000.0, target / 1000.0, beta=0.05, reduction="none")
    w = weight[mask]
    return (w * loss[mask]).sum() / w.sum().clamp_min(1.0)


def formant_objective(tract, ctrl: Tensor, f1_t, f2_t, f3_t, b1_t, b2_t, b3_t, vmask,
                      smooth_weight: float = 0.0) -> tuple[Tensor, Tensor]:
    """Supervise predicted F1-F3 (and weakly B1-B3) with Praat estimates on reliable frames.

    Frames are kept only if voiced, physiologically plausible and free of tracker jumps.
    Returns ``(formant_loss, residual_regularization)``.
    """
    p = tract.get_formant_params(ctrl)
    T = min(p["f1"].shape[1], f1_t.shape[1])
    f1p, f2p, f3p = (p[k][:, :T] for k in ("f1", "f2", "f3"))
    f1t, f2t, f3t, b1t, b2t, b3t = (t[:, :T] for t in (f1_t, f2_t, f3_t, b1_t, b2_t, b3_t))
    m = vmask[:, :T] > 0

    def jumps(t):
        return torch.diff(t, dim=1, prepend=t[:, :1]).abs() if t.shape[1] > 1 else torch.zeros_like(t)

    df1, df2, df3 = jumps(f1t), jumps(f2t), jumps(f3t)
    smooth_traj = (df1 <= 350.0) & (df2 <= 600.0)
    valid = [
        (f1t >= 100.0) & (f1t <= 1400.0),
        (f2t >= 500.0) & (f2t <= 3600.0) & (f2t >= f1t + 50.0),
        (f3t >= 1800.0) & (f3t <= 5000.0) & (f3t >= f2t + 50.0),
    ]
    clean = [
        m & valid[0] & smooth_traj & (b1t > 0) & (b1t < 400.0),
        m & valid[1] & smooth_traj & (b2t > 0) & (b2t < 400.0),
        m & valid[2] & (df3 <= 800.0) & (b3t > 0) & (b3t < 500.0),
    ]
    loss = ctrl.new_zeros(())
    for pred, target, ok, plausible, d in zip((f1p, f2p, f3p), (f1t, f2t, f3t), clean, valid,
                                              (df1, df2, df3)):
        mask = ok if ok.any() else (m & plausible)
        if mask.any():
            # Formant transitions (> 30 Hz per frame) count twice.
            loss = loss + _weighted_smooth_l1(pred, target, mask, torch.where(d > 30.0, 2.0, 1.0))

    # Weak log-bandwidth supervision on steady-state frames.
    steady = m & (df1 < 10.0)
    bw_loss, bw_count = ctrl.new_zeros(()), 0
    for k, bt in (("b1", b1t), ("b2", b2t), ("b3", b3t)):
        ok = steady & (bt >= 30.0) & (bt <= 600.0)
        if ok.any():
            bp = p[k][:, :T]
            bw_loss = bw_loss + (torch.log(bp[ok].clamp_min(1.0)) - torch.log(bt[ok].clamp_min(1.0))).abs().mean()
            bw_count += 1
    if bw_count:
        loss = loss + 0.15 * (bw_loss / bw_count)

    # Barrier keeping formant logits away from sigmoid saturation.
    barrier = ctrl.new_zeros(())
    for i in range(tract.n_formants):
        sig = torch.sigmoid(ctrl[..., 1 + 2 * i]).clamp(1e-5, 1.0 - 1e-5)
        barrier = barrier - (torch.log(sig) + torch.log(1.0 - sig)).mean()
    loss = loss + 1e-3 * barrier

    if smooth_weight > 0 and T > 1:
        adjacent = m[:, 1:] & m[:, :-1]
        if adjacent.any():
            def step(x, scale):
                return F.smooth_l1_loss(x[:, 1:][adjacent] / scale, x[:, :-1][adjacent].detach() / scale,
                                        beta=0.05)
            smooth = step(f1p, 1000.0) + step(f2p, 1000.0) + step(f3p, 1000.0)
            for k in ("b1", "b2", "b3"):
                smooth = smooth + 0.5 * step(p[k][:, :T], 100.0)
            loss = loss + smooth_weight * smooth

    return loss, tract.residual_regularization(ctrl)
