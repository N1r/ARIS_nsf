"""Source-filter decoder: y = g * h * (e_harmonic + e_noise)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from aris.audio_tensor import AudioTensor
from aris.dsp import window_fn
from aris.source import Component, GaussianNoise, GlottalSource
from aris.vocal_tract import VocalTract


class NoiseFilter(Component):
    """Time-varying zero-phase FIR shaping the noise, one log-magnitude envelope per frame.

    ``n_mag`` magnitudes (natural log, clamped to [-15, 3]) give a windowed kernel of
    ``2 * (n_mag - 1)`` taps. Frame ``i`` filters output samples ``[i*hop, (i+1)*hop)``
    (kernels are held within a frame).
    """

    def __init__(self, n_mag: int = 513, window: str = "hann"):
        super().__init__()
        self.split_size = (n_mag,)
        self.n_mag = n_mag
        self.window_fn = window_fn(window)

    def forward(self, ex: AudioTensor, log_mag: AudioTensor) -> AudioTensor:
        hop = log_mag.hop_length
        mag = torch.exp(torch.clamp(log_mag, min=-15.0, max=3.0)) + 0j
        kernel = torch.fft.fftshift(torch.fft.irfft(mag, dim=-1), dim=-1)
        kernel = (kernel * self.window_fn(kernel.shape[-1], device=kernel.device,
                                          dtype=kernel.dtype)).as_tensor()
        samples = min(ex.shape[1], kernel.shape[1] * hop)
        frames = (samples + hop - 1) // hop
        kernel = kernel[:, :frames]
        taps = kernel.shape[-1]
        left = taps // 2  # irfft + fftshift puts zero lag at taps // 2
        padded = F.pad(ex.as_tensor(), (left, taps - 1 - left + hop - 1))
        blocks = padded[:, : frames * hop + taps - 1].unfold(-1, taps + hop - 1, hop)
        out = F.conv1d(blocks.reshape(1, -1, blocks.shape[-1]), kernel.reshape(-1, 1, taps),
                       groups=kernel.shape[0] * kernel.shape[1])
        return AudioTensor(out.view(kernel.shape[0], -1)[:, :samples])


class SourceFilterSynth(torch.nn.Module):
    """Glottal pulse + filtered noise through one vocal-tract filter.

    The decoder has no trainable parameters; every control is a coefficient of
    one of its components, predicted per frame by the encoder.
    """

    def __init__(self, harm_oscillator: GlottalSource, noise_filter: NoiseFilter,
                 end_filter: VocalTract):
        super().__init__()
        # Order defines the channel order of the encoder output.
        self.harm_oscillator = harm_oscillator
        self.noise_generator = GaussianNoise()
        self.noise_filter = noise_filter
        self.end_filter = end_filter

    @property
    def components(self) -> list[tuple[str, Component]]:
        return [(n, m) for n, m in self.named_children() if isinstance(m, Component)]

    @property
    def control_channels(self) -> dict[str, tuple[int, ...]]:
        """Channel indices of each component (and named sub-part) in the flat control vector."""
        channels, offset = {}, 0
        for name, module in self.components:
            width = sum(module.split_size)
            if width:
                channels[name] = tuple(range(offset, offset + width))
                for part, local in module.control_partitions.items():
                    channels[f"{name}.{part}"] = tuple(offset + i for i in local)
            offset += width
        return channels

    @property
    def n_controls(self) -> int:
        return sum(sum(m.split_size) for _, m in self.components)

    def split_controls(self, logits: AudioTensor) -> dict[str, tuple[AudioTensor, ...]]:
        """Flat logits ``[B, frames, C]`` -> ``{"<component>_params": (args...)}``."""
        sizes = [s for _, m in self.components for s in m.split_size]
        chunks = [logits.new_tensor(torch.squeeze(t, 2)) for t in torch.split(logits, sizes, dim=2)]
        params, i = {}, 0
        for name, module in self.components:
            n = len(module.split_size)
            params[f"{name}_params"] = module.transform(*chunks[i:i + n])
            i += n
        return params

    def _harmonic(self, phase, harm_oscillator_params, voicing):
        # Binary voicing gate: the glottal source is on only where F0 > 0.
        harm = self.harm_oscillator(phase, *harm_oscillator_params)
        if voicing is not None:
            harm = harm * F.threshold(torch.clamp(voicing, 0.0, 1.0), 0.5, 0)
        return harm

    def forward(self, phase, harm_oscillator_params, noise_generator_params,
                noise_filter_params, end_filter_params, voicing=None) -> AudioTensor:
        harm = self._harmonic(phase, harm_oscillator_params, voicing)
        noise = self.noise_filter(self.noise_generator(harm, *noise_generator_params),
                                  *noise_filter_params)
        return self.end_filter(harm + noise, *end_filter_params)

    def decompose(self, phase, harm_oscillator_params, noise_generator_params,
                  noise_filter_params, end_filter_params, voicing=None) -> dict[str, AudioTensor]:
        """Like ``forward``, but also returns each branch before and after the tract."""
        harm = self._harmonic(phase, harm_oscillator_params, voicing)
        noise = self.noise_filter(self.noise_generator(harm, *noise_generator_params),
                                  *noise_filter_params)
        harm_speech = self.end_filter(harm, *end_filter_params)
        noise_speech = self.end_filter(noise, *end_filter_params)
        return {"full": harm_speech + noise_speech, "harmonics": harm_speech,
                "noise": noise_speech, "harm_source": harm, "noise_source": noise}
