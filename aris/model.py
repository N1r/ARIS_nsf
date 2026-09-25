"""ARIS analysis-by-synthesis model: neural encoder + deterministic source-filter decoder."""

from __future__ import annotations

from typing import Sequence

import lightning.pytorch as pl
import torch
from torch import nn

from aris.audio_tensor import AudioTensor
from aris.encoder import Encoder
from aris.losses import AdaptivePeriodicityObjective, formant_objective
from aris.vocoder import SourceFilterSynth


class ARIS(pl.LightningModule):
    """Trained to reconstruct its input with ``L_MSS + w_F L_F + w_P L_P``.

    ``head_components`` groups decoder components into encoder output heads,
    e.g. ``[["harm_oscillator"], ["end_filter"], ["noise_filter"]]``; names may
    address sub-parts such as ``end_filter.formants``.
    """

    def __init__(
        self,
        decoder: SourceFilterSynth,
        criterion: nn.Module | None = None,
        sample_rate: int = 16000,
        head_components: Sequence[Sequence[str]] = (("harm_oscillator",), ("end_filter",),
                                                    ("noise_filter",)),
        encoder: dict | None = None,
        formant_loss_weight: float = 2.0,
        formant_smooth_weight: float = 0.05,
        residual_reg_weight: float = 0.02,
        periodicity_loss_weight: float = 1.0,
        periodicity: dict | None = None,
        validation_seed: int | None = 0,
    ):
        super().__init__()
        self.decoder = decoder
        self.criterion = criterion
        self.sample_rate = sample_rate
        channels = decoder.control_channels
        head_indices = [[i for name in group for i in channels[name]] for group in head_components]
        self.encoder = Encoder(decoder.n_controls, head_indices, sample_rate=sample_rate,
                               **(encoder or {}))
        self.formant_loss_weight = formant_loss_weight
        self.formant_smooth_weight = formant_smooth_weight
        self.residual_reg_weight = residual_reg_weight
        self.periodicity_loss_weight = periodicity_loss_weight
        self.periodicity = (AdaptivePeriodicityObjective(sample_rate=sample_rate, **(periodicity or {}))
                            if periodicity_loss_weight > 0 else None)
        self.validation_seed = validation_seed

    # ---- analysis / synthesis ----------------------------------------------------

    def analyze(self, x: AudioTensor, f0: AudioTensor) -> dict:
        """Audio + sample-rate F0 -> decoder controls ``{"<component>_params": (...)}``."""
        return self.decoder.split_controls(self.encoder(x, f0))

    def render(self, controls: dict, f0: AudioTensor, unvoiced_f0=150.0, decompose=False):
        """Synthesize speech from controls with a (possibly edited) F0 contour per sample;
        F0 = 0 marks unvoiced samples.

        The glottal phase keeps running at ``unvoiced_f0`` through unvoiced regions, where the
        voicing gate silences it.
        """
        fill = unvoiced_f0 if torch.is_tensor(unvoiced_f0) else torch.full_like(f0, unvoiced_f0)
        phase = torch.where(f0 == 0, fill, f0) / self.sample_rate
        args = dict(controls, phase=phase, voicing=(f0 > 0).float())
        return self.decoder.decompose(**args) if decompose else self.decoder(**args)

    def forward(self, x: AudioTensor, f0: AudioTensor) -> AudioTensor:
        """Copy synthesis."""
        return self.render(self.analyze(x, f0), f0)

    # ---- training ----------------------------------------------------------------

    def _losses(self, batch, x_hat, controls):
        x, f0, *formant_targets = batch
        losses = {"spectral": self.criterion(x_hat[:, : x.shape[1]], x[:, : x_hat.shape[1]])}
        total = losses["spectral"]
        if self.periodicity is not None:
            hop = self.periodicity.periodicity.hop_length
            losses["periodicity"] = self.periodicity(
                x_hat.as_tensor()[:, : x.shape[1]], x.as_tensor()[:, : x_hat.shape[1]],
                f0.as_tensor()[:, ::hop])
            total = total + self.periodicity_loss_weight * losses["periodicity"]
        if self.formant_loss_weight > 0:
            ctrl = controls["end_filter_params"][0].as_tensor()
            losses["formant"], losses["residual"] = formant_objective(
                self.decoder.end_filter, ctrl, *formant_targets, smooth_weight=self.formant_smooth_weight)
            total = (total + self.formant_loss_weight * losses["formant"]
                     + self.residual_reg_weight * losses["residual"])
        losses["loss"] = total
        return losses

    def training_step(self, batch, batch_idx):
        x, f0 = AudioTensor(batch[0]), AudioTensor(batch[1])
        batch = (x, f0, *batch[2:])
        controls = self.analyze(x, f0)
        # Unvoiced samples get a random phase-driving F0 per item; the voicing gate mutes them.
        random_f0 = f0.as_tensor().new_empty(f0.shape[0], 1).uniform_(50, 500)
        x_hat = self.render(controls, f0, unvoiced_f0=random_f0)
        losses = self._losses(batch, x_hat, controls)
        self.log_dict({f"train_{k}": v for k, v in losses.items()}, prog_bar=False)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        x, f0 = AudioTensor(batch[0]), AudioTensor(batch[1])
        batch = (x, f0, *batch[2:])
        # Fixed noise per batch so validation losses are comparable across epochs.
        devices = [x.device.index] if x.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            if self.validation_seed is not None:
                torch.manual_seed(self.validation_seed + batch_idx)
            controls = self.analyze(x, f0)
            x_hat = self.render(controls, f0)
        losses = self._losses(batch, x_hat, controls)
        self.log_dict({f"val_{k}": v for k, v in losses.items()}, prog_bar=False)
