"""Smoke test: build the paper model, run a training step, copy synthesis and edits (CPU)."""

import numpy as np
import torch
import yaml

from aris import edit, resynthesize
from aris.audio_tensor import AudioTensor
from aris.edit import build

CONFIG = yaml.safe_load(open("configs/aris_16k.yaml"))["model"]


def test_train_step_and_edits():
    torch.manual_seed(0)
    model = build({**CONFIG, "encoder": {**CONFIG["encoder"], "model_dim": 32, "num_layers": 1,
                                         "num_heads": 2}})
    from aris.edit import _instantiate
    model.criterion = _instantiate(CONFIG["criterion"])
    sr, n = 16000, 16000
    t = np.arange(n) / sr
    f0 = np.where(t < 0.8, 200.0, 0.0).astype(np.float32)
    x = (0.3 * np.sin(2 * np.pi * np.cumsum(f0) / sr) + 0.01 * np.random.randn(n)).astype(np.float32)
    frames = n // 160
    tracks = [torch.full((1, frames), v) for v in (600.0, 1600.0, 2700.0, 80.0, 100.0, 150.0)]
    batch = (torch.from_numpy(x)[None], torch.from_numpy(f0)[None], *tracks, torch.ones(1, frames))
    loss = model.training_step(batch, 0)
    loss.backward()
    assert torch.isfinite(loss) and model.encoder.heads[0][-1].weight.grad is not None

    model.eval()
    f0_5ms = f0[::80]
    y = resynthesize(model, x, f0_5ms)
    assert y.shape == x.shape and np.isfinite(y).all()

    # Formant edits move exactly the requested section, within its range.
    xt, ft = AudioTensor(torch.from_numpy(x)[None]), AudioTensor(torch.from_numpy(f0)[None])
    controls = model.analyze(xt, ft)
    tract = model.decoder.end_filter
    before = tract.get_formant_params(controls["end_filter_params"][0].as_tensor())
    after = tract.get_formant_params(
        edit(model, controls, f2=1.1)["end_filter_params"][0].as_tensor())
    assert torch.allclose(after["f2"] / before["f2"], torch.tensor(1.1), atol=1e-3)
    assert torch.equal(after["f1"], before["f1"]) and torch.equal(after["f3"], before["f3"])

    # A region edit leaves the samples before the region untouched (same noise seed).
    y_local = resynthesize(model, x, f0_5ms, f1=1.2, rd=1.5, f0_scale=1.1, region=(0.4, 0.6))
    assert np.allclose(y_local[:4000], y[:4000], atol=1e-5)
    assert not np.allclose(y_local[7000:9000], y[7000:9000])
