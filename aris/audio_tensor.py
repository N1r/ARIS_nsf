"""AudioTensor: a ``torch.Tensor`` that knows its time resolution (``hop_length``).

The synthesizer mixes two rates. Audio-rate signals (waveforms, excitation, noise)
have ``hop_length = 1``; frame-rate controls (R_d, formants, filter coefficients,
noise envelopes) have, e.g., ``hop_length = 160`` (10 ms at 16 kHz). Time is
dimension 1.

When AudioTensors with different hop lengths meet in a torch operation, they are
linearly upsampled to their finest common rate and truncated to a common length,
so frame-rate controls and audio-rate signals can be combined directly. Based on
the AudioTensor of GOLF (https://github.com/yoyololicon/golf).
"""

from __future__ import annotations

import math
from functools import reduce
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


def linear_upsample(x: Tensor, hop_length: int) -> Tensor:
    """Linearly upsample the last dimension of a tensor by an integer factor.

    Frame ``i`` maps to sample ``i * hop_length`` (``align_corners=True``), so the
    first and last frames are reproduced exactly.

    Args:
        x: Input tensor of shape (..., time_frames).
        hop_length: Integer upsampling factor (e.g., 160).

    Returns:
        Upsampled tensor of shape (..., (time_frames - 1) * hop_length + 1).
    """
    orig_shape = x.shape
    flat_x = x.reshape(-1, 1, orig_shape[-1])
    target_len = (orig_shape[-1] - 1) * hop_length + 1
    upsampled = F.interpolate(
        flat_x,
        size=target_len,
        mode="linear",
        align_corners=True,
    )
    return upsampled.view(*orig_shape[:-1], target_len)


def check_hop_length(func: Callable) -> Callable:
    """Decorator to ensure the AudioTensor has a valid positive hop length."""

    def wrapper(self: AudioTensor, *args: Any, **kwargs: Any) -> Any:
        if self.hop_length < 0:
            raise ValueError(
                f"Cannot call {func.__name__} on an AudioTensor with invalid hop_length={self.hop_length}"
            )
        return func(self, *args, **kwargs)

    return wrapper


class AudioTensor(Tensor):
    """PyTorch Tensor subclass with multi-rate acoustic metadata (hop_length).

    Tracks temporal discretization rate and provides explicit upsampling/downsampling,
    temporal truncation, and multi-signal broadcasting.
    """

    hop_length: int

    def __new__(
        cls,
        x: Union[Tensor, np.ndarray, Sequence],
        hop_length: int = 1,
        *args: Any,
        requires_grad: Optional[bool] = None,
        **kwargs: Any,
    ) -> AudioTensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif not isinstance(x, Tensor):
            x = torch.as_tensor(x)

        if isinstance(x, Tensor) and requires_grad is None:
            res = x.as_subclass(cls)
        else:
            req_grad = x.requires_grad if requires_grad is None else bool(requires_grad)
            res = Tensor._make_subclass(cls, x, req_grad)

        res.hop_length = int(hop_length)
        return res

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __repr__(self) -> str:
        return f"AudioTensor(hop_length={self.hop_length}, shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device})"

    @check_hop_length
    def set_hop_length(self, target_hop: int) -> AudioTensor:
        """Change the hop length to ``target_hop`` by integer decimation or upsampling."""
        if target_hop > self.hop_length:
            assert target_hop % self.hop_length == 0, (
                f"Cannot decimate {self.hop_length} to {target_hop}"
            )
            return self.increase_hop_length(target_hop // self.hop_length)
        elif target_hop < self.hop_length:
            assert self.hop_length % target_hop == 0, (
                f"Cannot upsample {self.hop_length} to {target_hop}"
            )
            return self.reduce_hop_length(self.hop_length // target_hop)
        return self

    @check_hop_length
    def increase_hop_length(self, factor: int) -> AudioTensor:
        """Keep every ``factor``-th step (no anti-aliasing filter); hop_length grows."""
        assert factor > 0, "Decimation factor must be positive"
        if factor == 1:
            return self

        data = self[:, ::factor].clone()
        data.hop_length = self.hop_length * factor
        return data

    @check_hop_length
    def reduce_hop_length(self, factor: Optional[int] = None) -> AudioTensor:
        """Linearly upsample along time; hop_length becomes ``hop_length // factor``.

        With ``factor=None`` the result is at the audio sample rate (hop_length = 1).
        """
        if factor is None:
            factor = self.hop_length
        else:
            assert self.hop_length % factor == 0 and factor <= self.hop_length, (
                f"Invalid upsample factor {factor} for hop_length {self.hop_length}"
            )

        if factor == 1:
            return self

        # Temporal dimension is dimension 1 in standard (B, T) or (B, T, C)
        self_copy = self.clone()
        if self.ndim > 2:
            self_copy = self_copy.transpose(1, -1)
        expanded = linear_upsample(self_copy, factor)
        if self.ndim > 2:
            expanded = expanded.transpose(1, -1)

        expanded.hop_length = self.hop_length // factor
        return expanded

    def as_tensor(self) -> Tensor:
        """Zero-copy view of this AudioTensor as a standard torch.Tensor."""
        return self.as_subclass(Tensor)

    @property
    @check_hop_length
    def steps(self) -> int:
        """Number of temporal frames / samples along dimension 1."""
        if self.ndim < 2:
            return torch.iinfo(torch.int32).max
        return self.size(1)

    @check_hop_length
    def truncate(self, target_steps: int) -> AudioTensor:
        """Truncate temporal dimension (dim 1) to match target_steps."""
        if target_steps >= self.steps or self.ndim < 2:
            return self
        return self.narrow(1, 0, target_steps)

    def new_tensor(self, data: Tensor) -> AudioTensor:
        """Wrap newly computed data with the same hop_length."""
        if isinstance(data, AudioTensor):
            return data
        return AudioTensor(data, hop_length=self.hop_length)

    @check_hop_length
    def unfold(self, size: int, step: int = 1) -> AudioTensor:
        """Sliding windows along dimension 1 (unlike ``Tensor.unfold``, the dimension is fixed)."""
        assert self.ndim == 2, "Unfold only supported for 2D tensors"
        data = super().unfold(1, size, step)
        data.hop_length = self.hop_length * step
        return data

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Run a torch function, aligning the rates of AudioTensor arguments first."""
        # Fast path for common unary and binary operations (avoids tree_flatten/tree_map).
        if not kwargs:
            nargs = len(args)
            if nargs == 1:
                arg0 = args[0]
                if isinstance(arg0, cls):
                    output_hop = arg0.hop_length
                    result = super().__torch_function__(func, types, args, kwargs)
                    if isinstance(result, cls) and output_hop > 0:
                        result.hop_length = output_hop
                    return result
            elif nargs == 2:
                a, b = args
                is_a = isinstance(a, cls)
                is_b = isinstance(b, cls)
                if is_a and not is_b:
                    output_hop = a.hop_length
                    result = super().__torch_function__(func, types, args, kwargs)
                    if isinstance(result, cls) and output_hop > 0:
                        result.hop_length = output_hop
                    return result
                elif is_b and not is_a:
                    output_hop = b.hop_length
                    result = super().__torch_function__(func, types, args, kwargs)
                    if isinstance(result, cls) and output_hop > 0:
                        result.hop_length = output_hop
                    return result
                elif is_a and is_b and a.hop_length == b.hop_length:
                    # Same hop length and same temporal length: direct dispatch
                    if a.steps == b.steps:
                        output_hop = a.hop_length
                        result = super().__torch_function__(func, types, args, kwargs)
                        if isinstance(result, cls) and output_hop > 0:
                            result.hop_length = output_hop
                        return result

        kwargs = kwargs or {}
        flatten, spec = tree_flatten((args, kwargs))
        mask = tuple(
            isinstance(t, AudioTensor) and getattr(t, "hop_length", -1) > 0 for t in flatten
        )
        audio_count = sum(mask)

        # Fast path: 0 or 1 AudioTensor requires no multi-rate temporal broadcasting
        if audio_count <= 1:
            output_hop = next((t.hop_length for t in flatten if isinstance(t, AudioTensor)), -1)
            result = super().__torch_function__(func, types, args, kwargs)
            if output_hop > 0:

                def tag_hop(t):
                    if isinstance(t, cls):
                        t.hop_length = output_hop
                    return t

                return tree_map(tag_hop, result)
            return result

        # Multi-audio path: align temporal rates and lengths
        audio_tensors = tuple(t for is_audio, t in zip(mask, flatten, strict=True) if is_audio)
        aligned_tensors = cls.broadcasting(*audio_tensors)
        min_steps = min(a.steps for a in aligned_tensors)
        truncated_tensors = tuple(a.truncate(min_steps) for a in aligned_tensors)

        audio_iter = iter(truncated_tensors)
        new_flatten = tuple(
            next(audio_iter) if is_audio else t for is_audio, t in zip(mask, flatten, strict=True)
        )
        output_hop = truncated_tensors[0].hop_length

        broadcasted_args, broadcasted_kwargs = tree_unflatten(new_flatten, spec)
        result = super().__torch_function__(func, types, broadcasted_args, broadcasted_kwargs)

        def post_process(t):
            if isinstance(t, cls):
                t.hop_length = output_hop
                if t.ndim == 1:
                    t.hop_length = -1
            return t

        return tree_map(post_process, result)

    @classmethod
    def broadcasting(cls, *tensors: AudioTensor) -> Tuple[AudioTensor, ...]:
        """Upsample to the greatest common divisor of the hop lengths and match ranks."""
        assert len(tensors) > 0, "Broadcasting requires at least one tensor"
        hop_lengths = tuple(t.hop_length for t in tensors)
        gcd_hop = math.gcd(*hop_lengths)
        upsampled = tuple(t.reduce_hop_length(t.hop_length // gcd_hop) for t in tensors)
        max_ndim = max(t.ndim for t in upsampled)
        aligned = tuple(
            (
                reduce(lambda x, _: x.unsqueeze(-1), [None] * (max_ndim - t.ndim), t)
                if t.ndim < max_ndim
                else t
            )
            for t in upsampled
        )
        return aligned
