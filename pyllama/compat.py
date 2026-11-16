"""Small helpers that keep the code working on both transformers 4.x and 5.x."""

from __future__ import annotations

import math

import transformers
from packaging.version import Version

_TF = Version(transformers.__version__)


def dtype_kwarg(dtype) -> dict:
    """`torch_dtype=` was renamed to `dtype=` in transformers 4.56 (the old name warns in 5.x)."""
    return {"dtype": dtype} if _TF >= Version("4.56") else {"torch_dtype": dtype}


def warmup_steps(n_examples: int, batch_size: int, grad_accum: int, epochs: float, ratio: float = 0.03) -> int:
    """Warmup as a step count: `warmup_ratio` was removed in transformers 5."""
    steps_per_epoch = math.ceil(n_examples / (batch_size * grad_accum))
    return max(1, round(steps_per_epoch * epochs * ratio))
