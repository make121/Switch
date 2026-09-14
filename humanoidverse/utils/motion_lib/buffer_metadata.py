"""Pure helpers for per-frame Skill Graph Buffer metadata."""

from typing import Optional, Tuple

import numpy as np


def prepare_buffer_metadata(
    is_buffer,
    kappa,
    num_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate Buffer metadata and derive legacy missing kappa countdowns.

    Older augmented datasets contain ``is_buffer`` but no ``kappa``. For
    each contiguous True run, derive N, N-1, ..., 1 so those datasets remain
    usable without weakening validation of newly generated data.
    """
    if is_buffer is None:
        mask = np.zeros(num_frames, dtype=bool)
    else:
        mask = np.asarray(is_buffer, dtype=bool)
    if mask.shape != (num_frames,):
        raise ValueError(
            f"is_buffer must have shape ({num_frames},), got {mask.shape}"
        )

    if kappa is None:
        countdown = np.zeros(num_frames, dtype=np.float32)
        start = 0
        while start < num_frames:
            if not mask[start]:
                start += 1
                continue
            end = start + 1
            while end < num_frames and mask[end]:
                end += 1
            countdown[start:end] = np.arange(
                end - start, 0, -1, dtype=np.float32
            )
            start = end
    else:
        countdown = np.asarray(kappa, dtype=np.float32)

    if countdown.shape != (num_frames,):
        raise ValueError(
            f"kappa must have shape ({num_frames},), got {countdown.shape}"
        )
    if np.any(countdown < 0):
        raise ValueError("kappa must be non-negative")
    if np.any((countdown > 0) != mask):
        raise ValueError(
            "kappa and is_buffer disagree: kappa must be positive exactly "
            "on buffer frames"
        )
    return mask, countdown


def buffer_target_indices(is_buffer) -> np.ndarray:
    """Map each frame to the reward/guidance frame used by Switch.

    Plain frames map to themselves. Every contiguous Buffer run maps to its
    first non-Buffer successor, i.e. the cross-skill transition endpoint.
    A run at the end of a motion is invalid because it has no target state.
    """
    mask = np.asarray(is_buffer, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(f"is_buffer must be 1-D, got shape {mask.shape}")

    targets = np.arange(len(mask), dtype=np.int64)
    start = 0
    while start < len(mask):
        if not mask[start]:
            start += 1
            continue
        end = start + 1
        while end < len(mask) and mask[end]:
            end += 1
        if end == len(mask):
            raise ValueError(
                "Buffer segment reaches the end of the motion and has no "
                "non-Buffer successor target"
            )
        targets[start:end] = end
        start = end
    return targets
