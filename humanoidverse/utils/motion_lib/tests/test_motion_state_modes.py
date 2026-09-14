"""Regression tests for Buffer physical/guidance separation and RSI.

Run directly (pytest is not required):
    python humanoidverse/utils/motion_lib/tests/test_motion_state_modes.py
"""

import torch

from humanoidverse.utils.motion_lib.motion_lib_base import MotionLibBase


def make_toy_motion_lib():
    """Four frames: source, two Buffer frames, transition endpoint."""
    lib = MotionLibBase.__new__(MotionLibBase)
    lib._device = torch.device("cpu")
    lib._motion_lengths = torch.tensor([0.3])
    lib._motion_num_frames = torch.tensor([4])
    lib._motion_dt = torch.tensor([0.1])
    lib.length_starts = torch.tensor([0])
    lib._motion_is_buffer = torch.tensor([False, True, True, False])
    lib._motion_kappas = torch.tensor([0.0, 2.0, 1.0, 0.0])
    lib._motion_buffer_targets = torch.tensor([0, 3, 3, 3])

    lib.dof_pos = torch.arange(4, dtype=torch.float32).view(4, 1)
    lib.dvs = torch.arange(4, dtype=torch.float32).view(4, 1)
    lib.gts = torch.zeros(4, 1, 3)
    lib.gts[:, 0, 0] = torch.arange(4, dtype=torch.float32)
    lib.grs = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(4, 1, 1)
    lib.gvs = torch.zeros(4, 1, 3)
    lib.gavs = torch.zeros(4, 1, 3)
    lib._motion_aa = torch.arange(4, dtype=torch.float32).view(4, 1)
    lib._motion_bodies = torch.zeros(1, 17)
    lib.has_contact_mask = None
    return lib


def test_buffer_physical_and_guidance_states_are_distinct():
    lib = make_toy_motion_lib()
    motion_ids = torch.tensor([0])
    # Halfway between the two stored Buffer frames.
    motion_times = torch.tensor([0.15])
    physical = lib.get_physical_state(motion_ids, motion_times)
    guidance = lib.get_guidance_state(motion_ids, motion_times)

    assert physical["is_buffer"].item()
    assert guidance["is_buffer"].item()
    assert abs(physical["dof_pos"].item() - 1.5) < 1e-5
    assert guidance["dof_pos"].item() == 3.0
    assert physical["kappa"].item() == guidance["kappa"].item() == 2.0


def test_rsi_never_samples_buffer_frame():
    lib = make_toy_motion_lib()
    motion_ids = torch.zeros(20_000, dtype=torch.long)
    motion_times = lib.sample_time(motion_ids)
    frame0, _, _ = lib._calc_frame_blend(
        motion_times,
        lib._motion_lengths[motion_ids],
        lib._motion_num_frames[motion_ids],
        lib._motion_dt[motion_ids],
    )
    assert not lib._motion_is_buffer[frame0].any()


if __name__ == "__main__":
    test_buffer_physical_and_guidance_states_are_distinct()
    print("PASS test_buffer_physical_and_guidance_states_are_distinct")
    test_rsi_never_samples_buffer_frame()
    print("PASS test_rsi_never_samples_buffer_frame")
