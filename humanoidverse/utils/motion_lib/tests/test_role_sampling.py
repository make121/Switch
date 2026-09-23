"""Regression tests for task/recovery/transition bucket sampling."""

import torch

from humanoidverse.utils.motion_lib.motion_lib_base import MotionLibBase


def make_role_sampling_lib():
    lib = MotionLibBase.__new__(MotionLibBase)
    lib._device = torch.device("cpu")
    lib._num_unique_motions = 10
    lib._role_indices = {
        "task_skill": [0, 1, 2],
        "recovery_skill": [3, 4, 5, 6, 7],
        "task_transition": [8],
        "recovery_transition": [9],
    }
    return lib


def test_role_sampling_assigns_bucket_mass_not_file_count():
    lib = make_role_sampling_lib()
    expected = {
        "task_skill": 0.40,
        "recovery_skill": 0.35,
        "task_transition": 0.15,
        "recovery_transition": 0.10,
    }
    lib._build_role_sampling_prob(expected)

    assert abs(float(lib._sampling_prob.sum()) - 1.0) < 1e-6
    for bucket, indices in lib._role_indices.items():
        actual = float(lib._sampling_prob[indices].sum())
        assert abs(actual - expected[bucket]) < 1e-6, (bucket, actual)


def test_role_sampling_rejects_mass_for_empty_bucket():
    lib = make_role_sampling_lib()
    lib._role_indices["recovery_transition"] = []
    try:
        lib._build_role_sampling_prob({
            "task_skill": 0.4,
            "recovery_skill": 0.4,
            "task_transition": 0.1,
            "recovery_transition": 0.1,
        })
    except ValueError as exc:
        assert "no motion entries" in str(exc)
    else:
        raise AssertionError("nonzero mass for an empty bucket must fail")


def test_recovery_rsi_biases_only_recovery_single_skills():
    lib = make_role_sampling_lib()
    lib._motion_lengths = torch.ones(10)
    lib._curr_motion_ids = torch.arange(10)
    lib._recovery_motion_mask = torch.zeros(10, dtype=torch.bool)
    lib._recovery_motion_mask[3:8] = True
    lib._recovery_rsi_enable = True
    lib._recovery_rsi_entry_probability = 1.0
    lib._recovery_rsi_entry_fraction = 0.2

    ids = torch.arange(10)
    times = lib.sample_time(ids)
    assert torch.all(times[3:8] <= 0.2)
    # Transitions are not classified as recovery single skills.
    assert times[8:].shape[0] == 2


if __name__ == "__main__":
    test_role_sampling_assigns_bucket_mass_not_file_count()
    print("PASS test_role_sampling_assigns_bucket_mass_not_file_count")
    test_role_sampling_rejects_mass_for_empty_bucket()
    print("PASS test_role_sampling_rejects_mass_for_empty_bucket")
    test_recovery_rsi_biases_only_recovery_single_skills()
    print("PASS test_recovery_rsi_biases_only_recovery_single_skills")
