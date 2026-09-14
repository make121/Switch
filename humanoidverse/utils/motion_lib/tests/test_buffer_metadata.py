import numpy as np
import pytest

from humanoidverse.utils.motion_lib.buffer_metadata import (
    buffer_target_indices,
    prepare_buffer_metadata,
)


def test_plain_motion_defaults_to_zero_metadata():
    mask, kappa = prepare_buffer_metadata(None, None, 4)
    np.testing.assert_array_equal(mask, [False, False, False, False])
    np.testing.assert_array_equal(kappa, [0, 0, 0, 0])


def test_legacy_mask_derives_countdown_per_contiguous_run():
    mask, kappa = prepare_buffer_metadata(
        [False, True, True, True, False, True, False], None, 7
    )
    np.testing.assert_array_equal(mask, [False, True, True, True, False, True, False])
    np.testing.assert_array_equal(kappa, [0, 3, 2, 1, 0, 1, 0])


def test_explicit_kappa_is_preserved():
    _, kappa = prepare_buffer_metadata(
        [False, True, True, False], [0, 2, 1, 0], 4
    )
    np.testing.assert_array_equal(kappa, [0, 2, 1, 0])


@pytest.mark.parametrize(
    "mask,kappa",
    [
        ([False, True], [0, 0]),
        ([False, False], [0, 1]),
        ([False, True], [0, -1]),
    ],
)
def test_inconsistent_metadata_is_rejected(mask, kappa):
    with pytest.raises(ValueError):
        prepare_buffer_metadata(mask, kappa, 2)


def test_buffer_frames_target_first_non_buffer_successor():
    targets = buffer_target_indices(
        [False, True, True, False, True, False, False]
    )
    np.testing.assert_array_equal(targets, [0, 3, 3, 3, 5, 5, 6])


def test_buffer_run_requires_a_successor_target():
    with pytest.raises(ValueError, match="no non-Buffer successor"):
        buffer_target_indices([False, True, True])
