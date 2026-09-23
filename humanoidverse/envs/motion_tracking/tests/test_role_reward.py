"""Role-aware reward indexing uses source motion IDs, not environment slots."""

from types import SimpleNamespace

import torch

from humanoidverse.envs.motion_tracking.general_tracking import (
    LeggedRobotGeneralTracking,
)


def test_role_rewards_with_more_envs_than_unique_motions():
    env = LeggedRobotGeneralTracking.__new__(LeggedRobotGeneralTracking)
    env.device = "cpu"
    env._motion_lib = SimpleNamespace(
        _num_unique_motions=4,
        _role_indices={
            "task_skill": [0],
            "task_transition": [1],
            "recovery_skill": [2],
            "recovery_transition": [3],
        },
    )
    env.motion_ids = torch.arange(8)  # loaded-motion slots, not source IDs
    env.curr_motion_ids = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    env.penalised_contact_indices = torch.tensor([0])
    env.simulator = SimpleNamespace(contact_forces=torch.ones(8, 1, 3))

    assert env._reward_task_collision().tolist() == [1, 1, 0, 0, 1, 1, 0, 0]
    assert env._recovery_reward_mask().tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
