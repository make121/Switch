"""Verify the actor boundary at recovery -> Buffer -> task without a simulator."""
import unittest
from unittest.mock import Mock
from types import SimpleNamespace as NS
import numpy as np
import torch
from humanoidverse.agents.callbacks.skill_scheduler_callback import SkillSchedulerEvalCallback


class PolicyRoutingTest(unittest.TestCase):
    def test_collect_entry_without_threshold_preserves_physics(self):
        cb = SkillSchedulerEvalCallback.__new__(SkillSchedulerEvalCallback)
        torch.nn.Module.__init__(cb)
        cb._slot = 0
        cb.injection_history = []
        cb._update_recovery_slot_role = Mock()
        cb._align_to_robot = Mock()
        cb.graph = NS(is_buffer=np.array([False]), skill_ids=np.array([0]),
                      frame_idxs=np.array([0]), skill_names=['getup'])
        cb.scheduler = NS(last_trigger='recovery_entry', A=1., current_cmd=0,
                          last_best_sim=100.)
        cb.recovery_fsm = NS(selected=NS(distance=100.), threshold=None)
        cb.ref_builder = NS(build_trajectory=lambda path: {'fps': 30., 'dof': np.zeros((2, 1))})
        lib = NS(_motion_data_list=[None], load_motions=Mock(),
                 _curr_motion_ids=[0], curr_motion_keys=['getup'], _motion_lengths=[1.])
        cb.env = NS(_motion_lib=lib, motion_start_times=torch.zeros(1),
                    episode_length_buf=torch.tensor([50]), dt=.02,
                    motion_len=torch.zeros(1), common_step_counter=50,
                    reset_envs_idx=Mock())
        cb._inject([0], physical_reset_mode='never')
        cb.env.reset_envs_idx.assert_not_called()
        self.assertIsNone(cb.injection_history[-1]['entry_threshold'])
        self.assertFalse(cb.injection_history[-1]['physical_reset'])

    def test_recovery_buffer_and_task_use_expected_weights(self):
        cb = SkillSchedulerEvalCallback.__new__(SkillSchedulerEvalCallback)
        torch.nn.Module.__init__(cb)
        cb._policy_role = None
        cb._recovery_policy = lambda obs: torch.tensor([[2.]])
        cb.training_loop = NS(eval_policy=lambda obs: torch.tensor([[1.]]))
        cb.env = NS(action_queue=torch.ones(1, 2, 1))
        cb.graph = NS(skill_ids=np.array([0, 1, -1]), skill_roles=['task', 'recovery'])
        events = []
        cb.recovery_fsm = NS(active=True, damping_active=False,
                             record=lambda event, **fields: events.append(fields))
        guidance = NS(node_id=1)
        cb.scheduler = NS(current_guidance=lambda: guidance)
        state = {'obs': {}}
        self.assertEqual(cb._apply_recovery_policy(state)['actions'].item(), 2.)
        self.assertEqual(cb.env.action_queue.sum().item(), 0.)
        guidance.node_id = 2  # Exit Buffer has no skill ID, even while FSM active.
        self.assertEqual(cb._apply_recovery_policy(state)['actions'].item(), 1.)
        guidance.node_id = 0
        self.assertEqual(cb._apply_recovery_policy(state)['actions'].item(), 1.)
        guidance.node_id = 1
        cb.recovery_fsm.active = False
        self.assertEqual(cb._apply_recovery_policy(state)['actions'].item(), 1.)
        self.assertEqual([e['policy'] for e in events], ['recovery', 'task_and_buffer'])


if __name__ == '__main__':
    unittest.main()
