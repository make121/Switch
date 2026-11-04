"""Pre-step reference resets must reach physics and avoid stale FK inference."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

import torch

from humanoidverse.agents.callbacks.skill_scheduler_callback import SkillSchedulerEvalCallback


class SchedulerResetTest(unittest.TestCase):
    def callback(self):
        cb = SkillSchedulerEvalCallback.__new__(SkillSchedulerEvalCallback)
        torch.nn.Module.__init__(cb)
        cb._reference_dirty = True
        cb._reset_warmup_pending = False
        cb._apply_recovery_policy = Mock(side_effect=lambda state: state)
        cb.training_loop = NS(eval_policy=Mock(return_value=torch.tensor([[7.]])))
        cb.env = NS(
            device="cpu", reset_buf=torch.ones(1, dtype=torch.bool),
            need_to_refresh_envs=torch.zeros(1, dtype=torch.bool),
            action_queue=torch.ones(1, 2, 1), history=torch.ones(1, 10),
            config=NS(normalization=NS(clip_observations=100.)),
            obs_buf_dict={"actor_obs": torch.tensor([[3.]])},
            _pre_compute_observations_callback=Mock(), _compute_observations=Mock(),
        )
        return cb

    def test_reset_is_committed_before_a_refresh_can_discard_it(self):
        cb = self.callback()
        sim = NS(all_root_states=torch.zeros(1, 13), dof_state=torch.zeros(1, 2))
        physics = {}
        calls = []

        def reset(ids):
            calls.append("reset")
            sim.all_root_states[ids, :3] = torch.tensor([1., 2., .8])
            sim.dof_state[ids] = torch.tensor([.5, .1])
            cb.env.need_to_refresh_envs[ids] = True
            cb.env.history[ids] = 0

        def root(ids, state):
            calls.append("root")
            physics["root"] = state.clone()

        def dof(ids, state):
            calls.append("dof")
            physics["dof"] = state.clone()

        sim.set_actor_root_state_tensor = root
        sim.set_dof_state_tensor = dof
        cb.env.simulator = sim
        cb.env.reset_envs_idx = reset
        cb._reset_reference_episode()
        self.assertEqual(calls, ["reset", "root", "dof"])
        # Model a subsequent refresh from the already-committed physics state.
        sim.all_root_states.copy_(physics["root"])
        torch.testing.assert_close(sim.all_root_states[0, :3], torch.tensor([1., 2., .8]))
        torch.testing.assert_close(physics["dof"], torch.tensor([[.5, .1]]))
        self.assertFalse(bool(cb.env.need_to_refresh_envs.any()))
        self.assertEqual(float(cb.env.action_queue.sum()), 0.)
        self.assertEqual(float(cb.env.history.sum()), 0.)
        self.assertTrue(cb._reset_warmup_pending)

    def test_exactly_one_counted_step_before_using_new_rigid_body_observations(self):
        cb = self.callback()
        cb._reset_warmup_pending = True
        state = {"actions": torch.tensor([[9.]]), "obs": {"stale": True}}
        cb._refresh_reference_policy(state)
        self.assertEqual(float(state["actions"].sum()), 0.)
        self.assertFalse(bool(state["dones"].any()))
        cb.env._pre_compute_observations_callback.assert_not_called()
        cb.env._compute_observations.assert_not_called()
        cb.training_loop.eval_policy.assert_not_called()
        cb._apply_recovery_policy.assert_not_called()
        # The outer evaluator's next normal step supplies its fresh policy action.
        state["actions"] = torch.tensor([[4.]])
        cb._refresh_reference_policy(state)
        self.assertEqual(state["actions"].item(), 4.)

    def test_live_reference_change_reinfers_without_erasing_history(self):
        cb = self.callback()
        state = {"actions": torch.tensor([[9.]]), "obs": {"old": True}}
        cb._refresh_reference_policy(state)
        self.assertEqual(state["actions"].item(), 7.)
        cb.env._pre_compute_observations_callback.assert_called_once()
        cb.env._compute_observations.assert_called_once()
        self.assertEqual(float(cb.env.history.sum()), 10.)
        self.assertFalse(cb._reference_dirty)


if __name__ == "__main__":
    unittest.main()
