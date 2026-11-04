"""CPU regression tests; run with python -m unittest <module>."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from humanoidverse.deploy.skill_scheduler.recovery_selector import (
    RecoveryEntrySelector, gravity_in_body)
from humanoidverse.deploy.skill_scheduler.recovery_fsm import RecoveryFSM, RecoveryState


def fixtures(config=None):
    motions = []
    for sid in range(3):
        n = 50
        q = np.tile(np.arange(n)[:, None] * .01 + sid, (1, 3))
        motions.append(dict(dof=q, root_rot=np.tile(
            Rotation.from_euler("x", (sid - 1) * 1.5).as_quat(), (n, 1)),
            root_trans_offset=np.tile([0., 0., .3], (n, 1)),
            contact_mask=np.zeros((n, 2)), fps=30))
    graph = NS(skill_roles=["task", "recovery", "recovery"],
               skill_lengths=[50] * 3, skill_range=lambda sid: (sid * 50, (sid + 1) * 50),
               skill_index=int, skill_ids=np.repeat(np.arange(3), 50))
    root = torch.zeros(1, 13)
    root[0, 2], root[0, 6] = .3, 1.
    env = NS(num_envs=1, is_evaluating=True, dt=.02, sim_dt=.005,
             common_step_counter=0, env_origins=torch.zeros(1, 3),
             simulator=NS(robot_root_states=root, dof_pos=torch.ones(1, 3),
                          dof_vel=torch.zeros(1, 3)),
             torque_limits=torch.ones(3) * 10, d_gains=torch.ones(3) * 2,
             torques=torch.zeros(1, 3), contacts=torch.zeros(1, 2),
             reset_buf=torch.zeros(1, dtype=torch.bool),
             time_out_buf=torch.zeros(1, dtype=torch.bool), reset_buf_terminate_by={},
             episode_length_buf=torch.ones(1), motion_start_times=torch.zeros(1),
             actions=torch.ones(1, 3), actions_after_delay=torch.ones(1, 3),
             action_queue=torch.ones(1, 2, 3))
    scheduler = NS(current_cmd=0, pending_cmd=None, planner_type="graph_search",
                   _clear_pending_command=Mock(), _queue_command=Mock(),
                   current_guidance=lambda: NS(node_id=0))
    host = NS(env=env, graph=graph, scheduler=scheduler,
              ref_builder=NS(motions=motions), install_recovery_entry=Mock(),
              prepare_recovery_hard_reset=Mock())
    cfg = dict(selector=dict(
        recovery_skill_ids=[1, 2], entry_frames={1: 0, 2: 0},
        admission_mode="enforce", max_entry_distance=5),
        damping=dict(settle_hold_s=.06))
    cfg.update(config or {})
    fsm = RecoveryFSM(host, cfg)
    return fsm, env, host


def post_step(fsm, env, reason=None):
    env.common_step_counter += 1
    env.reset_buf[:] = bool(reason)
    env.time_out_buf[:] = False
    env.reset_buf_terminate_by = {reason: torch.ones(1, dtype=torch.bool)} if reason else {}
    fsm.intercept_terminations(env)


class RecoveryTests(unittest.TestCase):
    def test_detailed_trace_is_observational_and_preserves_hard_failure(self):
        f, e, _ = fixtures(dict(entry_diagnostics=dict(enabled=True, detailed=True)))
        e.config = NS(termination_scales=NS(terminate_by_body_z_threshold=.25),
                      robot=NS(dof_names=['a', 'b', 'c'], motion={}))
        e.simulator._body_list = ['body_' + str(i) for i in range(24)]
        e.simulator.contact_forces = torch.zeros(1, 24, 3)
        e.termination_dif_local_body_pos = torch.zeros(1, 27, 3)
        e.termination_dif_local_body_pos[0, 10, 2] = .1
        e._rigid_body_pos_extend = torch.ones(1, 27, 3) * .6
        e.env_origins[0, 2] = .5
        e.ref_is_buffer = torch.zeros(1, 1, dtype=torch.bool)
        e.ref_joint_pos = torch.ones(1, 3) * .8
        e.ref_contact_mask = torch.zeros(1, 2)
        f.request_recovery('test')
        for _ in range(3):
            post_step(f, e)
        f.before_scheduler(None)
        physical = [t.clone() for t in (e.simulator.robot_root_states,
                    e.simulator.dof_pos, e.simulator.dof_vel)]
        post_step(f, e)
        row = [r for r in f.events if r['event'] == 'entry_error_sample'][-1]
        self.assertAlmostEqual(row['actual_body_z_m'][1], .1)
        self.assertAlmostEqual(row['termination_reference_body_z_m'][1], .2)
        self.assertAlmostEqual(row['joint_error_rad'][0], .2)
        self.assertFalse(row['is_buffer'])
        for old, new in zip(physical, (e.simulator.robot_root_states,
                            e.simulator.dof_pos, e.simulator.dof_vel)):
            self.assertTrue(torch.equal(old, new))
        previous = sum(r['event'] == 'entry_error_sample' for r in f.events)
        e.simulator.dof_pos[0, 0] = float('nan')
        post_step(f, e)
        self.assertEqual(f.state, RecoveryState.HARD_RESET)
        self.assertEqual(previous, sum(r['event'] == 'entry_error_sample' for r in f.events))

    def test_entry_trace_records_actual_termination_metric_before_failure(self):
        f, e, _ = fixtures(dict(entry_grace_s=.75,
                               entry_diagnostics=dict(enabled=True, interval_s=.1)))
        e.config = NS(termination_scales=NS(terminate_by_body_z_threshold=.25))
        e.termination_dif_local_body_pos = torch.zeros(1, 27, 3)
        e.termination_dif_local_body_pos[0, 24, 2] = -.4
        # Deliberately different guidance error: diagnostics must use termination.
        e.dif_local_body_pos = torch.ones(1, 27, 3)
        f.request_recovery('test')
        for _ in range(3):
            post_step(f, e)
        f.before_scheduler(None)
        entry_step = e.common_step_counter
        post_step(f, e, 'body_z')
        self.assertEqual(f.state, RecoveryState.RECOVERY_TRACKING)
        first = [r for r in f.events if r['event'] == 'entry_error_sample'][-1]
        self.assertAlmostEqual(first['body_z_max_abs_m'], .4)
        self.assertEqual(first['body_z_exceed_indices'], [24])
        self.assertTrue(first['within_grace'])
        # Force the terminal sample even though it is inside the logging interval.
        e.common_step_counter = entry_step + 37
        post_step(f, e, 'body_z')
        final = [r for r in f.events if r['event'] == 'entry_error_sample'][-1]
        self.assertFalse(final['within_grace'])
        self.assertAlmostEqual(final['elapsed_s'], .76)
        self.assertEqual(f.state, RecoveryState.DAMPING)

    def test_entry_trace_stops_after_window_and_does_not_mask_hard_failure(self):
        f, e, _ = fixtures(dict(entry_grace_s=.75, entry_diagnostics=dict(enabled=True)))
        e.config = NS(termination_scales=NS(terminate_by_body_z_threshold=.25))
        e.termination_dif_local_body_pos = torch.zeros(1, 27, 3)
        f.request_recovery('test')
        for _ in range(3):
            post_step(f, e)
        f.before_scheduler(None)
        e.common_step_counter += 51
        post_step(f, e)
        self.assertFalse(any(r['event'] == 'entry_error_sample' for r in f.events))
        post_step(f, e, 'time_out')
        self.assertEqual(f.state, RecoveryState.HARD_RESET)

    def test_soft_failure_is_intercepted_and_clock_paused(self):
        f, e, h = fixtures()
        post_step(f, e, "body_z")
        self.assertEqual(f.state, RecoveryState.DAMPING)
        self.assertFalse(e.reset_buf[0])
        self.assertTrue(e.reset_buf_terminate_by["body_z"][0])
        f.pre_control_step(e)
        self.assertAlmostEqual(float(e.motion_start_times[0]), -.02)
        self.assertEqual(float(e.action_queue.sum()), 0)
        h.prepare_recovery_hard_reset.assert_not_called()

    def test_static_hold_must_be_continuous(self):
        f, e, _ = fixtures()
        f.request_recovery("test")
        post_step(f, e)
        post_step(f, e)
        e.simulator.robot_root_states[0, 7] = 1
        post_step(f, e)
        self.assertEqual(f.stable_steps, 0)
        e.simulator.robot_root_states[0, 7] = 0
        post_step(f, e)
        post_step(f, e)
        self.assertEqual(f.state, RecoveryState.DAMPING)
        post_step(f, e)
        self.assertEqual(f.state, RecoveryState.SELECTING)

    def test_damping_passive_limited_and_slew_scaled(self):
        f, e, _ = fixtures()
        f.request_recovery("test")
        e.simulator.dof_vel[:] = 100
        tau = f.override_torque(e, torch.ones(1, 3) * 100)
        self.assertTrue(torch.all(tau * e.simulator.dof_vel <= 0))
        self.assertTrue(torch.all(tau.abs() <= .25 + 1e-6))
        for _ in range(20):
            tau = f.override_torque(e, torch.ones(1, 3) * 100)
        self.assertTrue(torch.all(tau.abs() <= 2.5 + 1e-6))
        e.simulator.dof_vel[:] = -100
        tau = f.override_torque(e, tau)
        self.assertTrue(torch.all(tau * e.simulator.dof_vel <= 0))

    def test_timeout_hard_reset_is_executed_once(self):
        f, e, h = fixtures(dict(damping=dict(timeout_s=.03)))
        f.request_recovery("test")
        e.simulator.dof_vel[:] = 1
        post_step(f, e)
        post_step(f, e)
        self.assertEqual(f.state, RecoveryState.HARD_RESET)
        self.assertTrue(e.reset_buf[0])
        h.prepare_recovery_hard_reset.assert_called_once_with(0)
        f.before_scheduler(None)
        self.assertEqual(f.state, RecoveryState.TRACKING)
        post_step(f, e)
        h.prepare_recovery_hard_reset.assert_called_once()

    def test_pending_hard_reset_not_cleared_before_physics(self):
        f, e, h = fixtures(dict(selector=dict(
            admission_mode="enforce", max_entry_distance=.001)))
        f.request_recovery("test")
        for _ in range(3): post_step(f, e)
        self.assertTrue(f.before_scheduler(None))
        self.assertEqual(f.state, RecoveryState.HARD_RESET)
        self.assertTrue(f.before_scheduler(None))
        h.prepare_recovery_hard_reset.assert_not_called()
        post_step(f, e)
        h.prepare_recovery_hard_reset.assert_called_once()

    def test_hard_failure_wins_over_soft(self):
        f, e, h = fixtures()
        e.reset_buf_terminate_by = {k: torch.ones(1, dtype=torch.bool)
                                  for k in ["body_z", "time_out"]}
        f.intercept_terminations(e)
        self.assertTrue(e.reset_buf[0])
        h.prepare_recovery_hard_reset.assert_called_once()

    def test_nan_is_hard_failure(self):
        f, e, h = fixtures()
        e.simulator.dof_pos[0, 0] = float("nan")
        post_step(f, e)
        h.prepare_recovery_hard_reset.assert_called_once()

    def test_no_reset_on_admitted_recovery_and_queue_original_task(self):
        f, e, h = fixtures()
        f.request_recovery("test")
        for _ in range(3): post_step(f, e)
        root, q = e.simulator.robot_root_states.clone(), e.simulator.dof_pos.clone()
        self.assertFalse(f.before_scheduler(None))
        h.install_recovery_entry.assert_called_once()
        self.assertEqual(f.state, RecoveryState.RECOVERY_TRACKING)
        self.assertTrue(torch.equal(root, e.simulator.robot_root_states))
        self.assertTrue(torch.equal(q, e.simulator.dof_pos))
        h.prepare_recovery_hard_reset.assert_not_called()
        h.scheduler.current_cmd = 1
        f.before_scheduler(0)
        h.scheduler._queue_command.assert_called_with(0)

    def test_retries_bounded_and_reset_only_after_success(self):
        f, e, h = fixtures()
        for attempt in (1, 2):
            f.request_recovery("test")
            for _ in range(3): post_step(f, e)
            f.before_scheduler(None)
            self.assertEqual(f.attempts, attempt)
        f.request_recovery("third_failure")
        self.assertEqual(f.state, RecoveryState.HARD_RESET)

    def test_success_requires_target_dwell(self):
        f, e, h = fixtures()
        f.state = RecoveryState.RECOVERY_TRACKING
        f.attempts = 1
        f.after_scheduler()
        self.assertEqual(f.state, RecoveryState.RETURNING_TO_TASK)
        self.assertEqual(f.attempts, 1)
        e.common_step_counter = 51
        f.after_scheduler()
        self.assertEqual(f.state, RecoveryState.TRACKING)
        self.assertEqual(f.attempts, 0)

    def test_selector_fixed_entries_and_yaw_invariance(self):
        f, e, h = fixtures()
        s = f.selector
        self.assertEqual(s.entries, [(1, 0, 50), (2, 0, 100)])
        for sid in (1, 2):
            m = h.ref_builder.motions[sid]
            live = dict(q=m["dof"][0], qdot=np.ones(3)*.3,
                        projected_gravity=gravity_in_body(m["root_rot"][0]),
                        root_height=np.asarray([.3]), foot_contact=np.zeros(2))
            entry = s.rank(live)[0]
            self.assertEqual((entry.skill_id, entry.frame_idx), (sid, 0))
            self.assertAlmostEqual(entry.distance, 0)
            quat = (Rotation.from_euler("z", 2.3) * Rotation.from_quat(m["root_rot"][0])).as_quat()
            live["projected_gravity"] = gravity_in_body(quat)
            self.assertEqual(s.rank(live)[0].node_id, entry.node_id)
            self.assertNotEqual(s.rank(live, [(sid, 0)])[0].skill_id, sid)

    def test_attempt_labels_follow_mode_and_capture_failure(self):
        # Default logging modes=[collect], so enforce runs stay unlabelled.
        enforce, env, _ = fixtures()
        enforce.request_recovery("test")
        for _ in range(3): post_step(enforce, env)
        enforce.before_scheduler(None)
        self.assertFalse(any(e["event"].startswith("attempt_")
                             for e in enforce.events))

        collect, env, _ = fixtures(dict(
            selector=dict(recovery_skill_ids=[1, 2],
                          entry_frames={1: 0, 2: 0},
                          admission_mode="collect",
                          max_entry_distance=None),
            attempt_logging=dict(enabled=True, modes=["collect"])))
        collect.request_recovery("initial_fall")
        for _ in range(3): post_step(collect, env)
        collect.before_scheduler(None)
        starts = [e for e in collect.events if e["event"] == "attempt_start"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0]["fall_event_id"], 1)
        self.assertEqual(starts[0]["attempt_id"], 1)
        self.assertIn("initial_distance", starts[0])
        collect.request_recovery("body_z")
        results = [e for e in collect.events if e["event"] == "attempt_result"]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["result"], "failure")
        self.assertEqual(results[0]["reason"], "body_z")
        self.assertFalse(results[0]["returned_to_task"])

    def test_attempt_success_label_is_end_to_end(self):
        f, e, h = fixtures(dict(
            selector=dict(recovery_skill_ids=[1, 2],
                          entry_frames={1: 0, 2: 0},
                          admission_mode="collect",
                          max_entry_distance=None),
            attempt_logging=dict(enabled=True, modes=["collect"])))
        f.request_recovery("test")
        for _ in range(3): post_step(f, e)
        f.before_scheduler(None)
        f.after_scheduler()
        e.common_step_counter += 51
        f.after_scheduler()
        results = [event for event in f.events
                   if event["event"] == "attempt_result"]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["result"], "success")
        self.assertTrue(results[0]["returned_to_task"])

    def test_training_rejected(self):
        f, e, h = fixtures()
        e.is_evaluating = False
        with self.assertRaises(ValueError): RecoveryFSM(h, {})

    def test_motion_end_ignored_during_damping_but_real_timeout_is_hard(self):
        f, e, h = fixtures()
        f.request_recovery("test")
        post_step(f, e, "motion_end")
        self.assertFalse(e.reset_buf[0])
        h.prepare_recovery_hard_reset.assert_not_called()
        post_step(f, e, "time_out")
        self.assertTrue(e.reset_buf[0])

    def test_threshold_rejects_without_reference_install(self):
        f, e, h = fixtures(dict(selector=dict(max_entry_distance=.001)))
        f.request_recovery("test")
        for _ in range(3): post_step(f, e)
        f.before_scheduler(None)
        self.assertEqual(f.state, RecoveryState.HARD_RESET)
        h.install_recovery_entry.assert_not_called()

    def test_collect_mode_ignores_threshold_and_installs_nearest_entry(self):
        f, e, h = fixtures(dict(selector=dict(
            recovery_skill_ids=[1, 2], entry_frames={1: 0, 2: 0},
            admission_mode="collect", max_entry_distance=.001)))
        f.request_recovery("test")
        for _ in range(3): post_step(f, e)
        f.before_scheduler(None)
        self.assertEqual(f.state, RecoveryState.RECOVERY_TRACKING)
        h.install_recovery_entry.assert_called_once()

    def test_enforce_mode_requires_threshold(self):
        with self.assertRaises(ValueError):
            fixtures(dict(selector=dict(
                recovery_skill_ids=[1, 2], entry_frames={1: 0, 2: 0},
                admission_mode="enforce", max_entry_distance=None)))

    def test_buffer_node_is_not_success(self):
        f, e, h = fixtures()
        f.state = RecoveryState.RECOVERY_TRACKING
        h.graph.skill_ids[0] = -1
        f.after_scheduler()
        e.common_step_counter = 500
        f.after_scheduler()
        self.assertNotEqual(f.state, RecoveryState.TRACKING)
        self.assertEqual(f.summary()["recoveries"], 0)

    def test_current_task_command_cancels_stale_pending_return(self):
        f, e, h = fixtures()
        f.state = RecoveryState.RETURNING_TO_TASK
        h.scheduler.pending_cmd = 1
        f.before_scheduler(0)
        h.scheduler._clear_pending_command.assert_called_once()

    def test_injection_preserves_physics_and_reports_correct_admission_metric(self):
        from humanoidverse.agents.callbacks.skill_scheduler_callback import SkillSchedulerEvalCallback
        e = NS(device="cpu", dt=.02, common_step_counter=10,
               episode_length_buf=torch.tensor([80]), motion_start_times=torch.zeros(1),
               motion_len=torch.zeros(1), reset_envs_idx=Mock())
        lib = NS(_motion_data_list=[None], load_motions=Mock(),
                 _curr_motion_ids=torch.tensor([0]), curr_motion_keys=["test"],
                 _motion_lengths=torch.tensor([2.]))
        e._motion_lib = lib
        callback = SkillSchedulerEvalCallback({}, NS(env=e, device="cpu"))
        callback.graph = NS(is_buffer=np.zeros(2, dtype=bool), skill_ids=np.ones(2, dtype=int),
                            frame_idxs=np.arange(2), skill_names=["task", "recovery"])
        callback.ref_builder = NS(build_trajectory=lambda path: dict(fps=30., dof=np.zeros((len(path), 3))))
        callback.scheduler = NS(last_trigger="recovery_entry", last_best_sim=100.,
                                 switch_entry_A=11., A=6., current_cmd=1)
        callback.recovery_fsm = NS(selected=NS(distance=.2), threshold=.5)
        callback._slot = 0
        callback._align_to_robot = Mock()
        callback._update_recovery_slot_role = Mock()
        callback._inject([0, 1], 0., np.zeros(2), physical_reset_mode="never")
        e.reset_envs_idx.assert_not_called()
        self.assertAlmostEqual(float(e.motion_start_times[0]), -1.6, places=6)
        self.assertEqual(callback.injection_history[-1]["entry_metric"], "recovery_nn")
        self.assertEqual(callback.injection_history[-1]["best_sim"], .2)
        self.assertFalse(callback.injection_history[-1]["physical_reset"])


    def test_real_graph_recovery_return_uses_trained_buffer(self):
        """Ideal tracking isolates FSM/planning correctness from policy ability."""
        from pathlib import Path
        import yaml
        from humanoidverse.deploy.skill_scheduler.graph_data import SkillGraphData
        from humanoidverse.deploy.skill_scheduler.reference_builder import ReferenceBuilder
        from humanoidverse.deploy.skill_scheduler.scheduler import SkillGraphScheduler
        root = Path(__file__).resolve().parents[4]
        graph_file = root / "Switch_data/unified_graph_contact_refined/skill_graph.json"
        if not graph_file.exists():
            self.skipTest("Unified graph data unavailable")
        cfg = yaml.safe_load((root / "Switch_data/skill_graph/scheduler_config.yaml").read_text())
        g = SkillGraphData.from_json(str(graph_file), sigma_q=cfg["sigma_q"],
            sigma_qdot=cfg["sigma_qdot"], sigma_p=cfg["sigma_p"], lambda_sw=5., w_p=0.)
        files = [root / f"Switch_data/skills/retargeted/141_{c}_poses_retarget.pkl"
                 for c in ("12", "14", "19")]
        files += [root / f"Switch_data/recovery/retargeted_contact_refined/140_{c}_poses_retarget.pkl"
                  for c in ("01", "02", "04", "08", "09")]
        rb = ReferenceBuilder.from_pkl_files(g, [str(p) for p in files])
        s = SkillGraphScheduler(g, "graph_search", 6.6014, 14.4499, 1., .2, 5)
        s.set_command(0)
        s.install_reference_path(s.pure_skill_path(0), 0.)
        _, env, _ = fixtures()
        env.dt = 1. / 30
        m = rb.motions[5]
        env.simulator.dof_pos = torch.tensor(m["dof"][:1])
        env.simulator.dof_vel = torch.tensor(np.gradient(m["dof"], 1./30, axis=0)[:1])
        env.simulator.robot_root_states[0, :3] = torch.tensor(m["root_trans_offset"][0])
        env.simulator.robot_root_states[0, 3:7] = torch.tensor(m["root_rot"][0])
        env.contacts = torch.tensor(m["contact_mask"][:1])
        host = NS(env=env, graph=g, scheduler=s, ref_builder=rb,
                  prepare_recovery_hard_reset=Mock())
        def install(entry):
            s.set_command(entry.skill_id)
            _, end = g.skill_range(entry.skill_id)
            s.install_reference_path(list(range(entry.node_id, end)), 0.)
        host.install_recovery_entry = install
        f = RecoveryFSM(host, dict(selector=dict(max_entry_distance=.01)))
        f.state, f.attempts = RecoveryState.SELECTING, 1
        f.before_scheduler(None)
        self.assertEqual(f.state, RecoveryState.RECOVERY_TRACKING)
        self.assertEqual(f.selected.skill_id, 5)
        buffers = set()
        for step in range(600):
            env.common_step_counter = step
            f.before_scheduler(None)
            current = s.current_guidance()
            s.step(g.nodes[current.node_id], None, f.now)
            buffers.update(n for n in s.current_path if g.is_buffer[n])
            f.after_scheduler()
            if f.state == RecoveryState.TRACKING:
                break
            s.advance()
        self.assertTrue(buffers)
        self.assertEqual(f.state, RecoveryState.TRACKING)
        self.assertEqual(s.current_cmd, 0)
        self.assertEqual(f.summary()["recoveries"], 1)
        host.prepare_recovery_hard_reset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
