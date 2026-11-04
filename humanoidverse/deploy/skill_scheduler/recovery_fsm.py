"""Evaluation-only recovery controller. Physics hooks run BEFORE reset.

The simulator remains authoritative: reference installation never sets a
physical pose. Only the explicit HARD_RESET fallback requests a reset.
"""

from dataclasses import asdict
from enum import Enum
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from .recovery_selector import RecoveryEntrySelector, gravity_in_body


class RecoveryState(str, Enum):
    TRACKING = "TRACKING"
    DAMPING = "DAMPING"
    SELECTING = "SELECTING"
    RECOVERY_TRACKING = "RECOVERY_TRACKING"
    RETURNING_TO_TASK = "RETURNING_TO_TASK"
    HARD_RESET = "HARD_RESET"


class RecoveryFSM:
    def __init__(self, host, config):
        self.host, self.env, self.config = host, host.env, config
        if self.env.num_envs != 1 or not self.env.is_evaluating:
            raise ValueError("Recovery FSM currently supports single-env evaluation only")
        if host.scheduler.planner_type != "graph_search":
            raise ValueError("Recovery NN entry selection requires graph_search for trained exits")
        self.selector = RecoveryEntrySelector(host.graph, host.ref_builder.motions,
                                              config.get("selector", {}))
        self.damping = config.get("damping", {})
        self.admission_mode = str(
            config.get("selector", {}).get("admission_mode", "enforce")
        ).lower()
        if self.admission_mode not in {"collect", "enforce"}:
            raise ValueError("selector.admission_mode must be collect or enforce")
        self.threshold = config.get("selector", {}).get("max_entry_distance", None)
        if self.threshold is not None:
            self.threshold = float(self.threshold)
            if not np.isfinite(self.threshold) or self.threshold <= 0:
                raise ValueError("max_entry_distance must be positive or null")
        if self.admission_mode == "enforce" and self.threshold is None:
            raise ValueError(
                "selector.max_entry_distance is required in enforce mode"
            )
        self.max_attempts = int(config.get("max_recovery_attempts", 2))
        if self.max_attempts < 1:
            raise ValueError("max_recovery_attempts must be >= 1")
        attempt_logging = config.get("attempt_logging", {})
        self.attempt_logging_modes = {
            str(mode).lower() for mode in attempt_logging.get("modes", ["collect"])
        }
        unknown_logging_modes = self.attempt_logging_modes - {"collect", "enforce"}
        if unknown_logging_modes:
            raise ValueError(
                f"Unknown attempt_logging modes: {sorted(unknown_logging_modes)}"
            )
        self.attempt_logging_enabled = (
            bool(attempt_logging.get("enabled", True)) and
            self.admission_mode in self.attempt_logging_modes
        )
        for key, default in (("entry_grace_s", .25), ("policy_blend_s", .2),
                             ("success_hold_s", 1.), ("recovery_timeout_s", 20.),
                             ("max_distance_from_origin_m", 20.)):
            value = float(config.get(key, default))
            if not np.isfinite(value) or value < 0 or (key.endswith("timeout_s") and value == 0):
                raise ValueError("Invalid recovery parameter: " + key)
        for k, default in (("kd_scale", .5), ("torque_limit_ratio", .25),
                           ("torque_slew_limit_ratio", .1), ("timeout_s", 3.),
                           ("settle_hold_s", .4), ("root_lin_vel_threshold", .15),
                           ("root_ang_vel_threshold", .3), ("dof_vel_rms_threshold", .5)):
            if not np.isfinite(float(self.damping.get(k, default))) or float(self.damping.get(k, default)) <= 0:
                raise ValueError("Invalid damping parameter: " + k)
        self.soft_reasons = set(config.get("recoverable_terminations", [
            "ref_pos_z", "ref_ori", "body_z", "motion_far", "gravity", "contact"]))
        self.hard_reasons = set(config.get("hard_terminations", [
            "time_out", "dof_pos_limit", "dof_vel_limit", "torque_limit"]))
        self.state = RecoveryState.TRACKING
        self.resume_skill_id = int(host.scheduler.current_cmd)
        fallback = int(config.get("fallback_task_skill", 0))
        if host.graph.skill_roles[fallback] != "task":
            raise ValueError("fallback_task_skill must be a task")
        if host.graph.skill_roles[self.resume_skill_id] != "task":
            self.resume_skill_id = fallback
        self.attempts = 0
        self.fall_event_id = 0
        self.attempt_started_at = None
        self._attempt_open = False
        self.excluded = []
        self.selected = None
        self.events = []
        self.stable_steps = 0
        self.entered_at = self.now
        self.recovery_started_at = self.now
        self.target_since = None
        self.last_torque = None
        self._hard_reset_executed = False
        self.blend_until = 0.
        self.last_raw_terminations = []
        self.entry_diagnostics = config.get("entry_diagnostics", {})
        self._last_entry_diagnostic_at = None
        if bool(self.entry_diagnostics.get("enabled", False)):
            for key, default in (("window_s", 1.), ("interval_s", .02)):
                value = float(self.entry_diagnostics.get(key, default))
                if not np.isfinite(value) or value <= 0:
                    raise ValueError("Invalid entry_diagnostics parameter: " + key)
        self.output = config.get("output_path", None)
        if self.output:
            Path(self.output).parent.mkdir(parents=True, exist_ok=True)
        self.record("initialized", admission_mode=self.admission_mode,
                    threshold=self.threshold, candidates=len(self.selector.entries),
                    entry_grace_s=float(config.get("entry_grace_s", .25)),
                    attempt_logging=self.attempt_logging_enabled,
                    attempt_logging_modes=sorted(self.attempt_logging_modes))

    @property
    def now(self):
        return int(self.env.common_step_counter) * float(self.env.dt)

    @property
    def active(self):
        return self.state != RecoveryState.TRACKING

    @property
    def damping_active(self):
        return self.state in (RecoveryState.DAMPING, RecoveryState.SELECTING,
                              RecoveryState.HARD_RESET)

    def record(self, event, **fields):
        row = dict(step=int(self.env.common_step_counter), time_s=self.now,
                   event=event, state=self.state.value, attempt=self.attempts,
                   resume_skill=self.resume_skill_id, **fields)
        self.events.append(row)
        logger.info("RecoveryFSM: {}", json.dumps(row, ensure_ascii=False))
        if self.output:
            with Path(self.output).open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")

    def transition(self, state, **fields):
        previous = self.state.value
        self.state, self.entered_at = state, self.now
        self.record("state_change", previous=previous, **fields)

    def _start_attempt_label(self):
        self.attempt_started_at = self.now
        self._last_entry_diagnostic_at = None
        self._attempt_open = True
        if self.attempt_logging_enabled:
            self.record(
                "attempt_start", fall_event_id=self.fall_event_id,
                attempt_id=self.attempts, admission_mode=self.admission_mode,
                threshold=self.threshold, entry=asdict(self.selected),
                initial_distance=self.selected.distance,
                distance_components=self.selected.components,
                entry_grace_s=float(self.config.get("entry_grace_s", .25)),
                initial_state={k: np.asarray(v).tolist() for k, v in self.live_features().items()},
            )

    def _finish_attempt_label(self, result, reason=None):
        if not self._attempt_open or self.selected is None:
            return
        if self.attempt_logging_enabled:
            duration = max(0., self.now - float(self.attempt_started_at))
            self.record(
                "attempt_result", fall_event_id=self.fall_event_id,
                attempt_id=self.attempts, admission_mode=self.admission_mode,
                result=str(result), reason=reason,
                duration_s=duration,
                failure_stage=(None if result == "success" else
                               "return_to_task" if self.state == RecoveryState.RETURNING_TO_TASK else
                               "entry" if duration <= float(self.config.get("entry_grace_s", .25)) + self.env.dt else
                               "recovery_tracking"),
                raw_terminations=list(self.last_raw_terminations),
                entry=asdict(self.selected),
                initial_distance=self.selected.distance,
                distance_components=self.selected.components,
                returned_to_task=result == "success",
            )
        self._attempt_open = False
        self.attempt_started_at = None

    def _record_rejected_attempt(self, entry, reason):
        if self.attempt_logging_enabled:
            self.record(
                "attempt_rejected", fall_event_id=self.fall_event_id,
                attempt_id=self.attempts, admission_mode=self.admission_mode,
                result="rejected", reason=reason, threshold=self.threshold,
                entry=asdict(entry), initial_distance=entry.distance,
                distance_components=entry.components,
            )

    def accept_command(self, cmd):
        if cmd is None:
            return
        sid = self.host.graph.skill_index(cmd)
        if self.host.graph.skill_roles[sid] != "task":
            self.record("ignored_non_task_command", requested_skill=sid)
            return
        self.resume_skill_id = sid
        self.target_since = None
        self.record("resume_command", requested_skill=sid)

    def request_recovery(self, reason):
        if self.damping_active or self.state == RecoveryState.HARD_RESET:
            return
        self._finish_attempt_label("failure", reason=reason)
        if self.attempts >= self.max_attempts:
            self.transition(RecoveryState.HARD_RESET, reason="max_attempts")
            return
        if self.state == RecoveryState.TRACKING:
            self.fall_event_id += 1
        self.remember_task()
        if self.selected is not None:
            self.excluded.append((self.selected.skill_id, self.selected.frame_idx))
        self.attempts += 1
        self.stable_steps = 0
        self.target_since = None
        self.last_torque = torch.zeros_like(self.env.torques[0])
        self.host.scheduler._clear_pending_command()
        self.transition(RecoveryState.DAMPING, reason=reason)

    def remember_task(self):
        if self.state == RecoveryState.TRACKING:
            s = self.host.scheduler
            cmd = s.pending_cmd if s.pending_cmd is not None else s.current_cmd
            if self.host.graph.skill_roles[cmd] == "task":
                self.resume_skill_id = int(cmd)

    def pre_control_step(self, env):
        if self.damping_active:
            # episode time advances in post_physics_step; compensate exactly.
            env.motion_start_times[0] -= env.dt
            # These are policy actions, not executed position commands in damping.
            env.actions.zero_()
            env.actions_after_delay.zero_()
            if hasattr(env, "action_queue"):
                env.action_queue.zero_()

    def override_torque(self, env, torque):
        if self.damping_active:
            if self.last_torque is None:
                self.last_torque = torch.zeros_like(torque[0])
            limits = env.torque_limits
            cap = limits * float(self.damping.get("torque_limit_ratio", .25))
            kd = env.d_gains * float(self.damping.get("kd_scale", .5))
            target = torch.clamp(-kd * env.simulator.dof_vel[0], -cap, cap)
            # Ratio means per CONTROL step, scaled for physics substeps.
            delta = limits * float(self.damping.get("torque_slew_limit_ratio", .1)) * env.sim_dt / env.dt
            target = torch.clamp(target, self.last_torque - delta, self.last_torque + delta)
            # Slew limiting must not turn damping into positive mechanical work
            # when a joint reverses velocity after contact.
            target = torch.where(target * env.simulator.dof_vel[0] > 0,
                                 torch.zeros_like(target), target)
            self.last_torque = target.detach().clone()
            torque[0] = target
        elif self.now < self.blend_until and self.last_torque is not None:
            delta = env.torque_limits * float(self.damping.get("torque_slew_limit_ratio", .1)) * env.sim_dt / env.dt
            torque[0] = torch.clamp(torque[0], self.last_torque - delta, self.last_torque + delta)
            self.last_torque = torque[0].detach().clone()
        return torque

    def _hard_failure(self):
        sim = self.env.simulator
        tensors = (sim.robot_root_states[0], sim.dof_pos[0], sim.dof_vel[0])
        if not all(bool(torch.isfinite(t).all()) for t in tensors):
            return "nan_or_inf"
        if float(sim.robot_root_states[0, 3:7].norm()) < 1e-6:
            return "invalid_root_quaternion"
        origin = self.env.env_origins[0]
        if float(torch.linalg.vector_norm(sim.robot_root_states[0, :2] - origin[:2])) > float(self.config.get("max_distance_from_origin_m", 20.)):
            return "out_of_bounds"
        return None

    def _record_entry_diagnostic(self, env, names, force=False):
        """Observe the same errors used by termination, before any retry/reset."""
        if not bool(self.entry_diagnostics.get("enabled", False)) or not self._attempt_open:
            return
        elapsed = self.now - float(self.attempt_started_at)
        if elapsed > float(self.entry_diagnostics.get("window_s", 1.)) + 1e-6:
            return
        interval = float(self.entry_diagnostics.get("interval_s", .02))
        if not force and self._last_entry_diagnostic_at is not None \
                and self.now - self._last_entry_diagnostic_at < interval - 1e-6:
            return
        errors = getattr(env, "termination_dif_local_body_pos", None)
        if errors is None:
            errors = env.dif_local_body_pos
        indices = list(getattr(env, "BODY_Z_TERMINATION_INDICES", (4, 10, 24, 25, 26)))
        z = errors[0, indices, -1].detach().cpu()
        # A non-finite state is handled by the existing hard-reset check.
        if not bool(torch.isfinite(z).all()):
            return
        scales = env.config.termination_scales
        threshold = float(getattr(scales, "terminate_by_body_z_threshold", .25))
        body_names = list(getattr(env.simulator, "_body_list", []))
        motion_cfg = getattr(getattr(env.config, "robot", None), "motion", {})
        body_names += [item["joint_name"] for item in motion_cfg.get("extend_config", [])]
        names_by_index = [body_names[i] if i < len(body_names) else "body_" + str(i) for i in indices]
        absolute = z.abs()
        details = {}
        if bool(self.entry_diagnostics.get("detailed", False)):
            # Observational only: physical states and the reference currently
            # used for termination, captured before any retry/reset.
            physical = (env.simulator.robot_root_states[0],
                        env.simulator.dof_pos[0], env.simulator.dof_vel[0])
            if not all(bool(torch.isfinite(t).all()) for t in physical):
                # JSON diagnostics must not interrupt the hard-failure handler.
                return
            origin_z = float(env.env_origins[0, 2])
            actual_z = env._rigid_body_pos_extend[0, indices, -1].detach().cpu() - origin_z
            details.update(
                actual_body_z_m=actual_z.tolist(),
                termination_reference_body_z_m=(actual_z + z).tolist(),
                reference_time_s=float((env.episode_length_buf[0] + 1) * env.dt + env.motion_start_times[0]),
                is_buffer=bool(env.ref_is_buffer[0].any()),
                root_state=env.simulator.robot_root_states[0].detach().cpu().tolist(),
                joint_names=list(env.config.robot.dof_names),
                q=env.simulator.dof_pos[0].detach().cpu().tolist(),
                reference_q=env.ref_joint_pos[0].detach().cpu().tolist(),
                qdot=env.simulator.dof_vel[0].detach().cpu().tolist(),
                joint_error_rad=(env.simulator.dof_pos[0] - env.ref_joint_pos[0]).detach().cpu().tolist(),
                actual_foot_contacts=env.contacts[0].detach().cpu().tolist(),
                reference_foot_contacts=env.ref_contact_mask[0].detach().cpu().tolist(),
                contact_body_names=list(env.simulator._body_list),
                contact_force_norm_n=env.simulator.contact_forces[0].norm(dim=-1).detach().cpu().tolist(),
            )
        self.record(
            "entry_error_sample", fall_event_id=self.fall_event_id, attempt_id=self.attempts,
            elapsed_s=elapsed, entry_grace_s=float(self.config.get("entry_grace_s", .25)),
            within_grace=elapsed < float(self.config.get("entry_grace_s", .25)),
            body_indices=indices, body_names=names_by_index,
            body_z_error_m=z.tolist(), body_z_max_abs_m=float(absolute.max()),
            body_z_threshold_m=threshold,
            body_z_exceed_indices=[i for i, e in zip(indices, absolute.tolist()) if e > threshold],
            raw_terminations=sorted(names),
            **details,
        )
        self._last_entry_diagnostic_at = self.now

    def intercept_terminations(self, env):
        """Called after termination calculation, before reward/reset dispatch."""
        names = {k for k, v in env.reset_buf_terminate_by.items() if bool(v[0])}
        self.last_raw_terminations = sorted(names)
        hard = self._hard_failure()
        hard_names = (names & self.hard_reasons) | (names - self.soft_reasons - {"motion_end"})
        grace_expired = self.now - self.recovery_started_at >= float(self.config.get("entry_grace_s", .25))
        self._record_entry_diagnostic(env, names, force=bool(hard or hard_names or
                                     (names & self.soft_reasons and grace_expired)))
        if hard or hard_names:
            self.remember_task()
            reason = hard or ",".join(sorted(hard_names))
            self._finish_attempt_label("failure", reason=reason)
            self.transition(RecoveryState.HARD_RESET, reason=reason)
        elif self.state == RecoveryState.TRACKING and names & self.soft_reasons:
            self.request_recovery(",".join(sorted(names & self.soft_reasons)))
        elif self.state in (RecoveryState.RECOVERY_TRACKING, RecoveryState.RETURNING_TO_TASK):
            # Bounded admission grace for a close, but non-identical, real pose.
            grace = float(self.config.get("entry_grace_s", .25))
            if names & self.soft_reasons and self.now - self.recovery_started_at >= grace:
                self.request_recovery(",".join(sorted(names & self.soft_reasons)))
            elif self.now - self.recovery_started_at > float(self.config.get("recovery_timeout_s", 20.)):
                self._finish_attempt_label("failure", reason="recovery_timeout")
                self.transition(RecoveryState.HARD_RESET, reason="recovery_timeout")
            elif "motion_end" in names:
                self._finish_attempt_label(
                    "failure", reason="recovery_reference_exhausted"
                )
                self.transition(RecoveryState.HARD_RESET, reason="recovery_reference_exhausted")

        if self.state == RecoveryState.DAMPING:
            v = env.simulator.robot_root_states[0, 7:13]
            qv = env.simulator.dof_vel[0]
            speeds = (float(v[:3].norm()), float(v[3:].norm()), float(qv.square().mean().sqrt()))
            thresholds = (float(self.damping.get("root_lin_vel_threshold", .15)),
                          float(self.damping.get("root_ang_vel_threshold", .3)),
                          float(self.damping.get("dof_vel_rms_threshold", .5)))
            stationary = all(a < b for a, b in zip(speeds, thresholds))
            self.stable_steps = self.stable_steps + 1 if stationary else 0
            if self.stable_steps * env.dt >= float(self.damping.get("settle_hold_s", .4)):
                self.transition(RecoveryState.SELECTING, settle_s=self.now - self.entered_at)
            elif self.now - self.entered_at >= float(self.damping.get("timeout_s", 3.)):
                self.transition(RecoveryState.HARD_RESET, reason="settle_timeout")

        if self.state == RecoveryState.HARD_RESET:
            # Prepare the fallback reference, then let the environment perform
            # its SINGLE normal reset and physics-state tensor flush.
            self.host.prepare_recovery_hard_reset(self.resume_skill_id)
            env.time_out_buf[0] = False
            env.reset_buf[0] = True
            env.reset_buf_terminate_by["recovery_hard_reset"] = torch.ones_like(env.reset_buf, dtype=torch.bool)
            self.record("hard_reset", physical_reset=True)
            self._hard_reset_executed = True
            return
        if self.active:
            env.reset_buf[0] = False
            env.time_out_buf[0] = False

    def live_features(self):
        env = self.env
        root = env.simulator.robot_root_states[0].detach().cpu().numpy()
        return dict(q=env.simulator.dof_pos[0].detach().cpu().numpy(),
                    qdot=env.simulator.dof_vel[0].detach().cpu().numpy(),
                    projected_gravity=gravity_in_body(root[3:7]),
                    root_height=np.asarray([root[2] - float(env.env_origins[0, 2])]),
                    foot_contact=env.contacts[0].detach().cpu().numpy())

    def before_scheduler(self, cmd):
        """Return True while the FSM owns reference scheduling."""
        if self.state == RecoveryState.HARD_RESET and self._hard_reset_executed:
            self._hard_reset_executed = False
            self.last_torque, self.blend_until = None, 0.
            self.attempts, self.excluded, self.selected = 0, [], None
            self.transition(RecoveryState.TRACKING, reason="hard_reset_completed")
            self.host._prev_ep_len = int(self.env.episode_length_buf[0])
            self.host._just_injected = False
        if not self.active:
            return False
        self.accept_command(cmd)
        if self.state == RecoveryState.SELECTING:
            live = self.live_features()
            ranked = self.selector.rank(live, self.excluded)
            self.record("selection", candidates=[asdict(e) for e in ranked[:int(self.config.get("selector", {}).get("top_k", 3))]],
                        settled_state={k: np.asarray(v).tolist() for k, v in live.items()},
                        root_state=self.env.simulator.robot_root_states[0].detach().cpu().tolist())
            if not ranked:
                self.transition(RecoveryState.HARD_RESET, reason="no_recovery_entry")
                return True
            if (self.admission_mode == "enforce" and
                    ranked[0].distance > self.threshold):
                self._record_rejected_attempt(ranked[0], "no_admissible_entry")
                self.transition(RecoveryState.HARD_RESET,
                                reason="no_admissible_entry")
                return True
            self.selected = ranked[0]
            self.host.install_recovery_entry(self.selected)
            self.recovery_started_at = self.now
            self._start_attempt_label()
            self.blend_until = self.now + float(self.config.get("policy_blend_s", .2))
            self.transition(RecoveryState.RECOVERY_TRACKING, entry=asdict(self.selected), physical_reset=False)
        if self.damping_active or self.state == RecoveryState.HARD_RESET:
            return True
        # Queue original (or latest user) task. Existing scheduler waits for a
        # trained future Buffer source; no runtime synthetic task->recovery edge.
        if self.host.scheduler.current_cmd != self.resume_skill_id:
            self.host.scheduler._queue_command(self.resume_skill_id)
        elif self.host.scheduler.pending_cmd is not None:
            self.host.scheduler._clear_pending_command()
        return False

    def after_scheduler(self):
        if self.state not in (RecoveryState.RECOVERY_TRACKING, RecoveryState.RETURNING_TO_TASK):
            return
        s = self.host.scheduler
        if s.current_cmd == self.resume_skill_id:
            if self.state == RecoveryState.RECOVERY_TRACKING:
                self.record("recovery_exit_started", physical_reset=False)
                self.transition(RecoveryState.RETURNING_TO_TASK)
            guidance = s.current_guidance()
            on_target = guidance is not None and int(self.host.graph.skill_ids[guidance.node_id]) == self.resume_skill_id
            if on_target:
                if self.target_since is None:
                    self.target_since = self.now
                if self.now - self.target_since >= float(self.config.get("success_hold_s", 1.)):
                    self._finish_attempt_label("success")
                    self.record("recovery_success", physical_reset=False)
                    self.attempts, self.excluded, self.selected = 0, [], None
                    self.transition(RecoveryState.TRACKING)
            else:
                self.target_since = None

    def summary(self):
        attempt_results = [e for e in self.events if e["event"] == "attempt_result"]
        return dict(events=len(self.events),
                    recoveries=sum(e["event"] == "recovery_success" for e in self.events),
                    hard_resets=sum(e["event"] == "hard_reset" for e in self.events),
                    labelled_attempts=len(attempt_results),
                    successful_attempts=sum(e["result"] == "success" for e in attempt_results),
                    attempt_logging=self.attempt_logging_enabled,
                    state=self.state.value)
