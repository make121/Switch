"""Eval callback that closes the loop for the online skill scheduler.

Per eval env step (on_pre_eval_env_step):
  1. read the skill command requested via keyboard (env.requested_skill_id,
     set in eval_agent.py),
  2. extract the live robot state x = (q, q_dot, root_z),
  3. run SkillGraphScheduler.step() (entry check / planning / triggers),
  4. when the scheduler installs a NEW path (path_version bump), assemble the
     corresponding motion trajectory (ReferenceBuilder), align it to the
     robot's current yaw / x-y, and hot-swap it into the motion library so
     the policy's obs/rewards track the planned path with ZERO changes to
     the env code,
  5. keep the scheduler's guidance pointer in lockstep with the env's
     reference clock.

Enable during eval with a config override, e.g.:

    +algo.eval_callbacks.skill_scheduler._target_=\
humanoidverse.agents.callbacks.skill_scheduler_callback.SkillSchedulerEvalCallback \
    +algo.eval_callbacks.skill_scheduler.config.skill_graph_path=\
SG_build/sg_output_V2/skill_graph.json \
    +algo.eval_callbacks.skill_scheduler.config.scheduler_config_path=\
SG_build/sg_output_V2/scheduler_config.yaml \
    +algo.eval_callbacks.skill_scheduler.config.skill_pkl_files=\
[example/motion_data/Horse-stance_pose.pkl,example/motion_data/Horse-stance_punch.pkl]

Optional config keys: planner_type, tau, top_k, lambda_sw, lambda_cost,
A, B (default: from scheduler_config.yaml candidates), switch_entry_A,
enabled.
"""

import copy
import numpy as np
import torch
import yaml
from loguru import logger
from scipy.spatial.transform import Rotation as sRot

from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.deploy.skill_scheduler.distance import NodeState
from humanoidverse.deploy.skill_scheduler.graph_data import SkillGraphData
from humanoidverse.deploy.skill_scheduler.reference_builder import ReferenceBuilder
from humanoidverse.deploy.skill_scheduler.scheduler import SkillGraphScheduler


class SkillSchedulerEvalCallback(RL_EvalCallback):
    def __init__(self, config, training_loop):
        super().__init__(config, training_loop)
        self.env = self.training_loop.env
        self.enabled = bool(getattr(config, "enabled", True))
        self._initialized = False
        self._installed_version = -1
        self._traj_fps = 30.0
        self._installed_is_transition = False
        self._prev_ep_len = None
        self._just_injected = False
        self._reset_detect_cooldown = 0
        self._seen_estop_count = 0
        # Benchmark diagnostics.  This is deliberately observational: it does
        # not alter planning or reset behaviour.
        self.injection_history = []
        self.recovery_fsm = None
        self._reference_dirty = False
        self._reset_warmup_pending = False
        self._recovery_policy = None
        self._policy_role = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _lazy_init(self):
        cfg = self.config
        with open(cfg.scheduler_config_path) as f:
            sg_cfg = yaml.safe_load(f)

        lambda_sw = float(getattr(cfg, "lambda_sw", sg_cfg["lambda_sw"][1]))
        self.graph = SkillGraphData.from_json(
            cfg.skill_graph_path,
            sigma_q=sg_cfg["sigma_q"], sigma_qdot=sg_cfg["sigma_qdot"],
            sigma_p=sg_cfg["sigma_p"], lambda_sw=lambda_sw,
            w_q=float(sg_cfg.get("w_q", 1.0)),
            w_qdot=float(getattr(cfg, "w_qdot", sg_cfg.get("w_qdot", 1.0))),
            w_p=float(getattr(cfg, "w_p", sg_cfg.get("w_p", 1.0))),
        )
        A = float(getattr(cfg, "A", sg_cfg["A_candidates"][0]))
        B = float(getattr(cfg, "B", sg_cfg["B_candidates"][-1]))
        default_switch_entry_A = float(
            sg_cfg.get("switch_entry_A", min(11.0, B)))
        self.scheduler = SkillGraphScheduler(
            self.graph,
            planner_type=str(getattr(cfg, "planner_type", "graph_search")),
            A=A, B=B,
            lambda_cost=float(getattr(cfg, "lambda_cost", sg_cfg["lambda_cost"][1])),
            tau=float(getattr(cfg, "tau", sg_cfg["tau"][1])),
            top_k=int(getattr(cfg, "top_k", sg_cfg["top_k"][1])),
            grace_s=float(getattr(cfg, "grace_s", 1.0)),
            candidate_pool=str(getattr(cfg, "candidate_pool", "all")),
            enable_safety_replan=bool(getattr(cfg, "enable_safety_replan", False)),
            switch_entry_A=float(getattr(
                cfg, "switch_entry_A", default_switch_entry_A)),
        )
        self.ref_builder = ReferenceBuilder.from_pkl_files(
            self.graph, list(cfg.skill_pkl_files))

        # Headless / scripted switching (also used by the future grid-search
        # harness, spec 6.2): cycle through auto_switch_skills every
        # auto_switch_interval_steps env steps. -1 disables (keyboard only).
        self._auto_interval = int(getattr(cfg, "auto_switch_interval_steps", -1))
        self._auto_skills = list(getattr(cfg, "auto_switch_skills", []))
        self._auto_idx = 0

        # Eval runs with num_envs=1; the injection slot is the motion entry
        # the env currently tracks. Every switch replaces THIS slot.
        self._slot = int(self.env._motion_lib._curr_motion_ids[0].item())

        # Start tracking the skill the robot was reset into.
        initial_skill = int(getattr(cfg, "initial_skill", 0))
        self.scheduler.set_command(initial_skill)
        # The eval motion_file is the canonical initial skill.  Mirror that
        # already-loaded reference in the scheduler instead of planning an
        # arbitrary graph route (and resetting) during the first eval step.
        initial_path = self.scheduler.pure_skill_path(initial_skill)
        self.scheduler.install_reference_path(initial_path, 0.0)
        self._installed_version = self.scheduler.path_version
        self._initialized = True
        recovery_cfg = getattr(cfg, "recovery_fsm", {})
        if recovery_cfg.get("enabled", False):
            from humanoidverse.deploy.skill_scheduler.recovery_fsm import RecoveryFSM
            self.recovery_fsm = RecoveryFSM(self, recovery_cfg)
            self.env.recovery_controller = self.recovery_fsm
            self._update_recovery_slot_role(initial_path)
            recovery_checkpoint = recovery_cfg.get("policy_checkpoint", None)
            if recovery_checkpoint:
                # Independent weights; the task/Buffer actor remains untouched.
                model = copy.deepcopy(self.training_loop.alg)
                checkpoint = torch.load(recovery_checkpoint, map_location=self.device)
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                model.eval()
                self._recovery_policy = model.act_inference
                self.recovery_fsm.record("recovery_policy_loaded", checkpoint=str(recovery_checkpoint))
        logger.info(
            f"SkillScheduler ready: {self.graph.skill_names}, slot={self._slot}, "
            f"A={A:.3f}, switch_entry_A={self.scheduler.switch_entry_A:.3f}, "
            f"B={B:.3f}, lambda_sw={lambda_sw}, tau={self.scheduler.tau}, "
            f"top_k={self.scheduler.top_k}. Press 7/8/9/0 to command skill 0/1/2/3."
        )

    # ------------------------------------------------------------------
    # Live state extraction
    # ------------------------------------------------------------------

    def _extract_state(self):
        """x = (q, q_dot, root_z). q / q_dot are yaw-invariant joint
        quantities; p_hat is z-only by design (consistent with graph
        construction and calibration), so no x-y / yaw normalization of the
        live state is needed."""
        sim = self.env.simulator
        q = sim.dof_pos[0].detach().cpu().numpy().astype(np.float64)
        q_dot = sim.dof_vel[0].detach().cpu().numpy().astype(np.float64)
        root = sim.robot_root_states[0].detach().cpu().numpy()
        # root layout: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
        qx, qy, qz, qw = root[3:7]
        yaw = float(np.arctan2(2 * (qw * qz + qx * qy),
                               1 - 2 * (qy * qy + qz * qz)))
        origin = self.env.env_origins[0].detach().cpu().numpy()
        xy_rel = root[:2] - origin[:2]
        x = NodeState(q=q, q_dot=q_dot,
                      p_hat=np.array([0.0, 0.0, root[2]], dtype=np.float64))
        return x, yaw, xy_rel

    # ------------------------------------------------------------------
    # Reference injection
    # ------------------------------------------------------------------

    def _update_recovery_slot_role(self, path):
        """Hot-swapped eval entries must not retain their old reward role."""
        lib = self.env._motion_lib
        source = next(int(self.graph.skill_ids[n]) for n in path
                      if int(self.graph.skill_ids[n]) >= 0)
        is_transition = bool(np.any(self.graph.is_buffer[path]))
        role = self.graph.skill_roles[source]
        bucket = ("recovery_transition" if role == "recovery" else "task_transition") \
            if is_transition else role + "_skill"
        for indices in lib._role_indices.values():
            if self._slot in indices:
                indices.remove(self._slot)
        lib._role_indices[bucket].append(self._slot)
        lib._is_transition_entry[self._slot] = is_transition
        lib._recovery_motion_mask[self._slot] = role == "recovery" and not is_transition
        self.env.is_cross_skill_env[0] = is_transition
        for name in ("_task_collision_motion_mask", "_recovery_reward_motion_mask"):
            if hasattr(self.env, name):
                delattr(self.env, name)

    def _align_to_robot(self, traj, robot_yaw, robot_xy_rel):
        """Rotate the assembled trajectory (graph-normalized: yaw ~ 0, x-y
        centered) into the robot's current heading and shift it so the first
        frame sits at the robot's current (env-origin-relative) x-y."""
        pose_aa = traj["pose_aa"]  # (T, J, 3), row 0 = root orientation
        yaw0 = sRot.from_rotvec(pose_aa[0, 0]).as_euler("xyz")[2]
        delta = sRot.from_euler("z", robot_yaw - yaw0)
        pose_aa[:, 0] = (delta * sRot.from_rotvec(pose_aa[:, 0])).as_rotvec()

        trans = traj["root_trans_offset"]
        trans[:] = delta.apply(trans)
        trans[:, :2] += robot_xy_rel - trans[0, :2]

        if "root_rot" in traj:  # PBHC and SciPy both use xyzw
            rr = sRot.from_quat(traj["root_rot"])
            traj["root_rot"] = (delta * rr).as_quat()

    def _inject(self, path, robot_yaw=None, robot_xy_rel=None, align=True,
                note="", physical_reset_mode="auto"):
        if physical_reset_mode not in ("auto", "never"):
            raise ValueError("Invalid physical_reset_mode")
        env = self.env
        lib = env._motion_lib
        # A graph path containing Buffer nodes is a one-shot transition
        # reference.  It must never be replayed after motion_end; once it is
        # exhausted the target skill becomes the persistent reference.
        self._installed_is_transition = bool(
            np.any(self.graph.is_buffer[np.asarray(path, dtype=np.int64)]))
        traj = self.ref_builder.build_trajectory(path)
        if align:
            # Transition reference: anchor the first frame to the robot's
            # current yaw / x-y so the reference continues smoothly.
            self._align_to_robot(traj, robot_yaw, robot_xy_rel)
        # else: pure-skill playback keeps canonical coords, i.e. the skill
        # starts at the env origin -- the play area stays put across resets.
        self._traj_fps = float(traj["fps"])

        lib._motion_data_list[self._slot] = traj
        lib.load_motions(random_sample=False, start_idx=self._slot,
                         target_heading=None)
        env.curr_motion_ids = lib._curr_motion_ids
        env.curr_motion_keys = lib.curr_motion_keys
        if self.recovery_fsm is not None:
            self._update_recovery_slot_role(path)

        trigger = str(getattr(self.scheduler, "last_trigger", "?"))
        entry_threshold = self.scheduler.switch_entry_A \
            if trigger == "cmd_change" else self.scheduler.A
        entry_sim = float(getattr(self.scheduler, "last_best_sim", np.inf))
        entry_metric = "graph_sim"
        if trigger == "recovery_entry" and self.recovery_fsm is not None:
            entry_sim = self.recovery_fsm.selected.distance
            entry_threshold = self.recovery_fsm.threshold
            entry_metric = "recovery_nn"
        close_entry = entry_threshold is None or entry_sim <= entry_threshold
        preserve_physics = physical_reset_mode == "never" or (align and close_entry)
        # FSM-enabled reference switches must not hide failures with respawns.
        if self.recovery_fsm is not None and align:
            preserve_physics = True
        did_reset = not preserve_physics
        if preserve_physics:
            # TRANSITION at its time-aligned source: path[0] is the current
            # reference frame and the live error passed the applicable entry
            # threshold, so DO NOT physically reset -- just restart the
            # reference clock at the new trajectory's frame 0.
            env.motion_start_times[0] = \
                -float(env.episode_length_buf[0]) * env.dt
            env.motion_len[0] = lib._motion_lengths[0]
        else:
            # Pure-skill playback, OR a transition whose entry is beyond its
            # applicable threshold: respawn into the reference's first
            # frame so reference and robot agree (a far entry without reset
            # makes the robot chase a mismatched pose -> instant
            # body_z/motion_far termination, observed in the 3-skill eval).
            self._reset_reference_episode()
            self._reset_detect_cooldown = 5  # ignore the env's follow-up reset
        env._kick_motion_res_counter = -1
        self._just_injected = True
        self._reference_dirty = True

        n_cross = sum(1 for u, v in zip(path[:-1], path[1:])
                      if self.graph.skill_ids[u] != self.graph.skill_ids[v])
        self.injection_history.append({
            "step": int(env.common_step_counter),
            "command": int(self.scheduler.current_cmd),
            "trigger": str(getattr(self.scheduler, "last_trigger", "?")),
            "best_sim": entry_sim,
            "entry_metric": entry_metric,
            "entry_threshold": None if entry_threshold is None else float(entry_threshold),
            "path_nodes": int(len(path)),
            "cross_hops": int(n_cross),
            "buffer_nodes": int(self.graph.is_buffer[path].sum()),
            "aligned": bool(align),
            "physical_reset": bool(did_reset),
            "reset_committed_before_physics": bool(did_reset),
        })
        logger.info(
            f"Injected path: {len(path)} nodes "
            f"({self.graph.skill_names[self.scheduler.current_cmd]}), "
            f"trigger={getattr(self.scheduler, 'last_trigger', '?')}{note}, "
            f"best_sim={entry_sim:.3f}, entry_metric={entry_metric}, "
            f"entry_threshold={entry_threshold}, "
            f"start=node{path[0]}(skill {self.graph.skill_ids[path[0]]}, "
            f"frame {self.graph.frame_idxs[path[0]]}), "
            f"cross_hops={n_cross}, "
            f"buffers={int(self.graph.is_buffer[path].sum())}, "
            f"physical_reset={did_reset}, "
            f"traj={traj['dof'].shape[0]} frames @ {self._traj_fps}fps"
        )

    def _inject_pure_skill(self, skill_id: int, t: float):
        """Install the canonical full-skill playback (frame 0 -> end) and
        respawn the robot at the skill's initial pose. Used when a transition
        finishes (loop the target skill) and after any natural env reset
        (fall / termination): 'back to the skill start and replay'."""
        if self.recovery_fsm is not None and self.recovery_fsm.active:
            from humanoidverse.deploy.skill_scheduler.recovery_fsm import RecoveryState
            self.recovery_fsm.transition(RecoveryState.HARD_RESET,
                                         reason="recovery_reference_exhausted")
            return
        path = self.scheduler.pure_skill_path(skill_id)
        self.scheduler.install_reference_path(path, t)
        # Legacy playback loops still respawn at the skill start. Active
        # autonomous recovery is handled above and never counted as a loop.
        self.scheduler.estopped = False
        self._inject(path, align=False, note=", pure_skill")
        self._installed_version = self.scheduler.path_version

    def _reset_reference_episode(self):
        """Commit a pre-step reset before physics can overwrite its tensors."""
        env = self.env
        ids = torch.tensor([0], device=env.device)
        env.reset_envs_idx(ids)  # Clears the old episode's history and actions.
        env.simulator.set_actor_root_state_tensor(ids, env.simulator.all_root_states)
        env.simulator.set_dof_state_tensor(ids, env.simulator.dof_state)
        env.need_to_refresh_envs[ids] = False
        if hasattr(env, "action_queue"):
            env.action_queue[ids] = 0
        # Gym rigid-body FK tensors remain stale until physics advances.
        # Use one ordinary, counted zero-action step, as env.reset_all does;
        # its post-step computes observations/history from the new episode.
        self._reset_warmup_pending = True

    def _refresh_reference_policy(self, actor_state):
        if self._reset_warmup_pending:
            actor_state["actions"] = torch.zeros_like(actor_state["actions"])
            actor_state["dones"] = torch.zeros_like(self.env.reset_buf, dtype=torch.bool)
            self._reset_warmup_pending = False
            self._reference_dirty = False
            return actor_state
        if self._reference_dirty:
            # Live reference switches preserve physical history; only refresh
            # the target and inference, without appending history twice.
            self.env._pre_compute_observations_callback()
            self.env._compute_observations()
            clip = self.env.config.normalization.clip_observations
            actor_state["obs"] = {k: torch.clamp(v, -clip, clip)
                                  for k, v in self.env.obs_buf_dict.items()}
            actor_state["actions"] = self.training_loop.eval_policy(actor_state["obs"])
            self._reference_dirty = False
        return self._apply_recovery_policy(actor_state)

    # ------------------------------------------------------------------
    # Per-step hook
    # ------------------------------------------------------------------

    def on_pre_eval_env_step(self, actor_state):
        if not self.enabled:
            return actor_state
        if not self._initialized:
            self._lazy_init()
        fsm = self.recovery_fsm
        if fsm is not None:
            requested_push = bool(getattr(self.env, "requested_fall_push", False))
            self.env.requested_fall_push = False
            push_step = int(fsm.config.get("fall_push_step", -1))
            push_due = push_step >= 0 and self.env.common_step_counter >= push_step \
                and not getattr(self, "_fall_push_applied", False)
            if requested_push or push_due:
                self._fall_push_applied = True
                if not fsm.active:
                    velocity = torch.as_tensor(fsm.config.get("fall_push_velocity_xy", [1.5, 0.0]),
                                               device=self.env.device, dtype=torch.float)
                    if velocity.shape != (2,) or not bool(torch.isfinite(velocity).all()):
                        raise ValueError("fall_push_velocity_xy must contain two finite velocities")
                    self.env.simulator.robot_root_states[0, 7:9] += velocity
                    # Flush BEFORE physics: a post-step refresh would overwrite
                    # this velocity before the ordinary reset dispatcher flushes.
                    self.env.simulator.set_actor_root_state_tensor(
                        torch.zeros(1, dtype=torch.long, device=self.env.device),
                        self.env.simulator.all_root_states)
                    fsm.record("fall_push", delta_velocity_xy=velocity.cpu().tolist(),
                               physical_reset=False)
                else:
                    fsm.record("fall_push_ignored", reason="recovery_active")
            # Safety monitoring must have priority over pending user commands.
            # Old scheduler.step() returns early while a command is queued.
            if not fsm.active and self.scheduler.enable_safety_replan \
                    and fsm.now >= self.scheduler._grace_until:
                guidance = self.scheduler.current_guidance()
                x, _, _ = self._extract_state()
                if guidance is not None and self.graph.sim_to(x, guidance.node_id) >= self.scheduler.B:
                    fsm.request_recovery("scheduler_B")
            requested = bool(getattr(self.env, "requested_recovery", False))
            self.env.requested_recovery = False
            test_step = int(fsm.config.get("diagnostic_trigger_step", -1))
            test_due = test_step >= 0 and self.env.common_step_counter >= test_step \
                and not getattr(self, "_recovery_test_triggered", False)
            if requested or test_due:
                self._recovery_test_triggered = True
                fsm.request_recovery("manual_test" if requested else "scripted_test")
            was_active = fsm.active
            command = getattr(self.env, "requested_skill_id", None)
            blocked = fsm.before_scheduler(command if was_active else None)
            if was_active:
                self.env.requested_skill_id = None
                # The environment has already performed the one hard reset.
                if not fsm.active:
                    actor_state["dones"] = torch.zeros(1, dtype=torch.bool,
                                                      device=self.env.device)
            if blocked:
                return actor_state
        actor_state = self._scheduler_step(actor_state)
        if fsm is not None:
            if self.scheduler.estopped:
                fsm.request_recovery("scheduler_estop")
            fsm.after_scheduler()
        return self._refresh_reference_policy(actor_state)

    def _apply_recovery_policy(self, actor_state):
        """Recovery actor owns recovery frames; unified actor owns all Buffers."""
        if self._recovery_policy is None:
            return actor_state
        guidance = self.scheduler.current_guidance()
        skill_id = -1 if guidance is None else int(self.graph.skill_ids[guidance.node_id])
        use_recovery = (self.recovery_fsm is not None and self.recovery_fsm.active
                        and not self.recovery_fsm.damping_active and skill_id >= 0
                        and self.graph.skill_roles[skill_id] == "recovery")
        role = "recovery" if use_recovery else "task_and_buffer"
        if role != self._policy_role:
            self.recovery_fsm.record("policy_switch", policy=role, skill_id=skill_id)
            self._policy_role = role
            # Do not execute delayed actions from the other policy.
            if hasattr(self.env, "action_queue"):
                self.env.action_queue.zero_()
        with torch.no_grad():
            policy = self._recovery_policy if use_recovery else self.training_loop.eval_policy
            actor_state["actions"] = policy(actor_state["obs"])
        return actor_state

    def install_recovery_entry(self, entry):
        _, end = self.graph.skill_range(entry.skill_id)
        path = list(range(entry.node_id, end))
        self.scheduler.estopped = False
        self.scheduler._clear_pending_command()
        self.scheduler.set_command(entry.skill_id)
        self.scheduler.last_trigger = "recovery_entry"
        x, yaw, xy = self._extract_state()
        self.scheduler.last_best_sim = self.graph.sim_to(x, entry.node_id)
        self.scheduler.install_reference_path(path, self.env.common_step_counter * self.env.dt)
        self._inject(path, yaw, xy, physical_reset_mode="never", note=", recovery_entry")
        self._installed_version = self.scheduler.path_version
        # No stale delayed task action may be sent at recovery startup.
        if hasattr(self.env, "action_queue"):
            self.env.action_queue.zero_()

    def prepare_recovery_hard_reset(self, skill_id):
        """Only prepare the reference; env's reset dispatcher owns physics."""
        self.scheduler.estopped = False
        self.scheduler._clear_pending_command()
        self.scheduler.set_command(skill_id)
        path = self.scheduler.pure_skill_path(skill_id)
        self.scheduler.last_trigger = "recovery_hard_reset"
        self.scheduler.install_reference_path(path, self.env.common_step_counter * self.env.dt)
        self._inject(path, align=False, physical_reset_mode="never",
                     note=", preparing_hard_reset")
        self._installed_version = self.scheduler.path_version

    def on_post_evaluate_policy(self):
        if self.recovery_fsm is not None:
            logger.info("RecoveryFSM summary: {}", self.recovery_fsm.summary())
            self.env.recovery_controller = None

    def _scheduler_step(self, actor_state):
        if not self.enabled:
            return actor_state
        if not self._initialized:
            self._lazy_init()

        env = self.env
        user_cmd = getattr(env, "requested_skill_id", None)
        # Keyboard requests are edge-triggered events. Keeping the value on
        # the env made the same command appear on every subsequent frame and
        # obscured whether a queued switch had actually been consumed.
        if hasattr(env, "requested_skill_id"):
            env.requested_skill_id = None
        if self._auto_interval > 0 and self._auto_skills \
                and not (self.recovery_fsm is not None and self.recovery_fsm.active) \
                and env.common_step_counter > 0 \
                and env.common_step_counter % self._auto_interval == 0:
            user_cmd = self._auto_skills[self._auto_idx % len(self._auto_skills)]
            self._auto_idx += 1
            logger.info(f"Auto-switch command -> skill {user_cmd} "
                        f"({self.graph.skill_names[user_cmd]})")
        x, robot_yaw, robot_xy_rel = self._extract_state()
        t = env.common_step_counter * env.dt

        # Track which termination fired (for distinguishing benign motion_end
        # timeouts from genuine tracking failures in the logs).
        term = getattr(env, "reset_buf_terminate_by", None)
        if term:
            fired = [k for k, v in term.items() if bool(v[0])]
            if fired:
                self._last_termination = ",".join(fired)

        # --- Reference lifecycle management (before the scheduler step) ---
        # (a) Natural env reset (fall / termination): episode_length wrapped
        #     without our own injection. The env has already reset against
        #     the installed reference, so rewind scheduler state only.
        ep_len = int(env.episode_length_buf[0])
        previous_dones = actor_state.get("dones")
        previous_done = previous_dones is not None \
            and bool(previous_dones[0])

        # Hand off a one-shot transition to the full target skill before the
        # next env step can terminate it.  The env checks motion_end after
        # incrementing episode_length_buf, so predict that exact next time
        # here.  This avoids an env reset against the short transition and,
        # more importantly, prevents that transition from being rewound and
        # replayed indefinitely.
        if self._installed_is_transition and not previous_done \
                and self.scheduler.current_cmd is not None:
            next_motion_time = (ep_len + 1) * float(env.dt) \
                + float(env.motion_start_times[0])
            installed_motion_len = float(env.motion_len[0])
            if next_motion_time > installed_motion_len:
                logger.info(
                    "Transition reference ending -> install full target "
                    f"skill (command {self.scheduler.current_cmd}: "
                    f"{self.graph.skill_names[self.scheduler.current_cmd]})")
                self._inject_pure_skill(self.scheduler.current_cmd, t)
                self._prev_ep_len = int(env.episode_length_buf[0])
                return actor_state

        natural_reset = previous_done or (
            self._prev_ep_len is not None
            and ep_len < self._prev_ep_len
            and not self._just_injected
            and self._reset_detect_cooldown == 0
        )
        self._just_injected = False
        if self._reset_detect_cooldown > 0:
            self._reset_detect_cooldown -= 1
        self._prev_ep_len = ep_len
        if natural_reset and self.scheduler.current_cmd is not None:
            ended_transition = self._installed_is_transition \
                and "motion_end" in getattr(
                    self, "_last_termination", "").split(",")
            if ended_transition:
                # Fallback for a boundary/timing race: never rewind a
                # completed Buffer path.  The env has already reset once, so
                # this branch may perform a second reset, but it converts the
                # reference to a persistent pure-skill trajectory instead of
                # entering an infinite short-path reset loop.
                logger.info(
                    "Transition motion_end detected -> replace one-shot "
                    f"reference with full target skill (command "
                    f"{self.scheduler.current_cmd}: "
                    f"{self.graph.skill_names[self.scheduler.current_cmd]})")
                self._inject_pure_skill(self.scheduler.current_cmd, t)
                self._prev_ep_len = int(env.episode_length_buf[0])
                return actor_state
            logger.info(
                f"Env reset detected ({getattr(self, '_last_termination', '?')})"
                f" -> retry current loaded reference without MotionLib reload "
                f"(command {self.scheduler.current_cmd}: "
                f"{self.graph.skill_names[self.scheduler.current_cmd]})")
            # env.step() has already performed the physical reset using the
            # currently installed trajectory.  Reloading MotionLib and then
            # calling reset_envs_idx() here caused a second reset and repeated
            # native segfaults in the lie_down/get_up test.
            self.scheduler.rewind_reference(t)
            self._prev_ep_len = ep_len
            return actor_state

        # (b) Reference playback finished (transition completed / skill
        #     ended): loop the target skill from its first frame.
        if self.scheduler.guidance_seq and not self.scheduler.estopped \
                and self.scheduler.current_cmd is not None \
                and self.scheduler.pointer >= len(self.scheduler.guidance_seq) - 1:
            self._inject_pure_skill(self.scheduler.current_cmd, t)
            return actor_state

        # Diagnostics: remember what the safety check compared against.
        self._dbg_x = x
        if self.scheduler.guidance_seq:
            self._dbg_target = \
                self.scheduler.guidance_seq[self.scheduler.pointer].node_id

        self.scheduler.step(x, user_cmd, t)

        rejection_count = self.scheduler.entry_rejection_count
        if rejection_count != getattr(self, "_seen_entry_rejections", 0):
            self._seen_entry_rejections = rejection_count
            source = self.scheduler.last_rejected_source
            logger.warning(
                f"Rejected Buffer entry node{source} "
                f"(skill {self.graph.skill_ids[source]}, "
                f"frame {self.graph.frame_idxs[source]}): "
                f"entry_sim={self.scheduler.last_rejected_entry_sim:.3f} > "
                f"switch_entry_A={self.scheduler.switch_entry_A:.3f}; "
                f"continuing current reference")

        pending = self.scheduler.pending_cmd
        pending_source = self.scheduler.pending_source
        pending_state = (pending, pending_source)
        if pending_state != getattr(self, "_last_pending_state", None):
            self._last_pending_state = pending_state
            if pending is not None:
                if self.scheduler.transition_active():
                    logger.info(
                        f"Pending switch -> {self.graph.skill_names[pending]}: "
                        f"queued until the current transition lands in "
                        f"{self.graph.skill_names[self.scheduler.current_cmd]}")
                elif pending_source is None:
                    logger.info(
                        f"Pending switch -> {self.graph.skill_names[pending]}: "
                        f"waiting for a future Buffer macro source")
                else:
                    logger.info(
                        f"Pending switch -> {self.graph.skill_names[pending]}: "
                        f"future entry=node{pending_source} "
                        f"(skill {self.graph.skill_ids[pending_source]}, "
                        f"frame {self.graph.frame_idxs[pending_source]})")

        if self.scheduler.path_version != self._installed_version:
            if self.scheduler.current_path:
                self._inject(self.scheduler.current_path, robot_yaw, robot_xy_rel)
            self._installed_version = self.scheduler.path_version

        # Lockstep the guidance pointer with the env reference clock.
        if self.scheduler.guidance_seq:
            motion_time = float(env.episode_length_buf[0]) * env.dt \
                + float(env.motion_start_times[0])
            self.scheduler.pointer = min(
                int(motion_time * self._traj_fps),
                len(self.scheduler.guidance_seq) - 1,
            )

        if self.scheduler.estopped \
                and self.scheduler.estop_count != self._seen_estop_count:
            self._seen_estop_count = self.scheduler.estop_count
            breakdown = ""
            if getattr(self, "_dbg_target", None) is not None:
                node = self.graph.nodes[self._dbg_target]
                x0 = self._dbg_x
                d_q = np.abs(x0.q - node.q).sum() / self.graph.sigma_q
                d_qd = np.abs(x0.q_dot - node.q_dot).sum() / self.graph.sigma_qdot
                d_p = np.abs(x0.p_hat - node.p_hat).sum() / self.graph.sigma_p
                breakdown = (f" | target=node{self._dbg_target} "
                             f"(frame {self.graph.frame_idxs[self._dbg_target]}) "
                             f"d_q={d_q:.2f} d_qdot={d_qd:.2f} d_p={d_p:.2f} "
                             f"x.q.shape={x0.q.shape}")
            logger.warning(
                f"Skill scheduler E-STOP latched: "
                f"trigger={getattr(self.scheduler, 'last_trigger', '?')}, "
                f"reason={getattr(self.scheduler, 'last_estop_reason', '?')}, "
                f"best_sim_to_T={getattr(self.scheduler, 'last_best_sim', float('nan')):.3f}, "
                f"B={self.scheduler.B:.3f}{breakdown}"
            )
        return actor_state
