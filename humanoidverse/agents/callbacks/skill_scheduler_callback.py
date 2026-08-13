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
A, B (default: from scheduler_config.yaml candidates), enabled.
"""

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
        self._prev_ep_len = None
        self._just_injected = False
        self._reset_detect_cooldown = 0
        self._seen_estop_count = 0

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
        self._initialized = True
        logger.info(
            f"SkillScheduler ready: {self.graph.skill_names}, slot={self._slot}, "
            f"A={A:.3f}, B={B:.3f}, lambda_sw={lambda_sw}, tau={self.scheduler.tau}, "
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

        if "root_rot" in traj:  # stored wxyz
            rr = sRot.from_quat(traj["root_rot"][:, [1, 2, 3, 0]])
            traj["root_rot"] = (delta * rr).as_quat()[:, [3, 0, 1, 2]]

    def _inject(self, path, robot_yaw=None, robot_xy_rel=None, align=True,
                note=""):
        env = self.env
        lib = env._motion_lib
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

        close_entry = getattr(self.scheduler, "last_best_sim", np.inf) \
            <= self.scheduler.A
        if align and close_entry:
            # TRANSITION with a nearby entry (best_sim <= A): path[0] is the
            # robot's current pose, so DO NOT physically reset -- just
            # restart the reference clock at the new trajectory's frame 0.
            env.motion_start_times[0] = \
                -float(env.episode_length_buf[0]) * env.dt
            env.motion_len[0] = lib._motion_lengths[0]
        else:
            # Pure-skill playback, OR a transition whose entry is far from
            # the robot (best_sim > A): respawn into the reference's first
            # frame so reference and robot agree (a far entry without reset
            # makes the robot chase a mismatched pose -> instant
            # body_z/motion_far termination, observed in the 3-skill eval).
            env.reset_envs_idx(torch.tensor([0], device=env.device))
            self._reset_detect_cooldown = 5  # ignore the env's follow-up reset
        env._kick_motion_res_counter = -1
        self._just_injected = True

        n_cross = sum(1 for u, v in zip(path[:-1], path[1:])
                      if self.graph.skill_ids[u] != self.graph.skill_ids[v])
        logger.info(
            f"Injected path: {len(path)} nodes "
            f"({self.graph.skill_names[self.scheduler.current_cmd]}), "
            f"trigger={getattr(self.scheduler, 'last_trigger', '?')}{note}, "
            f"best_sim={getattr(self.scheduler, 'last_best_sim', float('nan')):.3f}, "
            f"start=node{path[0]}(skill {self.graph.skill_ids[path[0]]}, "
            f"frame {self.graph.frame_idxs[path[0]]}), "
            f"cross_hops={n_cross}, "
            f"buffers={int(self.graph.is_buffer[path].sum())}, "
            f"traj={traj['dof'].shape[0]} frames @ {self._traj_fps}fps"
        )

    def _inject_pure_skill(self, skill_id: int, t: float):
        """Install the canonical full-skill playback (frame 0 -> end) and
        respawn the robot at the skill's initial pose. Used when a transition
        finishes (loop the target skill) and after any natural env reset
        (fall / termination): 'back to the skill start and replay'."""
        path = self.scheduler.pure_skill_path(skill_id)
        self.scheduler.install_reference_path(path, t)
        # Respawning at the skill start IS the recovery action (until the
        # step-4 recovery FSM exists), so clear a latched e-stop.
        self.scheduler.estopped = False
        self._inject(path, align=False, note=", pure_skill")
        self._installed_version = self.scheduler.path_version

    # ------------------------------------------------------------------
    # Per-step hook
    # ------------------------------------------------------------------

    def on_pre_eval_env_step(self, actor_state):
        if not self.enabled:
            return actor_state
        if not self._initialized:
            self._lazy_init()

        env = self.env
        user_cmd = getattr(env, "requested_skill_id", None)
        if self._auto_interval > 0 and self._auto_skills \
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
        #     without our own injection. Respawn into the CURRENT commanded
        #     skill's canonical playback from frame 0 -- "back to the skill
        #     start pose and replay the skill".
        ep_len = int(env.episode_length_buf[0])
        natural_reset = (
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
            logger.info(
                f"Env reset detected ({getattr(self, '_last_termination', '?')})"
                f" -> respawn into pure skill "
                f"{self.scheduler.current_cmd} "
                f"({self.graph.skill_names[self.scheduler.current_cmd]})")
            self._inject_pure_skill(self.scheduler.current_cmd, t)
            self._prev_ep_len = 0
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
