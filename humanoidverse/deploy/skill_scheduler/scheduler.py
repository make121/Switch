"""Online skill scheduler (spec 2-5).

Deployment-time module: pure graph search + trigger logic. No training, no
RL. Implemented in this step:
  - entry_check (spec 3, attach / search / estop),
  - Graph-Search planning + path installation as guidance g_t = <s^g, kappa>,
  - step() main loop with the 4 trigger conditions (spec 5) for the
    attach / search paths.

Deliberately NOT implemented yet:
  - handle_safety_event + damping-controller FSM (step 4): an estop currently
    just latches `self.estopped` and stops emitting guidance,
  - the NN planner (step 3).
"""

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union

from .distance import NodeState
from .graph_data import SkillGraphData
from .planners import GraphSearchPlanner, NNPlanner


@dataclass
class Guidance:
    """g_t = <s_hat^g, kappa_t> fed to the downstream tracking policy."""
    node_id: int
    state: NodeState
    kappa: int


class SkillGraphScheduler:
    def __init__(self, graph: SkillGraphData, planner_type: str,
                 A: float, B: float, lambda_cost: float,
                 tau: float, top_k: int, grace_s: float = 0.0,
                 candidate_pool: str = "all",
                 enable_safety_replan: bool = False,
                 nn_buffer_base: float = 1.0, nn_max_buffer: int = 30,
                 switch_entry_A: Optional[float] = None):
        if planner_type not in ("graph_search", "nn"):
            raise ValueError(f"bad planner_type: {planner_type!r}")
        self.graph = graph
        self.planner_type = planner_type
        self.A = A
        self.B = B
        # A is calibrated for graph attachment/search.  A live policy does
        # not sit exactly on its reference node, so applying that same tight
        # threshold at a time-aligned Buffer source can reject every switch.
        # Keep a separate, moderately tolerant threshold for that operation.
        # It remains bounded by B so accepting a switch can never bypass the
        # scheduler's emergency-distance boundary.
        self.switch_entry_A = (
            min(11.0, float(B)) if switch_entry_A is None
            else float(switch_entry_A))
        if self.switch_entry_A <= 0:
            raise ValueError("switch_entry_A must be positive")
        if self.switch_entry_A > float(B):
            raise ValueError("switch_entry_A must be <= B")
        self.lambda_cost = lambda_cost
        self.tau = tau
        self.top_k = top_k
        # Safety auto-replan during tracking. DISABLED by default: with
        # pose-similarity entry selection, a replan can jump the reference in
        # TIME (pose-nearest != time-nearest for quasi-periodic skills), and
        # post-reset transients false-trigger B anyway. Turn this on only
        # together with the step-4 recovery FSM and calibrated A/B.
        self.enable_safety_replan = enable_safety_replan
        # "all": entry candidates span the whole graph, so planned paths
        # start near the robot's CURRENT pose and traverse cross-skill /
        # buffer edges into T_cmd (true transition trajectories).
        # "t_cmd": candidates are restricted to T_cmd itself (legacy; every
        # path starts at the target skill's opening frames).
        if candidate_pool not in ("all", "t_cmd"):
            raise ValueError(f"bad candidate_pool: {candidate_pool!r}")
        self.candidate_pool = candidate_pool
        # Safety-check grace period after a path install: right after an
        # env reset the live state carries reset noise (sim offset of ~3+
        # observed), so a too-eager safety_event would false-trigger.
        self.grace_s = grace_s
        self._grace_until = 0.0
        self.planner = GraphSearchPlanner(graph)
        self.nn_planner = NNPlanner(graph, buffer_base=nn_buffer_base,
                                    max_buffer=nn_max_buffer) \
            if planner_type == "nn" else None

        self.current_cmd: Optional[int] = None
        self.T_cmd: List[int] = []
        self.guidance_seq: List[Guidance] = []
        self.pointer: int = 0
        self.estopped: bool = False
        self.estop_count: int = 0  # increments on every latch (for logging)
        # Bookkeeping for the deployment loop: current_path is the node-id
        # path currently installed as reference; path_version increments on
        # every (re)plan so callers can detect the event and reload the
        # reference into the motion library.
        self.current_path: List[int] = []
        self.path_version: int = 0
        # A graph-search command change is intentionally two-phase.  The
        # command is queued first while the current reference keeps running;
        # it is installed only when the reference reaches a *future* source
        # node of a precomputed Buffer macro and the live robot is close to
        # that source.  This prevents attaching directly to a target-skill
        # middle frame or to a Buffer interior node.
        self.pending_cmd: Optional[int] = None
        self.pending_source: Optional[int] = None
        self.pending_path: List[int] = []
        self._pending_rejected_sources: Set[int] = set()
        self.entry_rejection_count: int = 0
        self.last_rejected_source: Optional[int] = None
        self.last_rejected_entry_sim: Optional[float] = None
        self._buffer_macros_by_source = {}
        for (src, dst), buffers in self.graph.buffer_chains.items():
            self._buffer_macros_by_source.setdefault(src, []).append(
                (dst, buffers))

    # ------------------------------------------------------------------
    # Command / target set
    # ------------------------------------------------------------------

    def set_command(self, skill: Union[str, int]):
        sid = self.graph.skill_index(skill)
        self.current_cmd = sid
        self.T_cmd = self.graph.target_set(sid, self.tau)

    def _clear_pending_command(self):
        self.pending_cmd = None
        self.pending_source = None
        self.pending_path = []
        self._pending_rejected_sources.clear()

    def _queue_command(self, skill: Union[str, int]):
        sid = self.graph.skill_index(skill)
        if sid == self.current_cmd:
            self._clear_pending_command()
            return
        if sid != self.pending_cmd:
            self.pending_cmd = sid
            self.pending_source = None
            self.pending_path = []
            self._pending_rejected_sources.clear()
            self.last_trigger = "cmd_pending"

    # ------------------------------------------------------------------
    # Spec 3: entry check
    # ------------------------------------------------------------------

    def entry_check(self, x: NodeState, T_cmd: List[int]):
        if self.candidate_pool == "t_cmd":
            pool = list(T_cmd)
        else:
            pool = range(self.graph.num_nodes)
        sims = {v: self.graph.sim_to(x, v) for v in pool}
        best_v = min(sims, key=sims.get)
        best_sim = sims[best_v]
        self.last_best_sim = best_sim  # diagnostics for the deployment loop
        self.last_sims = sims          # for init fallback candidate ranking
        if best_sim <= self.A:
            return "attach", [best_v]
        if best_sim >= self.B:
            return "estop", None
        top = sorted(sims.items(), key=lambda kv: kv[1])[: self.top_k]
        return "search", [v for v, _ in top]

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _extend_to_skill_end(self, path: List[int]) -> List[int]:
        """Continue the reference along temporal edges until the skill ends,
        so the installed guidance covers the whole commanded skill, not just
        the route into its target set."""
        while True:
            nxt = self.graph.temporal_successor(path[-1])
            if nxt is None:
                return path
            path.append(nxt)

    def _buffer_path_from_source(
            self, source: int, target_set: List[int]) -> Optional[List[int]]:
        """Best target-reaching path that takes a Buffer macro immediately.

        The regular value function is reused unchanged.  This method merely
        constrains the first edge to one of ``source``'s precomputed Buffer
        chains, which is the online execution contract for command changes.
        """
        if self.planner_type != "graph_search":
            return None
        V, next_hop = self.planner.build_value_function(target_set)
        target_nodes = set(target_set)
        options: List[Tuple[float, List[int]]] = []
        for dst, buffers in self._buffer_macros_by_source.get(source, []):
            if dst not in V:
                continue
            suffix = self.planner.reconstruct_path_gs(
                dst, next_hop, target_nodes)
            if suffix is None:
                continue
            path = [source] + list(buffers) + suffix
            # macro cost + the already-computed cost-to-target from dst
            cost = self.graph.macro_edge_costs[(source, dst)] + V[dst]
            options.append((cost, path))
        if not options:
            return None
        _, best = min(options, key=lambda item: item[0])
        return self._extend_to_skill_end(best)

    def _find_forward_buffer_entry(self) -> Tuple[Optional[int], List[int]]:
        """Find the earliest usable Buffer source ahead in reference time."""
        if self.pending_cmd is None or not self.guidance_seq:
            return None, []
        target_set = self.graph.target_set(self.pending_cmd, self.tau)
        current_node = self.guidance_seq[self.pointer].node_id
        current_skill = int(self.graph.skill_ids[current_node])
        if current_skill < 0:
            # Never attach from a Buffer interior.  Let the installed
            # transition reach an original-skill frame first.
            return None, []

        for guidance in self.guidance_seq[self.pointer:]:
            nid = guidance.node_id
            sid = int(self.graph.skill_ids[nid])
            if sid < 0:
                continue
            if sid != current_skill:
                break
            if nid in self._pending_rejected_sources:
                continue
            path = self._buffer_path_from_source(nid, target_set)
            if path is not None:
                return nid, path
        return None, []

    def _step_pending_command(
            self, x: NodeState, t: float) -> Optional[Guidance]:
        """Keep current guidance until a safe future Buffer source arrives."""
        source, path = self._find_forward_buffer_entry()
        self.pending_source = source
        self.pending_path = path
        current = self.current_guidance()
        if source is None or current is None or current.node_id != source:
            return current

        entry_sim = self.graph.sim_to(x, source)
        self.last_best_sim = entry_sim
        if entry_sim > self.switch_entry_A:
            # Do not jump or reset.  Continue the current skill and try the
            # next future macro source instead.
            self._pending_rejected_sources.add(source)
            self.entry_rejection_count += 1
            self.last_rejected_source = source
            self.last_rejected_entry_sim = entry_sim
            self.pending_source = None
            self.pending_path = []
            self.last_trigger = "cmd_wait_entry_error"
            return current

        target = self.pending_cmd
        self.set_command(target)
        self.last_trigger = "cmd_change"
        self.install_reference_path(path, t)
        self._clear_pending_command()
        return self.guidance_seq[0]

    def rewind_reference(self, t: float):
        """Synchronize with an env reset without rebuilding MotionLib.

        The environment has already reset itself against the currently loaded
        trajectory.  Rewinding the scheduler avoids the previous second
        load_motions()+reset_envs_idx() pair, which could crash Isaac Gym.
        """
        self.pointer = 0
        self.estopped = False
        self._grace_until = t + self.grace_s
        self.pending_source = None
        self.pending_path = []
        self._pending_rejected_sources.clear()

    def plan(self, x: NodeState, candidates: List[int],
             T_cmd: List[int]) -> Optional[List[int]]:
        """Score candidates, walk next_hop from the best reachable one.

        If every top-k candidate is unreachable (e.g. dead-end tail frames
        of a skill -- forward-only temporal edges and boundary-excluded
        cross edges leave them with no route to T), fall back to the best
        REACHABLE node overall; V already encodes reachability, so this is
        the principled graph-search entry selection."""
        if self.planner_type == "nn":
            path = self.nn_planner.plan_nn(x, candidates, T_cmd, self.B)
            return self._extend_to_skill_end(path) if path else None
        V, next_hop = self.planner.build_value_function(T_cmd)
        T_set = set(T_cmd)
        ranked = sorted(
            candidates,
            key=lambda v: self.planner.score(x, v, V, self.lambda_cost),
        )
        for v in ranked:
            path = self.planner.reconstruct_path_gs(v, next_hop, T_set)
            if path is not None:
                return self._extend_to_skill_end(path)
        # Fallback: best reachable node in the whole graph
        reachable = [v for v in self.last_sims if v in V]
        if reachable:
            best = min(reachable, key=lambda v: self.planner.score(
                x, v, V, self.lambda_cost))
            path = self.planner.reconstruct_path_gs(best, next_hop, T_set)
            if path is not None:
                return self._extend_to_skill_end(path)
        return None

    def path_to_guidance(self, path: List[int]) -> List[Guidance]:
        """kappa is taken from the node's kappa field (spec 1.4): buffer
        nodes pass through their remaining-buffer-step count, regular nodes
        emit kappa=0."""
        return [
            Guidance(node_id=nid, state=self.graph.nodes[nid],
                     kappa=int(self.graph.kappas[nid]))
            for nid in path
        ]

    def install_reference_path(self, path: List[int], t: float):
        """Install a node-id path as the current reference. Single entry
        point used by step() and by the deployment loop (e.g. installing a
        pure-skill playback after a transition completes or after a fall)."""
        self.guidance_seq = self.path_to_guidance(path)
        self.current_path = list(path)
        self.path_version += 1
        self.pointer = 0
        self._grace_until = t + self.grace_s

    def pure_skill_path(self, skill_id: int) -> List[int]:
        """The whole skill from frame 0 to its end (canonical playback)."""
        start, end = self.graph.skill_range(skill_id)
        return list(range(start, end))

    # ------------------------------------------------------------------
    # Spec 5: main loop
    # ------------------------------------------------------------------

    def current_guidance(self) -> Optional[Guidance]:
        if not self.guidance_seq:
            return None
        return self.guidance_seq[self.pointer]

    def transition_active(self) -> bool:
        """Whether the installed path has not landed in its target skill.

        ``current_cmd`` is changed when a transition path is installed, but
        that path starts at the old skill and then traverses Buffer nodes.
        Until its current node belongs to ``current_cmd``, replacing the path
        would restart the reference clock and visually resemble a reset.
        """
        guidance = self.current_guidance()
        if guidance is None or self.current_cmd is None:
            return False
        return int(self.graph.skill_ids[guidance.node_id]) != self.current_cmd

    def advance(self) -> Optional[Guidance]:
        """Advance the reference by one frame (called once per control step
        by the deployment loop)."""
        if self.guidance_seq and self.pointer < len(self.guidance_seq) - 1:
            self.pointer += 1
        return self.current_guidance()

    def step(self, x: NodeState, user_cmd: Optional[Union[str, int]],
             t: float) -> Optional[Guidance]:
        if self.estopped:
            # Until the step-4 recovery FSM lands, a fresh user command
            # re-arms the scheduler; otherwise stay latched, emit nothing.
            if user_cmd is not None \
                    and self.graph.skill_index(user_cmd) != self.current_cmd:
                self.estopped = False
            else:
                return None

        # Graph-search command changes are deferred to a future Buffer macro
        # source.  Initialisation and the NN planner retain their established
        # immediate planning behaviour.
        if self.current_cmd is not None and self.guidance_seq \
                and self.planner_type == "graph_search":
            if user_cmd is not None:
                requested = self.graph.skill_index(user_cmd)
                if requested != self.current_cmd:
                    self._queue_command(requested)
                elif self.pending_cmd is not None:
                    self._clear_pending_command()
            if self.pending_cmd is not None:
                # A command arriving during an installed transition is
                # queued.  Do not search from its old-skill/Buffer prefix and
                # do not hot-swap the reference.  Process it once the current
                # path first lands in the command's target skill.
                if self.transition_active():
                    self.last_trigger = "cmd_queued_during_transition"
                    return self.current_guidance()
                return self._step_pending_command(x, t)

        trigger = None
        if self.current_cmd is None or not self.guidance_seq:
            trigger = "init"
        elif user_cmd is not None \
                and self.graph.skill_index(user_cmd) != self.current_cmd:
            trigger = "cmd_change"
        elif self.pointer >= len(self.guidance_seq) - 1:
            trigger = "ref_end"
        elif self.enable_safety_replan \
                and t >= self._grace_until \
                and self.graph.sim_to(x, self.guidance_seq[self.pointer].node_id) \
                >= self.B:
            trigger = "safety_event"

        if trigger is None:
            return self.guidance_seq[self.pointer]

        self.last_trigger = trigger  # diagnostics

        if user_cmd is not None and trigger in ("init", "cmd_change"):
            self.set_command(user_cmd)

        status, candidates = self.entry_check(x, self.T_cmd)
        entry_status = status  # diagnostics: estop provenance
        if status == "estop" and trigger in ("init", "cmd_change"):
            # Commanded-switch fallback: a fresh user command (or the init
            # spawn) must never deadlock on estop -- plan from the top-k
            # nearest candidates instead. E.g. right after a reset the live
            # state carries noise, and on cmd_change the robot can be far
            # from every node of a dynamic skill.
            ranked = sorted(self.last_sims.items(), key=lambda kv: kv[1])
            status = "search"
            candidates = [v for v, _ in ranked[: self.top_k]]
        path = None
        if status == "attach":
            # Walk from the attached entry THROUGH the graph (cross-skill /
            # buffer edges included) into T_cmd.
            if self.planner_type == "nn":
                path = self.nn_planner.short_hop_path(
                    candidates[0], self.T_cmd, self.B)
                path = self._extend_to_skill_end(path) if path else None
            else:
                V, next_hop = self.planner.build_value_function(self.T_cmd)
                path = self.planner.reconstruct_path_gs(
                    candidates[0], next_hop, set(self.T_cmd))
                if path is not None:
                    path = self._extend_to_skill_end(path)
            if path is None:
                # Attached entry cannot reach T (e.g. sits past the target
                # region on forward-only temporal edges): scored search.
                ranked = sorted(self.last_sims.items(), key=lambda kv: kv[1])
                status = "search"
                candidates = [v for v, _ in ranked[: self.top_k]]
        if status == "search" and path is None:
            path = self.plan(x, candidates, self.T_cmd)
            if path is None:
                status = "estop"

        if status == "estop":
            # Step 4 implements handle_safety_event; for now latch e-stop.
            self.estopped = True
            self.estop_count += 1
            self.guidance_seq = []
            self.current_path = []
            self.path_version += 1
            self.last_estop_reason = (
                "entry_check" if entry_status == "estop" else "plan_failed")
            return None

        # `entry_check` records the nearest candidate, but a reachability
        # fallback may select another path start. Consumers use this value to
        # decide whether a physical reset is required, so it must describe
        # the path that will actually be installed.
        self.last_best_sim = self.graph.sim_to(x, path[0])
        self.install_reference_path(path, t)
        return self.guidance_seq[0]
