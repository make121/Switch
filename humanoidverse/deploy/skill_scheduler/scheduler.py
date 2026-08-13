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
from typing import List, Optional, Union

from .distance import NodeState
from .graph_data import SkillGraphData
from .planners import GraphSearchPlanner


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
                 enable_safety_replan: bool = False):
        if planner_type != "graph_search":
            raise NotImplementedError(
                f"planner_type={planner_type!r} not implemented yet (step 3)")
        self.graph = graph
        self.planner_type = planner_type
        self.A = A
        self.B = B
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

    # ------------------------------------------------------------------
    # Command / target set
    # ------------------------------------------------------------------

    def set_command(self, skill: Union[str, int]):
        sid = self.graph.skill_index(skill)
        self.current_cmd = sid
        self.T_cmd = self.graph.target_set(sid, self.tau)

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

    def plan(self, x: NodeState, candidates: List[int],
             T_cmd: List[int]) -> Optional[List[int]]:
        """Score candidates, walk next_hop from the best reachable one.

        If every top-k candidate is unreachable (e.g. dead-end tail frames
        of a skill -- forward-only temporal edges and boundary-excluded
        cross edges leave them with no route to T), fall back to the best
        REACHABLE node overall; V already encodes reachability, so this is
        the principled graph-search entry selection."""
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
            # buffer edges included) into T_cmd via next_hop.
            V, next_hop = self.planner.build_value_function(self.T_cmd)
            path = self.planner.reconstruct_path_gs(
                candidates[0], next_hop, set(self.T_cmd))
            if path is not None:
                path = self._extend_to_skill_end(path)
            else:
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

        self.install_reference_path(path, t)
        return self.guidance_seq[0]
