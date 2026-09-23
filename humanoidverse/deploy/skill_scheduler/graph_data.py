"""Runtime skill-graph representation for the online scheduler (spec 1).

Loads the enhanced skill_graph.json (per-node features exported by
SG_build/skill_graph_V2.py) and precomputes:
  - forward / reverse adjacency with DEPLOY-TIME edge weights (spec 1.2;
    NOT the training-time weights stored in the json),
  - per-edge normalized sim distances d_uv (spec 1.3; single implementation
    lives in distance.py),
  - target sets T_cmd = first tau fraction of a skill's frames (spec 3).
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .distance import NodeState, sim

EDGE_SAME_SKILL = "same_skill_consecutive"
EDGE_CROSS = "cross_skill"
EDGE_BUFFER = "buffer"


def edge_class(edge: dict) -> str:
    if edge["is_buffer"]:
        return EDGE_BUFFER
    if edge["is_cross_skill"]:
        return EDGE_CROSS
    return EDGE_SAME_SKILL


def deploy_edge_weight(edge_type: str, d_uv: float,
                       src_skill: int, dst_skill: int,
                       lambda_sw: float) -> float:
    """Deploy-time edge weight (spec 1.2).

    same_skill_consecutive edges cost 1.0; cross_skill and buffer edges cost
    d_uv plus a lambda_sw penalty when the endpoint skill ids differ (buffer
    nodes carry skill_id=-1, so buffer hops always pay the penalty).
    """
    if edge_type == EDGE_SAME_SKILL:
        return 1.0
    penalty = lambda_sw if src_skill != dst_skill else 0.0
    return d_uv + penalty


@dataclass
class SkillGraphData:
    nodes: List[NodeState]
    skill_ids: np.ndarray    # (N,) int, -1 for buffer nodes
    frame_idxs: np.ndarray   # (N,) int, -1 for buffer nodes
    is_buffer: np.ndarray    # (N,) bool
    kappas: np.ndarray       # (N,) int
    buffer_src: np.ndarray   # (N,) int, src node of the buffer chain (-1 if N/A)
    buffer_dst: np.ndarray   # (N,) int, dst node of the buffer chain (-1 if N/A)
    skill_names: List[str]
    skill_lengths: List[int]
    skill_roles: List[str]   # one of {task, recovery}; old graphs default to task
    # adj[u] = [(v, weight, edge_type)]; rev_adj[v] = [(u, weight, edge_type)]
    adj: List[List[Tuple[int, float, str]]] = field(default_factory=list)
    rev_adj: List[List[Tuple[int, float, str]]] = field(default_factory=list)
    d_uv: Dict[Tuple[int, int], float] = field(default_factory=dict)
    sigma_q: float = 1.0
    sigma_qdot: float = 1.0
    sigma_p: float = 1.0
    w_q: float = 1.0
    w_qdot: float = 1.0
    w_p: float = 1.0
    lambda_sw: float = 5.0
    # A cross-skill edge is a macro edge.  When it owns Buffer nodes, the raw
    # direct edge is retained here as metadata but replaced in adj/rev_adj by
    # src -> buffers -> dst.  The expanded edge weights sum to the macro cost.
    buffer_chains: Dict[Tuple[int, int], Tuple[int, ...]] = field(
        default_factory=dict)
    macro_edge_costs: Dict[Tuple[int, int], float] = field(default_factory=dict)
    expanded_macro_edges: int = 0

    @classmethod
    def from_json(cls, path: str, sigma_q: float, sigma_qdot: float,
                  sigma_p: float, lambda_sw: float,
                  w_q: float = 1.0, w_qdot: float = 1.0,
                  w_p: float = 1.0, p_z_only: bool = True) -> "SkillGraphData":
        with open(path) as f:
            g = json.load(f)
        if "nodes" not in g:
            raise ValueError(
                f"{path} has no 'nodes' array. Rebuild the graph with the "
                f"updated SG_build/skill_graph_V2.py (node feature export)."
            )

        def _p_hat(n) -> np.ndarray:
            p = np.asarray(n["p_hat"], dtype=np.float64)
            if p_z_only:
                # Z-only, matching the graph-construction distance
                # (frame_distance uses root_z only) and the live robot state
                # (root height). Must stay consistent with calibrate.py.
                p[:2] = 0.0
            return p

        nodes = [
            NodeState(q=np.asarray(n["q"], dtype=np.float64),
                      q_dot=np.asarray(n["q_dot"], dtype=np.float64),
                      p_hat=_p_hat(n))
            for n in g["nodes"]
        ]
        buffer_src = np.full(len(nodes), -1, dtype=np.int64)
        buffer_dst = np.full(len(nodes), -1, dtype=np.int64)
        for bn in g.get("buffer_nodes", []):
            buffer_src[bn["global_id"]] = bn["src_node"]
            buffer_dst[bn["global_id"]] = bn["dst_node"]

        # Group Buffer nodes by the exact cross-skill macro edge they expand.
        # kappa is N..1 along a chain, hence descending kappa is path order.
        grouped_buffers: Dict[Tuple[int, int], List[int]] = {}
        for bn in g.get("buffer_nodes", []):
            src = int(bn["src_node"])
            dst = int(bn["dst_node"])
            if not (0 <= src < len(g["nodes"]) and
                    0 <= dst < len(g["nodes"])):
                continue
            grouped_buffers.setdefault((src, dst), []).append(
                int(bn["global_id"]))
        buffer_chains = {
            key: tuple(sorted(ids,
                              key=lambda nid: int(g["nodes"][nid]["kappa"]),
                              reverse=True))
            for key, ids in grouped_buffers.items()
        }
        skill_names = list(g["skill_names"])
        skill_roles = list(g.get("skill_roles", ["task"] * len(skill_names)))
        if len(skill_roles) != len(skill_names):
            raise ValueError(
                f"{path} has {len(skill_roles)} skill_roles for "
                f"{len(skill_names)} skills."
            )
        invalid_roles = sorted(set(skill_roles) - {"task", "recovery"})
        if invalid_roles:
            raise ValueError(
                f"{path} contains invalid skill_roles: {invalid_roles}; "
                "expected only 'task' or 'recovery'."
            )
        graph = cls(
            nodes=nodes,
            skill_ids=np.asarray([n["skill_id"] for n in g["nodes"]], dtype=np.int64),
            frame_idxs=np.asarray([n["frame_idx"] for n in g["nodes"]], dtype=np.int64),
            is_buffer=np.asarray([n["is_buffer"] for n in g["nodes"]], dtype=bool),
            kappas=np.asarray([n["kappa"] for n in g["nodes"]], dtype=np.int64),
            buffer_src=buffer_src,
            buffer_dst=buffer_dst,
            skill_names=skill_names,
            skill_lengths=[int(x) for x in g["skill_lengths"]],
            skill_roles=skill_roles,
            sigma_q=sigma_q, sigma_qdot=sigma_qdot, sigma_p=sigma_p,
            w_q=w_q, w_qdot=w_qdot, w_p=w_p, lambda_sw=lambda_sw,
            buffer_chains=buffer_chains,
        )
        n = len(nodes)
        graph.adj = [[] for _ in range(n)]
        graph.rev_adj = [[] for _ in range(n)]

        def add_runtime_edge(u: int, v: int, w: float, et: str):
            graph.adj[u].append((v, w, et))
            graph.rev_adj[v].append((u, w, et))
            graph.d_uv[(u, v)] = sim(
                nodes[u], nodes[v], sigma_q, sigma_qdot, sigma_p,
                w_q, w_qdot, w_p)

        # Compute the deployment-time total cost of every raw cross-skill
        # macro edge before constructing the searchable adjacency.
        for e in g["edges"]:
            if edge_class(e) != EDGE_CROSS:
                continue
            u, v = int(e["src"]), int(e["dst"])
            d = sim(nodes[u], nodes[v], sigma_q, sigma_qdot, sigma_p,
                    w_q, w_qdot, w_p)
            graph.macro_edge_costs[(u, v)] = deploy_edge_weight(
                EDGE_CROSS, d, int(graph.skill_ids[u]),
                int(graph.skill_ids[v]), lambda_sw)

        # Load temporal edges and cross-skill macros without a Buffer chain.
        # Raw Buffer edges are reconstructed below with cost conservation.
        for e in g["edges"]:
            u, v = int(e["src"]), int(e["dst"])
            et = edge_class(e)
            if et == EDGE_BUFFER:
                continue
            if et == EDGE_CROSS and (u, v) in graph.buffer_chains:
                graph.expanded_macro_edges += 1
                continue
            d = sim(nodes[u], nodes[v], sigma_q, sigma_qdot, sigma_p,
                    w_q, w_qdot, w_p)
            w = deploy_edge_weight(et, d, int(graph.skill_ids[u]),
                                   int(graph.skill_ids[v]), lambda_sw)
            add_runtime_edge(u, v, w, et)

        # Expand each buffered macro into N+1 searchable Buffer edges.  Use
        # equal shares so the path has exactly the same total cost as the raw
        # direct macro; lambda_sw is therefore charged once, not once per
        # boundary involving a skill_id=-1 Buffer node.
        for (src, dst), buffers in graph.buffer_chains.items():
            macro_cost = graph.macro_edge_costs.get((src, dst))
            if macro_cost is None:
                # Backward-compatible fallback for artifacts that contain a
                # Buffer chain but omitted its raw macro edge.
                d = sim(nodes[src], nodes[dst], sigma_q, sigma_qdot, sigma_p,
                        w_q, w_qdot, w_p)
                macro_cost = deploy_edge_weight(
                    EDGE_CROSS, d, int(graph.skill_ids[src]),
                    int(graph.skill_ids[dst]), lambda_sw)
                graph.macro_edge_costs[(src, dst)] = macro_cost
                graph.expanded_macro_edges += 1
            chain = (src,) + buffers + (dst,)
            hop_cost = macro_cost / (len(chain) - 1)
            for u, v in zip(chain[:-1], chain[1:]):
                add_runtime_edge(u, v, hop_cost, EDGE_BUFFER)
        return graph

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def sim_to(self, x: NodeState, node_id: int) -> float:
        """Normalized distance between live state x and a graph node."""
        return sim(x, self.nodes[node_id], self.sigma_q, self.sigma_qdot,
                   self.sigma_p, self.w_q, self.w_qdot, self.w_p)

    def skill_index(self, skill: Union[str, int]) -> int:
        if isinstance(skill, str):
            return self.skill_names.index(skill)
        sid = int(skill)
        if not 0 <= sid < len(self.skill_names):
            raise ValueError(f"Invalid skill id: {sid}")
        return sid

    def skill_range(self, skill_id: int) -> Tuple[int, int]:
        """[start, end) global ids of a skill's original (non-buffer) frames.

        Original frames are added before all buffer nodes, so each skill
        occupies one contiguous id range.
        """
        start = sum(self.skill_lengths[:skill_id])
        return start, start + self.skill_lengths[skill_id]

    def skills_with_role(self, role: str) -> List[int]:
        """Return skill ids carrying ``role`` in dataset order."""
        if role not in {"task", "recovery"}:
            raise ValueError(f"Invalid skill role: {role}")
        return [sid for sid, value in enumerate(self.skill_roles)
                if value == role]

    def target_set(self, skill_id: int, tau: float) -> List[int]:
        """Command targets: opening frames plus trained Buffer landings.

        A Buffer macro may deliberately land after the opening ``tau``
        fraction of a skill.  Its destination is still a valid trained entry
        and must therefore be a graph-search goal; otherwise the expanded
        Buffer chain can appear unreachable and Dijkstra may detour through a
        third skill merely to reach the opening window.
        """
        start, end = self.skill_range(skill_id)
        n = max(1, int(round((end - start) * tau)))
        targets = set(range(start, start + n))
        targets.update(
            dst for (_, dst) in self.buffer_chains
            if int(self.skill_ids[dst]) == skill_id
        )
        return sorted(targets)

    def temporal_successor(self, node_id: int) -> Optional[int]:
        """The unique same-skill forward neighbor, or None at a skill end."""
        for v, _, et in self.adj[node_id]:
            if et == EDGE_SAME_SKILL:
                return v
        return None

    def add_runtime_buffer_node(self, src: int, dst: int, kappa: int,
                                state: NodeState) -> int:
        """Append a runtime buffer node (NN planner, spec 2.2): an on-the-fly
        interpolated node between src and dst, not present at graph build
        time. Marked is_buffer with buffer_src/dst so ReferenceBuilder chain
        detection and kappa passthrough work exactly like build-time buffers.
        NOT wired into the adjacency lists (NN paths reference it directly).
        """
        gid = len(self.nodes)
        self.nodes.append(state)
        self.skill_ids = np.append(self.skill_ids, -1)
        self.frame_idxs = np.append(self.frame_idxs, -1)
        self.is_buffer = np.append(self.is_buffer, True)
        self.kappas = np.append(self.kappas, kappa)
        self.buffer_src = np.append(self.buffer_src, src)
        self.buffer_dst = np.append(self.buffer_dst, dst)
        self.adj.append([])
        self.rev_adj.append([])
        return gid
