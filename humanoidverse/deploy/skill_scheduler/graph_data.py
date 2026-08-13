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
        graph = cls(
            nodes=nodes,
            skill_ids=np.asarray([n["skill_id"] for n in g["nodes"]], dtype=np.int64),
            frame_idxs=np.asarray([n["frame_idx"] for n in g["nodes"]], dtype=np.int64),
            is_buffer=np.asarray([n["is_buffer"] for n in g["nodes"]], dtype=bool),
            kappas=np.asarray([n["kappa"] for n in g["nodes"]], dtype=np.int64),
            buffer_src=buffer_src,
            buffer_dst=buffer_dst,
            skill_names=list(g["skill_names"]),
            skill_lengths=[int(x) for x in g["skill_lengths"]],
            sigma_q=sigma_q, sigma_qdot=sigma_qdot, sigma_p=sigma_p,
            w_q=w_q, w_qdot=w_qdot, w_p=w_p, lambda_sw=lambda_sw,
        )
        n = len(nodes)
        graph.adj = [[] for _ in range(n)]
        graph.rev_adj = [[] for _ in range(n)]
        for e in g["edges"]:
            u, v = int(e["src"]), int(e["dst"])
            et = edge_class(e)
            d = sim(nodes[u], nodes[v], sigma_q, sigma_qdot, sigma_p,
                    w_q, w_qdot, w_p)
            w = deploy_edge_weight(et, d, int(graph.skill_ids[u]),
                                   int(graph.skill_ids[v]), lambda_sw)
            graph.adj[u].append((v, w, et))
            graph.rev_adj[v].append((u, w, et))
            graph.d_uv[(u, v)] = d
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

    def target_set(self, skill_id: int, tau: float) -> List[int]:
        """T_cmd: the first tau fraction of the skill's frames (spec 3)."""
        start, end = self.skill_range(skill_id)
        n = max(1, int(round((end - start) * tau)))
        return list(range(start, start + n))

    def temporal_successor(self, node_id: int) -> Optional[int]:
        """The unique same-skill forward neighbor, or None at a skill end."""
        for v, _, et in self.adj[node_id]:
            if et == EDGE_SAME_SKILL:
                return v
        return None
