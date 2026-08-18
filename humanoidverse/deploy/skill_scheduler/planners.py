"""Path planners for the online skill scheduler (spec 2).

Only the Graph-Search planner for now; the NN planner is added in step 3 and
must expose the same plan() interface for A/B switching via planner_type.
"""

import heapq
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .distance import NodeState
from .graph_data import SkillGraphData


class NNPlanner:
    """Nearest-neighbor planner (spec 2.2).

    No global search: the entry is the candidate nearest to the live state;
    the path hops directly (or via on-the-fly runtime buffer frames) into the
    nearest T_cmd node. Buffer frames are appended to the graph as runtime
    buffer nodes so kappa passthrough and reference assembly are reused.
    """

    def __init__(self, graph: SkillGraphData, buffer_base: float = 1.0,
                 max_buffer: int = 30):
        self.graph = graph
        self.buffer_base = buffer_base
        self.max_buffer = max_buffer
        # Cache runtime buffer chains per (entry, target) pair
        self._buf_cache: Dict[Tuple[int, int], List[int]] = {}

    def _runtime_buffers(self, src: int, dst: int, n_buf: int) -> List[int]:
        key = (src, dst)
        if key in self._buf_cache:
            return self._buf_cache[key]
        g = self.graph
        s, t = g.nodes[src], g.nodes[dst]
        ids = []
        for k in range(1, n_buf + 1):
            alpha = k / (n_buf + 1)
            state = NodeState(
                q=(1 - alpha) * s.q + alpha * t.q,
                q_dot=(1 - alpha) * s.q_dot + alpha * t.q_dot,
                p_hat=(1 - alpha) * s.p_hat + alpha * t.p_hat,
            )
            ids.append(g.add_runtime_buffer_node(src, dst,
                                                 kappa=n_buf - k + 1,
                                                 state=state))
        self._buf_cache[key] = ids
        return ids

    def short_hop_path(self, entry: int, T_cmd: List[int],
                       B: float) -> Optional[List[int]]:
        """entry -> (runtime buffers) -> nearest T node. Returns None when
        the direct jump exceeds the safe distance B."""
        g = self.graph
        T_set = set(T_cmd)
        if entry in T_set:
            return [entry]
        t_star = min(T_cmd, key=lambda t: g.sim_to(g.nodes[entry], t))
        d = g.sim_to(g.nodes[entry], t_star)
        if d > B:
            return None
        n_buf = min(max(0, int(d / self.buffer_base)), self.max_buffer)
        bufs = self._runtime_buffers(entry, t_star, n_buf) if n_buf else []
        return [entry] + bufs + [t_star]

    def plan_nn(self, x: NodeState, candidates: List[int],
                T_cmd: List[int], B: float) -> Optional[List[int]]:
        entry = min(candidates, key=lambda v: self.graph.sim_to(x, v))
        return self.short_hop_path(entry, T_cmd, B)

    def two_stage_nn(self, x: NodeState, T_rec: List[int],
                     T_cmd: List[int], B: float) -> Optional[List[int]]:
        """Unsafe direct jump (spec Problem B): hop to the recovery target
        first, then from there to the commanded target."""
        rec_entry = min(T_rec, key=lambda v: self.graph.sim_to(x, v))
        path_to_rec = self.short_hop_path(rec_entry, T_rec, B)
        if path_to_rec is None:
            return None
        rec_state = self.graph.nodes[path_to_rec[-1]]
        cmd_entry = min(range(self.graph.num_nodes),
                        key=lambda v: self.graph.sim_to(rec_state, v))
        path_to_cmd = self.short_hop_path(cmd_entry, T_cmd, B)
        if path_to_cmd is None:
            return None
        return path_to_rec + path_to_cmd


class GraphSearchPlanner:
    """Reverse multi-source Dijkstra over the skill graph (spec 2.1).

    V[v]       = min cumulative deploy cost from v to the target set T_cmd
    next_hop   = optimal next node on the way to T_cmd

    Both are cached per target set: recomputation is only needed when the
    command (and hence T_cmd) changes; replanning under the same command is
    just a walk along next_hop.
    """

    def __init__(self, graph: SkillGraphData):
        self.graph = graph
        self._cache_key: Optional[FrozenSet[int]] = None
        self._V: Dict[int, float] = {}
        self._next_hop: Dict[int, int] = {}

    def build_value_function(
        self, T_cmd: List[int]
    ) -> Tuple[Dict[int, float], Dict[int, int]]:
        key = frozenset(T_cmd)
        if key == self._cache_key:
            return self._V, self._next_hop
        g = self.graph
        V: Dict[int, float] = {t: 0.0 for t in T_cmd}
        next_hop: Dict[int, int] = {}
        pq = [(0.0, t) for t in T_cmd]
        heapq.heapify(pq)
        while pq:
            d, v = heapq.heappop(pq)
            if d > V.get(v, np.inf):
                continue
            for u, w, _ in g.rev_adj[v]:
                nd = d + w
                if nd < V.get(u, np.inf):
                    V[u] = nd
                    next_hop[u] = v
                    heapq.heappush(pq, (nd, u))
        self._cache_key, self._V, self._next_hop = key, V, next_hop
        return V, next_hop

    def reconstruct_path_gs(self, entry: int, next_hop: Dict[int, int],
                            T_set) -> Optional[List[int]]:
        """Walk next_hop from entry until reaching T (spec 2.1).

        Returns None when entry cannot reach T (forward-only temporal edges
        mean nodes past the target region of a skill have no path back).
        """
        path = [entry]
        while path[-1] not in T_set:
            nxt = next_hop.get(path[-1])
            if nxt is None or len(path) > self.graph.num_nodes:  # cycle guard
                return None
            path.append(nxt)
        return path

    def score(self, x: NodeState, v: int, V: Dict[int, float],
              lambda_cost: float) -> float:
        """Candidate score (spec 3): lambda_cost * sim(x, v) + V[v].

        Unreachable candidates (v not in V) score +inf and sink to the bottom
        of any sorted candidate list.
        """
        return lambda_cost * self.graph.sim_to(x, v) + V.get(v, np.inf)
