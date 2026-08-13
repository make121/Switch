"""Path planners for the online skill scheduler (spec 2).

Only the Graph-Search planner for now; the NN planner is added in step 3 and
must expose the same plan() interface for A/B switching via planner_type.
"""

import heapq
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .distance import NodeState
from .graph_data import SkillGraphData


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
