#!/usr/bin/env python3
"""Offline unit tests for the skill scheduler (graph-search planner).

Run directly:  python humanoidverse/deploy/skill_scheduler/tests/test_graph_search.py
No pytest required; torch NOT required (skill_scheduler is numpy-only).
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from humanoidverse.deploy.skill_scheduler.distance import NodeState
from humanoidverse.deploy.skill_scheduler.graph_data import (
    SkillGraphData,
    deploy_edge_weight,
    EDGE_BUFFER,
    EDGE_CROSS,
    EDGE_SAME_SKILL,
)
from humanoidverse.deploy.skill_scheduler.planners import GraphSearchPlanner
from humanoidverse.deploy.skill_scheduler.scheduler import SkillGraphScheduler

REAL_GRAPH = _repo_root / "SG_build" / "sg_output_V2" / "skill_graph.json"
REAL_CONFIG = _repo_root / "SG_build" / "sg_output_V2" / "scheduler_config.yaml"

LAMBDA_SW = 4.0


def zeros_node(kappa=0, skill_id=0, frame_idx=0, is_buffer=False):
    return {
        "node_id": -1,  # filled by caller
        "skill_id": skill_id,
        "frame_idx": frame_idx,
        "is_buffer": is_buffer,
        "kappa": kappa,
        "q": [0.0] * 23,
        "q_dot": [0.0] * 23,
        "p_hat": [0.0] * 3,
    }


def make_toy_graph() -> dict:
    """Skill A: nodes 0-4; skill B: nodes 5-9; buffer node 10 on 3->...->8.

    All features are zero, so every d_uv = 0 and cross/buffer edge weights
    reduce to the lambda_sw penalty (skills differ everywhere: buffer nodes
    have skill_id=-1).
    """
    nodes = []
    for i in range(5):
        nodes.append(zeros_node(skill_id=0, frame_idx=i))
    for i in range(5):
        nodes.append(zeros_node(skill_id=1, frame_idx=i))
    nodes.append(zeros_node(kappa=1, skill_id=-1, frame_idx=-1, is_buffer=True))
    for gid, n in enumerate(nodes):
        n["node_id"] = gid

    edges = []
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 4),
                 (5, 6), (6, 7), (7, 8), (8, 9)]:
        edges.append({"src": u, "dst": v, "weight": 1.0,
                      "is_cross_skill": False, "is_buffer": False})
    edges.append({"src": 2, "dst": 7, "weight": 0.0,
                  "is_cross_skill": True, "is_buffer": False})
    edges.append({"src": 3, "dst": 10, "weight": 0.0,
                  "is_cross_skill": False, "is_buffer": True})
    edges.append({"src": 10, "dst": 8, "weight": 0.0,
                  "is_cross_skill": False, "is_buffer": True})
    return {
        "num_nodes": 11, "num_original_nodes": 10, "num_buffer_nodes": 1,
        "num_edges": len(edges), "num_buffer_edges": 2,
        "skill_names": ["skill_A", "skill_B"], "skill_lengths": [5, 5],
        "nodes": nodes, "edges": edges,
        "buffer_nodes": [{"global_id": 10, "src_node": 3, "dst_node": 8}],
    }


def load_toy() -> SkillGraphData:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(make_toy_graph(), f)
        path = f.name
    return SkillGraphData.from_json(path, sigma_q=1.0, sigma_qdot=1.0,
                                    sigma_p=1.0, lambda_sw=LAMBDA_SW)


def test_deploy_edge_weight():
    assert deploy_edge_weight(EDGE_SAME_SKILL, 99.0, 0, 0, 5.0) == 1.0
    assert deploy_edge_weight(EDGE_CROSS, 2.0, 0, 1, 5.0) == 7.0
    assert deploy_edge_weight(EDGE_CROSS, 2.0, 0, 0, 5.0) == 2.0
    assert deploy_edge_weight(EDGE_BUFFER, 0.5, 0, -1, 5.0) == 5.5
    print("PASS test_deploy_edge_weight")


def test_value_function_hand_computed():
    g = load_toy()
    planner = GraphSearchPlanner(g)
    V, next_hop = planner.build_value_function([9])
    # Hand-computed with lambda_sw=4, all d_uv=0. Note: node 4 is skill A's
    # LAST frame -- it has no outgoing edges and cannot reach T at all.
    # V[8]=1, V[7]=2, V[6]=3, V[5]=4, V[10]=4+1=5,
    # V[3]=4+5=9 via the buffer chain (temporal route dead-ends at node 4),
    # V[2]=min(1+9, 4+2)=6 via cross edge to node 7,
    # V[1]=7, V[0]=8.
    expected_V = {9: 0, 8: 1, 7: 2, 6: 3, 5: 4, 10: 5, 3: 9, 2: 6,
                  1: 7, 0: 8}
    assert V == expected_V, f"V mismatch: {V}"
    assert 4 not in V            # dead end: skill A's last frame
    assert next_hop[2] == 7      # cross-skill edge wins for node 2
    assert next_hop[3] == 10     # node 3's only route is the buffer chain
    path = planner.reconstruct_path_gs(0, next_hop, {9})
    assert path == [0, 1, 2, 7, 8, 9], f"path: {path}"

    # Cache: same T returns identical objects without recomputation.
    V2, nh2 = planner.build_value_function([9])
    assert V2 is V and nh2 is next_hop

    # Unreachable: T={0} has no incoming edges; node 5 cannot reach it.
    V0, nh0 = planner.build_value_function([0])
    assert V0 == {0: 0.0}
    assert planner.reconstruct_path_gs(5, nh0, {0}) is None
    assert np.isinf(planner.score(g.nodes[5], 5, V0, lambda_cost=1.0))
    print("PASS test_value_function_hand_computed")


def test_entry_check_branches():
    g = load_toy()
    sched = SkillGraphScheduler(g, planner_type="graph_search",
                                A=0.5, B=2.0, lambda_cost=1.0,
                                tau=0.2, top_k=3)
    sched.set_command("skill_B")
    assert sched.T_cmd == [5]  # first tau=0.2 fraction of skill B = 1 frame

    # candidate_pool="all" (default): best node can be ANY graph node
    x = NodeState(q=np.zeros(23), q_dot=np.zeros(23), p_hat=np.zeros(3))
    status, cand = sched.entry_check(x, sched.T_cmd)
    assert status == "attach" and cand == [0]  # node 0, not a T_cmd member

    x_mid = NodeState(q=np.full(23, 0.05), q_dot=np.zeros(23),
                      p_hat=np.zeros(3))  # sim = 23*0.05 = 1.15 in (A, B)
    status, cand = sched.entry_check(x_mid, sched.T_cmd)
    assert status == "search" and cand == [0, 1, 2]

    x_far = NodeState(q=np.full(23, 10.0), q_dot=np.zeros(23),
                      p_hat=np.zeros(3))
    status, cand = sched.entry_check(x_far, sched.T_cmd)
    assert status == "estop" and cand is None
    print("PASS test_entry_check_branches")


def test_scheduler_step_flow():
    g = load_toy()
    # tau=1.0 so T spans all of skill B (reachable from skill A via the
    # cross edge), letting the attach path traverse the switching edge.
    # enable_safety_replan=True to exercise the safety_event -> estop path
    # (it is OFF by default).
    sched = SkillGraphScheduler(g, planner_type="graph_search",
                                A=0.5, B=2.0, lambda_cost=1.0,
                                tau=1.0, top_k=3, enable_safety_replan=True)
    x = NodeState(q=np.zeros(23), q_dot=np.zeros(23), p_hat=np.zeros(3))

    # init: attaches at node 0 (nearest overall) and walks THROUGH the
    # cross-skill edge 2->7 into skill B
    gd = sched.step(x, user_cmd="skill_B", t=0.0)
    assert gd is not None and gd.node_id == 0 and gd.kappa == 0
    assert sched.current_path == [0, 1, 2, 7, 8, 9], sched.current_path
    # no trigger: same guidance returned
    assert sched.step(x, user_cmd="skill_B", t=0.02).node_id == 0
    # cmd_change: re-plans onto skill_A (already inside T -> temporal walk)
    gd = sched.step(x, user_cmd="skill_A", t=0.04)
    assert gd.node_id == 0
    assert sched.current_path == [0, 1, 2, 3, 4], sched.current_path
    # safety_event: huge deviation latches e-stop, guidance stops
    x_far = NodeState(q=np.full(23, 10.0), q_dot=np.zeros(23),
                      p_hat=np.zeros(3))
    assert sched.step(x_far, user_cmd="skill_A", t=0.06) is None
    assert sched.estopped
    assert sched.step(x, user_cmd="skill_A", t=0.08) is None  # stays latched
    print("PASS test_scheduler_step_flow")


def test_safety_replan_disabled_by_default():
    """With the default enable_safety_replan=False, a huge tracking
    deviation must NOT trigger any replan: the reference stays put (no
    time-jumping the reference mid-tracking)."""
    g = load_toy()
    sched = SkillGraphScheduler(g, planner_type="graph_search",
                                A=0.5, B=2.0, lambda_cost=1.0,
                                tau=1.0, top_k=3)
    x = NodeState(q=np.zeros(23), q_dot=np.zeros(23), p_hat=np.zeros(3))
    gd = sched.step(x, user_cmd="skill_B", t=0.0)
    assert gd is not None
    version = sched.path_version

    x_far = NodeState(q=np.full(23, 10.0), q_dot=np.zeros(23),
                      p_hat=np.zeros(3))
    for i in range(5):
        gd = sched.step(x_far, user_cmd="skill_B", t=1.0 + i * 0.02)
        assert gd is not None, "guidance must continue despite deviation"
        assert not sched.estopped
    assert sched.path_version == version, "no replan should have happened"
    print("PASS test_safety_replan_disabled_by_default")


def _load_real_graph() -> SkillGraphData:
    import yaml
    with open(REAL_CONFIG) as f:
        cfg = yaml.safe_load(f)
    return SkillGraphData.from_json(
        str(REAL_GRAPH), sigma_q=cfg["sigma_q"], sigma_qdot=cfg["sigma_qdot"],
        sigma_p=cfg["sigma_p"], lambda_sw=5.0)


def test_real_graph_paths():
    if not REAL_GRAPH.exists() or not REAL_CONFIG.exists():
        print("SKIP test_real_graph_paths (sg_output_V2 artifacts missing)")
        return
    g = _load_real_graph()
    planner = GraphSearchPlanner(g)
    T = g.target_set(1, tau=0.2)
    V, next_hop = planner.build_value_function(T)
    T_set = set(T)
    edge_set = {(u, v) for u, outs in enumerate(g.adj) for v, _, _ in outs}

    reachable = [v for v in V if v not in T_set]
    print(f"  reachable nodes outside T: {len(reachable)}/{g.num_nodes}")
    rng = np.random.default_rng(0)
    sample = rng.choice(reachable, size=min(50, len(reachable)),
                        replace=False)
    for entry in sample:
        path = planner.reconstruct_path_gs(int(entry), next_hop, T_set)
        assert path is not None
        for u, v in zip(path[:-1], path[1:]):
            assert (u, v) in edge_set, f"missing edge {u}->{v}"
        assert path[-1] in T_set
    print("PASS test_real_graph_paths")


def test_kappa_passthrough():
    if not REAL_GRAPH.exists() or not REAL_CONFIG.exists():
        print("SKIP test_kappa_passthrough (sg_output_V2 artifacts missing)")
        return
    g = _load_real_graph()
    sched = SkillGraphScheduler(g, planner_type="graph_search",
                                A=3.6, B=3.6, lambda_cost=1.0,
                                tau=0.2, top_k=5)
    # Walk one buffer chain: src -> buf_1 ... buf_k -> dst via buffer edges.
    buf_gid = int(np.nonzero(g.is_buffer)[0][0])
    # find chain start: the non-buffer predecessor
    src = next(u for u, w, et in g.rev_adj[buf_gid]
               if et == EDGE_BUFFER and not g.is_buffer[u])
    path = [src]
    while True:
        nxt = None
        for v, _, et in g.adj[path[-1]]:
            if et == EDGE_BUFFER:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        if not g.is_buffer[nxt]:
            break
    assert any(g.is_buffer[n] for n in path), f"no buffer node in {path}"

    guidance = sched.path_to_guidance(path)
    for gd in guidance:
        if g.is_buffer[gd.node_id]:
            assert gd.kappa == int(g.kappas[gd.node_id]) > 0
        else:
            assert gd.kappa == 0
    # kappa strictly decreases along the buffer segment
    buf_kappas = [gd.kappa for gd in guidance if gd.kappa > 0]
    assert buf_kappas == sorted(buf_kappas, reverse=True)
    print(f"PASS test_kappa_passthrough (chain: {path}, "
          f"kappas: {buf_kappas})")


def test_extend_to_skill_end():
    if not REAL_GRAPH.exists() or not REAL_CONFIG.exists():
        print("SKIP test_extend_to_skill_end (artifacts missing)")
        return
    g = _load_real_graph()
    sched = SkillGraphScheduler(g, planner_type="graph_search",
                                A=3.6, B=3.6, lambda_cost=1.0,
                                tau=0.2, top_k=5)
    start, end = g.skill_range(0)
    path = sched._extend_to_skill_end([start + 10])
    assert path[-1] == end - 1  # last frame of skill 0
    assert path == list(range(start + 10, end))
    print("PASS test_extend_to_skill_end")


def test_attach_path_crosses_skills():
    """With candidate_pool='all', attaching from mid skill 0 must produce a
    path that starts at the robot's current frame and traverses cross-skill
    (or buffer) edges into T_cmd -- not a jump to the skill start."""
    if not REAL_GRAPH.exists() or not REAL_CONFIG.exists():
        print("SKIP test_attach_path_crosses_skills (artifacts missing)")
        return
    g = _load_real_graph()
    sched = SkillGraphScheduler(g, planner_type="graph_search",
                                A=1.889, B=5.0, lambda_cost=1.0,
                                tau=0.2, top_k=5)
    entry_gid = 100  # skill 0, frame 100
    x = g.nodes[entry_gid]
    sched.set_command(1)
    gd = sched.step(x, user_cmd=None, t=0.0)  # init trigger
    assert gd is not None, "scheduler e-stopped on an exact graph state"
    path = sched.current_path
    T_set = set(sched.T_cmd)
    assert path[0] == entry_gid, f"path should start at the entry, got {path[0]}"
    assert path[-1] in T_set or g.skill_ids[path[-1]] == 1
    assert any(nid >= 210 or g.is_buffer[nid] for nid in path), \
        "path never leaves skill 0"
    # consecutive nodes must be graph edges
    edge_set = {(u, v) for u, outs in enumerate(g.adj) for v, _, _ in outs}
    for u, v in zip(path[:-1], path[1:]):
        assert (u, v) in edge_set, f"missing edge {u}->{v}"
    n_cross = sum(1 for u, v in zip(path[:-1], path[1:])
                  if g.skill_ids[u] != g.skill_ids[v])
    print(f"PASS test_attach_path_crosses_skills "
          f"(path len {len(path)}, cross-skill hops {n_cross}, "
          f"buffers {int(g.is_buffer[path].sum())})")


def test_cmd_change_estop_fallback():
    """A commanded switch must never deadlock on entry estop: cmd_change
    downgrades estop -> search and still installs a path."""
    g = load_toy()
    sched = SkillGraphScheduler(g, planner_type="graph_search",
                                A=0.5, B=2.0, lambda_cost=1.0,
                                tau=1.0, top_k=3)
    x = NodeState(q=np.zeros(23), q_dot=np.zeros(23), p_hat=np.zeros(3))
    assert sched.step(x, user_cmd="skill_B", t=0.0) is not None

    # Robot far from every graph node, then a command arrives: estop at
    # entry check must be downgraded, not latched.
    x_far = NodeState(q=np.full(23, 10.0), q_dot=np.zeros(23),
                      p_hat=np.zeros(3))
    gd = sched.step(x_far, user_cmd="skill_A", t=1.0)
    assert gd is not None, "cmd_change estop should fall back to search"
    assert not sched.estopped
    assert sched.current_path  # some path installed
    print("PASS test_cmd_change_estop_fallback")


def main():
    test_deploy_edge_weight()
    test_value_function_hand_computed()
    test_entry_check_branches()
    test_scheduler_step_flow()
    test_safety_replan_disabled_by_default()
    test_cmd_change_estop_fallback()
    test_real_graph_paths()
    test_kappa_passthrough()
    test_extend_to_skill_end()
    test_attach_path_crosses_skills()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
