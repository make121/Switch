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
SWITCH_GRAPH = _repo_root / "Switch_data" / "skill_graph" / "skill_graph.json"
SWITCH_CONFIG = _repo_root / "Switch_data" / "skill_graph" / "scheduler_config.yaml"

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
    # Raw direct macro for the Buffer chain below. It remains in the JSON for
    # distance/cost calculation but is not exposed in runtime adjacency.
    edges.append({"src": 3, "dst": 8, "weight": 0.0,
                  "is_cross_skill": True, "is_buffer": False})
    # Reverse direction has no Buffer coverage and therefore keeps its direct
    # edge as the connectivity fallback.
    edges.append({"src": 7, "dst": 2, "weight": 0.0,
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


def test_skill_roles_load_and_backward_compatibility():
    # Existing graphs have no skill_roles field and must retain their old
    # behaviour: every skill is treated as a normal task skill.
    old_graph = load_toy()
    assert old_graph.skill_roles == ["task", "task"]
    assert old_graph.skills_with_role("task") == [0, 1]
    assert old_graph.skills_with_role("recovery") == []

    recovery_json = make_toy_graph()
    recovery_json["skill_roles"] = ["task", "recovery"]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(recovery_json, f)
        path = f.name
    recovery_graph = SkillGraphData.from_json(
        path, sigma_q=1.0, sigma_qdot=1.0, sigma_p=1.0,
        lambda_sw=LAMBDA_SW)
    assert recovery_graph.skill_roles == ["task", "recovery"]
    assert recovery_graph.skills_with_role("task") == [0]
    assert recovery_graph.skills_with_role("recovery") == [1]
    print("PASS test_skill_roles_load_and_backward_compatibility")


def test_value_function_hand_computed():
    g = load_toy()
    planner = GraphSearchPlanner(g)
    V, next_hop = planner.build_value_function([9])
    # Hand-computed with lambda_sw=4, all d_uv=0. Note: node 4 is skill A's
    # LAST frame -- it has no outgoing edges and cannot reach T at all.
    # Macro 3->8 costs lambda_sw=4 and expands into two Buffer hops costing
    # 2 each. V[8]=1, V[10]=2+1=3, V[3]=2+3=5, then temporal costs.
    expected_V = {9: 0, 8: 1, 7: 2, 6: 3, 5: 4, 10: 3, 3: 5,
                  2: 6, 1: 7, 0: 8}
    assert V == expected_V, f"V mismatch: {V}"
    assert 4 not in V            # dead end: skill A's last frame
    assert next_hop[2] == 3      # advance temporally to the Buffer entry
    assert next_hop[3] == 10     # node 3's only route is the buffer chain
    path = planner.reconstruct_path_gs(0, next_hop, {9})
    assert path == [0, 1, 2, 3, 10, 8, 9], f"path: {path}"

    # Cache: same T returns identical objects without recomputation.
    V2, nh2 = planner.build_value_function([9])
    assert V2 is V and nh2 is next_hop

    # Unreachable: T={0} has no incoming edges; node 5 cannot reach it.
    V0, nh0 = planner.build_value_function([0])
    assert V0 == {0: 0.0}
    assert planner.reconstruct_path_gs(5, nh0, {0}) is None
    assert np.isinf(planner.score(g.nodes[5], 5, V0, lambda_cost=1.0))
    print("PASS test_value_function_hand_computed")


def test_buffer_macro_expansion_preserves_cost():
    g = load_toy()
    assert g.buffer_chains == {(3, 8): (10,)}
    assert g.expanded_macro_edges == 1
    assert g.macro_edge_costs[(3, 8)] == LAMBDA_SW
    cross_edges = [(u, v) for u, outs in enumerate(g.adj)
                   for v, _, et in outs if et == EDGE_CROSS]
    assert cross_edges == [(7, 2)]

    chain_weights = []
    for u, v in [(3, 10), (10, 8)]:
        weights = [w for nxt, w, et in g.adj[u]
                   if nxt == v and et == EDGE_BUFFER]
        assert len(weights) == 1
        chain_weights.append(weights[0])
    assert chain_weights == [LAMBDA_SW / 2, LAMBDA_SW / 2]
    assert sum(chain_weights) == g.macro_edge_costs[(3, 8)]

    # Without Buffer metadata, the raw macro remains a searchable direct edge.
    raw = make_toy_graph()
    raw["buffer_nodes"] = []
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(raw, f)
        path = f.name
    fallback = SkillGraphData.from_json(
        path, sigma_q=1.0, sigma_qdot=1.0, sigma_p=1.0,
        lambda_sw=LAMBDA_SW)
    assert fallback.buffer_chains == {}
    assert fallback.expanded_macro_edges == 0
    assert any(v == 8 and et == EDGE_CROSS for v, _, et in fallback.adj[3])
    print("PASS test_buffer_macro_expansion_preserves_cost")


def test_entry_check_branches():
    g = load_toy()
    sched = SkillGraphScheduler(g, planner_type="graph_search",
                                A=0.5, B=2.0, lambda_cost=1.0,
                                tau=0.2, top_k=3)
    sched.set_command("skill_B")
    # Opening frame plus the trained 3->8 Buffer landing.
    assert sched.T_cmd == [5, 8]

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
    # Buffer chain), letting the attach path traverse the switching edge.
    # enable_safety_replan=True to exercise the safety_event -> estop path
    # (it is OFF by default).
    sched = SkillGraphScheduler(g, planner_type="graph_search",
                                A=0.5, B=2.0, lambda_cost=1.0,
                                tau=1.0, top_k=3, enable_safety_replan=True)
    x = NodeState(q=np.zeros(23), q_dot=np.zeros(23), p_hat=np.zeros(3))

    # init: attaches at node 0 (nearest overall) and walks THROUGH the
    # precomputed Buffer chain into skill B
    gd = sched.step(x, user_cmd="skill_B", t=0.0)
    assert gd is not None and gd.node_id == 0 and gd.kappa == 0
    assert sched.current_path == [0, 1, 2, 3, 10, 8, 9], sched.current_path
    assert any(g.is_buffer[n] for n in sched.current_path)
    # no trigger: same guidance returned
    assert sched.step(x, user_cmd="skill_B", t=0.02).node_id == 0
    # A command change is queued instead of attaching to an arbitrary target
    # frame.  The current reference keeps running because no forward Buffer
    # macro from skill B to skill A exists in this toy graph.
    version = sched.path_version
    gd = sched.step(x, user_cmd="skill_A", t=0.04)
    assert gd.node_id == 0
    assert sched.pending_cmd == 0
    assert sched.path_version == version
    # Re-issuing the active command cancels the pending request, after which
    # the independent safety-event behaviour can be tested.
    assert sched.step(x, user_cmd="skill_B", t=0.05).node_id == 0
    assert sched.pending_cmd is None
    # safety_event: huge deviation latches e-stop, guidance stops
    x_far = NodeState(q=np.full(23, 10.0), q_dot=np.zeros(23),
                      p_hat=np.zeros(3))
    assert sched.step(x_far, user_cmd="skill_B", t=0.06) is None
    assert sched.estopped
    assert sched.step(x, user_cmd="skill_B", t=0.08) is None  # stays latched
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


def _load_switch_graph() -> SkillGraphData:
    import yaml
    with open(SWITCH_CONFIG) as f:
        cfg = yaml.safe_load(f)
    return SkillGraphData.from_json(
        str(SWITCH_GRAPH), sigma_q=cfg["sigma_q"],
        sigma_qdot=cfg["sigma_qdot"], sigma_p=cfg["sigma_p"],
        lambda_sw=cfg["lambda_sw"][1], w_q=cfg.get("w_q", 1.0),
        w_qdot=cfg.get("w_qdot", 1.0), w_p=cfg.get("w_p", 1.0))


def test_switch_graph_mid_skill_buffer_is_direct_target():
    """Regression: skill0/frame105 -> skill1/frame72 must not detour via 2."""
    if not SWITCH_GRAPH.exists() or not SWITCH_CONFIG.exists():
        print("SKIP test_switch_graph_mid_skill_buffer_is_direct_target "
              "(Switch_data artifacts missing)")
        return
    g = _load_switch_graph()
    source = next(
        nid for nid in range(g.num_nodes)
        if int(g.skill_ids[nid]) == 0 and int(g.frame_idxs[nid]) == 105)
    targets = g.target_set(1, tau=0.2)
    landing = next(
        dst for src, dst in g.buffer_chains
        if src == source and int(g.skill_ids[dst]) == 1)
    assert landing in targets

    sched = SkillGraphScheduler(
        g, planner_type="graph_search", A=6.6014, B=14.4499,
        lambda_cost=1.0, tau=0.2, top_k=5, switch_entry_A=11.0)
    path = sched._buffer_path_from_source(source, targets)
    assert path is not None
    non_buffer_skills = {
        int(g.skill_ids[nid]) for nid in path if not g.is_buffer[nid]
    }
    assert non_buffer_skills == {0, 1}, non_buffer_skills
    assert path[0] == source and landing in path
    print("PASS test_switch_graph_mid_skill_buffer_is_direct_target "
          f"(source {source}, landing {landing}, path len {len(path)})")


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
    entry_gid = 50  # reachable skill-0 frame before a Buffer-chain source
    x = g.nodes[entry_gid]
    gd = sched.step(x, user_cmd=1, t=0.0)  # init trigger
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
    """A far command entry must wait; it must never jump or reset."""
    g = load_toy()
    sched = SkillGraphScheduler(g, planner_type="graph_search",
                                A=0.5, B=2.0, lambda_cost=1.0,
                                tau=1.0, top_k=3)
    x = NodeState(q=np.zeros(23), q_dot=np.zeros(23), p_hat=np.zeros(3))
    assert sched.step(x, user_cmd="skill_B", t=0.0) is not None

    # Put the active reference on skill B and request skill A.  There is no
    # buffered B->A macro, so the command remains pending regardless of the
    # live-state distance.
    x_far = NodeState(q=np.full(23, 10.0), q_dot=np.zeros(23),
                      p_hat=np.zeros(3))
    version = sched.path_version
    gd = sched.step(x_far, user_cmd="skill_A", t=1.0)
    assert gd is not None
    assert not sched.estopped
    assert sched.pending_cmd == 0
    assert sched.path_version == version
    print("PASS test_cmd_change_waits_without_buffer")


def test_cmd_change_waits_for_forward_buffer_source():
    g = load_toy()
    sched = SkillGraphScheduler(g, planner_type="graph_search",
                                A=0.5, B=2.0, lambda_cost=1.0,
                                tau=1.0, top_k=3)
    sched.set_command("skill_A")
    sched.install_reference_path(sched.pure_skill_path(0), t=0.0)
    version = sched.path_version
    x = NodeState(q=np.zeros(23), q_dot=np.zeros(23), p_hat=np.zeros(3))

    # Command at frame 1: keep frames 1->2->3; node 3 is the earliest future
    # source of the buffered 3->8 macro.
    sched.pointer = 1
    assert sched.step(x, user_cmd="skill_B", t=0.02).node_id == 1
    assert sched.pending_cmd == 1 and sched.pending_source == 3
    assert sched.path_version == version
    sched.pointer = 2
    # A one-shot keyboard/automatic command remains pending; callers do not
    # need to resend it on every control step.
    assert sched.step(x, user_cmd=None, t=0.04).node_id == 2
    assert sched.path_version == version

    # Only at the exact source, with entry error <= A, install the Buffer
    # chain.  The path can neither start in Buffer node 10 nor target frame 8.
    sched.pointer = 3
    gd = sched.step(x, user_cmd=None, t=0.06)
    assert gd.node_id == 3
    assert sched.current_path == [3, 10, 8, 9]
    assert sched.current_cmd == 1 and sched.pending_cmd is None
    assert sched.last_trigger == "cmd_change"
    assert sched.path_version == version + 1
    print("PASS test_cmd_change_waits_for_forward_buffer_source")


def test_cmd_change_rejects_bad_live_entry():
    g = load_toy()
    sched = SkillGraphScheduler(g, planner_type="graph_search",
                                A=0.5, B=2.0, lambda_cost=1.0,
                                tau=1.0, top_k=3,
                                switch_entry_A=0.5)
    sched.set_command("skill_A")
    sched.install_reference_path(sched.pure_skill_path(0), t=0.0)
    sched.pointer = 3
    version = sched.path_version
    x_bad = NodeState(q=np.full(23, 0.05), q_dot=np.zeros(23),
                      p_hat=np.zeros(3))  # sim=1.15 > A
    gd = sched.step(x_bad, user_cmd="skill_B", t=0.02)
    assert gd.node_id == 3
    assert sched.current_cmd == 0 and sched.pending_cmd == 1
    assert sched.path_version == version
    assert sched.last_trigger == "cmd_wait_entry_error"
    print("PASS test_cmd_change_rejects_bad_live_entry")


def test_command_during_transition_is_queued():
    """A second command must not replace an in-flight Buffer reference."""
    g = load_toy()
    sched = SkillGraphScheduler(g, planner_type="graph_search",
                                A=0.5, B=2.0, lambda_cost=1.0,
                                tau=1.0, top_k=3)
    sched.set_command("skill_A")
    sched.install_reference_path(sched.pure_skill_path(0), t=0.0)
    x = NodeState(q=np.zeros(23), q_dot=np.zeros(23), p_hat=np.zeros(3))

    sched.pointer = 3
    sched.step(x, user_cmd="skill_B", t=0.02)
    assert sched.current_path == [3, 10, 8, 9]
    version = sched.path_version
    assert sched.transition_active()

    # Request A while still at the old-skill prefix, then cross the Buffer.
    gd = sched.step(x, user_cmd="skill_A", t=0.04)
    assert gd.node_id == 3 and sched.pending_cmd == 0
    assert sched.path_version == version
    sched.pointer = 1
    assert sched.step(x, user_cmd=None, t=0.06).node_id == 10
    assert sched.path_version == version and sched.pending_cmd == 0

    # The request becomes eligible only after landing in B. The toy graph
    # has no trained B->A Buffer, so it stays pending without a hot-swap.
    sched.pointer = 2
    assert not sched.transition_active()
    assert sched.step(x, user_cmd=None, t=0.08).node_id == 8
    assert sched.path_version == version and sched.pending_cmd == 0
    print("PASS test_command_during_transition_is_queued")


def test_default_switch_entry_threshold():
    g = load_toy()
    sched = SkillGraphScheduler(g, planner_type="graph_search",
                                A=0.5, B=14.4499, lambda_cost=1.0,
                                tau=1.0, top_k=3)
    assert sched.switch_entry_A == 11.0
    clipped = SkillGraphScheduler(g, planner_type="graph_search",
                                  A=0.5, B=2.0, lambda_cost=1.0,
                                  tau=1.0, top_k=3)
    assert clipped.switch_entry_A == 2.0
    try:
        SkillGraphScheduler(g, planner_type="graph_search",
                            A=0.5, B=2.0, lambda_cost=1.0,
                            tau=1.0, top_k=3, switch_entry_A=3.0)
    except ValueError as exc:
        assert "<= B" in str(exc)
    else:
        raise AssertionError("switch_entry_A > B must be rejected")
    print("PASS test_default_switch_entry_threshold")


def test_cmd_change_uses_separate_live_entry_threshold():
    """A time-aligned Buffer entry may tolerate ordinary tracking error
    without weakening the graph-wide attachment threshold A."""
    g = load_toy()
    sched = SkillGraphScheduler(g, planner_type="graph_search",
                                A=0.5, B=2.0, lambda_cost=1.0,
                                tau=1.0, top_k=3,
                                switch_entry_A=1.5)
    sched.set_command("skill_A")
    sched.install_reference_path(sched.pure_skill_path(0), t=0.0)
    sched.pointer = 3
    version = sched.path_version
    x_tracking_error = NodeState(
        q=np.full(23, 0.05), q_dot=np.zeros(23), p_hat=np.zeros(3))

    gd = sched.step(x_tracking_error, user_cmd="skill_B", t=0.02)
    assert gd.node_id == 3
    assert sched.last_best_sim > sched.A
    assert sched.last_best_sim <= sched.switch_entry_A
    assert sched.current_path == [3, 10, 8, 9]
    assert sched.current_cmd == 1 and sched.pending_cmd is None
    assert sched.path_version == version + 1
    assert sched.last_trigger == "cmd_change"
    print("PASS test_cmd_change_uses_separate_live_entry_threshold")


def test_nn_planner_direct_hop():
    g = load_toy()
    sched = SkillGraphScheduler(g, planner_type="nn",
                                A=0.5, B=2.0, lambda_cost=1.0,
                                tau=1.0, top_k=3)
    x = NodeState(q=np.zeros(23), q_dot=np.zeros(23), p_hat=np.zeros(3))
    gd = sched.step(x, user_cmd="skill_B", t=0.0)
    assert gd is not None
    # entry node 0 (sim 0), nearest T node = 5, distance 0 -> direct hop,
    # then temporal walk to the end of skill B
    assert sched.current_path == [0, 5, 6, 7, 8, 9], sched.current_path
    print("PASS test_nn_planner_direct_hop")


def test_nn_planner_runtime_buffers():
    g = load_toy()
    for i in range(5, 10):  # skill B frames now differ by 0.1 per dof
        g.nodes[i].q = g.nodes[i].q + 0.1
    sched = SkillGraphScheduler(g, planner_type="nn",
                                A=0.5, B=5.0, lambda_cost=1.0,
                                tau=1.0, top_k=3)
    x = NodeState(q=np.zeros(23), q_dot=np.zeros(23), p_hat=np.zeros(3))
    n0 = g.num_nodes
    gd = sched.step(x, user_cmd="skill_B", t=0.0)
    assert gd is not None
    # entry 0 -> t_star 5, d = 23*0.1 = 2.3 -> n_buf = 2 runtime buffers
    path = sched.current_path
    assert g.num_nodes == n0 + 2
    assert path[0] == 0 and path[-1] == 9
    buf_ids = path[1:3]
    assert all(g.is_buffer[b] for b in buf_ids)
    assert [int(g.kappas[b]) for b in buf_ids] == [2, 1]
    assert np.allclose(g.nodes[buf_ids[0]].q, 0.1 / 3)
    assert np.allclose(g.nodes[buf_ids[1]].q, 0.2 / 3)
    # buffer chain caches and reuses nodes on a second identical plan
    path2 = sched.nn_planner.short_hop_path(0, sched.T_cmd, 5.0)
    assert path2[1:3] == buf_ids and g.num_nodes == n0 + 2
    print("PASS test_nn_planner_runtime_buffers")


def test_nn_planner_unsafe_jump():
    g = load_toy()
    for i in range(5, 10):
        g.nodes[i].q = g.nodes[i].q + 10.0  # way beyond B
    sched = SkillGraphScheduler(g, planner_type="nn",
                                A=0.5, B=2.0, lambda_cost=1.0,
                                tau=1.0, top_k=3)
    nn = sched.nn_planner
    assert nn.short_hop_path(0, [5, 6, 7, 8, 9], 2.0) is None
    print("PASS test_nn_planner_unsafe_jump")


def main():
    test_deploy_edge_weight()
    test_skill_roles_load_and_backward_compatibility()
    test_buffer_macro_expansion_preserves_cost()
    test_value_function_hand_computed()
    test_entry_check_branches()
    test_scheduler_step_flow()
    test_safety_replan_disabled_by_default()
    test_cmd_change_estop_fallback()
    test_cmd_change_waits_for_forward_buffer_source()
    test_cmd_change_rejects_bad_live_entry()
    test_command_during_transition_is_queued()
    test_default_switch_entry_threshold()
    test_cmd_change_uses_separate_live_entry_threshold()
    test_nn_planner_direct_hop()
    test_nn_planner_runtime_buffers()
    test_nn_planner_unsafe_jump()
    test_switch_graph_mid_skill_buffer_is_direct_target()
    test_real_graph_paths()
    test_kappa_passthrough()
    test_extend_to_skill_end()
    test_attach_path_crosses_skills()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
