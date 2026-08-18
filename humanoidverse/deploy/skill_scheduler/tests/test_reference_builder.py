#!/usr/bin/env python3
"""Offline tests for the deploy-time reference builder.

Run directly:
    python humanoidverse/deploy/skill_scheduler/tests/test_reference_builder.py
"""

import sys
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import yaml

from SG_build.skill_graph_V2 import interpolate_frame
from humanoidverse.deploy.skill_scheduler.graph_data import SkillGraphData
from humanoidverse.deploy.skill_scheduler.reference_builder import ReferenceBuilder

GRAPH = _repo_root / "SG_build" / "sg_output_V2" / "skill_graph.json"
CONFIG = _repo_root / "SG_build" / "sg_output_V2" / "scheduler_config.yaml"
PKLS = [
    _repo_root / "example" / "motion_data" / "Horse-stance_pose.pkl",
    _repo_root / "example" / "motion_data" / "Horse-stance_punch.pkl",
]


def setup():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    graph = SkillGraphData.from_json(
        str(GRAPH), sigma_q=cfg["sigma_q"], sigma_qdot=cfg["sigma_qdot"],
        sigma_p=cfg["sigma_p"], lambda_sw=5.0,
        w_q=cfg.get("w_q", 1.0), w_qdot=cfg.get("w_qdot", 1.0),
        w_p=cfg.get("w_p", 1.0))
    rb = ReferenceBuilder.from_pkl_files(graph, [str(p) for p in PKLS])
    return graph, rb


def test_original_path():
    graph, rb = setup()
    # A plain run inside skill 1: nodes 210..219 (skill 1 frames 0..9)
    path = list(range(210, 220))
    traj = rb.build_trajectory(path)
    motion = rb.motions[1]
    assert traj["dof"].shape == (10, 23)
    assert traj["pose_aa"].shape[0] == 10
    assert traj["fps"] == motion["fps"]
    assert not traj["is_buffer"].any()
    assert np.array_equal(traj["node_ids"], np.arange(210, 220))
    for i, f in enumerate(range(0, 10)):
        assert np.allclose(traj["dof"][i], motion["dof"][f])
        assert np.allclose(traj["root_trans_offset"][i],
                           motion["root_trans_offset"][f])
    print("PASS test_original_path")


def test_buffer_chain():
    graph, rb = setup()
    # First buffer chain in the graph: src -> buffers -> dst
    buf_gid = int(np.nonzero(graph.is_buffer)[0][0])
    src = int(graph.buffer_src[buf_gid])
    dst = int(graph.buffer_dst[buf_gid])
    chain = [src]
    node = buf_gid
    while True:
        chain.append(node)
        nxt = [v for v, _, et in graph.adj[node]
               if et == "buffer" and v != node]
        if not nxt or not graph.is_buffer[nxt[0]]:
            if nxt:
                chain.append(nxt[0])
            break
        node = nxt[0]
    assert chain[-1] == dst, f"chain end {chain[-1]} != dst {dst}"
    n_buf = len(chain) - 2

    traj = rb.build_trajectory(chain)
    assert traj["dof"].shape[0] == len(chain)
    assert traj["is_buffer"].tolist() == \
        [False] + [True] * n_buf + [False]

    # Buffer frame k must equal interpolate_frame with alpha = k/(N+1)
    s_skill, s_f = int(graph.skill_ids[src]), int(graph.frame_idxs[src])
    d_skill, d_f = int(graph.skill_ids[dst]), int(graph.frame_idxs[dst])
    for k in range(1, n_buf + 1):
        expected = interpolate_frame(rb.motions[s_skill], rb.motions[d_skill],
                                     s_f, d_f, k / (n_buf + 1))
        assert np.allclose(traj["dof"][k], expected["dof"])
        assert np.allclose(traj["pose_aa"][k], expected["pose_aa"])
        # contact_mask of buffer frames comes from the DST frame
        if "contact_mask" in traj:
            assert np.array_equal(traj["contact_mask"][k],
                                  rb.motions[d_skill]["contact_mask"][d_f])
    # Endpoints are the exact source frames
    assert np.allclose(traj["dof"][0], rb.motions[s_skill]["dof"][s_f])
    assert np.allclose(traj["dof"][-1], rb.motions[d_skill]["dof"][d_f])
    print(f"PASS test_buffer_chain (chain len {len(chain)}, "
          f"{n_buf} buffers, src skill {s_skill} -> dst skill {d_skill})")


def test_contact_mask_present():
    graph, rb = setup()
    path = list(range(210, 220))
    traj = rb.build_trajectory(path)
    # MotionLib requires contact_mask (T, 2) if the original motions have it
    assert "contact_mask" in traj
    assert traj["contact_mask"].shape == (10, 2)
    print("PASS test_contact_mask_present")


def test_xy_continuity_across_skills():
    """Skills are x-y centered INDEPENDENTLY at graph build time, so a path
    crossing a skill boundary has a discontinuous global x-y in the raw
    frames. build_trajectory must re-anchor such jumps (max_xy_step)."""
    from humanoidverse.deploy.skill_scheduler.scheduler import \
        SkillGraphScheduler
    graph, rb = setup()
    sched = SkillGraphScheduler(graph, planner_type="graph_search",
                                A=1.889, B=5.0, lambda_cost=1.0,
                                tau=0.2, top_k=5)
    x = graph.nodes[100]  # robot standing exactly at skill 0 frame 100
    sched.set_command(1)
    assert sched.step(x, user_cmd=None, t=0.0) is not None
    path = sched.current_path
    assert any(nid >= 210 for nid in path), "test path must cross into skill 1"

    # Raw frames WOULD jump at the boundary; the assembled trajectory must not.
    traj = rb.build_trajectory(path, max_xy_step=0.3)
    rt = traj["root_trans_offset"]
    steps = np.linalg.norm(np.diff(rt[:, :2], axis=0), axis=1)
    assert steps.max() <= 0.3 + 1e-9, f"xy jump remains: {steps.max():.3f}m"
    # z must be untouched
    raw_first_z = rb.motions[int(graph.skill_ids[path[0]])] \
        ["root_trans_offset"][int(graph.frame_idxs[path[0]]), 2]
    assert np.isclose(rt[0, 2], raw_first_z)
    print(f"PASS test_xy_continuity_across_skills "
          f"(max xy step after fix: {steps.max():.3f}m)")


def test_nn_path_builds_trajectory():
    """NN planner paths (with runtime buffer nodes) must assemble into
    trajectories through the same ReferenceBuilder machinery."""
    from humanoidverse.deploy.skill_scheduler.scheduler import \
        SkillGraphScheduler
    graph, rb = setup()
    n0 = graph.num_nodes
    sched = SkillGraphScheduler(graph, planner_type="nn",
                                A=1.889, B=10.0, lambda_cost=1.0,
                                tau=0.2, top_k=5)
    x = graph.nodes[100]  # skill 0 frame 100
    sched.set_command(1)
    assert sched.step(x, user_cmd=None, t=0.0) is not None
    path = sched.current_path
    traj = rb.build_trajectory(path)
    assert traj["dof"].shape[0] == len(path)
    # if runtime buffers were created, kappas must decrease along them
    buf_kappas = [int(graph.kappas[n]) for n in path if graph.is_buffer[n]]
    assert buf_kappas == sorted(buf_kappas, reverse=True)
    steps = np.linalg.norm(
        np.diff(traj["root_trans_offset"][:, :2], axis=0), axis=1)
    assert steps.max() <= 0.3 + 1e-9
    print(f"PASS test_nn_path_builds_trajectory "
          f"(path {len(path)} nodes, {graph.num_nodes - n0} runtime buffers)")


def main():
    test_original_path()
    test_buffer_chain()
    test_contact_mask_present()
    test_xy_continuity_across_skills()
    test_nn_path_builds_trajectory()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
