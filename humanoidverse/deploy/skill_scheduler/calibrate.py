#!/usr/bin/env python3
"""Offline calibration for the online skill scheduler (spec 6.1).

One-shot script:
  1. Load an enhanced skill_graph.json (must contain the per-node "nodes"
     array exported by SG_build/skill_graph_V2.py).
  2. Split edges into three classes: same_skill_consecutive / cross_skill
     (no buffer) / buffer.
  3. Compute sigma_q, sigma_qdot, sigma_p: std of the per-component L1
     distance over ALL graph edges.
  4. Recompute normalized sim() distances per edge class and report their
     distributions.
  5. Suggest thresholds:
       A candidate: P25 of cross_skill (no buffer) edge distances
       B candidates: P75 and P90 of buffer edge distances
     These are starting points for the grid search (spec 6.2), not finals.
  6. Write scheduler_config.yaml with sigmas + A/B candidates + weights.

Usage:
    python humanoidverse/deploy/skill_scheduler/calibrate.py \
        SG_build/sg_output_V2/skill_graph.json \
        -o SG_build/sg_output_V2/scheduler_config.yaml
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def _load_distance_module():
    """Load distance.py directly, bypassing the humanoidverse.deploy package
    __init__ (which pulls in urcirobot -> torch). This offline script must
    stay numpy/yaml-only."""
    spec = importlib.util.spec_from_file_location(
        "sg_distance", Path(__file__).with_name("distance.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_dist = _load_distance_module()
NodeState = _dist.NodeState
sim = _dist.sim


def load_graph(json_path: str):
    with open(json_path) as f:
        g = json.load(f)
    if "nodes" not in g:
        raise ValueError(
            f"{json_path} has no 'nodes' array. Rebuild the graph with the "
            f"updated SG_build/skill_graph_V2.py (node feature export)."
        )
    nodes = {}
    for n in g["nodes"]:
        if not n["q"] or not n["q_dot"] or not n["p_hat"]:
            raise ValueError(f"Node {n['node_id']} is missing state features.")
        # p_hat is used Z-ONLY, matching the graph-construction distance
        # (frame_distance in SG_build/skill_graph_V2.py uses root_z only).
        # x-y is zeroed so sigmas and sim distances are consistent with the
        # deploy-time live state (root height only).
        p_hat = np.asarray(n["p_hat"], dtype=np.float64)
        p_hat[:2] = 0.0
        nodes[n["node_id"]] = NodeState(
            q=np.asarray(n["q"]),
            q_dot=np.asarray(n["q_dot"]),
            p_hat=p_hat,
        )
    return g, nodes


def edge_class(edge: dict) -> str:
    if edge["is_buffer"]:
        return "buffer"
    if edge["is_cross_skill"]:
        return "cross_skill"
    return "same_skill_consecutive"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("skill_graph", help="Path to enhanced skill_graph.json")
    parser.add_argument("-o", "--output", default=None,
                        help="Output scheduler_config.yaml path "
                             "(default: alongside the graph json)")
    parser.add_argument("--w-q", type=float, default=1.0)
    parser.add_argument("--w-qdot", type=float, default=1.0)
    parser.add_argument("--w-p", type=float, default=1.0,
                        help="Weight of the p_hat (root z) component. Use 0 "
                             "to exclude it: with z-only p_hat, sigma_p is "
                             "tiny (~0.025) and a few cm of live root-height "
                             "deviation dominates the distance.")
    args = parser.parse_args()

    g, nodes = load_graph(args.skill_graph)
    edges = g["edges"]
    print(f"Loaded {len(nodes)} nodes, {len(edges)} edges "
          f"({g['num_buffer_nodes']} buffer nodes)")

    # --- Step 1: per-component L1 distances over all edges ---
    d_q_all, d_qdot_all, d_p_all = [], [], []
    for e in edges:
        u, v = nodes[e["src"]], nodes[e["dst"]]
        d_q_all.append(np.abs(u.q - v.q).sum())
        d_qdot_all.append(np.abs(u.q_dot - v.q_dot).sum())
        d_p_all.append(np.abs(u.p_hat - v.p_hat).sum())

    sigma_q = float(np.std(d_q_all)) or 1.0
    sigma_qdot = float(np.std(d_qdot_all)) or 1.0
    sigma_p = float(np.std(d_p_all)) or 1.0
    print(f"sigma_q={sigma_q:.4f}  sigma_qdot={sigma_qdot:.4f}  "
          f"sigma_p={sigma_p:.4f}")

    # --- Step 2: normalized sim distances per edge class ---
    w = dict(w_q=args.w_q, w_qdot=args.w_qdot, w_p=args.w_p)
    sims = {"same_skill_consecutive": [], "cross_skill": [], "buffer": []}
    for e in edges:
        s = sim(nodes[e["src"]], nodes[e["dst"]],
                sigma_q, sigma_qdot, sigma_p, **w)
        sims[edge_class(e)].append(s)

    # Buffer *gap* distances: sim between the endpoints of each buffer chain
    # (deduplicated). Per-hop buffer edges are artificially small (that is the
    # point of interpolation), so the B threshold candidate must come from
    # the endpoint-to-endpoint gap, not from per-hop edges.
    gap_sims, seen_gaps = [], set()
    for bn in g.get("buffer_nodes", []):
        key = (bn["src_node"], bn["dst_node"])
        if key in seen_gaps:
            continue
        seen_gaps.add(key)
        gap_sims.append(sim(nodes[bn["src_node"]], nodes[bn["dst_node"]],
                            sigma_q, sigma_qdot, sigma_p, **w))
    sims["buffer_gap (endpoint)"] = gap_sims

    def pct(values, q):
        return float(np.percentile(values, q)) if len(values) else float("nan")

    print(f"\n{'edge class':<24}{'count':>7}{'mean':>9}{'P25':>9}{'P50':>9}"
          f"{'P75':>9}{'P90':>9}")
    for cls, values in sims.items():
        values = np.asarray(values)
        print(f"{cls:<24}{len(values):>7}{values.mean():>9.3f}"
              f"{pct(values, 25):>9.3f}{pct(values, 50):>9.3f}"
              f"{pct(values, 75):>9.3f}{pct(values, 90):>9.3f}")

    # --- Step 3: threshold candidates (spec 6.1) ---
    A_candidate = pct(sims["cross_skill"], 25)
    B_p75 = pct(sims["buffer_gap (endpoint)"], 75)
    B_p90 = pct(sims["buffer_gap (endpoint)"], 90)
    print(f"\nA candidate (cross_skill P25): {A_candidate:.3f}")
    print(f"B candidates (buffer_gap P75/P90): {B_p75:.3f} / {B_p90:.3f}")

    # --- Step 4: write scheduler_config.yaml ---
    config = {
        "sigma_q": sigma_q,
        "sigma_qdot": sigma_qdot,
        "sigma_p": sigma_p,
        "w_q": args.w_q,
        "w_qdot": args.w_qdot,
        "w_p": args.w_p,
        # Grid-search starting points (spec 6.2); not yet calibrated.
        "A_candidates": [round(A_candidate, 4)],
        "B_candidates": [round(B_p75, 4), round(B_p90, 4)],
        "tau": [0.1, 0.2, 0.3],
        "top_k": [3, 5, 10],
        "lambda_sw": [1.0, 5.0, 10.0],
        "lambda_cost": [0.1, 1.0, 10.0],
    }
    out_path = args.output or str(Path(args.skill_graph).with_name(
        "scheduler_config.yaml"))
    with open(out_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"\nscheduler_config.yaml written to {out_path}")


if __name__ == "__main__":
    main()
