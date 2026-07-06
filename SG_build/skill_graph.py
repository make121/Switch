"""
Skill Graph (SG) construction for humanoid multi-skill transitions.

Based on "Switch: Learning Agile Skills Switching for Humanoid Robots"
(arXiv:2604.14834)

Core algorithm:
  1. Represent all motion frames as graph nodes
  2. Connect within-skill frames with temporal edges (weight=1)
  3. Find cross-skill connections via kinematic similarity (L1 distance
     on joint positions + velocities + root translation)
  4. Insert buffer nodes for large-gap transitions via interpolation
  5. Output augmented dataset + graph structure for training

Data format: PBHC .pkl files (joblib-serialized dicts)
  Each motion dict value has: dof, root_trans_offset, root_rot,
  pose_aa, smpl_joints, fps, contact_mask
"""

import copy
import itertools
import json
import math
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_motion_pkl(pkl_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Load a PBHC-format .pkl file (joblib or pickle)."""
    p = Path(pkl_path)
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return joblib.load(p)


def save_motion_pkl(data: Dict[str, Any], path: str):
    """Save motion data as pickle."""
    with open(path, "wb") as f:
        pickle.dump(data, f)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@dataclass
class FrameFeatures:
    """Precomputed per-frame features for distance computation."""

    dof: np.ndarray  # (23,) joint positions
    dof_vel: np.ndarray  # (23,) joint velocities
    root_trans: np.ndarray  # (3,) root translation offset
    fps: float

    @classmethod
    def from_motion(cls, motion: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract feature array for all frames of a motion.

        Returns array of FrameFeatures, shape (num_frames,)
        """
        dof = motion["dof"]  # (T, 23)
        fps = motion["fps"]

        # Joint velocities: finite difference + pad first frame
        dof_vel = np.diff(dof, axis=0) * fps
        dof_vel = np.vstack([dof_vel[:1], dof_vel])

        root = motion.get("root_trans_offset", np.zeros((dof.shape[0], 3)))

        features = np.empty(dof.shape[0], dtype=object)
        for t in range(dof.shape[0]):
            features[t] = cls(
                dof=dof[t].astype(np.float64),
                dof_vel=dof_vel[t].astype(np.float64),
                root_trans=root[t].astype(np.float64),
                fps=fps,
            )
        return features


# ---------------------------------------------------------------------------
# Distance metric  (Sec III-A.2, Eq. 2)
# ---------------------------------------------------------------------------

def frame_distance(fa: FrameFeatures, fb: FrameFeatures) -> float:
    """L1 distance between two frames (local frame, global x-y removed).

    Per Switch paper Sec III-A.2:
    "In the local frame (with global x–y translation and yaw/twist removed)"

    d(s_m, s_n) = ||q_m - q_n||_1 + ||q̇_m - q̇_n||_1 + |r_z_m - r_z_n|

    Only root height (z) is kept; global x-y discarded so the distance
    is invariant to where the motion capture was performed.
    """
    d_q = float(np.sum(np.abs(fa.dof - fb.dof)))
    d_qv = float(np.sum(np.abs(fa.dof_vel - fb.dof_vel)))
    d_root_z = float(np.abs(fa.root_trans[2] - fb.root_trans[2]))
    return d_q + d_qv + d_root_z


# ---------------------------------------------------------------------------
# Skill Graph data structures
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    skill_id: int       # -1 for buffer nodes
    frame_idx: int      # local frame index within the skill, -1 for buffer
    global_id: int      # unique global id
    is_buffer: bool = False
    buffer_src: int = -1   # source global id (buffer nodes only)
    buffer_dst: int = -1   # target global id (buffer nodes only)

    def __hash__(self):
        return hash(self.global_id)


@dataclass
class GraphEdge:
    src: int  # global node id
    dst: int  # global node id
    weight: float
    is_cross_skill: bool = False
    is_buffer: bool = False

    def to_dict(self) -> dict:
        return {
            "src": int(self.src),
            "dst": int(self.dst),
            "weight": float(round(self.weight, 4)),
            "is_cross_skill": bool(self.is_cross_skill),
            "is_buffer": bool(self.is_buffer),
        }


@dataclass
class SkillGraph:
    """Directed graph G = (V, E, w) for multi-skill transitions."""

    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    skill_names: List[str] = field(default_factory=list)
    skill_lengths: List[int] = field(default_factory=list)  # frames per skill
    buffer_count: int = 0  # number of buffer nodes added
    buffer_edge_count: int = 0  # number of edges involving buffer nodes
    # Adjacency: node_id -> list of (dst_node_id, edge)
    _adj: Dict[int, List[Tuple[int, GraphEdge]]] = field(default_factory=dict)

    def add_node(self, skill_id: int, frame_idx: int, is_buffer: bool = False,
                 buffer_src: int = -1, buffer_dst: int = -1) -> int:
        gid = len(self.nodes)
        self.nodes.append(GraphNode(skill_id, frame_idx, gid,
                                    is_buffer=is_buffer,
                                    buffer_src=buffer_src,
                                    buffer_dst=buffer_dst))
        return gid

    def add_edge(self, src: int, dst: int, weight: float, is_cross: bool = False, is_buffer: bool = False):
        e = GraphEdge(src, dst, weight, is_cross, is_buffer)
        self.edges.append(e)
        self._adj.setdefault(src, []).append((dst, e))

    def get_skill_start_gid(self, skill_id: int) -> int:
        """Global node id of the first frame of a skill."""
        offset = sum(self.skill_lengths[:skill_id])
        return offset

    def get_skill_end_gid(self, skill_id: int) -> int:
        offset = sum(self.skill_lengths[: skill_id + 1])
        return offset - 1

    def to_dict(self) -> dict:
        buf_nodes = [n for n in self.nodes if n.is_buffer]
        return {
            "num_nodes": len(self.nodes),
            "num_original_nodes": len(self.nodes) - len(buf_nodes),
            "num_buffer_nodes": len(buf_nodes),
            "num_edges": len(self.edges),
            "num_buffer_edges": self.buffer_edge_count,
            "skill_names": self.skill_names,
            "skill_lengths": [int(x) for x in self.skill_lengths],
            "edges": [e.to_dict() for e in self.edges],
            "buffer_nodes": [
                {
                    "global_id": int(n.global_id),
                    "src_node": int(n.buffer_src),
                    "dst_node": int(n.buffer_dst),
                }
                for n in buf_nodes
            ],
        }

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)


# ---------------------------------------------------------------------------
# Skill Graph Builder
# ---------------------------------------------------------------------------

class SkillGraphBuilder:
    """Builds a Skill Graph from a set of PBHC motion files.

    Usage:
        builder = SkillGraphBuilder(config)
        graph, augmented_data = builder.build()
        builder.save("output_dir/")
    """

    def __init__(
        self,
        motion_files: List[str],  # paths to .pkl files, one per skill
        skill_labels: Optional[List[str]] = None,
        cross_skill_topk: int = 5,  # top-K nearest neighbours per frame
        cross_skill_threshold: float = 30.0,  # max L1 distance for an edge
        buffer_base_threshold: float = 1.0,  # distance per buffer node
        max_buffer_nodes: int = 30,  # cap on buffer nodes per edge
        subsample_stride: int = 3,  # stride for source-frame subsampling
        exclude_boundary_frames: int = 10,  # exclude transitions near skill boundaries
        max_trajectories_per_pair: int = 10,  # max trajectories per skill pair
        fps: Optional[float] = None,  # override fps (default: from data)
    ):
        self.motion_files = motion_files
        self.skill_labels = skill_labels or [f"skill_{i}" for i in range(len(motion_files))]
        self.cross_skill_topk = cross_skill_topk
        self.cross_skill_threshold = cross_skill_threshold
        self.buffer_base_threshold = buffer_base_threshold
        self.max_buffer_nodes = max_buffer_nodes
        self.subsample_stride = subsample_stride
        self.exclude_boundary_frames = exclude_boundary_frames
        self.max_trajectories_per_pair = max_trajectories_per_pair
        self.fps = fps

        # Internal state
        self._motions: List[Dict[str, np.ndarray]] = []
        self._features: List[np.ndarray] = []  # per-skill feature arrays
        self._graph: Optional[SkillGraph] = None
        self._augmented_motions: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Step 1: Load & extract
    # ------------------------------------------------------------------

    def load(self):
        """Load all motion files, normalize away global x-y and yaw.

        Each skill may have been captured at a different position and
        facing direction. We center x-y to origin and align yaw to zero
        so that cross-skill transitions only require pose change, not
        large spatial displacement or rotation.  Height (z) is preserved.
        """
        print(f"Loading {len(self.motion_files)} motion files...")
        for i, path in enumerate(self.motion_files):
            data = load_motion_pkl(path)
            for key, motion in data.items():
                if self.fps is not None:
                    motion["fps"] = self.fps
                # Remove global x-y offset and yaw from each skill.
                # Use circular mean of yaw (handles ±180° wrap-around).
                if "root_rot" in motion and "root_trans_offset" in motion:
                    rto = motion["root_trans_offset"].copy()
                    rr = motion["root_rot"].copy()
                    r = R.from_quat(rr[:, [1, 2, 3, 0]])  # xyzw→wxyz
                    yaws = r.as_euler('xyz')[:, 2]

                    # Circular mean: atan2(mean(sin), mean(cos))
                    mean_yaw = np.arctan2(np.sin(yaws).mean(), np.cos(yaws).mean())
                    yaw_correction = R.from_euler('z', -float(mean_yaw))

                    # Apply to root_rot (all frames aligned to yaw≈0)
                    r_aligned = yaw_correction * r
                    motion["root_rot"] = r_aligned.as_quat()[:, [3, 0, 1, 2]]  # wxyz→xyzw

                    # Apply to root_trans x-y (rotate, then center)
                    rot_xy_3d = yaw_correction.apply(
                        np.column_stack([rto[:, :2], np.zeros(len(rto))])
                    )
                    rto[:, 0] = rot_xy_3d[:, 0] - rot_xy_3d[:, 0].mean()
                    rto[:, 1] = rot_xy_3d[:, 1] - rot_xy_3d[:, 1].mean()
                    motion["root_trans_offset"] = rto
                self._motions.append(motion)
                feats = FrameFeatures.from_motion(motion)
                self._features.append(feats)
                print(f"  [{i}] {self.skill_labels[i]}: {motion['dof'].shape[0]} frames, "
                      f"fps={motion['fps']}")

    # ------------------------------------------------------------------
    # Step 2: Build base graph (within-skill edges)
    # ------------------------------------------------------------------

    def build_base_graph(self) -> SkillGraph:
        """Create nodes for every frame and add temporal within-skill edges."""
        graph = SkillGraph(skill_names=self.skill_labels)
        gid = 0
        for skill_id, feats in enumerate(self._features):
            n_frames = len(feats)
            graph.skill_lengths.append(n_frames)
            for t in range(n_frames):
                node = GraphNode(skill_id, t, gid)
                graph.nodes.append(node)
                if t > 0:
                    # Temporal edge: weight = 1 (paper Eq. 3)
                    graph.add_edge(gid - 1, gid, weight=1.0, is_cross=False)
                gid += 1
        print(f"Base graph: {graph.num_nodes} nodes, {graph.num_edges} temporal edges")
        return graph

    # ------------------------------------------------------------------
    # Step 3: Cross-skill edges  (Sec III-A.2)
    # ------------------------------------------------------------------

    def build_cross_skill_edges(self, graph: SkillGraph):
        """For each frame in each skill, find nearest frames in other skills."""
        num_skills = len(self._features)
        cross_edges_added = 0

        # Precompute a flat list of all feature arrays indexed by global node id
        all_features: List[FrameFeatures] = []
        for skill_id, feats in enumerate(self._features):
            all_features.extend(feats)

        for src_skill in range(num_skills):
            src_feats = self._features[src_skill]
            src_start = sum(graph.skill_lengths[:src_skill])

            for dst_skill in range(num_skills):
                if dst_skill == src_skill:
                    continue

                dst_feats = self._features[dst_skill]
                dst_start = sum(graph.skill_lengths[:dst_skill])

                # Subsample source frames for efficiency
                sample_indices = list(range(0, len(src_feats), self.subsample_stride))

                for t_src in sample_indices:
                    src_gid = src_start + t_src
                    src_f = src_feats[t_src]

                    # Compute distances to all frames in target skill
                    distances = np.array([
                        frame_distance(src_f, dst_feats[t])
                        for t in range(len(dst_feats))
                    ])

                    # Top-K nearest
                    topk = min(self.cross_skill_topk, len(dst_feats))
                    top_indices = np.argpartition(distances, topk)[:topk]
                    # Sort by distance
                    top_indices = top_indices[np.argsort(distances[top_indices])]

                    for t_dst in top_indices:
                        d = float(distances[t_dst])
                        if d > self.cross_skill_threshold:
                            continue
                        dst_gid = dst_start + t_dst
                        graph.add_edge(src_gid, dst_gid, weight=d, is_cross=True)
                        cross_edges_added += 1

        print(f"Cross-skill edges: {cross_edges_added} added "
              f"(threshold={self.cross_skill_threshold}, topk={self.cross_skill_topk}, "
              f"subsample_stride={self.subsample_stride})")

    # ------------------------------------------------------------------
    # Step 4: Buffer nodes  (Sec III-B.3)
    # ------------------------------------------------------------------

    def _is_boundary_node(self, global_id: int, graph: SkillGraph) -> bool:
        """Check if a node falls within the boundary exclusion zone of any skill.

        Boundary zones are the first N and last N frames of each skill.
        Returns True if the node should be excluded from buffer trajectory
        endpoints.
        """
        if self.exclude_boundary_frames <= 0:
            return False
        n = self.exclude_boundary_frames
        offset = 0
        for length in graph.skill_lengths:
            local_id = global_id - offset
            if 0 <= local_id < length:
                # First N frames or last N frames of this skill
                return local_id < n or local_id >= length - n
            offset += length
        return False

    def compute_buffer_count(self, distance: float) -> int:
        """N buffer nodes based on distance between endpoints."""
        n = int(distance / self.buffer_base_threshold)
        return min(max(0, n), self.max_buffer_nodes)

    def interpolate_frame(
        self,
        src_motion: Dict[str, np.ndarray],
        dst_motion: Dict[str, np.ndarray],
        t_src: int,
        t_dst: int,
        alpha: float,
    ) -> Dict[str, np.ndarray]:
        """Linearly interpolate a single frame between two source frames.

        alpha=0 → src frame, alpha=1 → dst frame.
        Rotation uses SLERP (spherical linear interpolation).
        """
        frame = {}

        # Joint positions: linear
        if "dof" in src_motion:
            frame["dof"] = (1 - alpha) * src_motion["dof"][t_src] + alpha * dst_motion["dof"][t_dst]

        # Root translation: linear
        frame["root_trans_offset"] = (
            (1 - alpha) * src_motion["root_trans_offset"][t_src]
            + alpha * dst_motion["root_trans_offset"][t_dst]
        )

        # Root rotation: SLERP
        if "root_rot" in src_motion:
            r_src = R.from_quat(src_motion["root_rot"][t_src][[1, 2, 3, 0]])  # xyzw→wxyz
            r_dst = R.from_quat(dst_motion["root_rot"][t_dst][[1, 2, 3, 0]])
            slerp = Slerp([0, 1], R.concatenate([r_src, r_dst]))
            r_interp = slerp(alpha).as_quat()[[3, 0, 1, 2]]  # wxyz→xyzw
            if r_interp.shape == (4,):
                frame["root_rot"] = r_interp
            else:
                frame["root_rot"] = r_interp[0]

        # pose_aa: linear interpolation (approximate)
        if "pose_aa" in src_motion:
            frame["pose_aa"] = (1 - alpha) * src_motion["pose_aa"][t_src] + alpha * dst_motion["pose_aa"][t_dst]

        # smpl_joints: linear interpolation
        if "smpl_joints" in src_motion:
            frame["smpl_joints"] = (1 - alpha) * src_motion["smpl_joints"][t_src] + alpha * dst_motion["smpl_joints"][t_dst]

        # contact_mask: use target skill's mask (paper: buffer uses target frame for reward)
        if "contact_mask" in src_motion:
            frame["contact_mask"] = dst_motion["contact_mask"][t_dst].copy()

        return frame

    def build_buffer_trajectories(self, graph: SkillGraph) -> Dict[str, Dict]:
        """Create augmented motion trajectories with buffer nodes.

        For each cross-skill edge with large distance, inserts N buffer
        frames between source and target, creating a smooth transition
        trajectory.

        Returns dict of augmented motion data, keyed by trajectory name.
        """
        augmented = {}
        traj_idx = 0

        # Group cross-skill edges by (src_skill, dst_skill) pairs
        for src_skill in range(len(self._features)):
            for dst_skill in range(len(self._features)):
                if src_skill == dst_skill:
                    continue

                src_start = sum(graph.skill_lengths[:src_skill])
                dst_start = sum(graph.skill_lengths[:dst_skill])

                # Collect cross-skill edges for this pair
                pair_edges = [
                    e for e in graph.edges
                    if e.is_cross_skill
                    and src_start <= e.src < src_start + graph.skill_lengths[src_skill]
                    and dst_start <= e.dst < dst_start + graph.skill_lengths[dst_skill]
                ]

                if not pair_edges:
                    continue

                # Sort by weight (distance), pick best edges
                pair_edges.sort(key=lambda e: e.weight)

                # Collect up to max_trajectories_per_pair valid transitions,
                # iterating through all edges until target is met or exhausted
                seen_src = set()
                collected = 0
                for edge in pair_edges:
                    if collected >= self.max_trajectories_per_pair:
                        break
                    if edge.src in seen_src:
                        continue

                    # Skip transitions where src or dst falls within boundary
                    # frames of any skill (e.g. first/last N frames of a skill)
                    if self._is_boundary_node(edge.src, graph) or self._is_boundary_node(edge.dst, graph):
                        continue

                    seen_src.add(edge.src)

                    t_src = edge.src - src_start
                    t_dst = edge.dst - dst_start
                    n_buffer = self.compute_buffer_count(edge.weight)

                    # ---- Add buffer nodes + edges to the graph ----
                    if n_buffer > 0:
                        prev_gid = edge.src
                        buf_gids = []
                        for k in range(1, n_buffer + 1):
                            buf_gid = graph.add_node(
                                skill_id=-1, frame_idx=-1, is_buffer=True,
                                buffer_src=edge.src, buffer_dst=edge.dst)
                            buf_gids.append(buf_gid)
                            # Edge from previous node to this buffer
                            graph.add_edge(prev_gid, buf_gid,
                                           weight=self.buffer_base_threshold,
                                           is_cross=False, is_buffer=True)
                            graph.buffer_edge_count += 1
                            prev_gid = buf_gid
                        # Edge from last buffer to target
                        graph.add_edge(prev_gid, edge.dst,
                                       weight=self.buffer_base_threshold,
                                       is_cross=False, is_buffer=True)
                        graph.buffer_edge_count += 1
                        graph.buffer_count += n_buffer
                    # ---- End graph buffer node insertion ----

                    # Build trajectory: src_frames + buffer + dst_frames
                    traj_dof = []
                    traj_root = []
                    traj_root_rot = []
                    traj_pose_aa = []
                    traj_smpl = []
                    traj_contact = []
                    traj_is_buffer = []  # metadata: which frames are buffer nodes
                    traj_source = []  # metadata: which skill each frame is from

                    src_mot = self._motions[src_skill]
                    dst_mot = self._motions[dst_skill]

                    # Source frames: take ~1 second before transition
                    n_src = min(30, t_src + 1)
                    src_slice = slice(max(0, t_src - n_src + 1), t_src + 1)

                    def append_frames(mot, slc, is_buf, source_skill):
                        if "dof" in mot:
                            traj_dof.append(mot["dof"][slc])
                        if "root_trans_offset" in mot:
                            traj_root.append(mot["root_trans_offset"][slc])
                        if "root_rot" in mot:
                            traj_root_rot.append(mot["root_rot"][slc])
                        if "pose_aa" in mot:
                            traj_pose_aa.append(mot["pose_aa"][slc])
                        if "smpl_joints" in mot:
                            traj_smpl.append(mot["smpl_joints"][slc])
                        if "contact_mask" in mot:
                            traj_contact.append(mot["contact_mask"][slc])
                        n = (slc.stop - slc.start) if isinstance(slc, slice) else mot["dof"][slc].shape[0]
                        traj_is_buffer.extend([is_buf] * n)
                        traj_source.extend([source_skill] * n)

                    # Source segment
                    append_frames(src_mot, src_slice, is_buf=False, source_skill=src_skill)

                    # Buffer nodes
                    for k in range(1, n_buffer + 1):
                        alpha = k / (n_buffer + 1)
                        buf = self.interpolate_frame(src_mot, dst_mot, t_src, t_dst, alpha)
                        for key in ["dof", "root_trans_offset", "root_rot", "pose_aa", "smpl_joints", "contact_mask"]:
                            if key in buf and key[:-1] + "_" not in str(type(buf[key])):
                                pass
                        if "dof" in buf:
                            traj_dof.append(buf["dof"][np.newaxis, :])
                        if "root_trans_offset" in buf:
                            traj_root.append(buf["root_trans_offset"][np.newaxis, :])
                        if "root_rot" in buf:
                            traj_root_rot.append(buf["root_rot"][np.newaxis, :])
                        if "pose_aa" in buf:
                            traj_pose_aa.append(buf["pose_aa"][np.newaxis, :])
                        if "smpl_joints" in buf:
                            traj_smpl.append(buf["smpl_joints"][np.newaxis, :])
                        if "contact_mask" in buf:
                            traj_contact.append(buf["contact_mask"][np.newaxis, :])
                        traj_is_buffer.append(True)
                        traj_source.append(-1)  # -1 = buffer

                    # Target segment: take ~2 seconds after transition point
                    n_dst = min(60, len(self._features[dst_skill]) - t_dst)
                    dst_slice = slice(t_dst, t_dst + n_dst)
                    append_frames(dst_mot, dst_slice, is_buf=False, source_skill=dst_skill)

                    # Assemble trajectory
                    traj_dict = {}
                    if traj_dof:
                        traj_dict["dof"] = np.concatenate(traj_dof, axis=0)
                    if traj_root:
                        traj_dict["root_trans_offset"] = np.concatenate(traj_root, axis=0)
                    if traj_root_rot:
                        traj_dict["root_rot"] = np.concatenate(traj_root_rot, axis=0)
                    if traj_pose_aa:
                        traj_dict["pose_aa"] = np.concatenate(traj_pose_aa, axis=0)
                    if traj_smpl:
                        traj_dict["smpl_joints"] = np.concatenate(traj_smpl, axis=0)
                    if traj_contact:
                        traj_dict["contact_mask"] = np.concatenate(traj_contact, axis=0)
                    traj_dict["fps"] = src_mot.get("fps", 30)
                    traj_dict["is_buffer"] = np.array(traj_is_buffer, dtype=bool)
                    traj_dict["source_skill"] = np.array(traj_source, dtype=np.int32)
                    traj_dict["src_skill"] = src_skill
                    traj_dict["dst_skill"] = dst_skill
                    traj_dict["transition_distance"] = edge.weight

                    name = (f"trans_{traj_idx:03d}_"
                            f"{self.skill_labels[src_skill]}_to_"
                            f"{self.skill_labels[dst_skill]}_"
                            f"dist{edge.weight:.1f}")
                    augmented[name] = traj_dict
                    traj_idx += 1
                    collected += 1

        print(f"Buffer trajectories: {len(augmented)} created")
        return augmented

    # ------------------------------------------------------------------
    # Main build pipeline
    # ------------------------------------------------------------------

    def build(self) -> Tuple[SkillGraph, Dict[str, Dict]]:
        """Run full SG construction pipeline."""
        if not self._features:
            self.load()

        # Step 2: base graph
        self._graph = self.build_base_graph()

        # Step 3: cross-skill edges
        self.build_cross_skill_edges(self._graph)

        # Step 4: buffer trajectories
        self._augmented_motions = self.build_buffer_trajectories(self._graph)

        return self._graph, self._augmented_motions

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save(self, output_dir: str, merge_with_original: bool = True):
        """Save graph, augmented data, and optionally a merged training file.

        Outputs:
          {output_dir}/
            skill_graph.json       - graph structure
            augmented_motions.pkl  - transition trajectories (pickle)
            merged_training.pkl    - original + augmented (for training)
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1. Graph JSON
        graph_path = out / "skill_graph.json"
        self._graph.save_json(str(graph_path))
        print(f"Graph saved to {graph_path}")

        # 2. Augmented transitions
        aug_path = out / "augmented_motions.pkl"
        save_motion_pkl(self._augmented_motions, str(aug_path))
        print(f"Augmented motions ({len(self._augmented_motions)} trajectories) saved to {aug_path}")

        # 3. Merged training file
        if merge_with_original:
            merged = dict(self._augmented_motions)
            # Add original single-skill motions
            for i, (label, mot) in enumerate(zip(self.skill_labels, self._motions)):
                merged[f"skill_{label}"] = mot
            merged_path = out / "merged_training.pkl"
            save_motion_pkl(merged, str(merged_path))
            print(f"Merged training data ({len(merged)} entries) saved to {merged_path}")

    def print_stats(self):
        """Print summary statistics."""
        if self._graph is None:
            print("Graph not built yet. Call build() first.")
            return
        g = self._graph
        cross_edges = [e for e in g.edges if e.is_cross_skill]
        buf_trajs = len(self._augmented_motions)
        total_buf_frames = sum(
            np.sum(m["is_buffer"]) for m in self._augmented_motions.values()
        )
        print(f"\n{'='*50}")
        print(f"Skill Graph Summary")
        print(f"{'='*50}")
        print(f"  Skills:          {len(g.skill_lengths)}")
        print(f"  Total frames:    {sum(g.skill_lengths)}")
        print(f"  Nodes (total):   {g.num_nodes}")
        print(f"  Nodes (original):{g.num_nodes - g.buffer_count}")
        print(f"  Nodes (buffer):  {g.buffer_count}")
        print(f"  Temporal edges:  {g.num_edges - len(cross_edges) - g.buffer_edge_count}")
        print(f"  Cross-skill edges: {len(cross_edges)}")
        print(f"  Buffer edges:    {g.buffer_edge_count}")
        print(f"  Buffer trajectories: {buf_trajs}")
        print(f"  Total buffer frames: {total_buf_frames}")
        for i, (name, length) in enumerate(zip(g.skill_names, g.skill_lengths)):
            print(f"  Skill[{i}] {name}: {length} frames, "
                  f"{length/30:.1f}s @ 30fps")
        print(f"{'='*50}")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def build_skill_graph(
    pkl_paths: List[str],
    skill_labels: Optional[List[str]] = None,
    output_dir: str = "./sg_output",
    cross_skill_threshold: float = 30.0,
    cross_skill_topk: int = 5,
    buffer_base_threshold: float = 1.0,
    subsample_stride: int = 3,
    exclude_boundary_frames: int = 10,
    max_trajectories_per_pair: int = 10,
    **kwargs,
) -> Tuple[SkillGraph, Dict[str, Dict]]:
    """One-shot skill graph construction.

    Args:
        pkl_paths: Paths to motion .pkl files (one per skill)
        skill_labels: Human-readable skill names
        output_dir: Where to save outputs
        cross_skill_threshold: Max L1 distance for a cross-skill edge
        cross_skill_topk: Number of nearest neighbors per frame
        buffer_base_threshold: L1 distance per buffer node
        subsample_stride: Stride for sampling source frames
        exclude_boundary_frames: Exclude transitions whose src or dst
            falls within N frames of any skill boundary (default: 10)
        max_trajectories_per_pair: Max trajectories per skill pair
            (default: 10). Iterates all edges until target is met.

    Returns:
        (graph, augmented_motions) tuple
    """
    builder = SkillGraphBuilder(
        motion_files=pkl_paths,
        skill_labels=skill_labels,
        cross_skill_threshold=cross_skill_threshold,
        cross_skill_topk=cross_skill_topk,
        buffer_base_threshold=buffer_base_threshold,
        subsample_stride=subsample_stride,
        exclude_boundary_frames=exclude_boundary_frames,
        max_trajectories_per_pair=max_trajectories_per_pair,
        **kwargs,
    )
    graph, augmented = builder.build()
    builder.print_stats()
    builder.save(output_dir)
    return graph, augmented


if __name__ == "__main__":
    # Example: build SG from two example motions
    example_dir = Path(__file__).resolve().parent.parent / "example" / "motion_data"
    files = [
        str(example_dir / "Horse-stance_pose.pkl"),
        str(example_dir / "Horse-stance_punch.pkl"),
    ]
    build_skill_graph(
        pkl_paths=files,
        skill_labels=["horse_pose", "horse_punch"],
        output_dir="./sg_output",
    )
