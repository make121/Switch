"""Build a trackable reference trajectory from a scheduler-planned path.

The scheduler outputs a path of graph node ids. The tracking policy needs
full motion frames (pose_aa / root_trans_offset / contact_mask ...), while
graph nodes only store the distance features (q / q_dot / p_hat). This module
resolves node ids back to complete frames:

  - original node -> (skill_id, frame_idx) indexes the source skill motion,
  - buffer node   -> (buffer_src, buffer_dst, chain position) re-interpolated
                     with the SAME interpolate_frame used at graph build time.

The assembled trajectory dict follows the PBHC pkl format so it can be
injected into MotionLibBase._motion_data_list and reloaded with
load_motions().
"""

from typing import Dict, List

import numpy as np

from SG_build.skill_graph_V2 import interpolate_frame, load_motion_pkl

from .graph_data import SkillGraphData

# Keys assembled into the output trajectory. pose_aa + root_trans_offset +
# fps are what MotionLibBase.load_motion_with_skeleton actually consumes;
# contact_mask must be present if the initially loaded motions have it.
FRAME_KEYS = ["dof", "root_trans_offset", "root_rot", "pose_aa",
              "smpl_joints", "contact_mask"]


class ReferenceBuilder:
    def __init__(self, graph: SkillGraphData,
                 motions: List[Dict[str, np.ndarray]]):
        """motions[i] = the motion dict of skill i, in the same order (and
        from the same files) the skill graph was built with."""
        self.graph = graph
        self.motions = motions

    @classmethod
    def from_pkl_files(cls, graph: SkillGraphData,
                       pkl_paths: List[str]) -> "ReferenceBuilder":
        """Load skills exactly like SG_build does: every entry inside a pkl
        file becomes one skill, in file order then dict order."""
        motions = []
        for path in pkl_paths:
            data = load_motion_pkl(path)
            for _, motion in data.items():
                motions.append(motion)
        if len(motions) != len(graph.skill_names):
            raise ValueError(
                f"{len(motions)} skills loaded from {pkl_paths}, but the "
                f"graph has {len(graph.skill_names)} skills. Pass the same "
                f"files the graph was built with, in the same order."
            )
        return cls(graph, motions)

    # ------------------------------------------------------------------
    # Single-frame resolution
    # ------------------------------------------------------------------

    def _original_frame(self, node_id: int) -> Dict[str, np.ndarray]:
        skill = int(self.graph.skill_ids[node_id])
        f = int(self.graph.frame_idxs[node_id])
        motion = self.motions[skill]
        return {k: motion[k][f] for k in FRAME_KEYS if k in motion}

    def _buffer_frame(self, src_node: int, dst_node: int,
                      alpha: float) -> Dict[str, np.ndarray]:
        s_skill, s_f = int(self.graph.skill_ids[src_node]), int(self.graph.frame_idxs[src_node])
        d_skill, d_f = int(self.graph.skill_ids[dst_node]), int(self.graph.frame_idxs[dst_node])
        return interpolate_frame(self.motions[s_skill], self.motions[d_skill],
                                 s_f, d_f, alpha)

    # ------------------------------------------------------------------
    # Path -> trajectory
    # ------------------------------------------------------------------

    def build_trajectory(self, path: List[int],
                         max_xy_step: float = 0.3) -> Dict[str, np.ndarray]:
        """Assemble the full trajectory for a planned node path.

        Buffer chains (consecutive buffer nodes sharing src/dst) are expanded
        with alpha = k / (N + 1), identical to graph construction.

        max_xy_step: each skill was independently x-y centered at graph
        build time, so raw root x-y jumps discontinuously at cross-skill
        boundaries (the reference ghost teleports and the robot can never
        catch up). Any per-frame x-y step larger than this is removed by
        re-anchoring all subsequent frames to the previous frame's x-y.
        Root z is absolute and left untouched.
        """
        g = self.graph
        frames: List[Dict[str, np.ndarray]] = []
        i = 0
        while i < len(path):
            nid = path[i]
            if not g.is_buffer[nid]:
                frames.append(self._original_frame(nid))
                i += 1
                continue
            src, dst = int(g.buffer_src[nid]), int(g.buffer_dst[nid])
            chain = []
            while i < len(path) and g.is_buffer[path[i]] \
                    and int(g.buffer_src[path[i]]) == src \
                    and int(g.buffer_dst[path[i]]) == dst:
                chain.append(path[i])
                i += 1
            n_buf = len(chain)
            for k in range(1, n_buf + 1):
                frames.append(self._buffer_frame(src, dst, k / (n_buf + 1)))

        first = path[0]
        first_skill = int(g.skill_ids[first]) if not g.is_buffer[first] \
            else int(g.skill_ids[int(g.buffer_src[first])])
        fps = float(self.motions[first_skill].get("fps", 30))

        traj = {"fps": fps}
        for key in FRAME_KEYS:
            if all(key in f for f in frames):
                traj[key] = np.stack([f[key] for f in frames], axis=0)
        traj["is_buffer"] = g.is_buffer[np.asarray(path)].copy()
        traj["node_ids"] = np.asarray(path, dtype=np.int64)

        # Enforce global x-y continuity across skill boundaries (see docstring)
        rt = traj["root_trans_offset"]
        for f in range(1, len(rt)):
            jump = rt[f, :2] - rt[f - 1, :2]
            if np.linalg.norm(jump) > max_xy_step:
                rt[f:, :2] -= jump
        return traj
