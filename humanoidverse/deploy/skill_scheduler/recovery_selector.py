"""Yaw/XY-invariant nearest recovery entry lookup (no simulator dependency)."""

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class RecoveryEntry:
    skill_id: int
    frame_idx: int
    node_id: int
    distance: float
    components: Dict[str, float]


def gravity_in_body(quat):
    """Quaternions in PBHC and SciPy are xyzw. Gravity removes heading."""
    return Rotation.from_quat(quat).inv().apply([0., 0., -1.])


class RecoveryEntrySelector:
    def __init__(self, graph, motions, config):
        self.graph = graph
        self.config = config
        ids = config.get("recovery_skill_ids", None)
        if ids is None:
            ids = [i for i, role in enumerate(graph.skill_roles) if role == "recovery"]
        if not ids:
            raise ValueError("Recovery FSM needs recovery-role skills")
        ids = [int(sid) for sid in ids]

        # One reviewed canonical entry per recovery clip. Keep the previous
        # fraction/stride sampler as a compatibility fallback.
        configured_frames = config.get("entry_frames", None)
        if configured_frames is not None:
            configured_frames = {
                int(sid): int(frame) for sid, frame in dict(configured_frames).items()
            }
            missing = sorted(set(ids) - set(configured_frames))
            extra = sorted(set(configured_frames) - set(ids))
            if missing or extra:
                raise ValueError(
                    f"entry_frames must contain exactly recovery_skill_ids; "
                    f"missing={missing}, extra={extra}"
                )
            frames_by_skill = {sid: [configured_frames[sid]] for sid in ids}
        else:
            fraction = float(config.get("entry_fraction", 0.2))
            stride = int(config.get("stride", 2))
            if not 0 < fraction <= 1 or stride < 1:
                raise ValueError("Invalid recovery entry fraction/stride")
            frames_by_skill = None
        # Unit-based scales keep distances reproducible across candidate subsets.
        # A separate admission threshold must be calibrated for this metric.
        self.scales = dict(q=0.5, qdot=2.0, projected_gravity=1.0,
                           root_height=0.2, foot_contact=1.0)
        self.scales.update(config.get("scales", {}))
        # Damping makes velocity a tie-breaker. Orientation and articulated
        # pose identify the get-up mode; foot support and height refine it.
        self.weights = dict(q=1.25, qdot=0.10, projected_gravity=4.0,
                            root_height=0.50, foot_contact=1.50)
        self.weights.update(config.get("weights", {}))
        if set(self.weights) != set(self.scales):
            raise ValueError("Unknown recovery distance feature")
        if any(not np.isfinite(v) or v <= 0 for v in self.scales.values()):
            raise ValueError("Recovery distance scales must be positive")
        if any(not np.isfinite(v) or v < 0 for v in self.weights.values()):
            raise ValueError("Recovery distance weights must be nonnegative")
        self.entries = []
        values = {k: [] for k in self.scales}
        for sid in ids:
            if not 0 <= sid < len(motions) or graph.skill_roles[sid] != "recovery":
                raise ValueError("Recovery candidate must have recovery role")
            m = motions[sid]
            q = np.asarray(m["dof"], dtype=np.float64)
            if len(q) != graph.skill_lengths[sid] or len(q) < 2:
                raise ValueError("Recovery motion length disagrees with graph")
            qdot = np.gradient(q, 1.0 / float(m["fps"]), axis=0)
            grav = Rotation.from_quat(m["root_rot"]).inv().apply(
                np.tile([0., 0., -1.], (len(q), 1)))
            contact = np.asarray(m["contact_mask"], dtype=np.float64)
            if contact.shape != (len(q), 2):
                raise ValueError("Recovery selector requires left/right foot contacts")
            start, _ = graph.skill_range(sid)
            if frames_by_skill is None:
                frames = range(
                    0, min(len(q) - 1, int(np.ceil(len(q) * fraction))), stride
                )
            else:
                frames = frames_by_skill[sid]
            for frame in frames:
                if not 0 <= frame < len(q):
                    raise ValueError(
                        f"Recovery entry frame {frame} outside skill {sid} "
                        f"with {len(q)} frames"
                    )
                self.entries.append((sid, frame, start + frame))
                for key, value in dict(q=q[frame], qdot=qdot[frame],
                        projected_gravity=grav[frame],
                        root_height=np.asarray([m["root_trans_offset"][frame, 2]]),
                        foot_contact=contact[frame]).items():
                    values[key].append(value)
        self.values = {k: np.asarray(v) for k, v in values.items()}
        if not all(np.isfinite(v).all() for v in self.values.values()):
            raise ValueError("Nonfinite recovery reference features")

    def rank(self, live, excluded=()):
        components = {}
        for key, refs in self.values.items():
            value = np.asarray(live[key], dtype=np.float64)
            if value.shape != refs.shape[1:] or not np.isfinite(value).all():
                return []
            components[key] = np.mean(np.abs(refs - value), axis=1) / self.scales[key]
        distances = sum(self.weights[k] * v for k, v in components.items())
        radius = int(self.config.get("exclude_radius_frames", 10))
        ranked = []
        for idx in np.argsort(distances, kind="stable"):
            sid, frame, node = self.entries[idx]
            if any(sid == es and abs(frame - ef) <= radius for es, ef in excluded):
                continue
            ranked.append(RecoveryEntry(sid, frame, node, float(distances[idx]),
                          {k: float(v[idx]) for k, v in components.items()}))
        return ranked
