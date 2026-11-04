"""Remove an isolated final orientation outlier from a non-Buffer reference."""
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation

# MotionLib computes these from poses. Retaining an old filtered derivative
# would preserve contamination from the deleted frame in earlier samples.
CACHED_DERIVATIVES = {
    "dof_vel", "dof_vels", "dof_velocities", "root_vel", "root_ang_vel",
    "body_vel", "body_ang_vel", "global_velocity", "global_angular_velocity",
    "global_root_velocity", "global_root_angular_velocity",
}


def endpoint_statistics(motion, window=15):
    quat = np.asarray(motion["root_rot"])
    count = len(np.asarray(motion["dof"]))
    fps = float(np.asarray(motion["fps"]).item())
    if (count < 4 or quat.shape != (count, 4) or not np.isfinite(quat).all()
            or np.any(np.linalg.norm(quat, axis=-1) < 1e-8)
            or not np.isfinite(fps) or fps <= 0 or window < 2):
        raise ValueError("Expected at least four finite reference frames and positive fps")
    rotations = Rotation.from_quat(quat)
    steps = (rotations[:-1].inv() * rotations[1:]).magnitude()
    recent = np.rad2deg(steps[:-1][-window:])
    last = float(np.rad2deg(steps[-1]))
    median = float(np.median(recent))
    return {
        "frames": count, "fps": fps, "duration_s": (count - 1) / fps,
        "last_rotation_step_deg": last,
        "previous_rotation_step_deg": float(recent[-1]),
        "recent_rotation_step_median_deg": median,
        "last_to_recent_median_ratio": last / max(median, .1),
        "last_max_joint_step_rad": float(np.abs(np.diff(motion["dof"], axis=0)[-1]).max()),
        "last_root_step_m": float(np.linalg.norm(np.diff(motion["root_trans_offset"], axis=0)[-1])),
    }


def repair_terminal_frame(motion, min_jump_deg=15., min_ratio=10.):
    """Trim exactly one outlier frame, preserving every retained pose sample."""
    if (not np.isfinite(min_jump_deg) or min_jump_deg <= 0
            or not np.isfinite(min_ratio) or min_ratio <= 1):
        raise ValueError("Expected positive jump threshold and ratio greater than one")
    if bool(np.any(motion.get("is_buffer", False))):
        raise ValueError("Repair base references first; rebuild generated Buffer transitions")
    before = endpoint_statistics(motion)
    required = (before["last_rotation_step_deg"] > min_jump_deg
                and before["last_to_recent_median_ratio"] > min_ratio)
    result = deepcopy(motion)
    clipped, discarded = [], []
    if required:
        for key, value in list(result.items()):
            if key in CACHED_DERIVATIVES:
                del result[key]
                discarded.append(key)
            elif isinstance(value, np.ndarray) and value.ndim and len(value) == before["frames"]:
                result[key] = value[:-1].copy()
                clipped.append(key)
            elif isinstance(value, (list, tuple)) and len(value) == before["frames"]:
                result[key] = value[:-1]
                clipped.append(key)
        for key in ("dof", "pose_aa", "root_rot", "root_trans_offset", "contact_mask"):
            if key in motion and key not in clipped:
                raise ValueError("Required per-frame field has unexpected length: " + key)
    report = {
        "before": before, "after": endpoint_statistics(result),
        "frames_removed": int(required), "per_frame_fields_trimmed": clipped,
        "cached_derivative_fields_discarded": discarded,
        "thresholds": {"min_jump_deg": min_jump_deg, "min_ratio": min_ratio},
    }
    return result, report
