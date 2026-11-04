"""Keep G1 calibration poses identical to the poses consumed by MotionLib.

The canonical fields are dof, root_rot (XYZW), and root_trans_offset. Extended
pose_aa entries keep their existing meaning. This helper supports the positive
Cartesian hinge axes used by the project's pose.sum MotionLib implementation.
"""
from copy import deepcopy

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def joint_layout(model):
    joints = np.asarray(model.actuator_trnid[:, 0], dtype=int)
    if (model.nu != 23 or model.nq != 30 or model.nv != 29
            or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
            or not np.array_equal(model.jnt_qposadr[joints], np.arange(7, 30))
            or not np.all(model.jnt_type[joints] == mujoco.mjtJoint.mjJNT_HINGE)):
        raise ValueError("Expected G1's free root and 23 ordered hinge actuators")
    axes = model.jnt_axis[joints]
    if not (np.allclose(axes.sum(-1), 1) and np.allclose(axes.max(-1), 1)
            and np.all(axes >= 0)):
        raise ValueError("MotionLib pose.sum requires positive Cartesian joint axes")
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint))
             for joint in joints]
    return joints, names, axes


def validate_fields(model, motion):
    _, names, axes = joint_layout(model)
    dof = np.asarray(motion["dof"])
    pose = np.asarray(motion["pose_aa"])
    root = np.asarray(motion["root_trans_offset"])
    quat = np.asarray(motion["root_rot"])
    if (dof.ndim != 2 or dof.shape[1] != 23 or len(dof) < 3
            or pose.ndim != 3 or pose.shape[0] != len(dof)
            or pose.shape[1] < 24 or pose.shape[2] != 3
            or root.shape != (len(dof), 3) or quat.shape != (len(dof), 4)):
        raise ValueError("Expected dof(N,23), pose_aa(N,>=24,3), root(N,3), quaternion(N,4)")
    if not all(np.issubdtype(value.dtype, np.floating) and np.isfinite(value).all()
               for value in (dof, pose, root, quat)):
        raise ValueError("Pose fields must be finite floating-point arrays")
    fps = float(np.asarray(motion["fps"]).item())
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Expected positive finite fps")
    if np.any(np.linalg.norm(quat, axis=-1) < 1e-8):
        raise ValueError("Root quaternion has zero length")
    if "joint_names" in motion and list(motion["joint_names"]) != names:
        raise ValueError("Stored joint_names do not match model actuator order")
    return dof, pose, quat, axes


def pose_consistency(model, motion):
    dof, pose, quat, axes = validate_fields(model, motion)
    return {
        "dof_vs_pose_axis_max_error_rad": float(np.abs(pose[:, 1:24] - dof[..., None] * axes).max()),
        "pose_sum_vs_dof_max_error_rad": float(np.abs(pose[:, 1:24].sum(-1) - dof).max()),
        "root_rot_vs_pose_root_max_error_rad": float((
            Rotation.from_quat(quat).inv() * Rotation.from_rotvec(pose[:, 0])).magnitude().max()),
        "root_quaternion_norm_max_error": float(np.abs(np.linalg.norm(quat, axis=-1) - 1).max()),
    }


def require_consistent_pose(model, motion, tolerance=1e-5):
    report = pose_consistency(model, motion)
    failed = {key: value for key, value in report.items() if value > tolerance}
    if failed:
        raise ValueError("Calibration/runtime pose mismatch; synchronize from dof/root_rot first: " + str(failed))
    return report


def synchronize_pose(model, motion, *, copy_motion=True):
    """Rebuild root and all 23 joint entries; preserve extended pose entries."""
    dof, pose, quat, axes = validate_fields(model, motion)
    result = deepcopy(motion) if copy_motion else motion
    normalized = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    result["root_rot"] = normalized.astype(quat.dtype)
    result["pose_aa"][:, 0] = Rotation.from_quat(normalized).as_rotvec().astype(pose.dtype)
    result["pose_aa"][:, 1:24] = (dof[..., None] * axes).astype(pose.dtype)
    require_consistent_pose(model, result)
    return result
