"""Build isolated, collision-checked recovery clips with level support feet.

The height-only clips can put a toe on the floor while leaving the heel many
centimetres high.  This experiment levels each foot with bounded ankle IK
during the standing phase, then lowers the root to bring the lowest foot to
the floor.  It never changes hip/knee joints or the source files.  A contact
mask is true only for a near-flat, nearly stationary foot.
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, median_filter
from scipy.optimize import least_squares

if __package__:
    from .fix_recovery_foot_support import geometry_heights
    from .reference_pose import joint_layout, pose_consistency, synchronize_pose
else:
    from fix_recovery_foot_support import geometry_heights
    from reference_pose import joint_layout, pose_consistency, synchronize_pose


# Frame windows are limited to the rising/standing portion.  A lying or
# rolling foot must not be forced flat against the floor.
SPECS = {
    "140_01": (115, 150, 1),
    "140_02": (116, 150, 1),
    "140_04": (140, 185, 1),
    "140_08": (115, 165, 1),
    "140_09": (105, 150, 1),
}


def foot_model_ids(model):
    joints, geoms = [], []
    for side in ("left", "right"):
        names = [f"{side}_ankle_{axis}_joint" for axis in ("pitch", "roll")]
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                     for name in names]
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link"
        )
        support_ids = [
            geom for geom in range(model.ngeom)
            if model.geom_bodyid[geom] == body_id
            and model.geom_type[geom] in (
                mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE)
            and (model.geom_contype[geom] or model.geom_conaffinity[geom])
        ]
        collision_ids = [geom for geom in range(model.ngeom)
                         if model.geom_bodyid[geom] == body_id
                         and (model.geom_contype[geom] or model.geom_conaffinity[geom])]
        if not support_ids or support_ids != collision_ids or any(j < 0 for j in joint_ids):
            raise ValueError(f"Expected sphere/capsule foot collision geoms and two ankle joints on {side}")
        joints.append(joint_ids)
        geoms.append(support_ids)
    return joints, geoms


def foot_corner_heights(model, data, geom_ids):
    """Sole support heights: sphere bottoms or both capsule endpoint bottoms.

    MuJoCo stores capsule half-length along each geom's local Z axis. This
    preserves the four-sphere calibration and supports the physical rev_2
    model's seven capsules without substituting simplified collision shapes.
    """
    heights = []
    for geom in geom_ids:
        bottom = data.geom_xpos[geom, 2] - model.geom_size[geom, 0]
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            dz = data.geom_xmat[geom].reshape(3, 3)[2, 2] * model.geom_size[geom, 1]
            heights.extend((bottom - dz, bottom + dz))
        else:
            heights.append(bottom)
    return np.asarray(heights)


def smoothstep(frame, start, full):
    phase = np.clip((frame - start) / (full - start), 0.0, 1.0)
    return phase * phase * (3.0 - 2.0 * phase)


def refine_motion(model, source, spec, clearance=0.005, repair_ground_penetration=False):
    start, full, trim_last = spec
    length = len(source["dof"])
    motion = {
        key: (value[:-trim_last].copy() if trim_last and isinstance(value, np.ndarray)
              and value.ndim and len(value) == length else value.copy()
              if isinstance(value, np.ndarray) else value)
        for key, value in source.items()
    }
    # Canonicalize before any calibration, including the root and non-ankle
    # joints. Calibration must never consume a different pose from MotionLib.
    motion = synchronize_pose(model, motion, copy_motion=False)
    joints, support_geoms = foot_model_ids(model)
    data = mujoco.MjData(model)
    n = len(motion["dof"])
    joint_qpos = [[model.jnt_qposadr[j] for j in pair] for pair in joints]
    joint_dof = [[model.jnt_dofadr[j] - 6 for j in pair] for pair in joints]

    # Equalize sole support heights with ankle pitch/roll only. Capsule
    # endpoints also capture a tilted toe/heel; the tiny radius differences
    # in rev_2 remain part of the least-squares residual.
    for frame in range(n):
        weight = smoothstep(frame, start, full)
        if weight == 0.0:
            continue
        data.qpos[:3] = motion["root_trans_offset"][frame]
        data.qpos[3:7] = np.roll(motion["root_rot"][frame], 1)
        data.qpos[7:] = motion["dof"][frame]
        for side in range(2):
            qpos = joint_qpos[side]
            original = data.qpos[qpos].copy()
            joint_ids = joints[side]
            lower = model.jnt_range[joint_ids, 0] + 1e-5
            upper = model.jnt_range[joint_ids, 1] - 1e-5

            def residual(ankle):
                data.qpos[qpos] = ankle
                mujoco.mj_forward(model, data)
                heights = foot_corner_heights(model, data, support_geoms[side])
                return np.r_[(heights - heights.mean()) * 10,
                             (ankle - original) * 0.15]

            result = least_squares(
                residual, np.clip(original, lower, upper),
                bounds=(lower, upper), max_nfev=30,
            )
            corrected = original + weight * (result.x - original)
            data.qpos[qpos] = corrected
            motion["dof"][frame, joint_dof[side]] = corrected

    motion = synchronize_pose(model, motion, copy_motion=False)

    # Recompute geometry after ankle IK.  Root-only lowering is limited by
    # the lowest *whole-body* collision geom, including hand/knee supports.
    all_before, feet_before, _ = geometry_heights(model, motion)
    if repair_ground_penetration:
        # A fitmotion-calibrated source can still penetrate the physical
        # model at its hands. Smooth a conservative lift envelope, then bound
        # it from below by the exact per-frame minimum required clearance.
        required = np.maximum(clearance - all_before, 0.0)
        envelope = maximum_filter1d(required, size=9, mode="nearest")
        lift = np.maximum(required, gaussian_filter1d(envelope, sigma=2.0, mode="nearest"))
        motion["root_trans_offset"][:, 2] += lift.astype(motion["root_trans_offset"].dtype)
        all_before, feet_before, _ = geometry_heights(model, motion)
    phase = smoothstep(np.arange(n), start, full)
    desired_drop = np.maximum(np.min(feet_before, axis=1) - clearance, 0.0)
    safe_drop = np.maximum(all_before - clearance, 0.0)
    raw_drop = np.minimum(phase * desired_drop, safe_drop)
    drop = gaussian_filter1d(raw_drop, sigma=2.0, mode="nearest")
    drop = np.clip(drop, 0.0, safe_drop)
    motion["root_trans_offset"][:, 2] -= drop.astype(
        motion["root_trans_offset"].dtype
    )
    all_after, _, foot_xy = geometry_heights(model, motion)
    if all_after.min() < -0.001:
        raise AssertionError(f"Ground penetration: {all_after.min():.4f} m")

    mujoco.mj_forward(model, data)
    sizes = [len(foot_corner_heights(model, data, ids)) for ids in support_geoms]
    if sizes[0] != sizes[1]:
        raise ValueError("Left and right feet must have equal support sample counts")
    corners = np.empty((n, 2, sizes[0]))
    for frame in range(n):
        data.qpos[:3] = motion["root_trans_offset"][frame]
        data.qpos[3:7] = np.roll(motion["root_rot"][frame], 1)
        data.qpos[7:] = motion["dof"][frame]
        mujoco.mj_forward(model, data)
        for side in range(2):
            corners[frame, side] = foot_corner_heights(model, data, support_geoms[side])

    fps = float(np.asarray(motion["fps"]).item())
    speed = np.zeros((n, 2))
    speed[1:] = np.linalg.norm(np.diff(foot_xy, axis=0), axis=-1) * fps
    speed[0] = speed[1]
    flat_contact = (corners.max(axis=-1) <= 0.025) & (speed <= 0.3)
    motion["contact_mask"] = median_filter(
        flat_contact.astype(np.float32), size=(3, 1), mode="nearest"
    )
    return motion, all_after, corners, drop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=Path("Switch_data/recovery/retargeted_h_fixed"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("Switch_data/recovery/retargeted_contact_refined"))
    parser.add_argument("--model", type=Path,
                        default=Path("description/robots/g1/g1_23dof_lock_wrist_fitmotionONLY.xml"))
    parser.add_argument("--repair-ground-penetration", action="store_true",
                        help="Lift a copied reference where the selected physical model penetrates the floor")
    args = parser.parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("Input and output directories must differ")
    model = mujoco.MjModel.from_xml_path(str(args.model))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged = {}
    provenance = {"model": str(args.model.resolve()),
                  "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  "repair_ground_penetration": args.repair_ground_penetration,
                  "clearance_m": 0.005, "clips": {}}
    _, joint_names, joint_axes = joint_layout(model)
    provenance["canonical_pose_fields"] = ["dof", "root_rot", "root_trans_offset"]
    provenance["joint_names"] = joint_names
    provenance["joint_axes"] = joint_axes.tolist()
    for clip_id, spec in SPECS.items():
        filename = f"{clip_id}_poses_retarget.pkl"
        with (args.input_dir / filename).open("rb") as stream:
            wrapper = pickle.load(stream)
        if not isinstance(wrapper, dict) or len(wrapper) != 1:
            raise ValueError(f"Expected one motion in {filename}")
        key, source = next(iter(wrapper.items()))
        motion, all_height, corners, drop = refine_motion(
            model, source, spec, repair_ground_penetration=args.repair_ground_penetration)
        with (args.output_dir / filename).open("wb") as stream:
            pickle.dump({key: motion}, stream, protocol=pickle.HIGHEST_PROTOCOL)
        tagged = dict(motion, skill_role="recovery", skill_label=f"cmu_{clip_id}")
        merged[f"skill_cmu_{clip_id}"] = tagged
        provenance["clips"][clip_id] = {
            "source": str((args.input_dir / filename).resolve()),
            "source_sha256": hashlib.sha256((args.input_dir / filename).read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256((args.output_dir / filename).read_bytes()).hexdigest(),
            "refinement_frames_start_full_trim": list(spec),
            "pose_consistency": pose_consistency(model, motion),
            "minimum_floor_distance_m": float(all_height.min()),
            "maximum_net_root_height_change_m": float(np.abs(
                motion["root_trans_offset"][:, 2] - source["root_trans_offset"][:len(motion["dof"]), 2]).max()),
        }
        tail = slice(int(0.8 * len(corners)), None)
        print(
            f"{clip_id}: {len(motion['dof'])} frames; lowest whole-body "
            f"height {all_height.min():.3f} m; tail sole-corner maximum median "
            f"{np.median(corners[tail].max(axis=-1), axis=0).round(3)} m; "
            f"tail contact fraction {motion['contact_mask'][tail].mean(axis=0).round(2)}; "
            f"maximum root lowering {drop.max():.3f} m"
        )
    with (args.output_dir / "merged_training.pkl").open("wb") as stream:
        pickle.dump(merged, stream, protocol=pickle.HIGHEST_PROTOCOL)
    subset = {key: value for key, value in merged.items()
              if key in {"skill_cmu_140_02", "skill_cmu_140_04"}}
    with (args.output_dir / "merged_02_04.pkl").open("wb") as stream:
        pickle.dump(subset, stream, protocol=pickle.HIGHEST_PROTOCOL)
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
