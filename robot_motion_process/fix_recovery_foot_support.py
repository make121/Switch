"""Prepare a two-clip, contact-aware recovery learnability experiment.

Start from the hip-corrected and non-penetrating recovery PKLs.  The existing
height fix only lifted the robot out of the floor; it could not bring a
floating support foot down.  During the observed stand-up phase, lower the
root until the lowest foot collision geom touches the floor, without allowing
any robot collision geom to penetrate it.  The joint angles are unchanged.

This is a narrowly scoped diagnostic correction for CMU 140_02 and 140_04,
not a general multi-contact retargeter.
"""

import argparse
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter


SPECS = {
    "140_02": {"start": 116, "full": 140, "trim_last": 0},
    "140_04": {"start": 150, "full": 200, "trim_last": 1},
}


def geometry_heights(model, motion):
    """Return signed floor distances of all collision geoms and both feet."""
    data = mujoco.MjData(model)
    plane = next(
        (g for g in range(model.ngeom)
         if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE), None
    )
    if plane is None:
        raise ValueError("Robot model has no ground plane")
    collision = [
        g for g in range(model.ngeom)
        if g != plane and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]
    foot_bodies = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                          f"{side}_ankle_roll_link")
        for side in ("left", "right")
    ]
    feet = [[g for g in collision if model.geom_bodyid[g] == body]
            for body in foot_bodies]
    if any(not geoms for geoms in feet):
        raise ValueError("Robot model is missing foot collision geoms")
    n = len(motion["dof"])
    if model.nq != 7 + motion["dof"].shape[1]:
        raise ValueError("Robot model and motion have different DoF counts")
    all_height = np.empty(n)
    foot_height = np.empty((n, 2))
    foot_xy = np.empty((n, 2, 2))
    fromto = np.empty(6)
    for t in range(n):
        data.qpos[:3] = motion["root_trans_offset"][t]
        data.qpos[3:7] = np.roll(motion["root_rot"][t], 1)
        data.qpos[7:] = motion["dof"][t]
        mujoco.mj_forward(model, data)
        distances = {
            g: mujoco.mj_geomDistance(model, data, plane, g, 10.0, fromto)
            for g in collision
        }
        all_height[t] = min(distances.values())
        for side in range(2):
            foot_height[t, side] = min(distances[g] for g in feet[side])
            foot_xy[t, side] = data.xpos[foot_bodies[side], :2]
    return all_height, foot_height, foot_xy


def fix_motion(model, motion, spec, clearance=0.005):
    """Return a corrected copy and pre/post geometry diagnostics."""
    original_length = len(motion["dof"])
    trim_last = spec["trim_last"]
    if trim_last:
        motion = {
            key: (value[:-trim_last].copy()
                  if isinstance(value, np.ndarray)
                  and value.ndim > 0 and len(value) == original_length
                  else value)
            for key, value in motion.items()
        }
    else:
        motion = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in motion.items()
        }

    all_before, feet_before, _ = geometry_heights(model, motion)
    n = len(motion["dof"])
    frame = np.arange(n, dtype=np.float64)
    phase = np.clip((frame - spec["start"]) /
                    (spec["full"] - spec["start"]), 0.0, 1.0)
    weight = phase * phase * (3.0 - 2.0 * phase)

    desired_drop = np.maximum(np.min(feet_before, axis=1) - clearance, 0.0)
    safe_drop = np.maximum(all_before - clearance, 0.0)
    raw_drop = np.minimum(weight * desired_drop, safe_drop)
    drop = gaussian_filter1d(raw_drop, sigma=2.0, mode="nearest")
    drop = np.minimum(np.maximum(drop, 0.0), safe_drop)
    motion["root_trans_offset"][:, 2] -= drop.astype(
        motion["root_trans_offset"].dtype, copy=False
    )

    all_after, feet_after, foot_xy = geometry_heights(model, motion)
    if np.min(all_after) < -0.001:
        raise AssertionError(f"Ground penetration after correction: {all_after.min()}")

    fps = float(np.asarray(motion["fps"]).item())
    foot_speed = np.zeros((n, 2))
    foot_speed[1:] = np.linalg.norm(np.diff(foot_xy, axis=0), axis=-1) * fps
    foot_speed[0] = foot_speed[1]
    foot_contact = (feet_after <= 0.025) & (foot_speed <= 0.3)
    motion["contact_mask"] = median_filter(
        foot_contact.astype(np.float32), size=(3, 1), mode="nearest"
    )
    return motion, all_before, feet_before, all_after, feet_after, drop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("Switch_data/recovery/retargeted_h_fixed"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("Switch_data/recovery/retargeted_support_fixed"),
    )
    parser.add_argument(
        "--model", type=Path,
        default=Path("description/robots/g1/g1_23dof_lock_wrist_fitmotionONLY.xml"),
    )
    args = parser.parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("Input and output directories must differ")
    model = mujoco.MjModel.from_xml_path(str(args.model))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged = {}
    for clip_id, spec in SPECS.items():
        name = f"{clip_id}_poses_retarget.pkl"
        with (args.input_dir / name).open("rb") as stream:
            wrapper = pickle.load(stream)
        if not isinstance(wrapper, dict) or len(wrapper) != 1:
            raise ValueError(f"Expected one motion in {name}")
        key, motion = next(iter(wrapper.items()))
        fixed, all_before, feet_before, all_after, feet_after, drop = fix_motion(
            model, motion, spec
        )
        with (args.output_dir / name).open("wb") as stream:
            pickle.dump({key: fixed}, stream, protocol=pickle.HIGHEST_PROTOCOL)
        tagged = dict(fixed)
        tagged["skill_role"] = "recovery"
        tagged["skill_label"] = f"cmu_{clip_id}"
        merged[f"skill_cmu_{clip_id}"] = tagged
        tail = slice(int(0.8 * len(feet_after)), None)
        print(
            f"{clip_id}: {len(fixed['dof'])} frames, "
            f"minimum collision z {all_before.min():.3f} -> {all_after.min():.3f} m, "
            f"last-20% support-foot median z "
            f"{np.median(np.min(feet_before[tail], axis=1)):.3f} -> "
            f"{np.median(np.min(feet_after[tail], axis=1)):.3f} m, "
            f"maximum downward correction {drop.max():.3f} m"
        )
    with (args.output_dir / "merged_training.pkl").open("wb") as stream:
        pickle.dump(merged, stream, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
