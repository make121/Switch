"""Geometry and bounded IK helpers for preparing trainable G1 references."""
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


def collision_pairs(model):
    geoms = [i for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]
    excluded = set(int(x) for x in model.exclude_signature)
    pairs = []
    for offset, first in enumerate(geoms):
        for second in geoms[offset + 1:]:
            a, b = int(model.body_weldid[model.geom_bodyid[first]]), int(model.body_weldid[model.geom_bodyid[second]])
            if not a or not b or a == b:
                continue
            parent_a = int(model.body_weldid[model.body_parentid[a]])
            parent_b = int(model.body_weldid[model.body_parentid[b]])
            if parent_a == b or parent_b == a or ((min(a, b) << 16) + max(a, b)) in excluded:
                continue
            if (model.geom_contype[first] & model.geom_conaffinity[second]
                    or model.geom_contype[second] & model.geom_conaffinity[first]):
                pairs.append((first, second))
    pairs += [tuple(sorted((int(a), int(b)))) for a, b in zip(model.pair_geom1, model.pair_geom2)
              if model.geom_bodyid[a] and model.geom_bodyid[b]]
    return sorted(set(pairs))


def distance_and_jacobian(model, data, pair, dofs):
    points = np.zeros(6)
    distance = float(mujoco.mj_geomDistance(model, data, *pair, .2, points))
    if distance >= .199999 or abs(distance) < 1e-9:
        return distance, np.zeros(len(dofs))
    normal = (points[3:] - points[:3]) / distance
    first, second = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, first, None, points[:3], int(model.geom_bodyid[pair[0]]))
    mujoco.mj_jac(model, data, second, None, points[3:], int(model.geom_bodyid[pair[1]]))
    return distance, normal @ (second - first)[:, dofs]


class UpperBodyCollisionProjector:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.joints = model.actuator_trnid[:, 0].astype(int)
        self.indices = np.arange(12, 23)
        self.qpos = model.jnt_qposadr[self.joints[self.indices]]
        self.dofs = model.jnt_dofadr[self.joints[self.indices]]
        arm_bodies = set()
        for joint in self.joints[self.indices]:
            arm_bodies.add(int(model.body_weldid[model.jnt_bodyid[joint]]))
        self.pairs = [pair for pair in collision_pairs(model) if any(
            int(model.body_weldid[model.geom_bodyid[g]]) in arm_bodies for g in pair)]
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.arm_geoms = sorted({g for pair in self.pairs for g in pair if
            int(model.body_weldid[model.geom_bodyid[g]]) in arm_bodies})
        self.floor_pairs = [(self.floor, geom) for geom in self.arm_geoms]
        self.bounds = (model.jnt_range[self.joints[self.indices], 0] + 1e-5,
                       model.jnt_range[self.joints[self.indices], 1] - 1e-5)

    def project(self, root, quaternion_wxyz, dof, previous_delta=None, clearance=.004):
        model, data = self.model, self.data
        data.qpos[:3], data.qpos[3:7] = root, quaternion_wxyz
        data.qpos[model.jnt_qposadr[self.joints]] = dof
        mujoco.mj_kinematics(model, data); mujoco.mj_comPos(model, data)
        reference = np.asarray(dof)[self.indices].copy()
        width = len(self.indices)
        previous_delta = np.zeros(width) if previous_delta is None else previous_delta
        tracking = np.r_[np.full(3, .4), np.full(8, .06)]
        smoothing = np.r_[np.full(3, .3), np.full(8, .10)]
        # Preserve the heights of hands already supporting the body, allowing
        # the lateral movement needed to separate a hand from a penetrating leg.
        hands = [g for g in self.arm_geoms if "hand" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
        anchors = []
        for geom in hands:
            height = distance_and_jacobian(model, data, (self.floor, geom), self.dofs)[0]
            if height < .03:
                anchors.append((geom, height))
        cache = {}

        def evaluate(value):
            if "value" in cache and np.array_equal(cache["value"], value):
                return cache["residual"], cache["jacobian"]
            data.qpos[self.qpos] = value
            mujoco.mj_kinematics(model, data); mujoco.mj_comPos(model, data)
            residuals = [*(tracking * (value - reference)), *(smoothing * (value - reference - previous_delta))]
            jac = [*np.diag(tracking), *np.diag(smoothing)]
            for pair, margin, weight in ([(pair, clearance, 35.) for pair in self.pairs]
                                         + [(pair, .005, 50.) for pair in self.floor_pairs]):
                distance, gradient = distance_and_jacobian(model, data, pair, self.dofs)
                residuals.append(weight * max(margin - distance, 0.))
                jac.append(-weight * gradient if distance < margin else np.zeros(width))
            for geom, target in anchors:
                distance, gradient = distance_and_jacobian(model, data, (self.floor, geom), self.dofs)
                residuals.append(8 * (distance - target)); jac.append(8 * gradient)
            cache.update(value=value.copy(), residual=np.asarray(residuals), jacobian=np.asarray(jac))
            return cache["residual"], cache["jacobian"]

        result = least_squares(lambda value: evaluate(value)[0], np.clip(reference + previous_delta, *self.bounds),
                               jac=lambda value: evaluate(value)[1], bounds=self.bounds, max_nfev=70,
                               ftol=1e-8, xtol=1e-8, gtol=1e-7)
        corrected = np.asarray(dof).copy(); corrected[self.indices] = result.x
        data.qpos[self.qpos] = result.x
        mujoco.mj_forward(model, data)
        distances = [float(c.dist) for c in data.contact if self.floor not in (c.geom1, c.geom2)]
        return corrected, {"minimum_self_distance_m": min(distances, default=0.),
                           "maximum_upper_body_change_rad": float(np.abs(result.x - reference).max()),
                           "maximum_waist_change_rad": float(np.abs(result.x[:3] - reference[:3]).max()),
                           "evaluations": result.nfev, "cost": float(result.cost)}


def append_standing_exit(model, motion, default_dof, transition_seconds=3., hold_seconds=3.):
    """Append a feet-constrained return to upright, then a stationary hold."""
    data = mujoco.MjData(model)
    joints = model.actuator_trnid[:, 0].astype(int)
    addresses = model.jnt_qposadr[joints]
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side + "_ankle_roll_link")
            for side in ("left", "right")]
    source_dof = np.asarray(motion["dof"][-1], dtype=float)
    source_root = np.asarray(motion["root_trans_offset"][-1], dtype=float)
    source_rot = Rotation.from_quat(motion["root_rot"][-1])
    yaw = float(source_rot.as_euler("xyz")[2])
    upright = Rotation.from_euler("z", yaw)
    data.qpos[:3], data.qpos[3:7], data.qpos[addresses] = source_root, np.roll(source_rot.as_quat(), 1), source_dof
    mujoco.mj_forward(model, data)
    initial_foot_pos = data.xpos[feet].copy()
    initial_foot_rot = Rotation.from_matrix(data.xmat[feet].reshape(2, 3, 3))
    final_foot_rot = [Rotation.from_euler("z", r.as_euler("xyz")[2]) for r in initial_foot_rot]
    foot_slerps = [Slerp([0, 1], Rotation.concatenate([initial_foot_rot[i], final_foot_rot[i]])) for i in range(2)]
    root_slerp = Slerp([0, 1], Rotation.concatenate([source_rot, upright]))
    root_body = int(model.jnt_bodyid[0])
    # Establish a comfortable reference height/COM offset with the same model.
    data.qpos[:3], data.qpos[3:7], data.qpos[addresses] = 0, np.roll(upright.as_quat(), 1), default_dof
    mujoco.mj_forward(model, data)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    active = [g for g in range(model.ngeom) if g != floor and (model.geom_contype[g] or model.geom_conaffinity[g])]
    lowest = min(mujoco.mj_geomDistance(model, data, floor, g, 10., None) for g in active)
    desired_com = initial_foot_pos[:, :2].mean(0) + upright.apply([.02, 0, 0])[:2]
    target_root = np.r_[desired_com - data.subtree_com[root_body, :2], .005 - lowest]
    fps = float(motion["fps"])
    steps = round(transition_seconds * fps)
    hold = round(hold_seconds * fps)
    lower = np.r_[source_root[:2] - .4, .35, model.jnt_range[joints[:12], 0] + 1e-4]
    upper = np.r_[source_root[:2] + .4, 1.1, model.jnt_range[joints[:12], 1] - 1e-4]
    value = np.r_[source_root, source_dof[:12]]
    records, errors = [], []
    for step in range(1, steps + 1):
        phase = step / steps
        blend = phase ** 3 * (10 - 15 * phase + 6 * phase ** 2)
        quaternion = root_slerp(blend).as_quat()
        current = source_dof * (1 - blend) + np.asarray(default_dof) * blend
        goal_root = source_root * (1 - blend) + target_root * blend
        desired_rot = [slerp(blend) for slerp in foot_slerps]
        desired_pos = initial_foot_pos.copy()
        for side, body in enumerate(feet):
            offsets = []
            for geom in active:
                if int(model.geom_bodyid[geom]) != body:
                    continue
                radius, half = model.geom_size[geom, :2]
                axis = Rotation.from_quat(model.geom_quat[geom][[1, 2, 3, 0]]).as_matrix()[:, 2]
                offsets += [desired_rot[side].apply(model.geom_pos[geom] + sign * half * axis)[2] - radius for sign in (-1, 1)]
            height = .005 - min(offsets)
            desired_pos[side, 2] = initial_foot_pos[side, 2] * (1 - blend) + height * blend

        def residual(candidate):
            data.qpos[:3], data.qpos[3:7] = candidate[:3], quaternion[[3, 0, 1, 2]]
            data.qpos[addresses] = current
            data.qpos[addresses[:12]] = candidate[3:]
            mujoco.mj_kinematics(model, data); mujoco.mj_comPos(model, data)
            rotation_error = np.concatenate([(desired_rot[i].inv() * Rotation.from_matrix(
                data.xmat[body].reshape(3, 3))).as_rotvec() for i, body in enumerate(feet)])
            return np.r_[300 * (data.xpos[feet] - desired_pos).ravel(), 100 * rotation_error,
                         .1 * (candidate[3:] - current[:12]), 3 * (candidate[:3] - goal_root),
                         25 * (data.subtree_com[root_body, :2] - desired_com) * blend]

        result = least_squares(residual, np.clip(value, lower, upper), bounds=(lower, upper), max_nfev=45,
                               ftol=1e-9, xtol=1e-9, gtol=1e-8)
        value = result.x
        residual(value)
        current[:12] = value[3:]
        errors.append(float(np.linalg.norm(data.xpos[feet] - desired_pos, axis=-1).max()))
        records.append((value[:3].copy(), quaternion.copy(), current.copy()))
    records += [records[-1]] * hold
    result = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in motion.items()}
    for key, column in [("root_trans_offset", 0), ("root_rot", 1), ("dof", 2)]:
        result[key] = np.concatenate([motion[key], np.array([record[column] for record in records], dtype=motion[key].dtype)])
    result["pose_aa"] = np.concatenate([motion["pose_aa"], np.repeat(motion["pose_aa"][-1:], len(records), axis=0)])
    result["contact_mask"] = np.concatenate([motion["contact_mask"], np.ones((len(records), 2), dtype=np.float32)])
    return result, {"transition_seconds": transition_seconds, "hold_seconds": hold_seconds,
                    "transition_start_frame": len(motion["dof"]), "hold_start_frame": len(motion["dof"]) + steps,
                    "maximum_foot_position_residual_m": max(errors)}
