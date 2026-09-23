# Switch dataset

This directory keeps the source AMASS/CMU motions and the final G1-retargeted
motions used by the Switch project in separate sets.

```text
Switch_data/
├── skills/
│   ├── source/       # original CMU 141 NPZ files
│   └── retargeted/   # G1 23-DoF PKL files
├── skill_graph/          # three-skill graph and transition training data
└── recovery/
    ├── source/       # original CMU 140 NPZ files
    ├── retargeted/   # hip-corrected G1 23-DoF PKL files
    └── retargeted_h_fixed/ # same motions with collision-ground height correction
```

The recovery outputs were copied from
`smpl_retarget/retargeted_motion_data/mink_knee_experiments/bilateral_hip_orientation_07`.
The skill outputs were generated from the selected CMU 141 source files with
the repository's Mink retargeting pipeline. All final files were validated as
30 FPS, 23 DoF, with finite values and the required motion fields.

The `retargeted_h_fixed/` recovery files preserve every field of the
hip-corrected originals except `root_trans_offset[:, 2]`. The height correction
uses the collision geoms of the training G1 model, keeps their signed distance
from the ground plane at least 5 mm, and smooths the time-varying lift. This is
run through the existing Mink retargeting script in height-only mode so the
earlier hip-joint correction is not lost:

```bash
cd smpl_retarget
python mink_retarget/convert_fit_motion.py ../Switch_data/recovery \
    --height-only-input-dir ../Switch_data/recovery/retargeted \
    --target-ground-correct \
    --pkl-output-dir ../Switch_data/recovery/retargeted_h_fixed
cd ..
```

The current manifest and unified graph still use `recovery/retargeted/`; rebuild
them explicitly before training with `retargeted_h_fixed/`.

## Isolated 140_02/140_04 support-foot experiment

`retargeted_h_fixed/` prevents floor penetration but leaves the standing tail
of 140_02 and 140_04 with both feet above the floor. The following script
creates **separate** copies: it lowers the root during the rise until the
lowest foot collision geom is 5 mm above the floor, never lowers another
collision geom through the floor, recomputes the two-foot contact masks from
the corrected motion, and removes 140_04's discontinuous final frame. Joint
angles remain unchanged, so the non-support foot may still be raised.

```bash
python robot_motion_process/fix_recovery_foot_support.py
```

The output is `recovery/retargeted_support_fixed/`, including the two
individual clips and `merged_training.pkl`. It is not part of the current
unified graph. Inspect the two individual clips before training:

```bash
for clip in 02 04; do
    python robot_motion_process/vis_q_mj.py \
        "+motion_file=Switch_data/recovery/retargeted_support_fixed/140_${clip}_poses_retarget.pkl" \
        +robot_xml=description/robots/g1/g1_23dof_lock_wrist_fitmotionONLY.xml \
        +speed=0.5 +vis_contact=True
done
```

For a controlled learnability test, use
`+rewards=motion_tracking/recovery_contact_02_04`: it retains the standard
tracking terms, removes the generic hand/knee collision penalty that conflicts
with getting up, and rewards agreement with the corrected foot contact masks.
This reward configuration is only for this two-clip test, not the task graph.

The follow-up five-clip contact refinement keeps the earlier two-clip dataset
untouched. It levels the four collision spheres under each foot with bounded
ankle pitch/roll IK during the rising phase, lowers the root only where all
robot collision geoms remain above ground, and labels a foot as supporting
only when **all four** sole corners are close to the floor and the foot is
nearly stationary. It removes one discontinuous final frame from each clip.
Hip/knee trajectories are preserved; this is not a full leg retargeting pass.

```bash
python robot_motion_process/refine_recovery_contacts.py

for clip in 01 02 04 08 09; do
    python robot_motion_process/vis_q_mj.py \
        "+motion_file=Switch_data/recovery/retargeted_contact_refined/140_${clip}_poses_retarget.pkl" \
        +robot_xml=description/robots/g1/g1_23dof_lock_wrist_fitmotionONLY.xml \
        +speed=0.5 +vis_contact=True
done
```

This produces `recovery/retargeted_contact_refined/merged_training.pkl` for an
isolated five-clip test and `merged_02_04.pkl` for a direct follow-up to the
earlier two-clip experiment; it does **not** update `unified_graph/`. Use
`+rewards=motion_tracking/recovery_contact_refined` for new training: this adds
joint-angle tracking to the contact reward, since close keypoints can hide a
folded leg. Existing checkpoints were trained with the earlier references and
must be evaluated against those original references for a fair comparison.

## MuJoCo visualization

Run from the repository root after activating the `pbhc` environment:

```bash
for motion_file in Switch_data/skills/retargeted/*.pkl; do
    echo "Inspecting skill: $motion_file"
    python robot_motion_process/vis_q_mj.py "+motion_file=$motion_file" \
        +speed=0.5 +vis_contact=False
done

for motion_file in Switch_data/recovery/retargeted/*.pkl; do
    echo "Inspecting recovery: $motion_file"
    python robot_motion_process/vis_q_mj.py "+motion_file=$motion_file" \
        +speed=0.5 +vis_contact=False
done

for motion_file in Switch_data/recovery/retargeted_h_fixed/*.pkl; do
    echo "Inspecting height-fixed recovery: $motion_file"
    python robot_motion_process/vis_q_mj.py "+motion_file=$motion_file" \
        +robot_xml=description/robots/g1/g1_23dof_lock_wrist_fitmotionONLY.xml \
        +speed=0.5 +vis_contact=False
done
```

Close the current MuJoCo window to advance to the next file. In the viewer,
Space pauses, left/right arrows step frames, `R` resets, `K` slows playback,
`L` speeds it up, and `Q` exits.

## Three-skill graph

The graph contains only `cmu_141_12`, `cmu_141_14`, and `cmu_141_19`;
recovery motions are intentionally excluded. It uses phase-stratified edge
selection with 16 source-time bins and one transition per directed skill pair
per bin, for 96 trained cross-skill transitions in total.

```bash
python SG_build/build_sg.py \
    --files Switch_data/skills/retargeted/141_12_poses_retarget.pkl \
            Switch_data/skills/retargeted/141_14_poses_retarget.pkl \
            Switch_data/skills/retargeted/141_19_poses_retarget.pkl \
    --labels cmu_141_12 cmu_141_14 cmu_141_19 \
    --transition-selection phase \
    --phase-bins 16 \
    --edges-per-bin 1 \
    --max-buffer 30 \
    --output Switch_data/skill_graph
```

The output contains `skill_graph.json` for online scheduling,
`augmented_motions.pkl` for transition-only inspection/training, and
`merged_training.pkl` for joint training of the three skills and all sampled
transitions. `scheduler_config.yaml` contains graph-derived normalization and
A/B grid-search candidates calibrated with `w_p=0`; these candidates still
need policy evaluation before they are treated as final thresholds.

For online graph search, the target set is the union of the skill's opening
window and every trained Buffer landing in that skill. This lets a Buffer
that lands in the middle of a motion terminate the macro directly instead of
forcing a detour through another skill. A new command received while a
transition is active is queued until the current Buffer path lands in its
original target skill; it does not hot-swap and restart the reference clock.
The default live Buffer-entry threshold is `switch_entry_A: 11.0` and must
not exceed `B`.

Buffer frames intentionally use the transition endpoint for policy guidance
and reward computation, while reference-relative termination checks use the
stored interpolated physical frame. Physical reset initialization also uses
the stored trajectory state; RSI currently excludes Buffer frames. Keeping
these references separate prevents a large source-to-endpoint body-height
difference from resetting an environment back to the first Buffer frame
indefinitely.

Every cross-skill edge uses an edge-local SE(2) transform: the destination
entry frame is anchored to the source frame's x-y position and yaw before
Buffer interpolation. Relative destination motion is preserved, including
the displacement of locomotion clips.

## Directed recovery graph

`unified_graph/` combines the three task skills and five recovery clips
without constructing a naive eight-skill all-to-all graph. Skill roles are
stored in `skill_graph.json`. Cross-skill construction permits task-to-task
and recovery-to-task edges only; task-to-recovery and recovery-to-recovery
edges are excluded. A runtime recovery selector is responsible for attaching
the settled physical robot state to a recovery entry.

Recovery exits use the final 20 percent of each recovery clip (excluding the
normal ten-frame boundary), target the first 20 percent of a task, and retain
at most three macros per recovery/task pair. The resulting graph contains 96
task transitions and 45 recovery exits. `merged_training.pkl` tags all 149
entries as one of `task_skill`, `recovery_skill`, `task_transition`, or
`recovery_transition`.

Enable role-balanced sampling when training this dataset:

```bash
robot.motion.motion_file=Switch_data/unified_graph/merged_training.pkl \
robot.motion.role_sampling_enable=true \
robot.motion.recovery_rsi_enable=true
```

The default category masses are 0.40 task skills, 0.35 recovery skills, 0.15
task transitions, and 0.10 recovery exits. They are divided uniformly within
each category, so five recovery files do not automatically outweigh the three
task files. With recovery RSI enabled, 70 percent of recovery resets sample
from the first 20 percent of a recovery clip; the remainder retain full-clip
coverage.

Warm-start the unified policy from the converged three-skill checkpoint. The
configured iteration count is **additional** training after the checkpoint's
stored iteration, rather than an absolute final iteration:

```bash
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
    +simulator=isaacgym \
    +exp=general_tracking \
    +terrain=terrain_locomotion_plane \
    project_name=MotionTracking \
    experiment_name=switch_3skill_5recovery_warmstart \
    num_envs=1024 \
    +obs=motion_tracking/obs_ppo_teacher_kappa \
    +robot=g1/g1_23dof_general \
    +domain_rand=main \
    +rewards=motion_tracking/general_main \
    robot.motion.motion_file=Switch_data/unified_graph/merged_training.pkl \
    robot.motion.role_sampling_enable=true \
    robot.motion.recovery_rsi_enable=true \
    checkpoint=logs/MotionTracking/20260914_144752-switch_3skill_se2_kappa30-motion_tracking-g1_23dof_lock_wrist/model_189000.pt \
    algo.config.num_learning_iterations=30000 \
    algo.config.save_interval=1000 \
    seed=1 \
    +device=cuda:0 \
    headless=True
```

This phase trains exact recovery references and their directed exits. It does
not yet turn ordinary task termination into autonomous recovery; that needs
the runtime settled-state classifier/selector and recovery state machine.

## Contact-refined unified graph

The five contact-refined recovery motions passed per-clip, 3000-step strict
evaluation with the recovery-only checkpoint `model_57000.pt`: every observed
episode of 01/02/04/08/09 reached `motion_end` without a tracking-error
termination. This validates tracking from each clip's defined start, **not**
selection from an arbitrary fallen state or online recovery after a task fall.

`unified_graph_contact_refined/` is a separate rebuild using the same three
task skills and the five contact-refined recovery files. It contains 96
task-to-task and 45 recovery-to-task macros, with 149 role-tagged training
entries. The original `unified_graph/` is unchanged. The combined dataset
retains the refined joint angles, contact masks, and root height; the builder
canonicalizes global XY/yaw when loading each skill.

For unified training, use
`+rewards=motion_tracking/task_recovery_contact_refined`. This keeps the
original non-foot collision penalty for task skills and task transitions,
while allowing hand/knee support and adding contact/joint tracking rewards
only for recovery skills and recovery exits. The five-only reward config would
disable the collision penalty on task skills and is not appropriate here.

For a transition-only PKL, `vis_q_mj.py` supports three Buffer views:

```bash
# Stored/physical interpolation: model and markers both follow stored frames.
python robot_motion_process/vis_q_mj.py \
    +motion_file=/tmp/fixed_transition_000.pkl +buffer_view=stored

# Policy guidance: Buffer frames are replaced by their fixed endpoint target.
python robot_motion_process/vis_q_mj.py \
    +motion_file=/tmp/fixed_transition_000.pkl +buffer_view=guidance

# Recommended: model follows stored interpolation; orange markers remain at
# the fixed endpoint target while is_buffer=True.
python robot_motion_process/vis_q_mj.py \
    +motion_file=/tmp/fixed_transition_000.pkl +buffer_view=dual
```
