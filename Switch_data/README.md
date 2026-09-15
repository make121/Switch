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
    └── retargeted/   # corrected G1 23-DoF PKL files
```

The recovery outputs were copied from
`smpl_retarget/retargeted_motion_data/mink_knee_experiments/bilateral_hip_orientation_07`.
The skill outputs were generated from the selected CMU 141 source files with
the repository's Mink retargeting pipeline. All final files were validated as
30 FPS, 23 DoF, with finite values and the required motion fields.

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
