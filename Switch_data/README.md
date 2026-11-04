# Switch dataset

## 本次基础功能发布：技能切换与倒地恢复尝试

本次以 `auto` 分支已有的 `unified_graph_contact_refined/` 为运行基线，
补齐三条任务动作、恢复 FSM、仿真启动脚本及回归测试。对应运行资产可用
`sha256sum -c Switch_data/CHECKSUMS.sha256` 核对。
后续章节保留研究记录；十月修复图谱、七恢复扩展和专项诊断工具不属于本次基础发布。

按根目录 `INSTALL.md` 安装 IsaacGym/PBHC 并激活对应 Python 环境后，提供已经训练的
Switch 教师检查点，以及放在检查点同目录（或上一级目录）的原始 `config.yaml`。
模型应与现有三任务、五恢复和 κ 观测配置兼容；本次不附带策略权重。

```bash
# 一个兼容的统一策略：任务切换和恢复均使用同一模型。
bash scripts/eval_recovery_fsm.sh /path/to/task_run/model_241000.pt

# 两个兼容模型：任务/Buffer 使用统一策略，恢复本体使用专门起身策略。
SWITCH_TASK_CHECKPOINT=/path/to/task_run/model_241000.pt \
SWITCH_RECOVERY_CHECKPOINT=/path/to/recovery_run/model_115000.pt \
  bash scripts/eval_real_fall_recovery.sh
```

界面中按 `7/8/9` 请求三个任务技能；按 `t` 施加真实仿真 XY 速度扰动，
由原有终止条件触发阻尼、落稳检查、入口选择和最多两次起身尝试。
`r` 仅诊断触发 FSM，不施加扰动。恢复中收到的新任务命令用于更新返回目标。
脚本默认不导出 ONNX；需要时追加 `++export_onnx=true`。

无界面自动推倒示例（使用上述两个检查点环境变量）：

```bash
bash scripts/eval_real_fall_recovery.sh \
  ++headless=true ++eval_steps=1500 +eval_seed=1 \
  ++algo.config.eval_callbacks.skill_scheduler.config.recovery_fsm.fall_push_step=122 \
  '++algo.config.eval_callbacks.skill_scheduler.config.recovery_fsm.fall_push_velocity_xy=[3.0,0.0]'
```

无界面技能切换可追加
`+algo.config.eval_callbacks.skill_scheduler.config.auto_switch_interval_steps=250`
及 `'+algo.config.eval_callbacks.skill_scheduler.config.auto_switch_skills=[1,2,0]'`。
每次执行的 `eval_log_dir` 保存 `recovery_events.jsonl` 和 `recovery_metrics.json`；
可用 `python scripts/summarize_real_fall_recovery.py EVENTS_FILE` 汇总推倒与恢复尝试。
起身尝试失败后的硬重置会独立记录，不能计作起身成功。

离线回归入口：

```bash
python -m unittest \
  humanoidverse.deploy.skill_scheduler.tests.test_recovery_fsm \
  humanoidverse.deploy.skill_scheduler.tests.test_recovery_policy_routing \
  humanoidverse.deploy.skill_scheduler.tests.test_scheduler_reset \
  humanoidverse.agents.ppo.tests.test_checkpoint_resume
python humanoidverse/deploy/skill_scheduler/tests/test_reference_builder.py
```

2026-11-04 在只含本次发布文件及 `auto` 基线的干净副本中，使用上述
241000 任务模型与 115000 起身模型完成了以下检查：

- 31 项 CPU 回归、7 项参考轨迹检查和 13 个动作/图谱文件的 SHA256 校验通过。
- 650 步 IsaacGym 自动切换中，两次任务切换经过训练过的 Buffer，未发生终止或物理重置。
- seed=1、1500 步、step=122 施加 `[3,0]` m/s 扰动：倒地后进入阻尼与入口选择，
  使用 `140_04` 起身并经 Buffer 返回任务，step=662 完成成功保持判定；该恢复过程
  没有硬重置。之后正常任务播放到末尾的循环复位仍保留。

这验证上述模型和单次扰动条件下的基础链路，不代表任意倒地状态的恢复成功率。

2026-10-29 扩展图谱已完成从零 10k 阶段，现按用户要求追加 90k 至累计 100k，
延续模型、两套优化器及计数，原五条组保持停止。详见 [扩展图谱 100k 续训记录](../docs/graph7_scratch_to100k_20261029.md)。

2026-10-28 已按用户要求停止原图谱训练，保留 100k 及最新已保存的 117.5k 检查点；
扩展图谱改为从零训练 10k，原七条 300k 排队作业取消。
详见 [扩展图谱 10k 训练记录](../docs/graph7_scratch_10k_20261028.md)。

2026-10-27 已启动五条/七条完整图谱从零等预算训练，两组各 300k 次 PPO 更新。
随机初始模型配对验证和两组短训练检查通过；五条正式训练先运行，七条随后自动启动。
详见 [从零训练记录与进度命令](../docs/graph5_vs7_scratch_training_20261027.md)。

2026-10-27 已在 Gym 完成独立任务倒地快照对比：冻结同一批 96 个状态，
最近邻入口一次起身成功数为五条 38/96、七条 31/96，当前未见扩集净改善。
新增 getup2 补到 2 个独有俯卧状态；状态恢复、重复回放和实验范围见
[独立倒地比较报告](../docs/gym_paired_falls_20261027.md)。

2026-10-27 两图谱各追加 10k 后的策略验收已完成：扩展策略七条起身在 Gym/MuJoCo 均通过，
原图谱 `140_01` 的 MuJoCo 起身和扩展图谱部分任务切换/返回任务链路尚未通过。
详见 [闭环验收结果](../docs/graph_policy_readiness_20261027.md)；当前未开始独立真实倒地快照对比。

2026-10-26 已启动原图谱和七条恢复扩展图谱各追加 10k 的联合训练，
两组短训练检查均通过，正式训练串行执行。详见 [联合训练记录](../docs/graph5_vs7_joint_training_20261026.md)，
实时状态见 `logs_eval/graph5_vs7_joint_20261026_190259/status.json`。

2026-10-26 已构建并验收扩展图谱 `graph_recovery7_20261026_v1/`：
3 个任务、7 条恢复、96 条任务过渡和 63 条恢复出口，合并共 169 段。
原图 149 个训练条目完全保留，新增两条恢复及 18 个出口。
169 段严格几何、159 条完整和 477 次局部运行时重建、CPU/CUDA 加载均通过；
配套七入口配置为 `+recovery=eval_fsm_recovery7`。完整图谱策略尚待联合训练。
详见 [七条恢复图谱验收报告](../docs/recovery7_skill_graph_20261026.md)，
运行时资产清单见 `graph_recovery7_20261026_v1/runtime_assets.yaml`。

2026-10-15 已从修复后的八段基础动作重建并验收完整图
`graph_repaired_20261015_v3/`，包括 96 条任务过渡、45 条恢复出口，合并共 149 段。
完整联合训练需在新启动器上增加 `--graph-dir Switch_data/graph_repaired_20261015_v3`。
详见 [技能图验收报告](../docs/rebuilt_skill_graph_acceptance_20261015.md)。

2026-10-15 新增 `reference_training_20261015_v3/`：继承姿态与末帧修复，
补齐地面/接触标签、恢复动作的深自穿透修正，以及直立和静止收尾。
八段基础动作通过严格验收、CPU/CUDA 加载与两组真实 GPU 短训练。
下一步使用 `bash scripts/train_teacher_repaired.sh --detach --iterations 10000`
让 241K 教师适应新数据；本阶段不混入旧图的过渡片段。
详见 [训练准备说明](../docs/training_reference_readiness_20261015.md)。

2026-10-14 新增 `reference_unified_20261014_v2/`，在 v1 的姿态统一基础上，
裁剪三个任务各自的异常最后一帧并同步合并文件。八个基础动作共 2310 帧，
五段恢复与 v1 单段文件完全相同。原始数据、v1 和旧图均保留。
详见 [任务末帧修复报告](../docs/task_endpoint_repair_20261014.md)。

2026-10-14 新增独立姿态统一版本 `reference_unified_20261014_v1/`：
`pose_only/` 保留八段基础动作的关节和位置轨迹，统一 `dof/pose_aa/root_rot`；
`calibrated/` 另外按训练对齐 XML 重新校准五段恢复。两份合并文件均只有
3 个任务和 5 个恢复，没有复制旧过渡。原数据、149 条旧图和默认配置保留。
CPU/GPU 实际加载检查通过；自接触、任务末帧/地面与稳定出口仍待处理。
详见 [姿态统一报告](../docs/reference_pose_unification_20261014.md)。

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
not train attachment from arbitrary fallen states. The opt-in evaluation FSM
below now provides runtime damping, entry selection, retries and return routing.

## Contact-refined unified graph

### Real-fall dual-policy evaluation

For a controlled entry-grace diagnostic, run
`bash scripts/eval_recovery_entry_grace.sh 0.25` and then
`bash scripts/eval_recovery_entry_grace.sh 0.75` in the `pbhc` environment.
Both default to eval seed 1, a one-shot `[3, 0]` m/s push at control step
122, and 1500 evaluation steps. Do not add manual pushes in the paired
experiment. Add `++headless=true` for unattended runs. Logs are under
`logs_eval/recovery_grace_025/` and `logs_eval/recovery_grace_075/`.
The global grace default stays 0.25 s. The override delays recoverable
tracking terminations only; hard failures retain their normal handling.

With entry diagnostics enabled, `entry_error_sample` records the exact
five `body_z` termination keypoints, signed Z errors, maximum absolute
error, the unchanged 0.25 m threshold, raw termination flags and grace
status. Samples cover the first second after each recovery entry and stop
when that attempt ends. Failure frames are sampled even between intervals.
No samples from damping/retries are appended to a completed attempt.
Use `python scripts/compare_recovery_entry_grace.py BASE_RUN EXTENDED_RUN --output-dir REPORT_DIR`
to create `comparison.json` and `body_z_traces.png`. Compare the first
attempt's initial pose/entry before drawing a causal conclusion; retry
states can differ. A falling error curve alone does not establish success:
check threshold crossing and the eventual attempt result as well.

For a matched first-entry comparison, run `bash scripts/inspect_recovery_entry.sh`.
This runs seeds 3 and 4 with the normal five-entry selector, then forces
`140_02` frame 0 from the same first fallen physical state. It enables
`entry_diagnostics.detailed=true` and `window_s=3.0`, recording actual and
termination-reference keypoint heights, joint poses/errors, foot contacts
and rigid-body contact forces before any retry/reset. Defaults and control
behavior are unchanged. Compare **first attempts only**: the forced-02
candidate pool has no alternate retry entry. Run
`python scripts/summarize_recovery_entry_inspection.py SUITE_DIR` to verify
initial-state equality and generate `summary.json` and
`first_entry_diagnostics.png`.

Run `bash scripts/eval_real_fall_recovery.sh` in the `pbhc` environment.
It loads the unified `model_241000.pt` for task/Buffer frames and the
stage-3 recovery `model_115000.pt` for recovery frames. Automatic random
pushes and initialization noise are disabled so each experiment starts
from a reproducible reference. Press `t` to add a physical XY velocity
impulse (default `[1.5, 0.0]` m/s); the existing termination checks trigger
damping, settle, selection and recovery. Press `7/8/9` for task commands.
The `r` key remains a diagnostic FSM request and does not apply a push.

Use `++algo.config.eval_callbacks.skill_scheduler.config.recovery_fsm.fall_push_velocity_xy=[-1.5,0.0]`
to change push direction, or `++algo.config.eval_callbacks.skill_scheduler.config.recovery_fsm.fall_push_step=100`
for an automatic one-shot push. Recovery uses `collect` admission and at
most two attempts. Events are stored under
`logs_eval/real_fall_dual_policy/<timestamp>/recovery_events.jsonl`.
Run `python scripts/summarize_real_fall_recovery.py EVENTS_FILE` for results.
Attempt labels include duration, raw terminations, initial state/distance,
and failure phase. An early failure is a timing classification, not proof
of an invalid initial contact. `recovery_exit_started` records scheduling
of the exit; it does not independently certify stable standing. An
unfinished attempt at evaluation shutdown is reported separately.

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

## 恢复 FSM：评估与诊断

配置 `humanoidverse/config/recovery/eval_fsm.yaml` 开启单环境 IsaacGym
评估期恢复。普通训练不注册 FSM。状态流为：

```text
TRACKING → DAMPING → SELECTING → RECOVERY_TRACKING
                              → RETURNING_TO_TASK → TRACKING
超时、无合格入口、最多两次恢复尝试失败 → HARD_RESET → TRACKING
```

`ref_pos_z/ref_ori/body_z/motion_far` 等跟踪失败在环境内部 reset 前被拦截。
阻尼力矩为限幅的 `-Kd*qdot`，不以零 action 代替阻尼；关节速度每个物理
子步更新。参考时钟暂停，持续检查 root 线速度、角速度及关节速度 RMS。
连续满足默认 0.4 秒后，只在五段恢复动作各自经过检查的第 0 帧中选择入口，
即技能 3/4/5/6/7 分别对应 frame 0。旧的前 20% 采样仅作为未配置
`entry_frames` 时的兼容回退。硬终止、数值异常和场地越界继续回退。

恢复距离为 `sum(weight * mean_absolute_error / physical_unit_scale)`，独立于图搜索
`sim`，其阈值不能照搬 `A/B/switch_entry_A`。默认尺度/权重为：

| 分量 | 尺度 | 权重 | 原因 |
|---|---:|---:|---|
| `projected_gravity` | 1.0 | 4.00 | 去除全局 yaw 后区分仰卧、俯卧和侧卧，是恢复模式的首要特征 |
| `q` | 0.5 rad | 1.25 | 区分四肢及躯干构型，但避免关节数量压倒身体朝向 |
| `foot_contact` | 1.0 | 1.50 | 区分双足是否支撑；使用离散接触差异 |
| `root_height` | 0.2 m | 0.50 | 辅助区分贴地与半跪，避免高度噪声主导选择 |
| `qdot` | 2.0 rad/s | 0.10 | 阻尼后应已近静止，只用于打破相近入口的平局 |

当前恢复 PKL 只提供双足 `contact_mask`，且现用锁腕模型的虚拟手部没有可靠碰撞
几何，因此本版没有伪造手掌/膝部接触标签。所有分量会写入选择日志；若成功与
失败距离明显重叠，再加入手膝近地高度/支撑标签并重新标定阈值。

恢复参考只做 XY/yaw 对齐；真实 root/关节状态、参考 z/roll/pitch 均不被覆盖。
参考切换后立即刷新观测并重新计算策略动作，保留实际历史观测。开始跟踪后有
0.25 秒的参考误差宽限和 0.2 秒的力矩变化率限制；硬故障不享受宽限。
恢复末段使用现有图搜索，等待训练过的未来 Buffer 起点后返回原任务。
进入目标任务并连续运行 1 秒后才记录 `recovery_success`，清除重试次数。

### 1. 默认采集模式（未标定准入阈值）

```bash
cd /home/ubuntu/xb/PBHC
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pbhc

CKPT=logs/MotionTracking/20260923_101737-switch3_recovery5_contact_refined_pilot-motion_tracking-g1_23dof_lock_wrist/model_241000.pt
bash scripts/eval_recovery_fsm.sh "$CKPT"
```

按 `7/8/9` 请求任务 0/1/2；恢复中只更新返回目标，不立即打断起身。
按 `r` 人为触发一次恢复流程（覆盖策略为阻尼，不施加推力或改变物理姿态）。
自然跟踪失败也会自动触发。FSM 开启时 `n` 被禁用，避免绕过调度器复位。
建议只使用任务按键 `7/8/9`；旧的 `0` 键仍代表技能 3。

默认 `selector.admission_mode=collect` 且 `max_entry_distance=null`：机器人落稳后
始终选择五个固定入口中距离最近的一个并尝试起身，不使用距离拒绝；两次尝试上限
保持不变。该模式用于采集阈值标定数据，不应作为最终部署配置。事件写到本次
`eval_log_dir/recovery_events.jsonl`，结束时汇总到
`recovery_metrics.json`。静止事件保存真实 root state、q/qdot、重力方向和双足接触，
用于后续入口容差标定与恢复初始化数据收集。事件文件采用追加方式，重复实验应换日志目录。

### 2. 尝试级标签与离线阈值标定

标签由两个参数共同控制：

```yaml
attempt_logging:
  enabled: true
  modes: [collect]
```

只有总开关为 true 且当前 `selector.admission_mode` 包含在 `modes` 中时才输出
`attempt_start/attempt_result/attempt_rejected`。默认只标记 `collect`，因此上面的默认
采集命令无需额外参数。显式启动参数为：

```bash
bash scripts/eval_recovery_fsm.sh "$CKPT" \
  algo.config.eval_callbacks.skill_scheduler.config.recovery_fsm.selector.admission_mode=collect \
  algo.config.eval_callbacks.skill_scheduler.config.recovery_fsm.attempt_logging.enabled=true \
  'algo.config.eval_callbacks.skill_scheduler.config.recovery_fsm.attempt_logging.modes=[collect]'
```

临时关闭标签：

```bash
bash scripts/eval_recovery_fsm.sh "$CKPT" \
  algo.config.eval_callbacks.skill_scheduler.config.recovery_fsm.attempt_logging.enabled=false
```

若需要同时审计 enforce 模式：

```bash
bash scripts/eval_recovery_fsm.sh "$CKPT" \
  algo.config.eval_callbacks.skill_scheduler.config.recovery_fsm.selector.admission_mode=enforce \
  algo.config.eval_callbacks.skill_scheduler.config.recovery_fsm.selector.max_entry_distance=3.0 \
  algo.config.eval_callbacks.skill_scheduler.config.recovery_fsm.attempt_logging.enabled=true \
  'algo.config.eval_callbacks.skill_scheduler.config.recovery_fsm.attempt_logging.modes=[collect,enforce]'
```

`attempt_start` 固化选中入口、初始总距离和各分量；`attempt_result` 的 `success` 只在
机器人完成恢复、经过恢复出口 Buffer 返回目标任务并保持 1 秒后产生。再次触发终止、
恢复超时、参考耗尽或硬故障标记为 `failure`。`attempt_rejected` 表示 enforce 阈值拒绝，
不属于实际执行过的成功率样本。评估中途结束而没有 result 的 start 也不会进入标定。

默认只用 collect 模式的第一次尝试进行标定：

```bash
python scripts/calibrate_recovery_entry_threshold.py \
  logs_eval/你的采集目录/recovery_events.jsonl \
  --target-success-rate 0.90 \
  --confidence 0.90 \
  --min-samples 30 \
  --output Switch_data/recovery_entry_calibration.json
```

可一次输入多次实验：

```bash
python scripts/calibrate_recovery_entry_threshold.py \
  logs_eval/recovery_collect_*/recovery_events.jsonl \
  --target-success-rate 0.90 --confidence 0.90 --min-samples 30 \
  --output Switch_data/recovery_entry_calibration.json
```

`--include-retries` 会把第二次尝试也纳入统计，默认关闭，因为第二次入口状态已受第一次
动作影响。脚本选择覆盖率最大的距离阈值，同时要求经验成功率和单侧 Wilson 置信下界
都不低于目标成功率；样本不足或没有候选满足条件时，`selected_threshold` 为 null。
输出还包含各恢复入口统计、成功/失败距离分布、距离判别 AUC、各距离分量分布和失败原因。

### 3. enforce 模式检验

```bash
# 3.0 仅用于流程试验，尚不是标定后的可靠阈值。
bash scripts/eval_recovery_fsm.sh "$CKPT" \
  algo.config.eval_callbacks.skill_scheduler.config.recovery_fsm.selector.admission_mode=enforce \
  algo.config.eval_callbacks.skill_scheduler.config.recovery_fsm.selector.max_entry_distance=3.0
```

`collect` 始终尝试最近入口；`enforce` 要求显式提供正数阈值，距离大于阈值时以
`no_admissible_entry` 进入硬复位。这样采集期不会因当前阈值造成截尾偏差，部署期
则能严格执行离线标定结果。

日志 `trigger=recovery_entry, entry_metric=recovery_nn, physical_reset=False`
表示从真实姿态接入起身；随后 `trigger=cmd_change, buffers>0` 表示使用恢复出口。
`preparing_hard_reset` 注入仅准备参考，其 `physical_reset=False` 不代表恢复成功；
实际复位由环境执行一次，并单独记录 `event=hard_reset, physical_reset=True`。

`enable_safety_replan` 仍默认关闭：FSM 使用已有环境误差条件触发恢复。
完成图搜索 B 阈值标定后，可追加
`+algo.config.eval_callbacks.skill_scheduler.config.enable_safety_replan=true`，
启用优先于待执行任务命令的 B 阈值检查。

### 4. 可重复的无界面流程检验

```bash
PYNPUT_BACKEND=dummy bash scripts/eval_recovery_fsm.sh "$CKPT" \
  ++headless=true ++eval_steps=700 \
  eval_log_dir=logs_eval/recovery_fsm_manual_check \
  algo.config.eval_callbacks.skill_scheduler.config.recovery_fsm.diagnostic_trigger_step=30 \
  algo.config.eval_callbacks.skill_scheduler.config.recovery_fsm.selector.admission_mode=enforce \
  algo.config.eval_callbacks.skill_scheduler.config.recovery_fsm.selector.max_entry_distance=3.0
```

`diagnostic_trigger_step` 使用环境累计步数，包含初始化步；默认 -1 禁用。
该测试验证控制流程，不等于随机外力下的恢复成功率测试。正常任务片段播放结束
仍沿用原评估器的复位/循环行为，不计为自主恢复成功。

### 当前验证范围

- 单元测试覆盖终止拦截、硬故障优先、连续静止、被动力矩/限幅、入口拒绝、
  无复位注入、有限次重试、返回目标保持、尝试级成功/失败标签及模式开关。
- 实际 IsaacGym 350 步诊断测试：阻尼约 1.94 秒后落稳、输出 NN 候选、一次回退。
- 实际 700 步任务初态/900 步恢复 02 初态测试：均完成无复位接入和两次尝试，
  策略最终仍因跟踪误差失败并回退。**尚未验证任意倒地状态下成功恢复的能力。**
- 真实统一图谱上的理想跟踪测试隔离策略误差，检查 FSM 是否经已有 Buffer 返回任务。

运行离线回归：

```bash
LOGURU_LEVEL=ERROR python -m unittest humanoidverse.deploy.skill_scheduler.tests.test_recovery_fsm -q
python humanoidverse/deploy/skill_scheduler/tests/test_graph_search.py
python humanoidverse/deploy/skill_scheduler/tests/test_reference_builder.py
```
