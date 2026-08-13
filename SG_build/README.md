# SG_build — Skill Graph Construction for Humanoid Multi-Skill Transitions

基于 **Switch** 论文 (arXiv:2604.14834) 的技能图谱构建工具。

## 概述

将多个独立的运动技能 (.pkl 文件) 通过运动学相似性自动连接，生成技能切换图谱和增强训练数据。

```
技能A (Horse-stance_pose)  ──┐
                              ├──→ Skill Graph (图谱) ──→ 增强数据集
技能B (Horse-stance_punch) ──┘      + buffer 过渡动作
```

## 快速使用

```bash
# 从合并的 .pkl 文件构建 (自动按 key 拆分技能)
python SG_build/build_sg.py \
    --merged example/motion_data/merged_two.pkl \
    --labels horse_pose horse_punch \
    -o SG_build/sg_output

# 从独立的 .pkl 文件构建
python SG_build/build_sg.py \
    --files pose.pkl punch.pkl kick.pkl \
    --labels pose punch kick \
    -o SG_build/sg_output

# 从文件夹构建 (所有 .pkl 文件各为一个技能)
python SG_build/build_sg.py \
    --folder example/motion_data/ \
    -o SG_build/sg_output
```

## 输出文件

```
sg_output/
├── skill_graph.json        # 图谱结构 (节点、边、权重)
│                           #   nodes[] 含逐节点部署特征: skill_id, frame_idx,
│                           #   is_buffer, kappa, q(dof), q_dot(dof_vel), p_hat(root_trans)
├── augmented_motions.pkl   # 过渡轨迹 (含 buffer 节点)
├── merged_training.pkl     # 原始动作 + 增强过渡 (PBHC 训练就绪)
└── scheduler_config.yaml   # 由 calibrate.py 生成 (σ 统计量 + A/B 阈值候选)
```

离线标定（在线调度器第一步，见 `switch_scheduler_spec.md` 6.1）：

```bash
python humanoidverse/deploy/skill_scheduler/calibrate.py sg_output/skill_graph.json
```

## 核心参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--threshold` | 30.0 | 跨技能边的最大 L1 距离 |
| `--topk` | 5 | 每帧在目标技能中的最近邻数 |
| `--buffer-base` | 1.0 | 每个 buffer 节点对应的 L1 距离 |
| `--max-buffer` | 30 | 单条边的最大 buffer 节点数 |
| `--subsample` | 3 | 源帧采样步长 (1=每帧, 3=每三帧) |
| `--exclude-boundary` | 10 | 排除起始/目标节点在技能前后 N 帧内的轨迹 |
| `--max-trajectories` | 10 | 每对技能最多收集的轨迹数，遍历所有边直到达标 |

**参数调节建议**：
- `--threshold` 越小 → 只连接非常相似的状态 → 过渡更可靠但更少
- `--buffer-base` 越小 → 更多 buffer 节点 → 过渡更平滑但训练数据更大
- `--subsample 1` → 每帧都计算跨技能边 → 全面但慢
- `--exclude-boundary 0` → 不排除边界帧，允许所有跨技能过渡

## 算法流程 (对应论文章节)

### Step 1: 状态特征提取
对每帧提取:
- `q`: 关节位置 (23-dim, dof)
- `q̇`: 关节速度 (23-dim, dof 差分)
- `root`: 根位移 (3-dim)

### Step 2: 基础图构建 (Sec III-A.1)
```
V = 所有技能的所有帧
E = 每个技能内的连续帧 → 权值 = 1
```

### Step 3: 跨技能边 (Sec III-A.2)
```
距离: d(s_m, s_n) = ||q_m - q_n||₁ + ||q̇_m - q̇_n||₁ + ||r_m - r_n||₁

对每帧 m (技能A)，在技能B中找到 top-K 最近邻:
  j = argmin d(s_m, s_n), n ∈ 技能B
  if d(s_m, s_j) ≤ threshold:
      E = E ∪ {(m, j)},  weight = d(m, j)
```

### Step 4: Buffer 过渡节点 (Sec III-B.3)
```
N_buffer = min(max(0, distance / buffer_base), max_buffer)

buffer_node[k] = lerp(src_frame, dst_frame, k/(N+1))
  - dof/root_trans: 线性插值
  - root_rot: SLERP 球面插值
```

### Step 5: 增强数据导出
每条过渡轨迹:
```
[源技能N帧] → [buffer_1, ..., buffer_N] → [目标技能M帧]
```
buffer 帧的 `is_buffer` 标记为 True，训练时用目标帧（首帧）计算 reward。

## 图谱统计解读

```
==================================================
Skill Graph Summary
==================================================
  Skills:          2              # 技能数量
  Total frames:    410            # 总帧数
  Nodes:           410            # 图节点数
  Temporal edges:  408            # 技能内时间边
  Cross-skill edges: 493          # 跨技能相似性边
  Buffer trajectories: 14         # 生成的过渡轨迹数
  Total buffer frames: 17         # buffer 节点总数
  Skill[0] horse_pose: 210 frames, 7.0s
  Skill[1] horse_punch: 200 frames, 6.7s
==================================================
```

## 使用增强数据训练

```bash
python humanoidverse/train_agent.py \
    +simulator=isaacgym +exp=general_tracking +terrain=terrain_locomotion_plane \
    project_name=MotionTracking num_envs=128 \
    +obs=motion_tracking/obs_ppo_teacher \
    +robot=g1/g1_23dof_general \
    +domain_rand=main \
    +rewards=motion_tracking/general_main \
    experiment_name=sg_augmented \
    robot.motion.motion_file="SG_build/sg_output/merged_training.pkl" \
    seed=1 +device=cuda:0
```

> **注意**: 当前 PBHC 的 buffer-aware imitation (用目标帧监督 buffer 节点) 尚未集成，需在 reward 函数中检查 `is_buffer` 标记并调整监督目标。

## 依赖

- numpy, scipy (已在 PBHC conda 环境中)
- joblib (已在 PBHC conda 环境中)
