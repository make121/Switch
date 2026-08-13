# 项目经历：基于 PBHC 的人形机器人多技能切换控制

## 项目背景

基于开源框架 **PBHC（KungfuBot / KungfuBot2，NeurIPS 2025）**——一套基于物理的人形机器人（Unitree G1）全身运动模仿控制框架（IsaacGym 训练 + PPO + MuJoCo 部署）——实现 Switch 论文提出的**技能图（Skill Graph）多技能切换**方法，目标是从"单技能运动跟踪"扩展到"多技能间平滑切换"。

## 主要工作

**1. 技能图离线构建管线（`SG_build/`）**

- 独立实现 Switch 论文的技能图构建算法（`skill_graph_V2.py` + `build_sg.py`）：提取机器人运动的帧级状态特征（关节角、关节速度、根节点位姿），在技能内建立时间边、跨技能间基于 L1 相似度建立切换边（threshold=30.0, topk=5），并通过线性插值 + SLERP 生成 buffer 过渡节点
- 支持首尾帧过滤（`--exclude-boundary`）与"仅采样切换前 N 帧"的切换轨迹生成策略
- 导出带完整元数据的增强训练数据：逐帧 `is_buffer` 布尔数组 + 标量 `transition_distance`，为多套实验（两技能 pose/punch、三技能 multi-motion）生成训练集

**2. 运动数据采集与重定向（`AMASS_data/`、`smpl_retarget/`）**

- 从 AMASS（MPI_HDM05）筛选 kick、punch、lie_down_stand_up 三类动作，经 mink 微分逆运动学管线重定向到 G1 机器人，作为技能图的技能来源
- 修正 `mink_retarget` 输出文件命名约定，统一为 `*_retarget.pkl`

**3. 跨技能训练框架改造（`humanoidverse/`）**

- 在运动库层（`motion_lib_base.py`）实现条目级分类：按元数据自动区分单技能轨迹与过渡轨迹，并实现**跨技能加权采样**——可配置比例（`cross_skill_ratio`）的环境采样过渡轨迹，支持运行时通过 `update_cross_skill_ratio()` 做课程式调整
- 在 KungfuBot2 通用跟踪任务（`general_tracking.py`）中维护逐环境 `is_cross_skill_env` 标志，在 reset/resample 时与运动库同步，为后续差异化 reward 预留接口

**4. 多技能策略训练与验证**

- 基于 KungfuBot2 的 general_tracking（teacher/student + DAgger）路线，完成从两技能（`sg_augmented`）到三技能（`mm_triple_skill`）的递进实验，最长训练 190k iteration

**5. 在线调度器设计（`switch_scheduler_spec.md`）**

- 撰写部署阶段 Skill Scheduler 的设计规格：基于图搜索 + 最近邻规划的技能切换路径规划、e-stop 状态机、超参数标定流程

## 待完成

- **Buffer-aware imitation**：将逐帧 `is_buffer` 引入 reward/obs，对 buffer 帧降低跟踪权重（当前 `is_cross_skill_env` 已就位但尚未被消费）
- 在线 Skill Scheduler 的代码实现（sim 距离函数、离线统计、图搜索规划器、e-stop 状态机）
