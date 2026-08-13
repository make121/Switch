# Switch 论文 Online Skill Scheduler 实现规格说明

> 来源论文：*Switch: Learning Agile Skills Switching for Humanoid Robots* (arXiv:2604.14834)
> 本文档面向代码实现，供 coding AI 直接产出代码使用。所有超参数/设计歧义已与项目负责人确认，见文末"设计决策记录"。

---

## 0. 背景与定位

Scheduler 是**部署阶段**（非训练阶段）运行的在线规划模块。它复用训练阶段已经构建好的 Skill Graph `G`（对应现有 `skill_graph.py` / `skill_graph.json` 的产出），在机器人运行时：

1. 根据用户指令或安全事件，在图上规划一条从当前状态到目标技能的路径；
2. 把路径转换成实时 guidance target `g_t = ⟨ŝ_t^g, κ_t⟩`，喂给下游 whole-body tracking policy；
3. 在检测到跟踪失败/外部扰动时自动触发恢复规划。

Scheduler 本身不训练、不涉及 RL，是纯粹的图搜索 + 状态机模块。

---

## 1. 数据结构

### 1.1 Skill Graph 输入格式

复用现有 `skill_graph.json`，需要包含（如现有格式不满足，需补充）：

- 节点列表：每个节点 = 一帧参考状态，字段包括：
  - `node_id`
  - `skill_id`（所属技能标签）
  - `frame_idx`（技能内帧序号）
  - `q`（关节位置，局部坐标系，去除全局 x-y 平移与 yaw/twist）
  - `q_dot`（关节速度）
  - `p_hat`（body 刚体位置，局部坐标系）
  - `is_buffer`（bool，是否为 buffer 插值节点）
  - `kappa`（int，若为 buffer 节点，距该 buffer 段结束的剩余步数；非 buffer 节点为 0）
- 边列表：`(u, v, edge_type)`，`edge_type ∈ {"same_skill_consecutive", "cross_skill", "buffer"}`

### 1.2 部署时边权重（与训练时权重公式不同，需重新计算，不要复用训练时 `w_train`）

```python
def deploy_edge_weight(u, v, edge_type, d_uv, lambda_sw):
    if edge_type == "same_skill_consecutive":
        return 1.0
    else:  # cross_skill 或 buffer 边
        skill_diff_penalty = lambda_sw if skill_id(u) != skill_id(v) else 0.0
        return d_uv + skill_diff_penalty
```

其中 `d_uv` 是下面 1.3 节定义的**归一化加权距离**（复用同一套距离函数，不要为训练时/部署时分别设计两套距离）。

### 1.3 状态距离 / 相似度函数（**已确认：采用方案2 —— 分量归一化加权**）

```python
def sim(x, node, sigma_q, sigma_qdot, sigma_p, w_q=1.0, w_qdot=1.0, w_p=1.0):
    d_q    = L1_norm(x.q     - node.q)     / sigma_q
    d_qdot = L1_norm(x.q_dot - node.q_dot) / sigma_qdot
    d_p    = L1_norm(x.p_hat - node.p_hat) / sigma_p
    return w_q * d_q + w_qdot * d_qdot + w_p * d_p
```

- `sigma_q, sigma_qdot, sigma_p`：**离线统计量**，取整个 `skill_graph.json` 数据集上对应分量 L1 距离的标准差（一次性计算，存入 config）。
- `w_q, w_qdot, w_p`：默认 1.0，作为可调超参数保留接口。
- **语义**：`sim` 越小越接近（本质是距离，不是 [0,1] 有界相似度）。论文中 `sim ≤ A` 表示"足够接近可直接切换"，`sim ≥ B` 表示"太远需要 e-stop"，实现时直接按这个方向判断，不需要做反转。
- 该函数同时用于：(a) 部署时跨技能边权重计算中的 `d_uv`；(b) 入口检查 entry check；(c) 安全阈值判断。三处保持同一实现，避免不一致。

### 1.4 Buffer 节点处理（**已确认：当作普通节点参与搜索**）

- Buffer 节点在最短路 / 最近邻搜索中与普通节点一视同仁地参与路径搜索，不做特殊跳过或后处理。
- 区别仅在于：当路径中的当前目标节点 `is_buffer=True` 时，输出的 guidance target 需要带上 `kappa_t = node.kappa`；普通节点 `kappa_t = 0`。
- 这与训练阶段"buffer-aware imitation"的语义一致（policy 侧本来就要处理 `κ_t > 0` 的情况）。

---

## 2. 规划器（**已确认：Graph-Search 与 NN 都实现，接口可切换**）

统一接口：

```python
class SkillGraphScheduler:
    def __init__(self, skill_graph, planner_type: Literal["graph_search", "nn"],
                 A: float, B: float, lambda_sw: float, lambda_cost: float,
                 tau: float, top_k: int,
                 sigma_q: float, sigma_qdot: float, sigma_p: float,
                 w_q: float = 1.0, w_qdot: float = 1.0, w_p: float = 1.0):
        ...
```

### 2.1 Graph-Search planner

- 给定目标集 `T`，从 `T` 做**反向多源最短路**（reverse Dijkstra，边权为 1.2 节定义的 `deploy_edge_weight`），得到：
  - 值函数 `V(v)`：v 到 T 的最小累计代价
  - `next_hop(v)`：v 的最优下一跳
- 目标集变化（新的 `T_cmd`）时才需要重新计算 `V`/`next_hop`；目标不变时的重规划只是"根据当前 entry 沿 next_hop 走一遍"，成本很低（缓存 `V`/`next_hop`）。
- 路径重建：从入口节点开始，反复 `next_hop`，直到进入 T。

```python
def build_value_function(self, T_cmd) -> Tuple[dict, dict]:  # (V, next_hop)
    ...  # 反向多源 Dijkstra

def reconstruct_path_gs(self, entry_node, next_hop) -> List[Node]:
    path = [entry_node]
    while path[-1] not in self.T_cmd:
        path.append(next_hop[path[-1]])
    return path
```

### 2.2 Nearest-Neighbor (NN) planner

- 不做全局搜索，直接在候选集合中按 `sim(x, node)` 选最近邻节点作为入口，单跳或短跳直接过渡到 `T_cmd`。
- 若直接跳转不安全（见 Problem B），采用两阶段：先到恢复目标 `T_rec`，再从 `T_rec` 重新规划到 `T_cmd`。

```python
def plan_nn(self, x, candidates, T_cmd) -> List[Node]:
    entry = min(candidates, key=lambda v: self.sim(x, v))
    return short_hop_path(entry, T_cmd)  # 直接连接或经1-2个中间节点

def two_stage_nn(self, x, T_rec, T_cmd) -> List[Node]:
    path_to_rec = self.plan_nn(x, candidates=T_rec_candidates, T_cmd=T_rec)
    path_to_cmd = self.plan_nn(x=T_rec_state, candidates=cmd_candidates, T_cmd=T_cmd)
    return path_to_rec + path_to_cmd
```

两种 planner 应可通过配置项 `planner_type` 切换，便于后续做 A/B 对比评估（对应论文 Table II 风格的 SSR / NR 对比表）。

---

## 3. Problem A：意图驱动切换（Intent-Driven Switching）

```python
def entry_check(self, x, T_cmd):
    similarities = {v: self.sim(x, v) for v in candidate_pool(T_cmd)}
    best_v, best_sim = min(similarities.items(), key=lambda kv: kv[1])

    if best_sim <= self.A:
        return "attach", [best_v]
    elif best_sim >= self.B:
        return "estop", None
    else:
        top_k_candidates = sorted(similarities.items(), key=lambda kv: kv[1])[:self.top_k]
        return "search", [v for v, _ in top_k_candidates]
```

- 目标集 `T_cmd`：用户指定技能序列的**前 τ 比例的帧**（`τ` 待标定，见第5节）。
- Top-k 候选评分（`A < sim < B` 区间时使用）：

```python
def score(self, x, v, V=None):
    cost_term = self.lambda_cost * self.sim(x, v)
    value_term = V[v] if V is not None else 0.0   # 仅 Graph-Search 使用；NN 时省略
    return cost_term + value_term
```

选出评分最优的候选，再交给对应 planner（`plan_graph_search` 或 `plan_nn`）生成完整路径并安装为参考。

---

## 4. Problem B：安全恢复（Emergency Stop）

触发条件（二选一即触发）：
- 跟踪过程中 `sim(x, current_target) ≥ B`
- 入口选择阶段，即使是最优候选 `best_sim ≥ B`

处理逻辑：

```python
def handle_safety_event(self, x, T_cmd, planner_type):
    if planner_type == "graph_search":
        # T 直接设为原始 T_cmd，恢复技能作为路径中间节点自然被涵盖
        V, next_hop = self.build_value_function(T_cmd)
        entry = best_entry_under_B(x, V)
        return self.reconstruct_path_gs(entry, next_hop)
    else:  # nn
        if is_direct_jump_safe(x, T_cmd):
            return self.plan_nn(x, candidates(T_cmd), T_cmd)
        else:
            T_rec = select_recovery_target(x)   # 如 get-up 技能的入口帧
            return self.two_stage_nn(x, T_rec, T_cmd)
```

e-stop 期间的动作覆盖：

```python
def during_estop(self, robot_state):
    action = damping_controller(robot_state)   # 例如：目标关节速度=0的PD阻尼，或论文所述的damping controller
    if is_stationary(robot_state, ang_vel_thresh=THRESH):
        return "ready_for_recovery_plan"
    return action
```

`is_stationary` 判据：机器人 root 角速度低于阈值（阈值待标定）。

---

## 5. 在线调度主循环（4种触发条件）

```python
def step(self, x, user_cmd, t):
    trigger = None
    if self.is_initialization(t):
        trigger = "init"
    elif user_cmd != self.current_cmd:
        trigger = "cmd_change"
    elif self.approaching_reference_end(t):
        trigger = "ref_end"
    elif self.sim_crosses_threshold(x):   # sim 穿越 A 或 B
        trigger = "safety_event"

    if trigger is None:
        return self.current_guidance   # 无需重规划，继续跟踪当前参考

    status, candidates = self.entry_check(x, self.T_cmd)
    if status == "attach":
        path = [candidates[0]] + suffix_to_target(candidates[0], self.T_cmd)
    elif status == "search":
        path = self.plan(x, candidates, self.T_cmd)   # 按 planner_type 分发
    else:  # estop
        path = self.handle_safety_event(x, self.T_cmd, self.planner_type)

    self.current_guidance = self.path_to_guidance(path)
    return self.current_guidance
```

`path_to_guidance`：把路径中当前应跟踪的节点转换为 `g_t = ⟨ŝ_t^g, κ_t⟩`（`κ_t` 取自节点的 `kappa` 字段，见 1.4 节）。

---

## 6. 超参数标定流程（**已确认：先在 IsaacGym 训练环境内直接评估**）

### 6.1 图结构统计（离线，一次性）

用一个独立脚本 `calibrate_scheduler_thresholds.py`：

1. 加载 `skill_graph.json`，分离三类边：`same_skill_consecutive` / `cross_skill`（无 buffer）/ `buffer`。
2. 计算 `sigma_q, sigma_qdot, sigma_p`：对全图所有边的三个分量 L1 距离分别求标准差。
3. 用归一化后的 `sim` 重新计算所有边的距离分布：
   - `A` 候选值：`cross_skill`（无 buffer）边距离分布的 P25
   - `B` 候选值：`buffer` 边距离分布的 P75（或 P90，作为网格搜索的起点范围而非固定值）

### 6.2 规划超参数网格搜索（IsaacGym 环境内评估）

复用现有 IsaacGym 训练 pipeline 做评估（不需要额外搭建 MuJoCo 环境），按论文 Easy/Medium/Hard 三档难度（对应切换1次/2次/3次）的思路组织测试用例：

- 网格：
  - `tau ∈ {0.1, 0.2, 0.3}`
  - `top_k ∈ {3, 5, 10}`
  - `lambda_sw ∈ {1, 5, 10}`
  - `lambda_cost ∈ {0.1, 1, 10}`
  - `A, B`：以 6.1 节统计值为中心，各测 2-3 个候选
- 评估指标（对齐论文 Table II）：
  - SSR（Skill Switching Success Rate）：body position error 相对 root 超过 0.5m 视为失败
  - NR（Normalized Reward）：每帧平均归一化奖励
- 流程：先用小样本（每组合 ~10 trials）粗筛，选出 top 2-3 组合后再跑满 50 trials 精确确认（对齐论文评估协议）。
- 输出：`scheduler_config.yaml`，包含标定后的全部超参数 + `sigma_q/sigma_qdot/sigma_p` + `w_q/w_qdot/w_p`。

---

## 7. 建议实现顺序

1. `sim()` 函数 + 离线统计脚本（第1.3、6.1节）—— 优先级最高，其他一切依赖它
2. `SkillGraphScheduler` 基础框架 + Graph-Search planner（逻辑更直接，便于先验证正确性）
3. NN planner（复用 Graph-Search 已验证的 `sim`/`entry_check`/触发逻辑，只替换路径生成部分）
4. e-stop / damping controller 状态机
5. 超参数网格搜索脚本，接入 IsaacGym 评估 pipeline

---

## 8. 设计决策记录（已与项目负责人确认，实现时不要偏离）

| 决策点 | 采用方案 |
|---|---|
| 规划器 | Graph-Search 与 NN 都实现，通过 `planner_type` 配置切换 |
| sim(x, node) 设计 | 方案2：分量归一化加权（各分量除以全图标准差后加权求和），不做核函数转换 |
| 超参数标定评估环境 | 先用 IsaacGym 训练环境直接评估，不额外搭建 MuJoCo |
| Buffer 节点处理 | 当作普通图节点参与搜索，只是携带 `kappa_t` 信息透传给下游策略，不做特殊跳过/补插逻辑 |
