# LogisticsVLA-Bench 论文大纲（基于 `isaaclab_logistics_vla`）

## 题目建议

- **LogisticsVLA-Bench: 面向仓储分拣与订单履约的 Isaac Lab 视觉-语言-动作评测基准**
- 备选：**LogiVLA-Bench: 具有场景随机化与可复现实验协议的物流机器人 VLA Benchmark**

---

## 1. 摘要（Abstract）

### 1.1 问题背景与动机

- 现有机器人 VLA 基准多聚焦通用抓放，缺少贴近物流履约流程的评测。
- 物流分拣需要同时满足：识别正确、抓放稳定、按单履约、效率可控。

### 1.2 现有挑战

- 缺少“订单约束 + 干扰物 + 场景复杂度变化”的统一 benchmark。
- 缺少可回放场景协议，导致跨模型对比复现性不足。

### 1.3 本文贡献

- 提出 `LogisticsVLA-Bench`，覆盖任务机制与场景难度的多维组合：
  - 任务族：`ss_st_series`、`ms_st_series`
  - 场景：`sparse`、`dense`、`stack`、`sparse_with_obstacles`
- 构建统一评估链路：`scripts/evaluate_vla.py` + `VLA_Evaluator` + `ObservationBuilder`
- 定义订单履约导向指标：`order_completion_rate`、`failure_rate`、`wrong_pick_rate`、`wrong_place_rate`、时间代价
- 提供 `from_json=0/1/2` 的记录-回放-随机协议，提升可复现性与公平性

### 1.4 主要发现（占位）

- 在密集场景和障碍场景下，错抓/错放显著上升。
- 多源任务 (`ms_st`) 对检索与规划能力要求更高。

### 1.5 价值

- 为物流场景 VLA 评测提供可复现、可解释、可扩展的标准化基准。

---

## 2. 引言（Introduction）

### 2.1 场景与问题定义

- 仓储分拣并非“单物体抓取”，而是“多 SKU + 订单约束 + 干扰抑制 + 时效”联合问题。

### 2.2 现有基准不足

- 缺少订单级别指标分解（是否完成、错抓、错放、失败）。
- 缺少稳定的评估协议（同场景回放对比）。
- 扰动维度覆盖不充分（纹理、障碍、密度、堆叠等）。

### 2.3 本文目标与贡献（详细）

1. 设计物流导向任务族与场景谱系。
2. 提供统一观测/动作接口，便于策略接入。
3. 建立可回放评测协议，提高跨模型可比性。
4. 支持多策略、多机器人一致评测。

### 2.4 论文结构

- 第 3 节相关工作，第 4 节基准设计，第 5 节评估协议，第 6-7 节实验与分析，第 8-9 节讨论与结论。

---

## 3. 相关工作（Related Work）

### 3.1 机器人操作 Benchmark

- 通用桌面操作 benchmark 的优势与局限。
- 与物流分拣任务需求的关键差异。

### 3.2 VLA 模型评测与鲁棒性

- 视觉扰动、语言泛化、执行误差评测现状。
- 错抓/错放等可解释错误类型在现有工作中的不足。

### 3.3 可复现实验协议

- 讨论随机评测在公平对比上的问题。
- 引出本工作 JSONL 场景记录/回放方案。

---

## 4. 基准设计与构建（Benchmark Design and Construction）

### 4.1 设计原则

- 真实性：贴近订单履约流程
- 可解释性：错误类型可分解
- 可复现性：场景可记录可回放
- 可扩展性：任务、机器人、策略可插拔

### 4.2 任务族定义

#### 4.2.1 单源单目标系列（`ss_st_series`）

- 典型任务：
  - `Spawn_ss_st_sparse_EnvCfg`
  - `Spawn_ss_st_dense_EnvCfg`
  - `Spawn_ss_st_sparse_with_obstacles_EnvCfg`
- 核心能力：目标识别、干扰抑制、精确放置

#### 4.2.2 多源单目标系列（`ms_st_series`）

- 典型任务：
  - `Spawn_ms_st_sparse_EnvCfg`
  - `Spawn_ms_st_dense_EnvCfg`
  - `Spawn_ms_st_stack_EnvCfg`
- 核心能力：跨源检索、路径规划、分拣一致性

### 4.3 场景元素与对象建模

- 统一基场景：`BaseOrderSceneCfg`
  - 地面、光照、传送带、工作台、源箱/目标箱
- SKU 对象动态注入：按 `scene_cfg.py` 中 `SKU_DEFINITIONS` 批量实例化
- 多难度场景差异：
  - `sparse`：低密度少干扰
  - `dense`：高密度多实例
  - `stack`：堆叠与遮挡增强
  - `with_obstacles`：障碍约束 + 尺度随机化

### 4.4 扰动与随机化设计

- 纹理随机化（箱体/障碍）
- 物体槽位随机化与姿态随机化
- 障碍物尺度随机化（`sparse_with_obstacles`）

### 4.5 观测与动作标准接口

- 观测结构（`evaluation/observation/schema.py`）：
  - `meta`
  - `robot_state`
  - `vision`
  - `point_cloud`
- 构建器：`ObservationBuilder`
- 动作支持：
  - `joint` 控制
  - `ee` 控制（可经 Curobo 转换为关节动作）

### 4.6 终止与回合管理

- `time_out`
- `order_success`（阈值触发）

---

## 5. 评估方法与协议（Evaluation Methodology）

### 5.1 统一评估入口

- 脚本：`scripts/evaluate_vla.py`
- 关键参数：
  - `--task_scene_name`
  - `--policy`
  - `--robot_id`
  - `--from_json`
  - `--num_envs`
  - `--sim_device` / `--curobo_device`

### 5.2 评估驱动逻辑

- 类：`VLA_Evaluator`
- 闭环流程：
  1. `reset`
  2. 构建 `ObservationDict`
  3. 策略输出动作
  4. 动作转换（必要时 EE -> joint）
  5. `env.step`

### 5.3 可复现实验协议（重点）

- `from_json=0`：记录场景到 JSONL
- `from_json=1`：从 JSONL 顺序回放
- `from_json=2`：纯随机采样

建议主报告使用 `from_json=1`，并以 `from_json=2` 作为泛化补充。

### 5.4 基线策略

- `random_policy`
- `openvla_remote_policy`
- `openpi_remote_policy`
- `curobo_reach_box_policy`

### 5.5 多机器人评测

- `evaluation/robot_registry.py` 提供统一配置映射：
  - `arm_dof`
  - 相机配置 key
  - 动作配置 key
  - Curobo 参数

---

## 6. 指标体系（Metrics）

### 6.1 核心指标

- `order_completion_rate`
- `failure_rate`
- `wrong_pick_rate`
- `wrong_place_rate`
- `mean_action_time` / `episode_physics_steps`

### 6.2 订单级与 SKU 级细粒度指标

- `completed_orders / total_orders`
- `correct / missing / extra`
- 每订单、每 SKU 的 `need` 与 `contain` 对齐误差

### 6.3 指标实现说明

- 指标由 `BaseOrderCommandTerm` 及其子类在 episode 生命周期中更新并落盘。
- 建议在论文中给出公式化定义与伪代码。

---

## 7. 实验设置（Experimental Setup）

### 7.1 仿真与实现细节

- 平台：Isaac Lab（版本需注明）
- 关键参数：`dt`、`decimation`、`num_envs`、`episode_length_s`

### 7.2 任务集合与划分

- 报告任务：
  - `ss_st_sparse`
  - `ss_st_dense`
  - `ss_st_sparse_with_obstacles`
  - `ms_st_sparse`
  - `ms_st_dense`
  - `ms_st_stack`

### 7.3 运行协议

- 每任务 episode 数、随机种子数
- JSONL 回放池规模
- 策略服务与 GPU 配置

---

## 8. 结果与分析（Experiments and Analysis）

### 8.1 主结果

- 表格报告各模型在各任务上的核心指标。
- 给出任务族聚合结果（`ss_st` vs `ms_st`）。

### 8.2 难度轴分析

- `sparse -> dense -> stack -> obstacle` 的性能衰减趋势。

### 8.3 协议与复现性分析

- `from_json=1` 与 `from_json=2` 在均值/方差上的差异。

### 8.4 消融实验（建议）

- 去除纹理随机化
- 去除障碍尺度随机化
- 多视角 vs 单视角
- joint 控制 vs ee+IK 控制

### 8.5 失败案例

- 错抓：目标-干扰混淆
- 错放：目标箱选择失败
- 失败：掉落/碰撞导致履约失败

---

## 9. 基准有效性验证（Benchmark Validity）

可加入独立小节，利用 `evaluation/tester`：

- 通过受控瞬移生成“应然”结果（错抓/错放/掉落比例可控）
- 与环境“实然”指标对比
- 验证指标实现可信度与一致性

---

## 10. 讨论（Discussion）

### 10.1 关键启示

- 物流 VLA 的瓶颈不仅是抓取成功，还包括订单语义一致性与流程鲁棒性。

### 10.2 局限性

- 语言复杂度仍可提升
- 动态场景与真实部署覆盖有限

### 10.3 未来方向

- 多机器人协同分拣
- 更复杂语言约束与长时序计划
- sim2real 与真实仓储数据闭环

---

## 11. 结论（Conclusion）

- 总结基准构建、评估协议、指标体系与主要实验发现。
- 强调该 benchmark 对物流场景 VLA 标准化评测的推动作用。

---

## 附录建议（Appendix）

- A. 全任务配置表（对象数、场景参数、回合时长）
- B. 指标公式与实现细节
- C. 评测命令与参数全表
- D. JSONL 数据格式说明
- E. 多机器人接入案例
- F. 额外失败案例可视化

---

## 图表规划（可直接用于写作 TODO）

- Figure 1：基准总体框架图（任务族 × 场景 × 扰动 × 指标）
- Figure 2：评估数据流（Env -> ObsBuilder -> Policy -> Action -> Env）
- Figure 3：代表性场景可视化（sparse/dense/stack/obstacle）
- Table 1：任务配置总表
- Table 2：指标定义表
- Table 3：主结果总表（模型 × 任务）
- Figure 4：错误类型分解柱状图（wrong pick/place/failure）
- Figure 5：随机 vs 回放的方差对比
- Figure 6：失败案例图集
