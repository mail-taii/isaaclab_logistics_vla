# IsaacLab Logistics VLA Benchmark 评估系统设计（讲解版：无实现代码）

面向对象：**环境侧同学 + 评估/模型侧同学**  
目标：讲清楚“谁负责什么、数据从哪来、怎么对齐、怎么跑（含两服务器）”。  
说明：本文**不包含具体实现代码**，但包含**类/模块职责、方法签名级别接口、数据 schema、对齐清单**。

---

## 0. 背景与关键约束

- **两服务器**：Benchmark Server（跑 IsaacLab 环境）与 Model Server（跑模型推理）不在同一台机器，需要网络 RPC。
- **指标只读**：**metrics 由环境内部计算**（例如 `OrderCommandTerm.metrics`），评估侧只做**读取/对齐/汇总/保存**，不做指标计算公式。
- **多模态观测**：模型可能需要多视角 `rgb/depth/seg/pcd` + 机器人状态 + 指令，需要一个类似 `get_observation()` 的聚合接口与稳定 schema。

---

## 1. 落地文件结构（建议的 `.py` 位置与名字）

> 这是你要给同学分工时最直观的“要写哪些文件、放哪”的结构图。

```
isaaclab_logistics_vla/evaluation/
├── evaluator/
│   ├── __init__.py
│   ├── base.py                 # Evaluator 基类：task/episode/step 主循环、错误处理、汇总
│   ├── vla_evaluator.py         # IsaacLab 环境评估实现：env 创建、循环调度、调用 policy
│   └── vlm_evaluator.py         # 可选：VLM-only（不跑环境）评估
│
├── policy/
│   ├── __init__.py
│   ├── base.py                 # Policy 接口（本地/远程统一）
│   ├── random_policy.py         # sanity check
│   ├── trajectory_policy.py     # 轨迹回放/示教策略（可选）
│   └── remote/
│       ├── __init__.py
│       ├── protocol.py          # 远程推理协议与 schema 常量（请求/响应字段、版本号）
│       ├── client.py            # RemoteClient：序列化/超时/重试/延迟统计
│       └── server_spec.md        # Model Server 端需要实现的接口说明（不一定放代码仓）
│
├── observation/
│   ├── __init__.py
│   ├── builder.py               # ObservationBuilder：类似 get_observation() 的聚合器
│   └── schema.py                # ObservationDict/ActionDict 的数据结构定义（typed dict/dataclass）
│
├── metrics/
│   ├── __init__.py
│   ├── reader.py                # MetricsReader：只读 env 的 metrics/state，统一成 MetricsReadout
│   └── schema.py                # MetricsReadout 结构定义 + reduce 约定
│
└── results/
    ├── __init__.py
    ├── saver.py                 # ResultSaver：episodes.jsonl/task_summary.json 落盘
    └── schema.py                # EpisodeReport/TaskReport/EvaluationReport 结构定义
```

---

## 2. 总体架构：模块与职责边界

### 2.1 逻辑模块图

```
┌────────────────────────────────────────────────────────────┐
│                       Benchmark Server                       │
│                                                            │
│  ┌──────────────┐     ┌──────────────────┐                 │
│  │ IsaacLab Env  │ --> │ ObservationBuilder│ --ObsDict-->   │
│  └──────┬───────┘     └────────┬─────────┘                 │
│         │ step(action)                  │                   │
│         │                               │ policy.predict()  │
│         │                               ↓                   │
│  ┌──────▼───────┐                ┌──────────────┐          │
│  │ MetricsReader │<-- metrics ---│   Policy      │          │
│  └──────┬───────┘                └──────┬───────┘          │
│         │                                 │ action          │
│         ▼                                 ▼                 │
│  ┌──────────────┐                ┌──────────────┐          │
│  │ ResultSaver   │                │ RemoteClient  │--RPC--> │
│  └──────────────┘                └──────────────┘          │
└────────────────────────────────────────────────────────────┘
                             │
                             │  TCP/IP
                             ▼
┌────────────────────────────────────────────────────────────┐
│                        Model Server                          │
│                                                            │
│  ┌──────────────┐    ┌──────────────────┐                  │
│  │ PolicyServer  │ -> │   VLA Model       │ -> action/chunk │
│  └──────────────┘    └──────────────────┘                  │
└────────────────────────────────────────────────────────────┘
```

### 2.2 谁负责什么（给环境侧看的重点）

- **环境侧必须提供**
  - 观测所需原始数据的权威来源（机器人状态、相机/传感器数据、指令等）
  - **metrics 的权威来源**：`command_term.metrics`（dict），以及必要状态量（例如 `order_completion`）
  - 统一的命名/访问路径（例如 command_term 名称、sensor 名称）

- **评估侧负责**
  - `ObservationBuilder`：把环境原始数据整理成稳定的 `ObservationDict`
  - `Policy`：本地或远程推理，输出 `ActionDict`
  - `MetricsReader`：从环境只读指标，输出 `MetricsReadout`
  - `ResultSaver`：结果落盘（episode/task 两级）
  - `RemoteClient`：RPC 超时/重试/延迟与失败统计

---

## 3. 关键类设计（职责 + 方法签名级别接口）

### 3.1 `Evaluator`（评估主循环基类）

**职责**
- 管理 task/episode/step 三层循环
- 统一错误处理（网络失败、环境异常、超时）
- 组织汇总与落盘（调用 `ResultSaver`）

**关键方法（签名级别）**
- `__init__(tasks, n_episodes, max_steps, num_envs, seed, save_dir, policy, observation_builder, metrics_reader, result_saver, ...)`
- `run() -> EvaluationReport`
- `run_task(task_name) -> TaskReport`
- `run_episode(task_name, episode_id) -> EpisodeReport`

---

### 3.2 `VLAEvaluator`（跑 IsaacLab 环境的评估实现）

**职责**
- 创建/配置 IsaacLab env（task、num_envs、device、headless）
- step 循环：`obs -> policy.predict -> action -> env.step`
- episode 结束：`MetricsReader.read(env)`（只读）+ 结果落盘

**关键方法**
- `make_env(task_name) -> env`
- `step_loop(env, ...) -> EpisodeRollout`（概念：记录 done、step 数、timing 等）

---

### 3.3 `Policy`（统一策略接口：本地/远程一致）

**职责**
- 输入：`ObservationDict`
- 输出：`ActionDict`

**关键方法**
- `name() -> str`
- `reset(context: EpisodeContext) -> None`
- `predict(observation: ObservationDict) -> ActionDict`

---

### 3.4 `RemoteClient`（远程推理客户端：网络层）

**职责**
- 序列化 `ObservationDict`，发到 Model Server，接收 `ActionDict`
- timeout/retry/backoff
- 统计 latency、失败原因

**关键方法**
- `request(observation: ObservationDict) -> ActionDict`
- `health_check() -> bool`（可选）

---

### 3.5 `ObservationBuilder`（观测聚合器：你要的 get_observation）

**职责**
- 从 env/scene/sensors 获取原始数据
- 按需开关：rgb/depth/seg/pcd
- 多相机选择：camera_names（顺序稳定）
- 输出稳定 `ObservationDict`

**关键方法**
- `build(env, task_name, episode_id, step_id, require: ObservationRequire, camera_names=None) -> ObservationDict`

**ObservationRequire（概念）**
- `require_rgb: bool`
- `require_depth: bool`
- `require_seg: bool`
- `require_pcd: bool`
- `pcd_frame: "world" | "camera"`

---

### 3.6 `MetricsReader`（指标读取器：只读）

**职责**
- 从环境读取 `metrics` dict（权威来源）
- 必要时读取 `order_completion/object_states` 等状态用于 success/调试
- 统一成 `MetricsReadout`（可序列化）

**关键方法**
- `read(env, command_term_name="order_command", reduce="none") -> MetricsReadout`

---

### 3.7 `ResultSaver`（结果落盘）

**职责**
- 写 `episodes.jsonl`（断点续写）
- 写 `task_summary.json`

**关键方法**
- `write_episode(episode_report: EpisodeReport) -> None`
- `write_task(task_report: TaskReport) -> None`

---

## 4. 时序：一次 episode 发生什么

```
run_episode()
  ├─ env.reset(seed)
  ├─ policy.reset(...)
  ├─ for step in range(max_steps):
  │    ├─ obs = ObservationBuilder.build(env, ...)
  │    ├─ action = Policy.predict(obs)            # 远程时会走 RemoteClient
  │    ├─ obs2, rew, terminated, truncated, info = env.step(action)
  │    └─ if done: break
  ├─ metrics_read = MetricsReader.read(env)       # 只读
  └─ ResultSaver.write_episode(...)
```

---

## 5. 数据 schema（观测 / 动作 / 指标）

### 5.1 `ObservationDict`（Benchmark -> Policy/Model）

- `meta`: `{task_name, episode_id, step_id, num_envs, timestamp?, units?}`
- `robot_state`（必选）
  - `qpos`: `(num_envs, J)` float
  - `qvel`: `(num_envs, J)` float
  - `qacc`: `(num_envs, J)` float（可选；没有就全 0 并在 meta 标注）
- `instruction`（可选）
  - `text`: `str` 或 `List[str]`
- `vision`（可选）
  - `cameras`: `List[str]`
  - `rgb`: `(C, num_envs, H, W, 3)` uint8/float（归一化约定写 meta）
  - `depth`: `(C, num_envs, H, W)` float32（单位 m）
  - `segmentation`: `(C, num_envs, H, W)` int32（instance/semantic 约定写 meta）
  - `robot_mask`: `(C, num_envs, H, W)` uint8（0/1）
  - `intrinsic`: `(C, num_envs, 3, 3)` float32
  - `extrinsic`: `(C, num_envs, 4, 4)` float32（方向约定写 meta）
- `point_cloud`（可选）
  - `masked_point_cloud`: `(num_envs, N, 3)` float32
  - `frame`: `"world" | "camera"`

### 5.2 `ActionDict`（Policy/Model -> Benchmark）

- `action`: `(num_envs, A)` float32
- `action_space`: `"joint"`（推荐统一 joint；若 ee，需要 Benchmark Server 做 IK）

可选提效：
- `action_chunk`: `(K, num_envs, A)` float32
- `chunk_horizon`: int = K

### 5.3 `MetricsReadout`（只读：Env -> Evaluator）

**权威入口约定**
- `env.unwrapped.command_manager.active_terms[command_term_name].metrics`（dict）
- 必要时补读：`order_completion/object_states`

**输出格式**
- `metrics`: `Dict[str, MetricValue]`
- `reduce`: `"none" | "mean" | "first"`（多 env 对齐方式）
- `num_envs`: int

`MetricValue`：
- 标量：float/int/bool
- 或 `List[scalar]`（长度 = num_envs）

---

## 6. 与环境侧同学的对齐清单（Checklist）

### 6.1 metrics 对齐
- [ ] command_term 名称是否统一（例如 `"order_command"`）
- [ ] `command_term.metrics` 至少包含：`object_success_rate`, `order_success_rate`（以及建议 `success`）
- [ ] metrics 在 `num_envs>1` 时的形态：每 env 一项还是已聚合？

### 6.2 observation 对齐
- [ ] 机器人：qpos/qvel/qacc 是否都能取到（没有 qacc 则约定为 0）
- [ ] 相机命名集合与顺序：camera_names
- [ ] RGB/Depth/Seg 的 dtype 与单位约定
- [ ] intrinsic/extrinsic 的方向约定（world->cam / cam->world）
- [ ] segmentation 的语义（instance/semantic）与 robot_mask 规则
- [ ] 点云坐标系（world/camera）与单位

### 6.3 远程推理对齐
- [ ] ObservationDict/ActionDict 的 schema 固化（第 5 节）
- [ ] payload 控制策略（分辨率/相机数/是否传 pcd）
- [ ] timeout/retry/backoff 策略

# IsaacLab Logistics VLA Benchmark 评估系统设计

---

## 0. 背景与关键约束

- **两服务器**：Benchmark Server（跑 IsaacLab 环境）与 Model Server（跑模型推理）不在同一台机器，需要网络 RPC。
- **指标只读**：**指标由环境内部计算**（例如 `OrderCommandTerm.metrics`），评估侧只做**读取/对齐/汇总/保存**。
- **多模态观测**：模型可能需要多视角 `rgb/depth/seg/pcd` + 机器人状态 + 指令，需要一个“类似 get_observation()”的聚合接口。

---

## 1. 总体架构：模块与职责边界

### 1.1 逻辑模块

```
┌────────────────────────────────────────────────────────────┐
│                       Benchmark Server                       │
│                                                            │
│  ┌──────────────┐     ┌──────────────────┐                 │
│  │ IsaacLab Env  │ --> │ ObservationBuilder│ --ObsDict-->   │
│  └──────┬───────┘     └────────┬─────────┘                 │
│         │ step(action)                  │                   │
│         │                               │ policy.predict()  │
│         │                               ↓                   │
│  ┌──────▼───────┐                ┌──────────────┐          │
│  │ MetricsReader │<-- metrics ---│   Policy      │          │
│  └──────┬───────┘                └──────┬───────┘          │
│         │                                 │ action          │
│         ▼                                 ▼                 │
│  ┌──────────────┐                ┌──────────────┐          │
│  │ ResultSaver   │                │ RemoteClient  │--RPC--> │
│  └──────────────┘                └──────────────┘          │
└────────────────────────────────────────────────────────────┘
                             │
                             │  TCP/IP
                             ▼
┌────────────────────────────────────────────────────────────┐
│                        Model Server                          │
│                                                            │
│  ┌──────────────┐    ┌──────────────────┐                  │
│  │ PolicyServer  │ -> │   VLA Model       │ -> action/chunk │
│  └──────────────┘    └──────────────────┘                  │
└────────────────────────────────────────────────────────────┘
```

### 1.2 谁负责什么（给环境侧看的最重要）

- **环境侧必须提供**：
  - 观测所需原始数据的“权威来源”（机器人状态、相机/传感器数据、指令等）
  - **metrics 的权威来源**：`command_term.metrics`（dict），以及必要的状态量（如 `order_completion`）
  - 统一的命名/访问路径（例如 command_term 名称、sensor 名称）

- **评估侧负责**：
  - 用 `ObservationBuilder` 把“环境原始数据”整理成稳定的 `ObservationDict`
  - 用 `Policy`（本地或远程）得到动作 `ActionDict`
  - 用 `MetricsReader` 从环境读取 metrics，统一成 `MetricsReadout`
  - 用 `ResultSaver` 按 schema 落盘（episode/task 两级）
  - 在两服务器模式下，负责 RPC 超时/重试/统计延迟

---

## 2. 关键类设计（无代码，只有职责与方法签名）

下面的类名是“概念设计”，落地时文件路径可调整，但**职责与接口建议保持**。

### 2.1 `Evaluator`（基类：评估主循环）

**职责**：
- 管理 task/episode/step 三层循环
- 统一结果结构与落盘
- 统一错误处理（网络失败、环境异常、超时等）

**关键方法（签名级别）**：
- `__init__(tasks, n_episodes, max_steps, num_envs, seed, save_dir, policy, observation_builder, metrics_reader, result_saver, ...)`
- `run() -> EvaluationReport`
- `run_task(task_name) -> TaskReport`
- `run_episode(task_name, episode_id) -> EpisodeReport`

**与环境侧关系**：不直接知道环境内部细节，只调用 `env.reset/step`，以及交给 `ObservationBuilder/MetricsReader` 去做“理解环境结构”的事情。

---

### 2.2 `VLAEvaluator`（实现：跑 IsaacLab 的评估器）

**职责**：
- 创建/配置 IsaacLab env（task 选择、num_envs、device、headless 等）
- 每步：
  1) 获取观测（通过 ObservationBuilder）
  2) 调用 policy（本地或远程）
  3) `env.step(action)`
- episode 结束：
  - `MetricsReader.read(env)` 读取 metrics（只读）
  - 写入 episode 结果

**关键点**：
- **policy 是可插拔的**：随机、轨迹、本地 VLA、远程 VLA 都可以
- **metrics 是可插拔的**：只要环境把指标放到约定位置（见第 5 节），评估侧不用改计算逻辑

---

### 2.3 `Policy`（统一策略接口：本地/远程一致）

**职责**：
- 输入：`ObservationDict`
- 输出：`ActionDict`

**关键方法**：
- `name() -> str`
- `reset(context: EpisodeContext) -> None`（可选：远程策略也可将 reset 透传给 server）
- `predict(observation: ObservationDict) -> ActionDict`

**约束**：
- `ActionDict.action` 必须能直接用于 IsaacLab `env.step(action)`（推荐统一 joint action）

---

### 2.4 `RemoteClient`（远程推理客户端：只处理网络）

**职责**：
- 负责把 `ObservationDict` 序列化、发送到 Model Server、接收 `ActionDict`
- 负责 timeout/retry/backoff
- 负责记录 latency、失败原因

**关键方法**：
- `request(observation: ObservationDict) -> ActionDict`
- `health_check() -> bool`（可选）

**协议选择**：
- HTTP / ZMQ / WebSocket 都可；但**schema 必须一致**（见第 4、5 节）

---

### 2.5 `PolicyServer`（Model Server 端：对外服务壳）

**职责**：
- 接收请求（ObservationDict）
- 调用实际模型（VLA Model）
- 返回动作（ActionDict / action_chunk）

**注意**：
- Model Server 只管推理，不知道 IsaacLab，不需要环境依赖
- Model Server 的输入输出完全由 schema 驱动

---

### 2.6 `ObservationBuilder`（观测聚合器：你要的 get_observation() 在这里）

**职责**：把 IsaacLab 环境里的“原始数据”聚合成统一 schema。

**关键方法**：
- `build(env, task_name, episode_id, step_id, require: ObservationRequire, camera_names: list[str] | None) -> ObservationDict`

**ObservationRequire**（概念）：
- `require_rgb: bool`
- `require_depth: bool`
- `require_seg: bool`
- `require_pcd: bool`
- `pcd_from: "env" | "rgbd"`（可选：点云在 env 生成还是用 rgbd 生成）

**环境侧需要配合的点**（非常关键）：
- 相机/传感器的命名与访问路径（例如 `scene.sensors["cam_front"]`）
- `rgb/depth/seg` 的 dtype/单位/坐标约定
- `intrinsic/extrinsic` 的方向约定（world->cam 还是 cam->world）
- robot_mask 的生成规则（instance id 范围？semantic label？）

---

### 2.7 `MetricsReader`（指标读取器：只读）

**职责**：从环境读取指标与必要状态，输出统一的 `MetricsReadout`。

**关键方法**：
- `read(env, command_term_name="order_command", reduce="none") -> MetricsReadout`

**它不做的事**：
- 不计算成功率/进度等逻辑（这些属于环境内部）

---

### 2.8 `ResultSaver`（结果落盘）

**职责**：
- 把 episode/task 级结构写入 `episodes.jsonl` / `task_summary.json`
- 支持断点续写（jsonl）

**关键方法**：
- `write_episode(episode_report: EpisodeReport) -> None`
- `write_task(task_report: TaskReport) -> None`

---

## 3. 时序：一次 episode 到底发生什么（讲解版）

```
Evaluator.run_episode()
  ├─ env.reset(seed)
  ├─ policy.reset(...)
  ├─ for step in range(max_steps):
  │    ├─ obs = ObservationBuilder.build(env, ...)
  │    ├─ action = Policy.predict(obs)
  │    ├─ env.step(action)
  │    └─ if done: break
  ├─ metrics = MetricsReader.read(env)      # 只读
  └─ ResultSaver.write_episode(...)
```

两服务器情况下：

```
Policy.predict(obs)
  └─ RemoteClient.request(obs)  --RPC-->  PolicyServer  -->  VLA Model
                                           <--ActionDict--
```

---

## 4. 数据 schema（观测 / 动作）：模型侧必须依赖这个

> 这一节用于你和模型侧同学对齐“输入输出长什么样”。  
> 实际序列化可 JSON/msgpack/自定义二进制，但**字段与 shape 必须一致**。

### 4.1 `ObservationDict`（Benchmark -> Policy/Model）

- **meta**
  - `task_name`: str
  - `episode_id`: int
  - `step_id`: int
  - `num_envs`: int
  - `timestamp`: float（可选）
  - `units`: dict（可选：depth 单位、pcd 坐标系、extrinsic 方向）

- **robot_state**（必选）
  - `qpos`: `(num_envs, J)` float
  - `qvel`: `(num_envs, J)` float
  - `qacc`: `(num_envs, J)` float（可选；没有就全 0，并在 meta 标注）

- **instruction**（可选）
  - `text`: `str` 或 `List[str]`

- **vision**（可选）
  - `cameras`: `List[str]`（顺序定义了以下数组第 0 维）
  - `rgb`: `(C, num_envs, H, W, 3)` uint8 或 float（是否归一化需在 meta 标注）
  - `depth`: `(C, num_envs, H, W)` float32（单位 m）
  - `segmentation`: `(C, num_envs, H, W)` int32（instance id / semantic id 需写入 meta）
  - `robot_mask`: `(C, num_envs, H, W)` uint8（0/1）
  - `intrinsic`: `(C, num_envs, 3, 3)` float32
  - `extrinsic`: `(C, num_envs, 4, 4)` float32（world->cam 或 cam->world 需写入 meta）

- **point_cloud**（可选）
  - `masked_point_cloud`: `(num_envs, N, 3)` float32
  - `frame`: `"world"` / `"camera"`（推荐 world）

### 4.2 `ActionDict`（Policy/Model -> Benchmark）

- `action`: `(num_envs, A)` float32
- `action_space`: `"joint"`（推荐统一 joint；若 ee，需要在 Benchmark Server 做 IK）

可选提效：
- `action_chunk`: `(K, num_envs, A)` float32
- `chunk_horizon`: int = K

---

## 5. Metrics：强调“环境计算，评估只读” + 读取接口与数据格式

### 5.1 环境侧需要提供的“权威入口”

评估侧约定从以下位置读取（你们可以统一命名）：

- `env.unwrapped.command_manager.active_terms[command_term_name].metrics`（dict）

并在必要时补读：
- `order_completion`（用于 success 判定/调试）
- `object_states`（用于调试/可视化）

> **关键：评估侧不做公式**。  
> 环境侧如果要新增指标，只需要往 `metrics` dict 里多加 key/value。

### 5.2 `MetricsReadout`（评估侧“已读指标”的统一格式）

评估侧输出（可序列化）：

- `metrics`: `Dict[str, MetricValue]`
- `reduce`: `"none" | "mean" | "first"`（多 env 时对齐方式）
- `num_envs`: int

`MetricValue` 允许：
- 标量：float/int/bool
- 多 env：`List[scalar]`（长度 = num_envs）

推荐策略：
- `episodes.jsonl`：`reduce="none"`（保留每个 env 值，后续离线聚合）
- `task_summary.json`：再做 `mean/std` 聚合

### 5.3 指标 key 命名建议（给环境侧同学的对齐清单）

你们当前已有：
- `object_success_rate`
- `order_success_rate`

强烈建议环境也直接提供：
- `success`（bool/0-1；否则评估侧会从 `order_completion.all()` 派生，但不如环境侧统一）

后续可选：
- `progress_score`
- `intention_score`
- 任何你们想要记录的 debug 指标（都放进 `metrics`）

---

## 6. 结果落盘：给评估侧/实验分析用

### 6.1 `episodes.jsonl`（每行一个 episode，便于断点续写）

每条记录建议包含：
- `task_name`
- `policy_name`
- `episode_id`
- `seed`
- `success`
- `episode_length`
- `metrics_read`: `MetricsReadout`（第 5 节）
- `timing`（两服务器必须要）：
  - `avg_latency_ms`
  - `p95_latency_ms`
  - `net_fail_count`
  - `error_types`（dict：timeout/conn_reset/...）

### 6.2 `task_summary.json`（每个 task 一份）

- `task_name`
- `policy_name`
- `n_episodes`
- `success_rate`
- `avg_episode_length`
- `metrics_agg`（mean/std 等）
- `failures`（网络/异常统计）
- `timing`（延迟统计）

---

## 7. 与环境侧同学的“接口对齐 checklist”（最实用）

### 7.1 metrics 对齐

- [ ] 确认 `command_term_name`（比如 `"order_command"`）在所有 task 中一致
- [ ] 确认 `command_term.metrics` 的 key 列表（至少包含 `object_success_rate/order_success_rate/success`）
- [ ] 确认 metrics 的 shape（单 env 标量 vs 多 env tensor），以及多 env 时是否每个 env 一项

### 7.2 observation 对齐

- [ ] 机器人状态来源：`qpos/qvel/qacc` 是否能稳定取到（没有 qacc 则统一为 0）
- [ ] 相机列表与命名：`camera_names` 的可用集合
- [ ] RGB/Depth/Seg 的 dtype 与单位
- [ ] `intrinsic/extrinsic` 的定义方向（world->cam or cam->world）
- [ ] segmentation 的语义（instance id / semantic id）与 robot_mask 的规则
- [ ] 点云坐标系与单位（world/camera）

### 7.3 远程推理对齐

- [ ] 观测/动作 schema 固化（第 4 节）
- [ ] 最大 payload 控制策略（分辨率/相机数/是否传 pcd）
- [ ] 超时/重试策略（timeout_ms, retries, backoff）

---

## 8. 最小可用实施路线（你可以用来排期）

- **阶段 A：闭环（不含视觉/远程）**
  - random/trajectory policy
  - metrics 只读 + episodes.jsonl 落盘

- **阶段 B：远程推理闭环**
  - 固化 schema（obs/action）
  - 打通一种协议（HTTP 或 ZMQ）
  - timing/failures 统计落盘

- **阶段 C：多模态观测逐步打开**
  - rgb -> depth/seg -> pcd（逐步增加，逐步压 payload）

---

## 9. 验收标准（讲解版）

- [ ] 环境侧能稳定提供 `metrics`（dict）与必要状态量（order_completion 等）
- [ ] 评估侧能在两服务器下跑完 N episodes，并输出：
  - episodes.jsonl（含 metrics_read + timing）
  - task_summary.json
- [ ] schema 稳定：换模型/换 task 时，不需要改评估主循环，只需要改 policy 或 env 配置

# IsaacLab Logistics VLA Benchmark 评估系统设计

---

## 0. 设计目标与关键约束

- **目标**：在 `isaaclab_logistics_vla` 上复用 VLABench 的评估思想：Evaluator/Policy 解耦、可插拔模型、可复现实验、结果可落盘与可分析。
- **约束 1（两服务器）**：benchmark（IsaacLab 环境）与模型推理**不在同一台服务器**，需要网络 RPC。
- **约束 2（指标只读）**：**metrics 由环境内部计算**（例如 `OrderCommandTerm.metrics`），评估模块只负责**读取/对齐/汇总/保存**，不负责计算逻辑。
- **约束 3（多模态观测）**：模型可能需要多视角 `rgb/depth/seg/pcd`，需要一个类似 `get_observation()` 的**观测聚合接口**与稳定 schema。

---

## 1. 模块总览（推荐目录结构）

```
isaaclab_logistics_vla/evaluation/
├── evaluator/
│   ├── base.md                # 评估主循环职责说明（无代码）
│   ├── vla_evaluator.md        # VLA 评估流程说明（无代码）
│   └── vlm_evaluator.md        # 可选：VLM-only 评估说明（无代码）
│
├── policy/
│   ├── policy_interface.md     # Policy 接口契约（核心）
│   └── remote_protocol.md      # 远程推理协议与 schema（核心）
│
├── observation/
│   ├── observation_builder.md  # 观测聚合接口设计（类似 get_observation）
│   └── observation_schema.md   # Observation schema（核心）
│
├── metrics/
│   ├── metrics_reader.md       # 如何从环境读取 metrics（核心）
│   └── metrics_schema.md       # Metrics schema（核心）
│
└── results/
    ├── result_schema.md        # 落盘文件结构与字段（核心）
    └── failure_and_timing.md    # 网络失败/延迟统计口径（推荐）
```

> 说明：此目录是“设计文档结构”。落地实现时可以把 `.md` 替换成 `.py`，但 **schema/契约优先**。

---

## 2. 两服务器运行形态（Benchmark Server + Model Server）

### 2.1 角色分工

- **Benchmark Server（跑 IsaacLab）**
  - 负责：环境 step、传感器数据产生、（可选）点云生成、**metrics 计算（在 env 内部）**、结果落盘。
  - 负责：将观测按 schema 打包发送给 Model Server，接收动作。

- **Model Server（只做推理）**
  - 负责：接收观测（序列化形式）、执行模型推理、返回动作（或 action chunk）。

### 2.2 评估数据流（高层）

1. `env.reset()` 得到初始状态
2. `observation_builder.build()` 生成 `ObservationDict`
3. `policy.predict(ObservationDict)`（本地或远程）
4. `env.step(action)`
5. episode 结束：`metrics_reader.read(env)` 读取**环境已计算**的 metrics
6. 写入结果文件（见第 6 节）

### 2.3 网络层要求（必须）

- **固定请求/响应 schema**（见第 4、5 节）
- **可配置**：
  - `timeout_ms`
  - `retries` / `backoff_ms`
  - `max_payload_bytes`
- **必须记录**：
  - 每次请求的 latency（至少平均 + P95）
  - 网络失败次数与失败原因（timeout/conn_reset/5xx 等）

---

## 3. 观测系统：如何在 IsaacLab 里做“get_observation()”

你给的 MuJoCo 风格 `get_observation()`，在 IsaacLab 里建议拆成两层：

### 3.1 环境侧：负责“产出/缓存”

- 通过 IsaacLab 的 sensor（Camera/TiledCamera 等）在仿真步内产生数据
- ObservationManager（或你自己的接口）提供**可直接访问的 tensor 数据**

### 3.2 评估侧：负责“聚合/裁剪/格式化”（推荐）

定义一个**观测聚合器**（概念名：`ObservationBuilder`），其职责是：

- 按需开关：`require_rgb/depth/seg/pcd`
- 多相机选择：`camera_names`（顺序稳定）
- 将各种来源的数据整理成稳定的 `ObservationDict`（见第 4 节）
- 控制数据体积：分辨率/相机数/是否下采样/是否只传关键视角

### 3.3 设计要点（强建议）

- **大数据可选**：默认只提供 state；需要多模态时显式开启。
- **点云生成位置**：
  - **优先**：Benchmark Server 生成（更贴近传感器侧，也减少模型侧依赖）
  - 可选：传 RGBD 到 Model Server 再生成（带宽更高、延迟更大）
- **Seg 规则要固化**：instance/semantic 的含义、robot_mask 生成规则都必须写进 schema，不然跨场景会漂。

---

## 4. Observation schema（Benchmark -> Policy/Model）

统一输出一个字典 `ObservationDict`。实现上可用 JSON/msgpack/自定义二进制，但**字段名、shape、dtype 约定不变**。

### 4.1 顶层结构

- **meta**
  - `task_name`: str
  - `episode_id`: int
  - `step_id`: int
  - `num_envs`: int
  - `timestamp`: float（可选）
  - `units`: dict（可选，例如 depth 单位、pcd 坐标系）

- **robot_state**（至少要有）
  - `qpos`: shape `(num_envs, J)`，float32/float64
  - `qvel`: shape `(num_envs, J)`，float32/float64
  - `qacc`: shape `(num_envs, J)`，float32/float64（可选：没有就置空或全 0，并在 meta 标注）

- **instruction**（可选）
  - `text`: str 或 `List[str]`（并行 env）

- **vision**（可选）
  - `cameras`: `List[str]`（相机名；其顺序定义了以下数组的第 0 维）
  - `rgb`: shape `(C, num_envs, H, W, 3)`，uint8 或 float32（必须在 meta 标注是否归一化）
  - `depth`: shape `(C, num_envs, H, W)`，float32（必须在 meta 标注单位：m）
  - `segmentation`: shape `(C, num_envs, H, W)`，int32（必须注明：instance id / semantic id）
  - `robot_mask`: shape `(C, num_envs, H, W)`，uint8（0/1）
  - `intrinsic`: shape `(C, num_envs, 3, 3)`，float32
  - `extrinsic`: shape `(C, num_envs, 4, 4)`，float32（必须注明方向：world->cam 或 cam->world）

- **point_cloud**（可选）
  - `masked_point_cloud`: shape `(num_envs, N, 3)`，float32
  - `frame`: `"world"` / `"camera"`（推荐 world）

### 4.2 体积控制建议（不写死，但要在实现里可配置）

- 限制 `C/H/W/N`
- 支持下采样（RGB/Depth/PCD）
- 支持只传关键视角（例如 front/top）

---

## 5. Action schema（Policy/Model -> Benchmark）

### 5.1 最小动作格式（推荐）

- `action`: shape `(num_envs, A)`，float32
- `action_space`: `"joint"`（推荐统一 joint，直接 `env.step(action)`）

### 5.2 可选扩展：action chunk（远程推理提效）

- `action_chunk`: shape `(K, num_envs, A)`，float32
- `chunk_horizon`: int = K

> 说明：chunk 可以显著减少网络往返次数，特别是高频控制时。

---

## 6. Metrics：只读读取接口 + 期望数据格式

### 6.1 强调（再次）

**metrics 不需要你在 evaluation 里做计算**。  
evaluation 模块只需要：

- 知道“从哪读”
- 读到后“如何变成可序列化格式”
- 如何在 episode/task 维度汇总并保存

### 6.2 从哪里读取（约定一个权威入口）

推荐统一从（概念）：

- `env.unwrapped.command_manager.active_terms[command_term_name].metrics`（dict）

并在必要时补读：

- `order_completion` / `object_states`（用于 success 判定或调试字段）

### 6.3 MetricsReadout schema（评估侧输出的“已读指标”）

统一输出一个可序列化字典 `MetricsReadout`：

- `metrics`: `Dict[str, MetricValue]`
- `reduce`: `"none"` / `"mean"` / `"first"`（当 `num_envs>1` 时如何对齐输出）
- `num_envs`: int

其中 `MetricValue` 允许：

- **标量**：float/int/bool（单环境或聚合后）
- **多环境列表**：`List[scalar]`（长度 = num_envs）

> 推荐：保存 episode 结果时，用 `reduce="none"`（保留每个 env 的值）；task_summary 再做聚合。

### 6.4 指标 key 命名建议（环境侧提供）

你环境里已有的建议保持：

- `object_success_rate`
- `order_success_rate`

另外建议环境也提供：

- `success`（bool 或 0/1；若没有，评估可从 `order_completion.all()` 派生，但更推荐环境直接写入 metrics）
- 可选：`progress_score`, `intention_score`（后续再加也不影响评估结构）

---

## 7. 结果落盘格式（Evaluator 输出）

只定义格式，不限制实现语言。

### 7.1 每个 task 输出

- `task_summary.json`
  - `task_name`
  - `policy_name`
  - `n_episodes`
  - `success_rate`
  - `avg_episode_length`
  - `metrics_agg`（对 `MetricsReadout` 的聚合结果，如 mean/std）
  - `failures`（网络/超时/异常计数）
  - `timing`（avg/p95 latency 等）

- `episodes.jsonl`（每行一个 episode，便于断点续写）
  - `episode_id`
  - `seed`
  - `success`
  - `episode_length`
  - `metrics_read`（即 `MetricsReadout`）
  - `timing`（可选：avg_latency_ms、p95_latency_ms、net_fail_count）

### 7.2 可选：step 级日志（默认关闭）

- `steps.jsonl`
  - 每 N 步记录一次：`step_id`、`metrics_snapshot`、`latency_ms` 等

---

## 8. 最小可用实施路线（MVP）

- **A：先打通闭环（不含视觉）**
  - Observation 仅 `robot_state` + `instruction`（可选）
  - Policy 用 random/trajectory
  - metrics 只读 + episodes.jsonl 落盘

- **B：打通远程推理**
  - 固化第 4/5 的 schema
  - 先支持一种协议（HTTP 或 ZMQ）
  - 加入 timing/failures 统计

- **C：逐步打开多模态**
  - 先 rgb，再 depth/seg，再 pcd
  - 每加一种都确认 payload 与延迟可接受

---

## 9. 验收标准（精简）

- **评估闭环**：能跑 task/episode/step，并按第 7 节落盘。
- **指标只读**：`metrics_read` 全部来自环境（evaluation 不写计算逻辑）。
- **远程可用**：两服务器部署下可稳定跑 N 个 episode，且输出 timing/failures。
- **schema 稳定**：观测/动作/指标/结果字段与 shape 不随实现变化。

---

## 附录

- `EVALUATION_ARCHITECTURE.md`（VLABench 评估理念参考）
- IsaacLab Sensors（Camera/TiledCamera）文档（用于实现侧）
