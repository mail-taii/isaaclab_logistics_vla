# Isaac Lab Logistics VLA 中的 CuRobo 封装与设计说明

本文面向需要在 **isaaclab_logistics_vla** 里使用或扩展 **CuRobo 运动规划** 的开发者，说明：

- 为什么要在仿真评估链路里接 CuRobo；
- **RoboTwin** 项目里可借鉴的封装范式；
- 本仓库 **升级前 / 升级后** 的架构差异；
- **`CuroboPlanner`** 的 API、数据流与配置入口；
- 后续可扩展方向。

---

## 一、问题背景：CuRobo 在 benchmark 里扮演什么角色

在 VLA（视觉-语言-动作）或末端空间（EE）策略的评估中，策略往往输出 **末端位移增量** 或 **目标位姿**，而 Isaac Lab 环境执行的是 **关节空间动作**。这中间需要一类 **「语义动作 → 可执行关节轨迹」** 的模块。

**NVIDIA CuRobo** 基于 GPU 提供碰撞感知下的 **MotionGen**（图搜索 + 轨迹优化），适合作为该环节的 **规划后端**。本仓库将其用于：

- **EE 控制模式**：`VLA_Evaluator` 在 `_convert_actions_by_control_mode` 中把策略的 EE 增量转为关节目标；
- **演示策略**：`CuroboReachBoxPolicy` 用固定箱子目标做多姿态尝试规划，验证场景与坐标系是否一致。

因此，CuRobo 在这里的定位是 **动作编译器 / 驱动器**：上层只关心「臂基系下末端位姿 + 当前关节角」，下层产出 **时间序列上的关节位置（及速度）**。

---

## 二、参考范式：RoboTwin 里怎么做

**RoboTwin**（Sapien 仿真 + 双臂任务）把 CuRobo 收进一个名为 **`CuroboPlanner`** 的类中，核心思想是：

1. **构造期**：读机器人 `curobo.yml`、构造简化的 world（如桌面 cuboid）、创建并 **warmup** `MotionGen`（单路 + batch 两实例）。
2. **规划期**：输入 **当前全关节向量** 与 **世界系下的目标位姿**（经 `Robot` 层做夹爪→末端 link 变换），在 planner 内做 **世界系 → 臂基系** 变换与 **frame_bias** 修正，再调用 `plan_single` / `plan_batch`。
3. **输出**：统一为 **Python dict**，`status` 为 `"Success"` / `"Fail"`，`position` / `velocity` 为 **CPU 上的 numpy 数组**，避免上层直接依赖 CuRobo 的 Tensor 类型。
4. **夹爪**：RoboTwin 里夹爪轨迹用 **线性插值**，不走 CuRobo；与本仓库「臂用 CuRobo、夹爪单独处理」的思路一致。
5. **双臂隔离**：左右臂若使用 **不同 yml**，通过 **子进程 + Pipe** 各跑一个 `CuroboPlanner`，避免单进程双实例问题。

这种设计的优点是：**任务脚本 / 评估循环只认「dict + numpy」**，换规划器或做单元测试时边界清晰。

---

## 三、本仓库升级前的形态（函数式封装）

在引入 **`CuroboPlanner` 类** 之前，本仓库采用 **`evaluation/curobo/planner.py`** 中的两个函数：

| 函数 | 作用 |
|------|------|
| `build_motion_gen(...)` | 按 `RobotEvalConfig` 加载 yml、覆盖 URDF/资产路径，按 `WorldMode` 组装 `WorldConfig`（桌子 / 箱子 cuboid / mesh / 空心箱），构建并 warmup `MotionGen`。 |
| `plan_single_ee_motion(motion_gen, q_start, target_pos_b, target_quat_b, ...)` | 假定目标已在 **臂基系**，封装张量形状与 `JointState` / `Pose` / `MotionGenPlanConfig`，返回 **CuRobo 原生 result 对象**。 |

**局限**：`vla_evaluator`、`curobo_reach_box_policy` 等上层需要：

- 持有 **`MotionGen` 实例**；
- 直接读 **`result.success`、`result.interpolated_plan.position`** 等 CuRobo 类型；
- 部分文件还曾 **`from curobo... import ...`**，封装层级相当于「**半封装**」：构建与单次调用集中了，但 **类型与返回值仍未与 CuRobo 解耦**。

---

## 四、升级后的核心：`CuroboPlanner` 与 `motion_gen_result_to_plan_dict`

升级目标是对齐 RoboTwin 的 **「类 + dict 输出」** 范式，同时 **保留** 原有 `build_motion_gen` / `plan_single_ee_motion`，供高级用法或测试复用。

### 4.1 类：`CuroboPlanner`

**位置**：`isaaclab_logistics_vla/evaluation/curobo/planner.py`

**构造参数**（与 benchmark 注册表一致）：

- `robot_eval_cfg`：`RobotEvalConfig`（`robot_registry` 中的 yml 名、资产目录、URDF 名等）；
- `curobo_device`：`torch.device` 或字符串（可与仿真 GPU 分离，减轻显存争抢）；
- `world_mode`：`"table_only" | "boxes_cuboid" | "boxes_mesh" | "boxes_hollow"`；
- `logger_name`：日志前缀。

**主要方法**：

- **`plan_ee(q_start, target_pos_b, target_quat_b, ...)`**  
  内部调用 `plan_single_ee_motion`，再经 **`motion_gen_result_to_plan_dict`** 转为标准 dict。  
  **约定**：`target_pos_b` / `target_quat_b` 已在 **CuRobo 臂基系**（与本仓库 EE 模式里 `subtract_frame_transforms` 后的语义一致）。

- **`reset(reset_seed=True)`**  
  转发 `motion_gen.reset`，与 RoboTwin 侧「重置规划器状态」的用法对齐。

- **`motion_gen`（只读属性）**  
  仅建议调试使用；业务代码应优先走 **`plan_ee`**。

### 4.2 结果字典：`motion_gen_result_to_plan_dict`

成功时典型结构：

```text
{
  "status": "Success",
  "position": np.ndarray,  # shape (T, dof), float32, CPU
  "velocity": np.ndarray or None,  # 与 position 同逻辑维度，若 CuRobo 提供
  "detail": str,  # 可选，来自 CuRobo 的 status 信息
}
```

失败时：

```text
{
  "status": "Fail",
  "position": None,
  "velocity": None,
  "detail": str,  # 若有则便于打印
}
```

这样 **评估器** 与 **策略** 只需判断 `result["status"] == "Success"`，并用 **`numpy` → `torch`** 拷回各自 device，无需 import `curobo`。

---

## 五、数据流：从策略到仿真一步

### 5.1 EE 模式（`VLA_Evaluator`）

1. 策略返回的动作 `actions` 前 3 维视为 **末端位移增量**（世界系）。
2. 从 `robot_data` 取当前 EE 位姿，得到 **目标 EE 世界坐标**。
3. 用 `arm_base_offset_in_root`、`platform_joint` 等与 **`combine_frame_transforms` / `subtract_frame_transforms`** 将目标变到 **臂基系** `target_ee_pos_b`、`ee_quat_b`。
4. 从 `robot_state["qpos"]` 取出左臂对应关节，调用 **`self._curobo_planner.plan_ee(...)`**。
5. 成功则取 **`result["position"][-1]`** 写回 `actions` 中左臂关节维度；失败则保持当前关节指令。

### 5.2 `CuroboReachBoxPolicy`

1. 将箱子目标点从世界系转到臂基系（与场景一致的简化平移或结合观测里的臂基位姿）。
2. 多组抓取姿态（欧拉角）循环调用 **`plan_ee`**，直到 `Success`。
3. 对 **`result["position"]`** 在时间上重采样到固定 horizon，填回全关节 `action` 向量中的左臂索引。

---

## 六、World 与配置入口

| 概念 | 说明 |
|------|------|
| **`RobotEvalConfig`** | `evaluation/robot_registry.py`：`curobo_yml_name`、`curobo_asset_folder`、`curobo_urdf_name`、`arm_dof`、`left_arm_joint_names`、`arm_base_offset_in_root`、`platform_joint_name` 等。 |
| **`WorldMode`** | 在 `planner.py` 的 `_build_world_cfg` 中分支：桌子 yaml、cuboid 箱子、`curobo_mesh_world` 的 mesh / 空心箱等。 |
| **环境变量** | 常见：`CUROBO_DEVICE`、`CUROBO_USE_MESH_OBSTACLES`、`CUROBO_HOLLOW_BOX`、`ASSET_ROOT_PATH`（mesh 资源根目录）。 |
| **评估脚本** | `scripts/evaluate_vla.py` 可将 CLI 参数写入上述环境变量，再构造 `VLA_Evaluator`。 |

机器人碰撞球、URDF 路径等仍由 **包内 `configs/robot_configs/`** 与 yml 共同决定，与升级前一致。

---

## 七、与 RoboTwin 的差异（读者需心中有数）

| 维度 | RoboTwin | 本仓库 |
|------|----------|--------|
| 仿真 | Sapien | Isaac Lab / PhysX |
| 目标位姿输入 | 经 `Robot` 做夹爪→link 变换 + 世界→基座 | EE 模式在 evaluator 内用 Isaac 链式变换到臂基系；ReachBox policy 用约定世界点 + 臂基平移 |
| World | 代码里写死桌面 cuboid 为主 | `WorldMode` + yaml + mesh 工具，更贴近当前 benchmark 场景 |
| 双臂 CuRobo | 双进程 + 双 yml | 当前评估以 **单臂规划** 为主（左臂），未做 RoboTwin 式双进程隔离 |
| Batch 规划 | `plan_batch` + 第二套 MotionGen | 可按同样模式扩展 `CuroboPlanner.plan_ee_batch`，尚未默认开启 |

---

## 八、扩展建议

1. **`plan_ee_batch`**：与 RoboTwin 一致，对多候选末端位姿一次 `plan_batch`，减少 Python 循环与多次单条规划开销（需额外 warmup 的 `MotionGen` 实例与 `CONFIGS.ROTATE_NUM` 类配置）。
2. **统一「世界系目标」入口**：在 `CuroboPlanner` 外层再包一层，输入世界系位姿 + 当前 `root/arm_base` 位姿，在类内完成变换，进一步减轻 evaluator 代码量（需注意与现有 Isaac 坐标约定严格一致）。
3. **后端可插拔**：定义抽象 `EeMotionPlanner`，`CuroboPlanner` 为实现类，便于接入 OMPL、纯 IK 等对比实验。
4. **动态障碍物**：若将来从场景更新点云 world，需在 planner 内暴露与 CuRobo API 一致的更新接口，并保证与 `WorldMode` 使用的 checker 类型匹配。

---

## 九、相关文件索引

| 路径 | 职责 |
|------|------|
| `isaaclab_logistics_vla/evaluation/curobo/planner.py` | `build_motion_gen`、`plan_single_ee_motion`、`motion_gen_result_to_plan_dict`、**`CuroboPlanner`** |
| `isaaclab_logistics_vla/utils/curobo_mesh_world.py` | 箱子 mesh / 空心 cuboid world 构造 |
| `isaaclab_logistics_vla/evaluation/evaluator/vla_evaluator.py` | 构造 **`CuroboPlanner`**，EE 模式调用 **`plan_ee`** |
| `isaaclab_logistics_vla/evaluation/models/policy/curobo_reach_box_policy.py` | Reach-box 演示策略，使用 **`CuroboPlanner`** |
| `isaaclab_logistics_vla/evaluation/robot_registry.py` | 机器人与 CuRobo 文件映射 |
| `scripts/evaluate_vla.py` | 评估入口与环境变量注入 |

---

## 十、小结

- CuRobo 在本 benchmark 中是 **EE 语义 → 关节轨迹** 的 **驱动后端**。
- 参考 **RoboTwin**，将「构建 + 规划 + 结果类型」收进 **`CuroboPlanner`**，对外只暴露 **`plan_ee` → dict + numpy**，有利于 **解耦、测试与后续换规划器**。
- **世界系 / 臂基系** 的变换仍在 Isaac 与任务侧完成，与 RoboTwin 在 `Robot` 层做变换的分工不同，但 **planner 边界**（臂基系进、轨迹出）一致。

若你后续在文档或论文中引用实现细节，可直接指向本文与 `planner.py` 中的类定义与 docstring。
