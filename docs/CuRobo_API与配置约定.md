# CuRobo 封装：API、配置与坐标系约定

本文档面向 **`isaaclab_logistics_vla.utils.curobo_planner`** 算法封装层，说明 **API 输入/输出**、**配置文件要求**、**使用前提**，并单独强调 **基座 / 坐标系** 应如何选择（常见误区：整车 root 与机械臂运动学基座混用）。

---

## 1. 基座与坐标系（必读）

### 1.1 两个「基座」不要混用

| 概念 | 含义 | 在本仓库中的典型对应 |
|------|------|----------------------|
| **A. cuRobo 运动学 `base_link`** | URDF 里 **双臂运动链的起始 link**，`MotionGen` / IK 中末端位姿、关节限位都相对 **该 link 固连的坐标系** 定义。 | `RobotSpec.base_link` 默认 **`dual_rm_75b_description_platform_base_link`**（Realman 双臂 URDF 上的 **平台/臂安装基座**）。 |
| **B. 仿真里整车的 `root`（或 world 中的底盘原点）** | Isaac 里 `robot.data.root_pos_w` 等，常对应 **移动底盘 / 整机根**。 | **不一定**等于 A；若臂装在升降平台上，A 往往随平台关节相对 B 有位移与姿态关系。 |

**结论：**

- **`RobotSpec.base_link` / `generate_robot_config_from_urdf` 的 `base_link` 必须选「机械臂（双臂）运动学链在 URDF 里从哪一节开始算」的那一节**，即 **臂系运动学基座**，而不是凭感觉选「整个机器人的世界根」。
- **`plan_dual` / `plan_single_arm` / `set_world` / `WorldSpec` 里给出的位置与四元数，必须与上述 A 在同一套「相对 base_link 的坐标约定」下表达**（见 1.2 与 1.3）。若你从仿真里读的是 **整车 root 系** 下的点，必须先通过 **`root → base_link`（或「臂基」）的刚体变换** 变到 **与 cuRobo `base_link` 一致** 的系，再送给封装；否则会出现 **IK_FAIL、目标不可达、障碍与几何不一致** 等。

本仓库策略层 **`CuRoboPlanPolicy`** 使用 `evaluation/robot_registry.py` 中的 **`arm_base_offset_in_root` + `platform_joint`**，用 `combine_frame_transforms` 把 Isaac 的 **root** 与 **臂安装偏置** 合成 **臂基世界位姿**，再用 `subtract_frame_transforms` 把手/ TCP 变到 **臂基系**，其目的正是 **与 A 对齐**（在 Realman 默认 URDF 下，该「臂基」应对齐 `base_link` 所在运动学意义）。

### 1.1.1 双人形、双臂：物理上有「左右肩两个基」，为何这里仍是一个 `base_link`？

直觉上左右臂各有一个肩关节，好像该有两个「臂基」；但在 **本封装 + cuRobo 当前双臂 `MotionGen` 模型** 里，用的是 **一棵运动学树、一个运动学根 `base_link`、两个末端**：

- URDF 里左臂链、右臂链会 **向上汇合** 到 **同一个祖先 link**（例如 **躯干 / 胸口 / 腰上部**）。**`base_link` 应选这个「双臂公共祖先」中、你希望对 IK 作为固定参考的那一节**（或整机关节已锁成姿态后，与该系固连的截面）。左肩、右肩 **不是** 两个独立的 `base_link`，而是 **从这同一个祖先分叉出去的两个子链**。
- **`plan_dual` 给出的左右末端位姿 `goal_poses['left']` / `['right']`**，必须在 **同一参考系** 下表达——也就是 **与这个唯一的 `base_link` 固连的坐标系**（再按 1.2 节决定是否做绕 z 旋转）。**不是**「左目标在左肩系、右目标在右肩系」各用各的，除非你在上层自己先把两边都变换到公共 `base_link` 系再传入。
- **若你的 URDF 把两条臂做成完全不相交的两棵树**（没有公共 link、或你希望两条链各自相对世界单独规划）：当前 **`CudaRobotGeneratorConfig` 单 `ee_link` + `link_names` 双臂 batch** 的用法 **可能不适用**，需要 **两条独立单臂配置、两次规划**，或 **先合并 URDF** 使双臂挂在同一躯干下。

因此：**人形双臂在配置里仍是「一个运动学 `base_link` + 两个 `*_ee_link`」**；物理上的「两个肩」体现在 **树的分叉**，而不是两个并列的 `base_link` 字段。

### 1.2 「机器人约定坐标系」与 `apply_robot_to_curobo_frame_transform`

封装文档中的 **「机器人约定坐标系」** 指：**与 URDF / Isaac 资产一致的、在施加可选旋转之前** 的那套轴约定。代码里若 **`apply_robot_to_curobo_frame_transform=True`**（`CuroboPlanner` 构造参数），会对 **末端目标位姿** 与 **长方体障碍位姿** 统一做 **绕 z 轴 -90°**（位置用旋转矩阵、姿态用四元数复合），以与 cuRobo 常用 **「前向为 x」** 的习惯对齐。

- 若你的 URDF 与 cuRobo 资产 **已经同轴**，应将该开关设为 **`False`**，否则目标会多转一次。
- **障碍与末端目标必须使用同一套输入约定**（开关对二者一致作用）。

### 1.3 四元数与关节顺序

- 所有 **`quaternion`** 均为 **Isaac 风格 `(w, x, y, z)`**。
- **`start_joint_positions` 与轨迹 `position`**：形状 **`(14,)`** 或 **`(T, 14)`**，顺序为 **左臂 7 关节 + 右臂 7 关节**（与当前 `RobotConfig` 双臂模型一致），**不是** Isaac 动作里常见的 `l1,r1,l2,r2,…` 交错顺序（交错由 **`CuRoboPlanPolicy`** 等策略层转换）。

---

## 2. 对外 API：输入与输出

### 2.1 `RobotSpec`（机器人描述，可选替代裸 `urdf_path`）

| 方法 / 字段 | 输入 | 说明 |
|-------------|------|------|
| `RobotSpec.from_urdf(urdf_path, *, cache_path=..., base_link=..., left_ee_link=..., right_ee_link=...)` | 路径与可选链名 | 主入口：**URDF** + 可选 **kinematics 缓存 YAML**；链名必须与 URDF 一致。 |
| `RobotSpec.from_robot_config_yaml(config_yaml, *, urdf_path=..., ...)` | YAML 路径 + URDF | `config_yaml` 存在则直接加载；不存在则用 URDF 生成并写入。 |
| 字段 `urdf_path`, `cache_path`, `base_link`, `left_ee_link`, `right_ee_link` | — | 冻结 dataclass，供 `CuroboPlanner(robot_spec=...)` 使用。 |

**输出：** 无；作为 **`CuroboPlanner`** 的构造输入。

---

### 2.2 `WorldSpec`（世界长方体障碍）

| 方法 | 输入 | 输出 |
|------|------|------|
| `WorldSpec.empty()` | — | `WorldSpec`（无障碍列表）。 |
| `WorldSpec.from_cuboids([{...}, ...])` | 字典列表，见下表 | `WorldSpec` |
| `WorldSpec.from_yaml(path)` | YAML 文件路径 | `WorldSpec` |
| `to_planner_obstacles()` | — | `List[dict]`，供内部 `set_world` 使用。 |

**每个长方体字典字段：**

| 键 | 类型 | 必填 |
|----|------|------|
| `position` | 长度为 3 的序列 | 是 |
| `size` 或 `dims` | 长度为 3 的序列 | 是其一 |
| `quaternion` | 长度为 4，wxyz | 否，默认单位四元数 |
| `name` | 字符串 | 否 |

坐标与 **`plan_dual` 的 `goal_poses` 使用同一套「机器人约定坐标系」**（再经 `apply_robot_to_curobo_frame_transform` 若开启）。

---

### 2.3 `CuroboPlanner.__init__(...)`

| 参数 | 类型 | 说明 |
|------|------|------|
| `robot_spec` | `RobotSpec \| None` | 若提供，优先生效；`urdf_path` / `cache_path` 由 spec 推导（见源码）。 |
| `urdf_path` | `str \| None` | 无 `robot_spec` 时使用；默认指向扩展内 Realman URDF。 |
| `device` | `str` | 如 `cuda:0`。 |
| `use_curobo_cache` | `bool` | `True` 且无显式缓存路径时，使用默认 `~/.cache/curobo_realman/realman_config_v2.yaml`。 |
| `cache_path` | `str \| None` | kinematics 缓存 YAML。 |
| `interpolation_dt` | `float` | 插值步长（秒）。 |
| `apply_robot_to_curobo_frame_transform` | `bool` | 是否对目标与障碍做绕 z -90° 对齐。 |
| `use_cuda_graph` | `bool` | 传给 `MotionGenConfig`。 |

**输出：** 无；构造副作用为加载模型、`warmup`。

---

### 2.4 `apply_world` / `set_world` / `clear_world`

| 方法 | 输入 | 输出 |
|------|------|------|
| `apply_world(spec: WorldSpec)` | `WorldSpec` | `None`；更新内部 `WorldConfig`。 |
| `set_world(obstacles: List[Dict])` | 与 `WorldSpec` 立方体字段一致 | `None` |
| `clear_world()` | — | `None`（空世界）。 |

---

### 2.5 `plan_dual(...)`

**输入：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `start_joint_positions` | `np.ndarray`，`(14,)` | 左 7 + 右 7，`float32` 可转。 |
| `goal_poses` | `dict` | 必须含键 **`left`**、**`right`**，每个为 `{"position": (3,), "quaternion": (4,)}`。 |
| `max_attempts` | `int` | 默认 60。 |
| `timeout` | `float` | 秒。 |
| `enable_graph` | `bool` | 默认 `True`。 |
| `enable_opt` | `bool` | 默认 `True`。 |

**输出：** `dict`，由 `result_utils.motion_gen_batch_result_to_plan_dict` 规整：

| 键 | 成功时 | 失败时 |
|----|--------|--------|
| `status` | `"Success"` | `"Fail"` |
| `position` | `np.ndarray` `(T, 14)`，`float32` | `None` |
| `velocity` | 与 `position` 同形或 `None` | `None` |
| `detail` | 可选，cuRobo 状态信息 | 常有失败原因 |
| `interpolation_dt` | 可选 `float` | 可能无 |

---

### 2.6 `plan_single_arm(...)`

**输入：**

| 参数 | 说明 |
|------|------|
| `start_joint_positions` | `(14,)`，完整双臂起始关节。 |
| `goal_pose` | 运动侧 `{"position": (3,), "quaternion": (4,)}`。 |
| `arm` | `"left"` 或 `"right"`。 |
| `fixed_arm_goal_pose` | 另一侧末端目标，同结构。 |
| `**kwargs` | 转发至 `plan_dual`（如 `max_attempts`、`timeout`、`enable_graph`、`enable_opt`）。 |

**输出：** 与 **`plan_dual`** 相同结构的 **`dict`**。

---

### 2.7 `plan(...)`、`plan_grippers(...)`、`reset(...)`

- **`plan`**：参数与 **`plan_dual`** 相同；默认返回 **`dict`**；仅当 `legacy_tuple_return=True` 时返回 `(bool, ndarray)`。
- **`plan_grippers(now_val, target_val, num_step=200)`**（静态方法）：**输入** 标量；**输出** `{"num_step", "per_step", "result": ndarray}`，**不经过 cuRobo**。
- **`reset(reset_seed=True)`**：无返回值。

---

## 3. 配置文件要求

### 3.1 机器人 kinematics 缓存 YAML（`cache_path` / `RobotSpec.cache_path`）

- **来源**：由本仓库 **`generate_robot_config_from_urdf`** 生成，内容为可 **`RobotConfig.from_dict`** 的 kinematics 字段；**不要**用 cuRobo 的 `RobotConfig.write_config` 写完整二进制式结构到同一用途文件（见原 `CuRobo规划器封装说明.md`）。
- **修改 URDF 或 `base_link` / 末端名后**：应删除或更换缓存文件，避免读到旧链。

### 3.2 世界障碍 YAML（`WorldSpec.from_yaml`）

根对象可为：

```yaml
version: 1
cuboids:
  - name: table
    position: [0.0, 0.0, 0.5]
    dims: [2.0, 1.0, 0.05]
    quaternion: [1.0, 0.0, 0.0, 0.0]
```

- 列表键名也可为 **`obstacles`**。
- **`position` / `dims`（或 `size`）** 为浮点序列；**`quaternion`** 可选，**wxyz**。

---

## 4. 算法使用前提（检查清单）

1. **`base_link` 与 `left_ee_link` / `right_ee_link` 在 URDF 中真实存在**，且 **`base_link` 是双臂运动学链的合理根**（一般为 **臂安装平台 link**，而非随意选整车 root）。
2. **目标位姿与障碍与 `base_link` 固连系一致**；若数据来自仿真 **整机 root**，必须先变换到 **与 cuRobo `base_link` 一致** 的系（可参考 `robot_registry` + `CuRoboPlanPolicy` 的做法）。
3. **`apply_robot_to_curobo_frame_transform`** 与资产轴约定一致，避免重复旋转。
4. **CUDA 与 cuRobo 已安装**；规划调用建议在 **`torch.inference_mode(False)` + `enable_grad`** 上下文外或按封装内 `_curobo_autograd_context` 使用（策略层已对 `env.step` 与规划做了区分）。
5. **关节初值在限位内**；目标在工作空间内，否则常见 **`IK_FAIL`**。

---

## 5. 使用样例

### 5.1 最小双臂规划（显式 `RobotSpec` + 空世界）

```python
import numpy as np
from isaaclab_logistics_vla.utils.curobo_planner import (
    CuroboPlanner,
    RobotSpec,
    WorldSpec,
)

robot = RobotSpec.from_urdf(
    "/path/to/realman_franka_ee.urdf",
    cache_path="/path/to/robot_kin_cache.yaml",
)
planner = CuroboPlanner(
    robot_spec=robot,
    device="cuda:0",
    apply_robot_to_curobo_frame_transform=False,
)
planner.apply_world(WorldSpec.empty())

q0 = np.zeros(14, dtype=np.float32)
goal = {
    "left": {
        "position": np.array([0.3, 0.2, 0.4]),
        "quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
    },
    "right": {
        "position": np.array([0.3, -0.2, 0.4]),
        "quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
    },
}
out = planner.plan_dual(q0, goal, max_attempts=24, timeout=5.0)
if out["status"] == "Success":
    traj = out["position"]  # (T, 14)
```

**注意**：上例中 `goal` 的数值仅为演示；**实际必须与你的 `base_link` 坐标系一致**。

### 5.2 从 YAML 加载世界障碍

```python
world = WorldSpec.from_yaml("/path/to/world_cuboids.yaml")
planner.apply_world(world)
out = planner.plan_dual(q0, goal_poses, ...)
```

### 5.3 单臂规划（另一臂保持当前末端目标）

```python
out = planner.plan_single_arm(
    q0,
    goal_pose={"position": p_l, "quaternion": q_l},
    arm="left",
    fixed_arm_goal_pose={"position": p_r, "quaternion": q_r},
    max_attempts=10,
    timeout=2.0,
    enable_graph=True,
    enable_opt=False,
)
```

---

## 6. 与策略层的关系

- **`CuRoboPlanPolicy`** 在仿真中负责 **臂基计算、手/ TCP 位姿、关节重排**，再调用 **`CuroboPlanner`**；评估器可通过 **`curobo_plan_kwargs`** 传入 **`robot_spec`**、**`world_spec`** 等。
- **纯算法集成**可只依赖 **`utils.curobo_planner`**，不引用 Isaac。

---

## 7. 相关文件

| 路径 | 说明 |
|------|------|
| `utils/curobo_planner/curobo_planner.py` | `CuroboPlanner` |
| `utils/curobo_planner/robot_spec.py` | `RobotSpec` |
| `utils/curobo_planner/world_spec.py` | `WorldSpec` |
| `utils/curobo_planner/config_generator.py` | URDF → `RobotConfig` |
| `utils/curobo_planner/result_utils.py` | 结果 dict |
| `evaluation/models/curobo_plan_policy.py` | Isaac 策略封装 |
| `evaluation/robot_registry.py` | `arm_base_offset_in_root` 等 |
| `docs/CuRobo规划器封装说明.md` | 更全面的模块说明与验证脚本 |

---

*文档版本与代码路径：`isaaclab_logistics_vla/utils/curobo_planner/`、`docs/CuRobo_API与配置约定.md`。*
