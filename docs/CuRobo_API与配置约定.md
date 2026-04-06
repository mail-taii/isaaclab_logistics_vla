# CuRobo 封装：API、配置与坐标系约定

本文档面向 **`isaaclab_logistics_vla.utils.curobo_planner`** 算法封装层，说明 **API 输入/输出**、**配置文件要求**、**使用前提**，并单独强调 **基座 / 坐标系** 应如何选择（常见误区：整车 root 与机械臂运动学基座混用）。

---

## 1. 基座与坐标系（必读）

### 1.1 两个「基座」不要混用

| 概念 | 含义 | 在本仓库中的典型对应 |
|------|------|----------------------|
| **A. cuRobo 运动学 `base_link`** | URDF 里 **双臂运动链的起始 link**，`MotionGen` / IK 中末端位姿、关节限位都相对 **该 link 固连的坐标系** 定义。 | 由 **生成 kinematics YAML 时** 在脚本里指定的 `base_link` 决定；`RobotSpec.base_link` 仅为字段默认/文档对齐，**不**在加载时改写 YAML。 |
| **B. 仿真里整车的 `root`（或 world 中的底盘原点）** | Isaac 里 `robot.data.root_pos_w` 等，常对应 **移动底盘 / 整机根**。 | **不一定**等于 A；若臂装在升降平台上，A 往往随平台关节相对 B 有位移与姿态关系。 |

**结论：**

- **生成 YAML 时**（`scripts/generate_curobo_robot_kinematics_yaml.py` 的 `--base-link`）**必须选「机械臂（双臂）运动学链在 URDF 里从哪一节开始算」的那一节**，即 **臂系运动学基座**。
- **`plan_dual` / `plan_single_arm` / `set_world` / `WorldSpec` 里给出的位置与四元数，必须与上述 A 在同一套「相对 base_link 的坐标约定」下表达**（见 1.2 与 1.3）。若你从仿真里读的是 **整车 root / World / Pelvis / Camera** 下的点，必须在 **你的集成代码**里先做 **`该系 → base_link 固连系`** 的刚体变换，再送给封装；否则会出现 **IK_FAIL、目标不可达、障碍与几何不一致** 等。

### 1.1.1 双人形、双臂：物理上有「左右肩两个基」，为何这里仍是一个 `base_link`？

直觉上左右臂各有一个肩关节，好像该有两个「臂基」；但在 **本封装 + cuRobo 当前双臂 `MotionGen` 模型** 里，用的是 **一棵运动学树、一个运动学根 `base_link`、两个末端**：

- URDF 里左臂链、右臂链会 **向上汇合** 到 **同一个祖先 link**（例如 **躯干 / 胸口 / 腰上部**）。**`base_link` 应选这个「双臂公共祖先」中、你希望对 IK 作为固定参考的那一节**（或整机关节已锁成姿态后，与该系固连的截面）。左肩、右肩 **不是** 两个独立的 `base_link`，而是 **从这同一个祖先分叉出去的两个子链**。
- **`plan_dual` 给出的左右末端位姿 `goal_poses['left']` / `['right']`**，必须在 **同一参考系** 下表达——也就是 **与这个唯一的 `base_link` 固连的坐标系**（再按 1.2 节决定是否做绕 z 旋转）。**不是**「左目标在左肩系、右目标在右肩系」各用各的，除非你在上层自己先把两边都变换到公共 `base_link` 系再传入。
- **若你的 URDF 把两条臂做成完全不相交的两棵树**（没有公共 link、或你希望两条链各自相对世界单独规划）：当前 **`CudaRobotGeneratorConfig` 单 `ee_link` + `link_names` 双臂 batch** 的用法 **可能不适用**，需要 **两条独立单臂配置、两次规划**，或 **先合并 URDF** 使双臂挂在同一躯干下。

因此：**人形双臂在配置里仍是「一个运动学 `base_link` + 两个 `*_ee_link`」**；物理上的「两个肩」体现在 **树的分叉**，而不是两个并列的 `base_link` 字段。

### 1.1.2 人形机器人：`base_link` 选在哪（与常见 cuRobo 实践对照）

任务/视觉给出的抓取目标往往在 **World** 系下，而本封装与 cuRobo 只吃 **相对 kinematics YAML 里固化的那一个 `base_link`** 的位姿（与生成脚本所选一致）。典型有两种搭法（可与 1.1、1.1.1 对照阅读）：

**策略一：只规划手臂（局部链 / arm-only）**

- **假设**：躯干相对「臂链根」可视为固定，或你在外部已锁死上半身关节。
- **`base_link`**：选在 **手臂与躯干连接的第一个 link**（例如右肩 `r_shoulder_*`）；单链场景下用生成脚本的 **`--ee-link`** 生成 **单臂** YAML，末端为手爪/TCP。
- **坐标变换（集成方必做）**：必须把目标从 World（或 Pelvis/Camera 等）变到 **该 `base_link` 系**。记 **`W T_B`** 为「从坐标系 B 到 World 的齐次变换」（4×4，`position` + `quaternion` 与此等价）；目标相对 World 为 **`W T_goal`**，`base_link` 相对 World 为 **`W T_base`**，则目标在 **`base_link` 系**下应为：
  ```
  base_T_goal = inv(W_T_base) @ W_T_goal
  ```
  （与「先把目标变到 World 再左乘 `base` 在 World 下位姿的逆」同义。）若省略此步，等价于把 **世界系坐标误当成 `base_link` 系坐标**，规划会整体错误。

**策略二：完整运动学树（躯干根 / root-based，更常见）**

- **`base_link`**：选在 **整机关节树的根**（如 **骨盆 `pelvis`、腰封 `base_link`** 等 URDF 根节），**末端**仍为左右手爪；URDF 中从根经躯干到肩、肘再到手的 **关节链由 cuRobo 在 FK/IK 内部走通**。
- **坐标变换**：你通常只需把目标统一到 **`base_link` 固连系`**（例如相对骨盆）；肩相对躯干的偏置 **不必**再手写一遍，由 URDF 连杆与关节体现。
- **浮动基座**：若骨盆/整机在 World 中运动，除手臂关节外，还需保证 **送入规划的状态与 URDF 一致地反映当前根位姿**（或在你的管线里用等价方式更新），否则模型与世界不同步。

**与本仓库双臂 `plan_dual` 的衔接**：策略二且双臂共根时，与 1.1.1 **单 `base_link` + 双末端** 一致；若两臂在 URDF 上 **无公共祖先**、又坚持两条独立肩基，则本仓库默认 **双臂 batch** 模型不适用，需 **两条单链配置（如两次 `plan_one_ee`）或合并 URDF**（见 1.1.1 末尾）。

### 1.2 「机器人约定坐标系」与 `apply_robot_to_curobo_frame_transform`

封装文档中的 **「机器人约定坐标系」** 指：**与 URDF / Isaac 资产一致的、在施加可选旋转之前** 的那套轴约定。代码里若 **`apply_robot_to_curobo_frame_transform=True`**（`CuroboPlanner` 构造参数），会对 **末端目标位姿** 与 **长方体障碍位姿** 统一做 **绕 z 轴 -90°**（位置用旋转矩阵、姿态用四元数复合），以与 cuRobo 常用 **「前向为 x」** 的习惯对齐。

- 若你的 URDF 与 cuRobo 资产 **已经同轴**，应将该开关设为 **`False`**，否则目标会多转一次。
- **障碍与末端目标必须使用同一套输入约定**（开关对二者一致作用）。

### 1.3 四元数与关节顺序

- 所有 **`quaternion`** 均为 **Isaac 风格 `(w, x, y, z)`**。
- **`start_joint_positions` 与轨迹 `position`**：长度/列数均为 **`CuroboPlanner.dof`**（由已加载的 `RobotConfig.kinematics` 推断，**不是**写死常数）。**默认 Realman 双臂** YAML 下为 **14**，即 **`(14,)`** / **`(T, 14)`**，顺序为 **左臂 7 + 右臂 7**（与生成脚本默认双臂 `link_names` 一致），**不是** 仿真里常见的 `l1,r1,l2,r2,…` **交错顺序**；若上游为交错顺序，须在调用 `plan_dual` / `plan_single_arm` **之前**自行对齐为与 cuRobo 关节向量一致。**单臂 YAML**（生成时指定 **`--ee-link`**）时 **`dof`** 一般为 **7**，应使用 **`plan_one_ee`**，不要用 **`plan_dual`**。

---

## 2. 对外 API：输入与输出

### 2.1 `RobotSpec`（指向已生成的 kinematics YAML）

| 方法 / 字段 | 输入 | 说明 |
|-------------|------|------|
| `RobotSpec.from_kinematics_yaml(kinematics_yaml, *, urdf_path=None, base_link=..., left_ee_link=..., right_ee_link=...)` | **YAML 路径**（须已存在）+ 可选元数据 | 推荐主入口。 |
| `RobotSpec.from_robot_config_yaml(config_yaml, *, urdf_path=None, ...)` | 同左 | 与 `from_kinematics_yaml` 同义（历史命名）。 |
| `RobotSpec.from_urdf(urdf_path, *, cache_path, ...)` | **必须**提供已存在的 `cache_path` YAML；`urdf_path` 仅作文档记录 | 兼容旧调用习惯。 |
| 字段 **`cache_path`** | — | **必填**，kinematics YAML 路径；`CuroboPlanner` 仅此字段参与加载。 |
| 字段 `urdf_path`, `base_link`, `left_ee_link`, `right_ee_link` | — | 可选元数据，供业务侧记录；**不参与**封装内加载。 |

**输出：** 无；作为 **`CuroboPlanner(robot_spec=...)`** 的构造输入。

---

### 2.2 `WorldSpec`（世界障碍）

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

加载顺序：**若传入 `robot_config` 则直接使用**；否则从 **磁盘 YAML** 路径解析（`RobotConfig.from_dict`）。**封装内不执行 URDF → YAML**。

| 参数 | 类型 | 说明 |
|------|------|------|
| `robot_config` | `curobo.types.robot.RobotConfig \| None` | 若提供，**跳过文件路径**，直接使用（须与 `device` 上张量约定一致）。 |
| `robot_spec` | `RobotSpec \| None` | 使用 `robot_spec.cache_path` 作为 YAML 路径；若同时传入 **`cache_path`** 关键字，则 **以 `cache_path` 覆盖** spec 中的路径。 |
| `cache_path` | `str \| None` | 无 `robot_spec` 时作为 YAML 路径；有 `robot_spec` 时仅当本参数非 `None` 时 **覆盖** `robot_spec.cache_path`。 |
| `device` | `str` | 如 `cuda:0`；用于 `TensorDeviceType` 与 `from_dict`。 |
| `use_curobo_cache` | `bool` | `True` 且未由 `robot_config` / `cache_path` / `robot_spec` 解析出路径时，使用默认 **`~/.cache/curobo_realman/realman_config_v2.yaml`**（**该文件须已预生成**）。 |
| `interpolation_dt` | `float` | 插值步长（秒）。 |
| `apply_robot_to_curobo_frame_transform` | `bool` | 是否对目标与障碍做绕 z -90° 对齐。 |
| `use_cuda_graph` | `bool` | 传给 `MotionGenConfig`。 |

**输出：** 无；构造副作用为加载模型、`warmup`。缺失 YAML 时抛出 **`FileNotFoundError`**，提示运行仓库生成脚本。

---

### 2.4 `apply_world` / `set_world` / `clear_world`

| 方法 | 输入 | 输出 |
|------|------|------|
| `apply_world(spec: WorldSpec)` | `WorldSpec` | `None`；更新内部 `WorldConfig`。 |
| `set_world(obstacles)` | `List[Dict]`，键与 `WorldSpec` 立方体一致（`position`、`size`/`dims`、可选 `quaternion`、`name`） | `None` |
| `clear_world()` | — | `None`（空世界）。 |

---

### 2.5 `plan_dual(...)`

**输入：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `start_joint_positions` | `np.ndarray`，**`(self.dof,)`** | 关节顺序与当前 `RobotConfig` 一致；默认双臂 Realman 为 14（左 7 + 右 7），`float32` 可转。 |
| `goal_poses` | `dict` | 必须含键 **`left`**、**`right`**，每个为 `{"position": (3,), "quaternion": (4,)}`。 |
| `max_attempts` | `int` | 默认 60。 |
| `timeout` | `float` | 秒。 |
| `enable_graph` | `bool` | 默认 `True`。 |
| `enable_opt` | `bool` | 默认 `True`。 |

**输出：** `dict`，由 `result_utils.motion_gen_batch_result_to_plan_dict` 规整：

| 键 | 成功时 | 失败时 |
|----|--------|--------|
| `status` | `"Success"` | `"Fail"` |
| `position` | `np.ndarray` `(T, dof)`，`float32` | `None` |
| `velocity` | 与 `position` 同形或 `None` | `None` |
| `detail` | 可选，cuRobo 状态信息 | 常有失败原因 |
| `interpolation_dt` | 可选 `float` | 可能无 |

---

### 2.6 `plan_single_arm(...)`

**输入：**

| 参数 | 说明 |
|------|------|
| `start_joint_positions` | **`(self.dof,)`**；本方法面向 **双臂** 模型（默认 14），须给出 **整臂** 起始关节向量。 |
| `goal_pose` | 运动侧 `{"position": (3,), "quaternion": (4,)}`。 |
| `arm` | `"left"` 或 `"right"`。 |
| `fixed_arm_goal_pose` | 另一侧末端目标，同结构。 |
| `**kwargs` | 转发至 `plan_dual`（如 `max_attempts`、`timeout`、`enable_graph`、`enable_opt`）。 |

**输出：** 与 **`plan_dual`** 相同结构的 **`dict`**。

---

### 2.7 `plan_one_ee(...)`

单末端、**单臂** kinematics（加载的 YAML / `robot_config` 为 **单 `link_names`**）使用；内部仍走 `plan_batch`，与 `plan_dual` 共用坐标与 `apply_robot_to_curobo_frame_transform` 约定。

| 参数 | 说明 |
|------|------|
| `start_joint_positions` | **`(self.dof,)`**（单臂时通常为 7）。 |
| `goal_pose` | `{"position": (3,), "quaternion": (4,)}`。 |
| `max_attempts` / `timeout` / `enable_graph` / `enable_opt` | 与 `plan_dual` 默认值相同（60 / 10.0 / True / True）。 |

**输出：** 与 **`plan_dual`** 相同结构的 **`dict`**（`position` 为 **`(T, dof)`**）。

---

### 2.8 `plan(...)`、`plan_grippers(...)`、`reset(...)`

- **`plan`**：参数与 **`plan_dual`** 相同；可选 **`dt`**，非 `None` 且与 `interpolation_dt` 不一致时发 **`UserWarning`**（已弃用，应用构造参数 `interpolation_dt`）。默认返回 **`dict`**；仅当 **`legacy_tuple_return=True`** 传入 **kwargs** 时返回 **`(bool, ndarray)`**。
- **`plan_grippers(now_val, target_val, num_step=200)`**（静态方法）：**输入** 标量；**输出** `{"num_step", "per_step", "result": ndarray}`，**不经过 cuRobo**。
- **`reset(reset_seed=True)`**：无返回值。

---

### 2.9 状态与调试（可选）

| 成员 | 说明 |
|------|------|
| **`dof`**（`@property`） | 当前模型关节维数，与 `plan_*` 的 `start_joint_positions` 一致。 |
| **`get_interpolated_trajectory()`** | 若上次规划成功，返回上次 **`plan_*`** 结果中的插值轨迹 `position`（`ndarray`），否则 `None`。 |
| **`get_optimized_trajectory()`** | 若存在 cuRobo **`optimized_plan`** 且成功，返回其位置张量转 CPU numpy，否则 `None`。 |
| **`is_success()`** | 是否 **`last_plan_dict["status"] == "Success"`**。 |
| **`solve_time`**（`@property`） | 上次求解时间（毫秒，来自 cuRobo；无则 `None`）。 |

---

## 3. 配置文件要求

### 3.1 生成 kinematics YAML（**不在** `curobo_planner` 包内）

由使用方在部署前或更新 URDF 后自行执行。本仓库提供独立脚本（**非** `utils.curobo_planner` 子包）：

```bash
python scripts/generate_curobo_robot_kinematics_yaml.py \
  --urdf /path/to/robot.urdf \
  --output /path/to/robot_kin.yaml \
  --base-link <URDF 中的 base link 名> \
  --left-ee-link <左末端 link> \
  --right-ee-link <右末端 link>
```

- **单臂**：增加 `--ee-link <末端 link>`，此时不读左右末端参数的双臂语义。
- **格式**：输出为根键 **`kinematics`** 的 YAML，可被 **`RobotConfig.from_dict`** 读回（与 cuRobo 约定一致）。
- **勿用** `RobotConfig.write_config` 混入不可 YAML 序列化的 `CudaRobotModelConfig` 字段当作本用途缓存（脚本内注释有说明）。

### 3.2 封装如何加载（`cache_path` / `RobotSpec.cache_path` / `robot_config`）

- **封装只做**：`yaml.safe_load` + **`RobotConfig.from_dict`**（或直接使用你传入的 **`robot_config`**）。
- **运维**：修改 **URDF** 或 **链名** 后须 **重新运行生成脚本** 并更新路径或覆盖 YAML；否则运动学与实物不一致。

### 3.3 世界障碍 YAML（`WorldSpec.from_yaml`）

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

1. **已生成** 与当前机器人一致的 kinematics YAML（或传入自建的 **`robot_config`**）；生成时 **`base_link` 与末端 link 名** 与 URDF 一致，且 **`base_link` 是双臂运动学链的合理根**（一般为 **臂安装平台 link**，而非随意选整车 root）。
2. **目标位姿与障碍与 `base_link` 固连系一致**；若数据来自 **World / root / Pelvis / Camera**，必须先变换到 **与 cuRobo `base_link` 一致** 的系（人形场景见 **1.1.2**）。
3. **`apply_robot_to_curobo_frame_transform`** 与资产轴约定一致，避免重复旋转。
4. **CUDA 与 cuRobo 已安装**；若外层使用 `torch.inference_mode(True)`，规划路径须按封装内 **`_curobo_autograd_context`** 在非纯 inference 下执行（见 `curobo_planner.py`）。
5. **关节初值在限位内**，且 `start_joint_positions` 长度等于 **`planner.dof`**；目标在工作空间内，否则常见 **`IK_FAIL`**。

---

## 5. 使用样例

### 5.1 最小双臂规划（显式 `RobotSpec` + 空世界）

事先生成 YAML（见 **§3.1**），例如 `robot_kin_cache.yaml`。

```python
import numpy as np
from isaaclab_logistics_vla.utils.curobo_planner import (
    CuroboPlanner,
    RobotSpec,
    WorldSpec,
)

robot = RobotSpec.from_kinematics_yaml(
    "/path/to/robot_kin_cache.yaml",
    urdf_path="/path/to/realman_franka_ee.urdf",  # 可选，仅记录
)
planner = CuroboPlanner(
    robot_spec=robot,
    device="cuda:0",
    apply_robot_to_curobo_frame_transform=False,
)
planner.apply_world(WorldSpec.empty())

q0 = np.zeros(planner.dof, dtype=np.float32)  # 默认双臂 Realman 下 dof == 14
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

## 6. 与仿真 / 任务集成

- 本仓库 **默认评估脚本** 当前 **不内置** Isaac 侧策略；在仿真或真机中使用时，由你在 **任务节点** 中完成：**World / Pelvis / root → `base_link` 系** 的位姿变换、**关节顺序** 与 **`start_joint_positions` 对齐**，再调用 **`CuroboPlanner`**。
- **仅依赖算法封装**时，可只 `import isaaclab_logistics_vla.utils.curobo_planner`，不必引用本扩展的 `evaluation` 包。

---

## 7. 相关文件

| 路径 | 说明 |
|------|------|
| `utils/curobo_planner/curobo_planner.py` | `CuroboPlanner`、`plan_dual` / `plan_single_arm` / `plan_one_ee` |
| `utils/curobo_planner/robot_spec.py` | `RobotSpec` |
| `utils/curobo_planner/world_spec.py` | `WorldSpec` |
| `utils/curobo_planner/result_utils.py` | 结果 dict |
| `utils/curobo_planner/example_usage.py` | 最小调用示例（需 GPU + cuRobo + 预生成 kinematics YAML） |
| `scripts/generate_curobo_robot_kinematics_yaml.py` | **包外**：URDF → kinematics YAML |

---

*文档版本与代码路径：`isaaclab_logistics_vla/utils/curobo_planner/`、`isaaclab_logistics_vla/scripts/generate_curobo_robot_kinematics_yaml.py`、`docs/CuRobo_API与配置约定.md`。*
