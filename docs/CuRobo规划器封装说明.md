# CuRobo 运动规划封装说明

本文介绍 `isaaclab_logistics_vla` 中 **`utils/curobo_planner`** 模块：在 NVIDIA cuRobo 之上提供一层与仿真/任务解耦的接口，风格对齐 RoboTwin 中的用法（**规划结果以 dict + NumPy 为主，不向上层暴露 cuRobo 的 Tensor 类型**）。

## 1. 定位与目标

| 目标 | 说明 |
|------|------|
| 抽象边界 | 上层只处理 `numpy`、标准 `dict`，便于单测、离线脚本与不同仿真后端对接 |
| 双臂规划 | 默认 **14 维**关节（左臂 7 + 右臂 7），通过 `MotionGen.plan_batch` 做双臂同时规划 |
| 夹爪 | 夹爪开合不经 cuRobo，采用与 RoboTwin 一致的 **线性插值** |
| 配置 | 从 URDF 生成 `RobotConfig`，支持缓存 YAML，避免每次冷启动都解析 URDF |

## 2. 目录与模块

```
isaaclab_logistics_vla/utils/curobo_planner/
├── __init__.py           # 导出 CuroboPlanner、工具函数
├── curobo_planner.py     # 主封装类 CuroboPlanner
├── config_generator.py   # URDF → RobotConfig，缓存读写
├── result_utils.py       # MotionGen 结果 → RoboTwin 风格 dict
└── example_usage.py      # 可运行的简单示例（需 GPU + cuRobo）
```

仓库根目录还提供验证脚本：

```
scripts/verify_curobo_planner.py
```

## 3. 环境与依赖

- **Python**：与 Isaac Lab / 项目约定一致（例如在 `env_isaaclab` 中已验证）。
- **PyTorch + CUDA**：cuRobo 在 GPU 上运行。
- **cuRobo**：需已安装并可 `import curobo`。

在 conda 环境中自检与冒烟（在仓库根目录 `isaaclab_logistics_vla` 下）：

```bash
conda activate env_isaaclab
python scripts/verify_curobo_planner.py
```

脚本会跑逻辑测试（部分不依赖 CUDA）以及可选的 **CuRobo 冒烟**（构造 `MotionGen` 并调用 `plan_dual`）。退出码：`0` 全部通过，`1` 失败，`2` 冒烟因环境缺失被跳过。

常用参数：

```bash
python scripts/verify_curobo_planner.py --device cuda:0 --urdf /path/to/robot.urdf
python scripts/verify_curobo_planner.py --no-smoke      # 只做轻量逻辑测试
python scripts/verify_curobo_planner.py --smoke-only
```

## 4. 机器人与 URDF 配置

### 4.1 默认 Realman + Franka 手 URDF

`config_generator.generate_robot_config_from_urdf` 与 `load_realman_config` 的默认 link 名称针对 **`realman_franka_ee.urdf`** 一类模型：

| 参数 | 默认值 |
|------|--------|
| `base_link` | `dual_rm_75b_description_platform_base_link` |
| 左末端 `ee_link`（主链） | `panda_left_hand` |
| 右末端（`link_names`） | `panda_right_hand` |

若使用 **其他 URDF**，必须在生成配置时传入匹配的 `base_link`、`left_ee_link`、`right_ee_link`（或等价地扩展 `load_realman_config` 的调用方式），否则 cuRobo 在解析运动链时可能出现 **`KeyError: '<link_name>'`**。

### 4.2 配置缓存

- 默认缓存目录：`~/.cache/curobo_realman/`。
- 默认缓存文件名：**`realman_config_v2.yaml`**（与旧版 `left_ee` / `right_ee` 等错误默认区分，避免误用旧缓存）。
- 缓存内容仅包含可重建的 **`CudaRobotGeneratorConfig` 字段**（通过 `dataclasses.asdict` 写入），与 `RobotConfig.from_dict` 的加载路径一致。**不要使用** cuRobo 自带的 `RobotConfig.write_config` 写本仓库生成的模型缓存：其对完整 `CudaRobotModelConfig` 做 `vars()`，会混入无法 YAML 序列化的对象。

修改 URDF 或 link 名后，若仍读旧文件，请删除对应缓存或更换 `cache_path`。

## 5. `CuroboPlanner` 使用说明

### 5.1 构造参数（摘要）

| 参数 | 含义 |
|------|------|
| `urdf_path` | URDF 路径 |
| `device` | 如 `cuda:0` |
| `use_curobo_cache` | 是否使用默认磁盘缓存 |
| `cache_path` | 显式指定缓存文件；`None` 时用默认 `realman_config_v2.yaml` |
| `interpolation_dt` | 插值时间步长（秒），影响稠密轨迹 |
| `apply_robot_to_curobo_frame_transform` | 是否将「机器人系 (x 右, y 前, z 上)」经绕 z 轴 **-90°** 对齐到 cuRobo 常用前向；若资产已与 cuRobo 一致可设为 `False` |
| `use_cuda_graph` | 是否启用 CUDA Graph（与 cuRobo 行为一致） |

内部使用 **`MotionGenConfig.load_from_robot_config(..., WorldConfig(), ...)`**：即使无障碍物，也会传入空的 `WorldConfig`，以便 cuRobo **创建世界碰撞检查器**；若传入 `world_model=None`，部分版本会导致 `world_coll_checker` 为空，随后在 `update_world` 时报错。

### 5.2 世界障碍物

- `set_world(obstacles)`：`obstacles` 为字典列表，字段包括 `position`（3）、`size` 或 `dims`（3）、可选 `quaternion`（wxyz）、可选 `name`。
- `clear_world()`：清空为无立方体障碍。
- 障碍物位姿与 `plan_dual` 的目标位姿使用 **同一套「机器人约定坐标系」**；若开启 `apply_robot_to_curobo_frame_transform`，封装内会一并变换到 cuRobo 帧。

### 5.3 主要 API

- **`plan_dual(start_joint_positions, goal_poses, ...)`**  
  - `start_joint_positions`：shape `(14,)`。  
  - `goal_poses`：`{'left': {'position', 'quaternion'}, 'right': {...}}`。  
  - 返回 **dict**（见下节）。

- **`plan(...)`**  
  - 默认返回 dict；若传入 `legacy_tuple_return=True`，兼容旧接口 `(success, trajectory)`。

- **`plan_single_arm(...)`**  
  - 只动一侧手臂时，调用方提供 **完整 14 维起始关节** 与 **固定侧末端目标位姿**（通常来自当前 FK），避免封装内再做 FK。

- **`plan_grippers(now_val, target_val, num_step)`**（静态方法）  
  - 夹爪线性插值，返回含 `result`、`num_step` 等的 dict。

- **`reset(reset_seed=True)`**  
  - 重置 MotionGen 内部状态（若底层支持）。

## 6. 规划返回 dict 约定

由 `result_utils.motion_gen_batch_result_to_plan_dict` 从 `plan_batch` 结果转换而来，常用键：

| 键 | 类型 | 说明 |
|----|------|------|
| `status` | `str` | `"Success"` 或 `"Fail"` |
| `position` | `ndarray` 或 `None` | 成功时一般为 `(T, n_dof)` 的 `float32` |
| `velocity` | `ndarray` 或 `None` | 成功且存在时与 `position` 时间维一致 |
| `detail` | 可选 | cuRobo 状态枚举名或字符串 |
| `interpolation_dt` | 可选 `float` | 插值步长 |

**注意**：在复杂场景或约束下，规划可能失败；冒烟测试允许 `Fail`，仅校验接口与结构；若需 CI 强约束成功，可使用验证脚本的严格模式（见脚本 `--help`）。

## 7. 工具函数导出

`isaaclab_logistics_vla.utils.curobo_planner` 包级导出：

- `CuroboPlanner`
- `motion_gen_batch_result_to_plan_dict`
- `plan_grippers_linear`（与 `plan_grippers` 逻辑一致，函数形式）

## 8. 运行示例

在已安装包且环境具备 cuRobo + CUDA 时：

```bash
python -m isaaclab_logistics_vla.utils.curobo_planner.example_usage
```

或在自己的代码中：

```python
from isaaclab_logistics_vla.utils.curobo_planner import CuroboPlanner
import numpy as np

planner = CuroboPlanner(urdf_path="...", device="cuda:0")
# planner.set_world([...])  # 可选
out = planner.plan_dual(start_joints, goal_poses)
```

## 9. 常见问题

1. **`KeyError` 与 link 名**  
   URDF 中不存在默认的 base / 末端 link 名时，在生成配置处改为实际名称。

2. **旧缓存**  
   删除 `~/.cache/curobo_realman/realman_config.yaml` 等旧文件，或统一使用 `realman_config_v2.yaml`。

3. **验证脚本的导入方式**  
   `verify_curobo_planner.py` 通过桩包只加载 `utils/curobo_planner`，**不执行**扩展根目录 `isaaclab_logistics_vla/__init__.py`，避免拉满 Isaac Lab；正常运行业务代码时仍按常规 `import isaaclab_logistics_vla` 即可。

4. **cuRobo 日志**  
   例如 batch 模式下 graph 与 `num_graph_seeds` 的提示来自 cuRobo 内部，一般不影响功能；若要消除，需按 cuRobo 文档调整 `MotionGenConfig` 相关种子参数。

5. **`RuntimeError: element 0 of tensors does not require grad`**  
   常见于外层包了 `torch.inference_mode()`（例如 `evaluate_vla.py` 主循环）。封装内在 `warmup` 与 `plan_batch` 处已用 `torch.inference_mode(False)` + `torch.enable_grad()` 包住 cuRobo 调用；若你在别处直接调 cuRobo API，需同样保证规划不在纯 inference 上下文中执行。

6. **`Inplace update to inference tensor outside InferenceMode`**  
   若在 `inference_mode` 下先构造了 `MotionGen`，其内部状态会成为 inference tensor，随后在关闭 inference 的 `warmup`/`plan` 里做 `copy_` 会失败。封装已将 **`load_realman_config`、`MotionGenConfig`、`MotionGen` 构造与 `warmup`** 放在同一 `_curobo_autograd_context()` 中；`evaluate_vla.py` 在 **`curobo_plan`** 策略下对 **`generate_action`** 不再包在 `inference_mode` 里（仅对 `env.step` 仍使用 inference）。

## 10. 与 `evaluate_vla.py` 联动的场景示例

仓库在 `isaaclab_logistics_vla/evaluation/models/curobo_plan_policy.py` 中提供 **`CuRoboPlanPolicy`**：在带 **`ee_frame`**（含 `left_ee_tcp` / `right_ee_tcp`）与 **Realman 双臂** 的任务里，用当前 TCP 在机器人根系下的位姿加上固定增量作为**双臂末端目标**，调用封装后的 `CuroboPlanner.plan_dual`，再把返回的关节轨迹写成环境的 **绝对关节位置动作**（与前 14 维 `arm_joints` 一致）。

启动示例（需正确配置 `ASSET_ROOT_PATH`，并安装 cuRobo + GPU；建议先 `--num_envs 1` 便于观察）：

```bash
./isaaclab.sh -p /path/to/isaaclab_logistics_vla/scripts/evaluate_vla.py \
  --policy curobo_plan \
  --task_scene_name Spawn_ms_st_dense_EnvCfg \
  --num_envs 1 \
  --device cuda:0
```

- 默认 URDF：`isaaclab_logistics_vla/assets/robots/realman/realman_franka_ee.urdf`；也可设环境变量 **`CUROBO_PLAN_URDF`** 覆盖。  
- 目标增量可在 `CuRoboPlanPolicy` 的 `left_goal_delta_base` / `right_goal_delta_base` 调整（机器人 **根** 坐标系下相对当前 TCP 的平移，单位米）；默认可达增量偏小，便于 dense 场景先规划成功。  
- cuRobo 轨迹关节顺序为 **左 7 + 右 7**，Isaac `arm_joints` 动作为 **l1,r1,…,l7,r7** 交错，策略内已做转换。  
- 当前实现用 **env 0** 的状态做规划，并将关节指令广播到所有并行环境。

评估器入口：`evaluation/evaluator/vla_evaluator.py` 在 `policy_name == "curobo_plan"` 时实例化上述策略；`--policy random`（默认）仍为原占位逻辑。

---

*文档版本与代码路径：`isaaclab_logistics_vla/utils/curobo_planner/`、`scripts/verify_curobo_planner.py`、`evaluation/models/curobo_plan_policy.py`。*
