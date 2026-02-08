# IsaacLab Logistics VLA：Evaluation 与摄像头观测工作总结

> 便于整理成 PPT 的 Markdown 提纲，每级标题可对应一页或一组幻灯片。  
> 已按当前版本更新：robot_registry（unnorm_key / camera_config_key / scene_robot_key）、configs/camera_configs、get_order_scene_cfg(robot_id)、OrderEnvCfg(robot_id)。

---

## 一、工作范围概览

- **Evaluation 模块**：VLA 评估流水线（观测构建、策略接口、结果落盘、EE/Joint 动作转换等）；按 **robot_id** 取机器人配置与相机配置。
- **Scene 摄像头观测**：相机配置放在 **configs/camera_configs/**（任务通用），按 **robot_id → camera_config_key** 动态取；场景通过 **get_order_scene_cfg(robot_id)** 挂载该机器人的机器人与相机。

---

## 二、Evaluation 块设计

### 2.1 设计原则

- 评估器**不包含策略逻辑**，只负责：创建环境 → 取观测 → 调用 policy → step → 记录。
- 策略接收的观测是 **ObservationBuilder** 产出的统一 **ObservationDict**，而非 env 原生的 group 观测。
- **robot_id** 贯穿：env_cfg、scene（机器人 + 相机）、评估侧配置（arm_dof、Curobo、unnorm_key）均按 robot_id 查表。

### 2.2 目录结构

| 路径 | 职责 |
|------|------|
| `evaluation/robot_registry.py` | 评估侧机器人注册表：robot_id → RobotEvalConfig（arm_dof、platform、Curobo、**unnorm_key**、**camera_config_key**、**scene_robot_key**） |
| `evaluation/evaluator/` | 评估驱动（VLA_Evaluator）+ VLA 专用环境封装（VLAIsaacEnv、关节映射） |
| `evaluation/observation/` | 观测 schema + ObservationBuilder 从 env 构建 ObservationDict |
| `evaluation/models/policy/` | Policy 抽象与多种实现（random、OpenVLA/OpenPI 远程等）；OpenVLA 的 unnorm_key 按 robot_id 从 registry 取 |
| `evaluation/result/` | EpisodeReport / TaskReport 落盘为 episodes.jsonl、task_summary_*.json |
| **configs/camera_configs/** | **按机器人区分的相机配置（任务通用）**：get_camera_config(key)，每个机器人一套 head/ee/top 相机，prim_path 绑定该机器人 link |

### 2.2.1 Evaluation 与相关目录结构

```
evaluation/
├── EVALUATION_AND_CAMERA_WORK_SUMMARY.md
├── robot_registry.py              # RobotEvalConfig：arm_dof, platform, Curobo, unnorm_key, camera_config_key, scene_robot_key
├── evaluator/
│   ├── vla_evaluator.py           # VLA_Evaluator；_make_policy_from_name(..., robot_eval_cfg) 传 unnorm_key 给 OpenVLA
│   └── VLAIsaacEnv.py
├── observation/
│   ├── builder.py
│   └── schema.py
├── models/policy/
│   ├── base.py
│   ├── openvla_remote_policy.py   # unnorm_key 由构造时传入（来自 registry）
│   └── ...
└── result/saver.py

configs/
├── robot_configs/                 # Curobo yml、spheres 等
└── camera_configs/                # 任务通用，按机器人 key 取相机
    ├── __init__.py                # get_camera_config(key), CAMERA_CONFIG_REGISTRY
    ├── realman.py                 # RealmanCameraConfig (head/ee/top，prim_path 绑 realman link)
    └── README.md
```

**组件关系（一页概括）：**

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    VLA_Evaluator                         │
                    │  (env_cfg, policy_name, robot_id) →  run_evaluation()    │
                    └─────────────────────────────────────────────────────────┘
                      │       │         │              │              │
                      ▼       ▼         ▼              ▼              ▼
              ┌──────────────┐  ┌───────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────────┐
              │robot_registry│  │VLAIsaacEnv │  │Observation   │  │ Policy   │  │ ResultSaver  │
              │robot_id 配置  │  │(关节映射)  │  │Builder       │  │.predict() │  │(episode/task)│
              │unnorm_key    │  └───────────┘  │+ schema 约定  │  │unnorm_key │  └─────────────┘
              │camera_config │        │        └──────────────┘  │← registry │
              │scene_robot   │        │              │            └──────────┘
              └──────┬───────┘        │              │
                     │                │              ▼
                     │         env_cfg.scene ← get_order_scene_cfg(robot_id)
                     │         (机器人+相机来自 registry)
                     └────────────────┴──────────────┴───────────────
                                                     │
                                                     ▼
                                              env.step(action)
```

### 2.3 观测 Schema（`observation/schema.py`）

- **MetaInfo**、**RobotState**（qpos/qvel）、**VisionData**（cameras、rgb、depth、intrinsic 等）、**PointCloudData**、**ObservationDict**（meta、robot_state、instruction、vision、point_cloud）、**ActionDict**、**ObservationRequire**：与旧版一致，策略只依赖这些键与 shape。

### 2.4 ObservationBuilder（`observation/builder.py`）

- 从 env（scene.sensors、observation_manager、scene["robot"].data）聚合成 ObservationDict；vision 按 camera_names 从 scene.sensors 取 rgb/depth/seg，多相机尺寸不一致时 resize 后堆叠。**相机本身由 scene 挂载，scene 按 robot_id 通过 get_order_scene_cfg(robot_id) 使用 configs/camera_configs 中该机器人的相机。**

### 2.5 VLAIsaacEnv（`evaluator/VLAIsaacEnv.py`）

- 在 ManagerBasedRLEnv 之上提供关节名 → 动作索引映射（_build_joint_mapping）；robot_entity = scene["robot"]；step 直接 super().step(action)。与旧版一致。

### 2.6 评估侧机器人注册表（`robot_registry.py`）

**作用**：多机器人接入时，按 **robot_id** 提供评估侧配置与**场景侧机器人/相机 key**，评估器与场景均不写死某一台。

- **RobotEvalConfig**（dataclass）：
  - **robot_id**：注册键。
  - **arm_dof**、**platform_joint_name**：IK、平台高度用。
  - **curobo_yml_name**、**curobo_asset_folder**、**curobo_urdf_name**：EE 模式 Curobo IK，None 表示不初始化。
  - **unnorm_key**：OpenVLA 等策略的 dataset_statistics key，与动作维度/归一化对应；None 时策略用默认（如 bridge_orig）。创建 OpenVLARemotePolicy 时从 registry 传入。
  - **camera_config_key**：对应 **configs/camera_configs** 中的 key，每个机器人一套相机（prim_path 绑定该机器人 link），**任务通用**。
  - **scene_robot_key**：场景中加载的机器人 asset 键，用于 `register.load_robot(scene_robot_key)`；None 表示沿用任务默认（如 realman_franka_ee）。
- **REGISTRY**、**get_robot_eval_config(robot_id)**、**list_registered_robots()**：与旧版一致。
- **接入新机器人**：在 REGISTRY 中新增一条（含 camera_config_key、scene_robot_key、unnorm_key 等）；在 **configs/camera_configs/** 中新增该机器人的相机模块并注册；若需 EE 模式，在 configs/robot_configs/ 及 spheres/ 下提供 Curobo yml；运行评估时传 `--robot_id <id>`。

### 2.7 VLA_Evaluator（`evaluator/vla_evaluator.py`）

- **构造**：接受 env_cfg、policy、trajectory_path、record_video、video_output_dir、**robot_id**。通过 **get_robot_eval_config(robot_id)** 得到 _robot_eval_cfg，用于 arm_dof、platform、Curobo、以及 **创建策略时传入 unnorm_key**（_make_policy_from_name(..., robot_eval_cfg=self._robot_eval_cfg)）。
- **run_evaluation()**：env.reset() → 循环：ObservationBuilder.build() → policy.predict(obs_dict) → _convert_actions_by_control_mode()（EE 时 Curobo IK）→ env.step(action) → 录视频、写结果。与旧版一致。
- **场景机器人说明**：启动时打印场景机器人 USD 路径（来自 env_cfg.scene，与 robot_registry 仅评估侧使用不同），便于核对。

### 2.7.1 vla_evaluator.py 模块展开（核心驱动）

本模块是评估流水线的**唯一驱动**：不实现任何策略逻辑，只负责「建环境 → 取观测 → 调策略 → 转动作（可选 IK）→ step → 录视频/写结果」。所有与 robot_id、相机、策略参数（如 unnorm_key）的衔接都在此完成。

#### 模块职责与定位

| 职责 | 说明 |
|------|------|
| **环境与策略** | 用 env_cfg 创建 VLAIsaacEnv；用 policy 名称或实例得到可调用的策略，策略的 unnorm_key 等来自 robot_registry。 |
| **观测** | 用 ObservationBuilder 从 env 构建 ObservationDict（meta / robot_state / vision / instruction），策略只消费该字典，不直接接触 env 的 group 观测。 |
| **动作** | 从策略取 action；若策略为 EE 模式，用 Curobo IK 将末端位移转为关节角后再交给 env.step。 |
| **结果** | 每步可选录多路相机视频；episode 结束时写 EpisodeReport / TaskReport 到 results/。 |

#### 构造函数 `__init__` 流程

1. **创建环境**：`self.env = VLAIsaacEnv(cfg=env_cfg)`。场景中的机器人与相机已由 env_cfg.scene 决定（来自 `get_order_scene_cfg(robot_id)`），此处不再次依赖 robot_id 改场景。
2. **取机器人评估配置**：`self._robot_eval_cfg = get_robot_eval_config(robot_id)`。得到 arm_dof、platform_joint_name、curobo_*、unnorm_key、camera_config_key、scene_robot_key 等，供后续 IK 与策略创建使用。
3. **创建策略**：若 `policy` 为字符串，调用 `_make_policy_from_name(policy, self.env, trajectory_path, robot_eval_cfg=self._robot_eval_cfg)`；否则直接使用传入的 policy 实例。OpenVLA 分支会从 `robot_eval_cfg.unnorm_key` 传入策略。
4. **ObservationBuilder**：`self._obs_builder = ObservationBuilder(self.env)`，以及 `ObservationRequire`、从 scene.sensors 收集的 `_camera_names`，用于 build 时指定需要哪些相机数据。
5. **视频与结果**：初始化 video_writers、video_output_dir、ResultSaver。
6. **Curobo IK（可选）**：若 `robot_eval_cfg` 中 curobo_yml_name / curobo_asset_folder / curobo_urdf_name 均非空，从 configs/robot_configs/ 和 assets/robots/ 加载 yml 与 URDF，创建 `IKSolver` 并保存 retract_config，供 EE 模式下的 `_convert_actions_by_control_mode` 使用；否则 EE 模式不可用。

#### 策略工厂 `_make_policy_from_name`

- **入参**：policy_name、env、trajectory_path、**robot_eval_cfg**（来自 `get_robot_eval_config(robot_id)`）。
- **作用**：根据 policy_name 实例化对应策略；从 env 取 `action_dim`、device。
- **与 robot_id 的衔接**：当 `policy_name == "openvla"` 时，`unnorm_key = getattr(robot_eval_cfg, "unnorm_key", None) or "bridge_orig"`，并传给 `OpenVLARemotePolicy(action_dim=..., device=..., unnorm_key=unnorm_key)`，保证**每种机器人用对动作归一化 key**。
- **支持名称**：random、openpi/pi0/openpi_remote、openvla_stub、openvla、rrt/trajectory；其他会 ValueError。

#### 观测与动作提取

- **观测**：`run_evaluation` 每步调用 `self._obs_builder.build(ctx, step_id, require=self._obs_require, camera_names=self._camera_names)`，得到 ObservationDict。相机数据来自 env 的 scene.sensors（即 scene_cfg 中按 robot_id 挂载的三路相机）。
- **动作**：`_get_action_from_policy(policy, obs_dict)` 统一接口：若策略有 `predict(obs)` 则用其返回值（tensor 或含 "action" 的 dict）；否则 `policy(obs)` 并取 "action"。保证不同策略实现都能被同一循环使用。

#### 动作转换 `_convert_actions_by_control_mode`

- **Joint 模式**：直接返回 `actions`，不做变换。
- **EE 模式**：
  1. 若无 `ik_solver` 或 obs_dict 缺少 `robot_state.qpos`，直接报错，避免静默错误。
  2. 从 env 的 robot data 取当前末端位姿（ee_pos, ee_quat）及 root 位姿；若有 `platform_joint_name`，臂基 = root + 平台高度偏移。
  3. 策略输出 `actions[:, :3]` 视为末端位移增量（米），得到目标世界系位置；再变换到臂基系，构造 Curobo `Pose`。
  4. 用多组 seed（retract_config、当前关节、零位）依次调用 `ik_solver.solve_single`；任一收敛则用解出的关节角替换 `actions[:, :arm_dof]` 并返回。
  5. 若全部未收敛，抛出 RuntimeError 并附带末端/目标/关节角等调试信息。  
  因此，**EE 模式依赖 robot_registry 的 arm_dof、platform_joint_name、Curobo 配置**，与 robot_id 一一对应。

#### 主循环 `run_evaluation`

1. `env.reset()`；若策略有 `reset()` 则调用。
2. 循环：  
   - `obs_dict = _obs_builder.build(...)`  
   - `actions = _get_action_from_policy(self.policy, obs_dict)`  
   - `actions = _convert_actions_by_control_mode(actions, obs_dict)`  
   - `obs, rew, terminated, truncated, info = self.env.step(actions)`  
   - `_record_frame_from_obs(obs_dict)`（按 vision 中的 cameras/rgb 写多路视频）  
   - 每 100 步打印一条进度；若 terminated 或 truncated 则退出循环。
3. 调用 `_save_evaluation_result` 写 EpisodeReport 与 TaskReport。
4. 若 KeyboardInterrupt 或异常，同样尝试保存结果并在 finally 中关闭视频写入器。

#### 视频录制与结果落盘

- **视频**：首次有观测时 `_init_video_writers(obs_dict)` 根据 vision.cameras / vision.rgb 为每个相机创建 imageio 写入器；之后每步 `_record_frame_from_obs` 写入一帧。相机名与数量来自 scene 中按 robot_id 挂载的传感器，与 configs/camera_configs 一致。
- **结果**：`_save_evaluation_result` 构造 EpisodeReport（episode_id、success、metrics_read、timing、task_name、episode_length），调用 `result_saver.write_episode` 与 `write_task`，结果落在 results/（episodes.jsonl、task_summary_*.json）。

#### 小结（可作一页 PPT）

- **vla_evaluator.py**：评估的单一入口驱动；**不包含策略逻辑**，只做「env + ObservationBuilder + policy + 动作转换 + step + 视频/结果」。
- **robot_id 贯穿**：通过 `get_robot_eval_config(robot_id)` 得到 _robot_eval_cfg，用于 Curobo IK（arm_dof、platform、yml/urdf）、策略创建（unnorm_key）、动作转换中的关节维度和臂基变换；场景与相机由 env_cfg.scene（即 get_order_scene_cfg(robot_id)）在构造 env 时已确定。
- **策略与观测**：策略只消费 ObservationBuilder 产出的 ObservationDict；OpenVLA 的 unnorm_key 在 _make_policy_from_name 中从 robot_eval_cfg 传入，保证多机器人时动作归一化正确。

### 2.8 策略层（`models/policy/`）

- **Policy**：name、control_mode（joint/ee）、reset、predict(obs) -> (num_envs, action_dim)。
- **OpenVLARemotePolicy**：**unnorm_key** 由构造时传入，**来自 robot_registry 的 robot_eval_cfg.unnorm_key**（_make_policy_from_name 在 policy_name=="openvla" 时取 getattr(robot_eval_cfg, "unnorm_key", None)，默认 "bridge_orig"），保证每个机器人用对动作归一化 key。
- **_make_policy_from_name(policy_name, env, trajectory_path=None, robot_eval_cfg=None)**：创建策略时传入 robot_eval_cfg，OpenVLA 使用其 unnorm_key。

### 2.9 ResultSaver（`result/saver.py`）

- 与旧版一致：EpisodeReport / TaskReport，episodes.jsonl、task_summary_*.json。

### 2.10 数据流（一页概括）

```
robot_id → get_robot_eval_config → (arm_dof, unnorm_key, camera_config_key, scene_robot_key)
                │
                ├→ 创建 OpenVLA 策略时传入 unnorm_key
                │
env_cfg = OrderEnvCfg(robot_id) → __post_init__: scene = get_order_scene_cfg(robot_id)(...)
                │
                └→ get_order_scene_cfg(robot_id) → get_camera_config(camera_config_key) + load_robot(scene_robot_key)
                           │
                           ▼
                    Scene：该机器人 + 该机器人三路相机 + 场景物体

ObservationBuilder.build → ObservationDict → Policy.predict → (EE 则 Curobo IK) → env.step → 录视频 / ResultSaver
```

---

## 三、按机器人区分的相机配置（configs/camera_configs）

### 3.1 设计原因

- 每个机器人的 link 名称不同（如 realman 的 head_link2、panda_left_hand，UR10e 的 ee_link、tool0），相机 **prim_path 必须绑定该机器人的 link**，否则挂载错误或无法渲染。
- 相机配置与**任务**无关，只与**机器人**有关，因此放在 **configs/camera_configs/**，**所有任务共用**。

### 3.2 目录与接口

- **configs/camera_configs/**：
  - **get_camera_config(key)**：根据 camera_config_key 返回该机器人的相机配置（含 .head_camera、.ee_camera、.top_camera）。
  - **CAMERA_CONFIG_REGISTRY**：key → 提供上述三路相机的类（如 realman.RealmanCameraConfig）。
  - **realman.py**：RealmanCameraConfig，head/ee/top 三路，prim_path 绑定 realman 的 head_link2、panda_left_hand 等。
  - **README.md**：接入新机器人时在本目录新增模块、在 REGISTRY 中注册，并在 robot_registry 中设置 camera_config_key。

### 3.3 Scene 中的使用

- **OrderSceneCfg**（默认 Realman）：head/ee/top 来自 `get_camera_config("realman")`。
- **get_order_scene_cfg(robot_id)**：根据 **get_robot_eval_config(robot_id)** 得到 **camera_config_key** 与 **scene_robot_key**；用 **get_camera_config(camera_config_key)** 取三路相机，用 **register.load_robot(scene_robot_key)** 加载机器人；动态构造场景类（该机器人 + 该相机 + OrderSceneCfg 的其余物体），返回该类。
- 任务（如 order_series）的 env_cfg 在 **__post_init__** 中调用 **get_order_scene_cfg(self.robot_id)** 得到场景类并实例化为 self.scene，从而**按 robot_id 动态挂载机器人与相机**。

### 3.4 观测组（observation_cfg.py）

- **ObservationsCfg.CamerasCfg**：仍按传感器名 head_camera、ee_camera、top_camera 引用；**传感器在 scene 中定义，scene 按 robot_id 从 configs/camera_configs 动态取**。observation_cfg 中不再保留 CameraConfig 别名，全部动态取。

### 3.5 工具函数（utils/util.py）

- euler_to_quat_isaac、camera_rot_look_along_parent_x/_z 等，与旧版一致，用于相机 offset.rot。

---

## 四、OpenVLA 接入

### 4.1 服务端：deploy.py

- **POST /act**，请求体：`{"image": np.ndarray (H,W,3 uint8), "instruction": str, "unnorm_key": Optional[str]}`，响应：`{"action": np.ndarray}`。
- **unnorm_key**：与数据集/机器人动作归一化对应；客户端（OpenVLARemotePolicy）的 unnorm_key 来自 **robot_registry 的 robot_eval_cfg.unnorm_key**，保证与当前 robot_id 一致。

### 4.2 客户端：OpenVLARemotePolicy

- 从 ObservationDict 取一张 RGB 与指令，按 prefer_camera 选相机，POST 到 /act，解析 action；**unnorm_key** 在构造时传入（由 _make_policy_from_name 从 robot_eval_cfg.unnorm_key 传入）。
- control_mode="ee"，评估器用 Curobo IK 转关节。host/port 可用环境变量 OPENVLA_HOST、OPENVLA_PORT。

### 4.3 步数与调用链

- 每一步：ObservationBuilder.build → OpenVLARemotePolicy.predict（带该机器人的 unnorm_key）→ POST /act → action →（EE 则 Curobo IK）→ env.step。**仿真每步 = 一次 OpenVLA 推理**。

---

## 五、评估使用方法（以 Realman + OpenVLA 为例）

### 5.1 前置条件

- 已安装本仓库与 IsaacLab；脚本内已设置 `args_cli.enable_cameras = True`。
- OpenVLA 服务端已运行（deploy.py），/act 可访问。

### 5.2 启动 OpenVLA 服务

```bash
cd /path/to/openvla
python vla-scripts/deploy.py --openvla_path openvla/openvla-7b --host 0.0.0.0 --port 8000
```

### 5.3 运行评估（Realman + OpenVLA）

```bash
# 在 isaaclab_logistics_vla 仓库根目录
python scripts/evaluate_vla.py --num_envs 1 --policy openvla --headless --robot_id realman_dual_left_arm
```

- **--robot_id realman_dual_left_arm**：使用 Realman 的评估配置与场景（scene_robot_key=realman_franka_ee，camera_config_key=realman，unnorm_key=bridge_orig）；不写则默认即为该值。
- **--policy openvla**：使用 OpenVLARemotePolicy，会请求上述 /act 服务。
- **--num_envs 1**、**--headless**：按需设置。

### 5.4 评估流程简述

1. 解析命令行 → **OrderEnvCfg(robot_id=args_cli.robot_id)**；在 __post_init__ 中 **self.scene = get_order_scene_cfg(self.robot_id)(num_envs=..., env_spacing=...)**，即场景按 robot_id 加载该机器人与该机器人的三路相机。
2. **VLA_Evaluator(env_cfg, policy="openvla", robot_id=...)**：通过 get_robot_eval_config(robot_id) 取配置；创建 OpenVLARemotePolicy 时传入 **unnorm_key=robot_eval_cfg.unnorm_key**。
3. run_evaluation()：env.reset() → 循环 build obs → policy.predict（带 unnorm_key）→ EE 则 Curobo IK → env.step → 录视频、写结果。

### 5.5 可选参数与扩展

- **--robot_id**：切换机器人；新机器人需在 robot_registry 的 REGISTRY 中注册（含 camera_config_key、scene_robot_key、unnorm_key），并在 **configs/camera_configs/** 中新增该机器人的相机模块并注册。
- IK 调试：**--ee_ik_fail_behavior hold**、**--ik_position_threshold**、**--ik_rotation_threshold**、**--ik_num_seeds** 等（见 vla_evaluator 与脚本入参）。
- OPENVLA_HOST、OPENVLA_PORT：策略连接的服务地址与端口。

---

## 六、接入新模型与接入新机器人

### 6.1 接入新机器人（完整步骤）

1. **configs/camera_configs/**：新增该机器人的模块（如 ur10e.py），定义 head_camera、ee_camera、top_camera（prim_path 绑定该机器人 link）；在 **CAMERA_CONFIG_REGISTRY** 中注册（如 `"ur10e": ur10e.UR10eCameraConfig`）。
2. **evaluation/robot_registry.py**：在 REGISTRY 中新增一条 RobotEvalConfig，填写 robot_id、arm_dof、platform_joint_name、**camera_config_key**（与 camera_configs 的 key 一致）、**scene_robot_key**（register.load_robot 的键）、**unnorm_key**（OpenVLA 用，若适用），以及 Curobo 相关（若需 EE 模式）。
3. **configs/robot_configs/**（若 EE 模式）：提供 Curobo yml 及 spheres；在 register 中注册该机器人的 asset 与 eeframe（命名约定如 `{scene_robot_key}_eeframe`）。
4. 运行评估时传 **--robot_id <id>** 即可；env 会通过 get_order_scene_cfg(robot_id) 加载该机器人与相机，策略会使用该机器人的 unnorm_key。

### 6.2 接入新模型（策略）

- 与旧版一致：实现 Policy 子类（name、predict、可选 control_mode、reset）；若为远程服务，仿 OpenVLARemotePolicy；在 _make_policy_from_name 中增加分支。若新策略需要“每机器人一个 key”（类似 unnorm_key），可在创建策略时传入 robot_eval_cfg 或从 registry 取对应字段。

### 6.3 注意事项

- **ObservationDict**：以 observation/schema.py 为准；策略只依赖 meta、robot_state、vision、instruction 等。
- **动作维度**：predict 返回的 action_dim 须与 env 的 total_action_dim 一致；EE 模式时评估器做 IK。
- **相机**：新机器人的相机必须在 configs/camera_configs 中单独定义，prim_path 绑定该机器人的 link，否则场景中相机位置错误或无法渲染。

---

## 七、评估侧多机器人接入（小结）

- **robot_registry**：robot_id → RobotEvalConfig，包含 **unnorm_key**（OpenVLA 等）、**camera_config_key**（configs/camera_configs 的 key）、**scene_robot_key**（场景机器人 asset）。
- **configs/camera_configs**：任务通用，按 key 提供每机器人的 head/ee/top 相机；scene 通过 get_order_scene_cfg(robot_id) 使用 get_camera_config(camera_config_key) 与 load_robot(scene_robot_key)。
- **OrderEnvCfg(robot_id)**：__post_init__ 中 self.scene = get_order_scene_cfg(self.robot_id)(...)，保证环境中的机器人与相机随 robot_id 切换。
- **VLA_Evaluator**：创建策略时传入 robot_eval_cfg，OpenVLA 使用 registry 中的 unnorm_key。
- 接入新机器人：在 camera_configs 中新增模块并注册 → 在 robot_registry 中新增一条（含 camera_config_key、scene_robot_key、unnorm_key 等）→ 必要时 Curobo/register → 运行 `--robot_id <id>`。

---

## 八、PPT 建议结构（每行为一页标题）

1. 标题：IsaacLab Logistics VLA — Evaluation 与摄像头观测工作总结
2. 工作范围概览
3. Evaluation 设计原则与 robot_id 贯穿
4. Evaluation 与 configs 目录结构
5. **Evaluation 块结构展示**（树状图 + 组件关系图）
6. 观测 Schema 与 ObservationDict
7. ObservationBuilder 与 scene 相机来源
8. VLAIsaacEnv 与关节映射
9. **评估侧机器人注册表**（arm_dof、unnorm_key、camera_config_key、scene_robot_key）
10. **VLA_Evaluator 模块展开**（构造、策略工厂、观测/动作、EE IK、主循环、视频与结果）
11. VLA_Evaluator 流程与策略创建（unnorm_key）
12. 策略层与 ResultSaver
13. 数据流总览（robot_id → scene / policy）
14. **按机器人区分的相机配置（configs/camera_configs）**
15. get_order_scene_cfg(robot_id) 与 Scene 挂载
16. OrderEnvCfg(robot_id) 与动态场景
17. **OpenVLA 接入：unnorm_key 按 robot_id**
18. **评估使用方法（Realman + OpenVLA 命令示例）**
19. **接入新机器人：camera_configs + robot_registry**
20. 总结：多机器人 + 任务通用相机 + 动态 scene/env

---

## 九、总结（可直接做结束页）

- **Evaluation 块**：通过 schema + ObservationBuilder、VLA_Evaluator（按 robot_id 取配置、创建带 unnorm_key 的策略）、多种 Policy、ResultSaver，形成与具体 env 解耦的 VLA 评估流水线。
- **多机器人融合**：**robot_registry** 提供 arm_dof、platform、Curobo、**unnorm_key**、**camera_config_key**、**scene_robot_key**；**configs/camera_configs** 任务通用，按机器人 key 提供三路相机；**get_order_scene_cfg(robot_id)** 与 **OrderEnvCfg(robot_id)** 保证场景与 env 按 robot_id 动态切换机器人与相机。
- **相机与场景**：相机配置与任务解耦，放在 configs/camera_configs；场景通过 robot_id → camera_config_key 与 scene_robot_key 动态挂载该机器人的机器人与三路相机，供观测、录视频与远程策略使用。
