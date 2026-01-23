## 单臂分拣任务中的随机化说明（Single Arm Sorting Randomization）

本文档说明当前 `single_arm_sorting` 任务（尤其是 Realman 配置）中**目标物随机化**的整体设计与使用方式，包括：

- 通用随机化（基础环境公共逻辑）
- Realman 专用多物体目标池随机化
- 如何开启 / 关闭 / 调整随机化

文中涉及的主要文件：

- `single_arm_sorting_env_cfg.py`：基础环境配置，包含通用随机化事件定义。
- `single_arm_sorting/mdp/randomization.py`：通用随机化函数（`randomize_object_positions`、`randomize_object_properties`）。
- `single_arm_sorting/object_randomization.py`：Realman 专用目标物池随机化实现（`randomize_target_objects`、`get_active_object_pose_w` 等）。
- `single_arm_sorting/config/realman/__init__.py`：Realman 特定配置，包含目标物池配置和事件注册。

---

## 1. 通用随机化（SingleArmSortingEnvCfg）

文件：`single_arm_sorting_env_cfg.py`

在基础环境配置中，`EventCfg` 定义了两类通用随机化事件：

```python
@configclass
class EventCfg:
    """Configuration for randomization events."""

    # 位置随机化：在 source_area 周围轻微抖动对象位置 / 姿态
    randomize_object_position = EventTerm(
        func=mdp.randomize_object_positions,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("object"),
            "source_cfg": SceneEntityCfg("source_area"),
            "pos_range": ((-0.1, 0.1), (-0.1, 0.1), (0.0, 0.1)),
        },
    )

    # 属性随机化（当前为占位实现）
    randomize_object_properties = EventTerm(
        func=mdp.randomize_object_properties,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("object"),
            "mass_range": (0.05, 0.2),
            "scale_range": (0.8, 1.2),
        },
    )
```

环境配置中挂载该事件配置：

```python
@configclass
class SingleArmSortingEnvCfg(ManagerBasedRLEnvCfg):
    ...
    events: EventCfg = EventCfg()
```

### 1.1 调用时机

- 每次 `env.reset()` 时，`EventManager` 会在 `mode="reset"` 下自动依次调用：
  - `mdp.randomize_object_positions(env, env_ids, object_cfg, source_cfg, pos_range)`
  - `mdp.randomize_object_properties(env, env_ids, object_cfg, mass_range, scale_range)`

### 1.2 实现细节（`mdp/randomization.py`）

#### 1.2.1 `randomize_object_positions`

签名：

```python
def randomize_object_positions(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    source_cfg: SceneEntityCfg = SceneEntityCfg("source_area"),
    pos_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = ...,
) -> None:
```

逻辑：

1. 解析 `env_ids`（若为 `None` 则默认所有环境）。
2. 从 `env.scene` 中取出：
   - `source = env.scene[source_cfg.name]`
   - `obj = env.scene[object_cfg.name]`
3. **类型判定**：
   - 若 `obj` 是 `RigidObjectCollection`，直接 `return`（集合对象由 Realman 专用逻辑控制；见第 2 节）。
   - 若 `obj` 是 `RigidObject`，才执行位置随机化。
4. 对选中的 `env_ids`，计算基于 `source.data.root_pos_w` 的随机偏移，并写回：

   ```python
   obj_pos_full = obj.data.root_pos_w.clone()
   obj_quat_full = obj.data.root_quat_w.clone()

   # 只修改 env_ids 对应的行
   obj_pos_full[env_ids] = obj_pos
   obj_quat_full[env_ids] = random_quat

   obj.set_root_pose_w(obj_pos_full, obj_quat_full)
   ```

#### 1.2.2 `randomize_object_properties`

签名：

```python
def randomize_object_properties(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    mass_range: tuple[float, float] = (0.05, 0.2),
    scale_range: tuple[float, float] = (0.8, 1.2),
) -> None:
```

逻辑：

1. 从 `env.scene` 中取出 `obj = env.scene[object_cfg.name]`。
2. 若 `obj` 是 `RigidObjectCollection`，直接 `return`，交由专用逻辑处理。
3. 若 `obj` 是 `RigidObject`，当前仅作为占位实现（`pass`），方便未来扩展质量 / 尺度等属性随机化。

### 1.3 如何关闭通用随机化

如果只希望使用 Realman 专用的目标物池随机化，而**不使用**通用位置 / 属性随机化，有两种方式：

1. **直接注释掉 `EventCfg` 中的两个 `EventTerm`**：

   ```python
   @configclass
   class EventCfg:
       pass
   ```

2. 或者在 `SingleArmSortingEnvCfg` 里不挂载 `events`（不推荐，易忘记）：

   ```python
   # events: EventCfg = EventCfg()
   ```

在当前实现中，即便开启通用随机化，对于 `RigidObjectCollection` 类型的 `scene.object`，上述函数也会自动跳过，不会和 Realman 的目标物随机化冲突。

---

## 2. Realman 专用目标物池随机化

文件：`single_arm_sorting/config/realman/__init__.py`  
辅助实现：`single_arm_sorting/object_randomization.py`

Realman 版本中，`scene.object` 不再是单一 `RigidObject`，而是一个可配置的 `RigidObjectCollection`（多物体池）。该物体池由以下几部分构成：

- `TargetObjectSpec`：描述一个可生成物体的规格。
- `TargetObjectRandomizationCfg`：描述整个物体池的配置（候选列表、生成范围等）。
- `build_object_collection_cfg`：根据配置构造 `RigidObjectCollectionCfg`。
- `randomize_target_objects`：在 reset 时从物体池中抽样若干物体放到台面上，并选定“当前目标物”。
- `get_active_object_pose_w`：在观测 / 策略中获取当前目标物的世界位姿。

### 2.1 目标物池配置（`TargetObjectRandomizationCfg`）

在 `RealmanSingleArmSortingEnvCfg.__post_init__` 中：

```python
if not hasattr(self, "object_randomization") or self.object_randomization is None:
    self.object_randomization = TargetObjectRandomizationCfg(
        asset_pool=[
            TargetObjectSpec(
                name="red_cube",
                size=(0.10, 0.10, 0.10),
                mass=0.12,
                color=(0.85, 0.2, 0.2),
                num_instances=2,
            ),
            TargetObjectSpec(
                name="green_cube",
                size=(0.08, 0.08, 0.08),
                mass=0.1,
                color=(0.2, 0.75, 0.35),
                num_instances=2,
            ),
            TargetObjectSpec(
                name="blue_cube",
                size=(0.06, 0.06, 0.06),
                mass=0.08,
                color=(0.2, 0.4, 0.8),
                num_instances=2,
            ),
        ],
        max_spawned_objects=3,
        pose_range={
            "x": (-0.08, 0.08),
            "y": (1.40, 1.56),
            "z": (0.92, 1.02),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-0.3, 0.3),
        },
        min_separation=0.04,
        offstage_pose=(4.0, 4.0, 4.0),
    )
```

含义：

- **`asset_pool`**：可供选择的物体类型列表（颜色、尺寸、质量、数量）。
  - 可以改成任意 USD 资源或不同形状，只需调整 `TargetObjectSpec`。
  - 若只想要"红色是真目标物"，可以只保留 `red_cube` 的规格。
- **`max_spawned_objects`**：每个环境最终会放到台面上的物体上限（干扰 + 目标）。
- **`pose_range`**：物体生成范围配置。
  - **严格模式（当前默认）**：`"x"` 和 `"y"` 会被忽略（由源区域的尺寸决定），只有 `"z"` 和 `"yaw"` 生效。
  - **通用模式**：所有坐标轴范围都生效，用于在世界坐标中采样。
  - 已调至合理的默认值，`"z"` 范围覆盖源区域高度附近，`"yaw"` 控制小角度旋转。
- **`min_separation`**：物体之间的最小距离，避免重叠（仅在通用模式下生效，严格模式下由源区域尺寸自然限制）。
- **`offstage_pose`**：未被选中的物体统一移出场景的远端坐标。

然后通过：

```python
self.scene.object = build_object_collection_cfg(
    self.object_randomization,
    prim_prefix="{ENV_REGEX_NS}/Object",
)
```

构建出对应的 `RigidObjectCollectionCfg` 并注入场景。

### 2.2 Reset 时的目标物随机化（`randomize_target_objects`）

事件配置：

```python
@configclass
class RealmanEventsCfg:
    randomize_targets: EventTerm = MISSING

...

self.events = RealmanEventsCfg()
self.events.randomize_targets = EventTerm(
    func=randomize_target_objects,
    mode="reset",
    params={
        "object_cfg": SceneEntityCfg("object"),
        "randomization_cfg": self.object_randomization,
        "source_cfg": SceneEntityCfg("source_area"),  # 源区域配置
        "source_size_xy": (0.5, 0.35),  # 源区域的 XY 尺寸（与 source_area.spawn.size 对应）
    },
)
```

**重要**：当提供了 `source_cfg` 和 `source_size_xy` 时，物体会**严格在源区域（红色区域）内部生成**，而不是基于 `pose_range` 的通用采样。

实现（`object_randomization.py` 中的核心逻辑）：

1. 根据 `randomization_cfg.asset_pool` 中声明的实例总数，`RigidObjectCollection` 会有固定数量的 object 槽位。
2. 每次 reset，对每个 env：
   - 随机决定要在台面上出现几个物体（不超过 `max_spawned_objects`）。
   - 在集合中随机选择对应数量的 index。
   - **位置采样方式**：
     - **严格版（当前默认）**：如果提供了 `source_cfg` 和 `source_size_xy`：
       - 读取 `source_area.data.root_pos_w[env_id]` 作为中心点。
       - 在源区域的 XY 半尺寸范围内均匀采样：`local_x ∈ [-size_x/2, size_x/2]`，`local_y ∈ [-size_y/2, size_y/2]`。
       - Z 坐标和 yaw 角仍从 `pose_range["z"]` 和 `pose_range["yaw"]` 中采样。
       - Roll 和 Pitch 固定为 0（物体平放在源区域上）。
       - **优点**：确保所有物体都落在红色源区域的可视范围内，不会超出边界。
     - **通用版（向后兼容）**：如果未提供 `source_cfg` 或 `source_size_xy`：
       - 退回到基于 `pose_range` 的通用采样，使用 `_sample_object_poses` 函数。
       - 仍然考虑 `min_separation` 来避免物体重叠。
   - 将被选中的 index 写入 `RigidObjectCollection` 的状态（其他 index 写到 `offstage_pose`）。
3. 选中的集合中**第一个 index** 被记录为“当前焦点/目标物”：

   ```python
   active_ids[env_id] = chosen[0]
   env._active_object_indices = active_ids
   ```

### 2.3 函数签名与参数说明

#### `randomize_target_objects`

```python
def randomize_target_objects(
    env,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    randomization_cfg: TargetObjectRandomizationCfg,
    source_cfg: SceneEntityCfg | None = None,
    source_size_xy: tuple[float, float] | None = None,
) -> None:
```

**参数说明**：
- `env`: 环境实例。
- `env_ids`: 需要随机化的环境 ID 张量（通常由 `EventManager` 自动传入）。
- `object_cfg`: 目标物体集合的配置（通常是 `SceneEntityCfg("object")`）。
- `randomization_cfg`: `TargetObjectRandomizationCfg` 实例，包含物体池、最大生成数、范围等配置。
- `source_cfg`: （可选）源区域的场景实体配置。如果提供，将启用“严格在源区域内部采样”模式。
- `source_size_xy`: （可选）源区域的 XY 尺寸元组 `(size_x, size_y)`。必须与 `source_cfg` 同时提供才会生效。

**采样行为**：
- 当 `source_cfg` 和 `source_size_xy` **都提供**时：严格在源区域的 XY 范围内采样，Z 和 yaw 仍由 `pose_range` 控制。
- 当两者**任一未提供**时：退回到基于 `pose_range` 的通用采样模式。

### 2.4 获取当前目标物位姿（`get_active_object_pose_w`）

为了统一单物体和多物体集合的访问方式，提供了一个辅助函数：

```python
def get_active_object_pose_w(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回当前任务关注的物体位姿 (pos_w, quat_w)."""
    asset = env.scene[object_cfg.name]
    if isinstance(asset, RigidObjectCollection):
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
        active_ids = getattr(env, "_active_object_indices", None)
        if active_ids is None or active_ids.numel() != env.scene.num_envs:
            active_ids = torch.zeros(env.scene.num_envs, dtype=torch.long, device=env.device)
        pos = asset.data.object_pos_w[env_ids, active_ids]
        quat = asset.data.object_quat_w[env_ids, active_ids]
    else:
        pos = asset.data.root_pos_w
        quat = asset.data.root_quat_w
    return pos, quat
```

在以下模块中，用它来统一获取“真正目标物”的位姿：

- 观测：`mdp/observations.py`（例如 `object_position_in_robot_root_frame`）
- 度量 / 终止：`mdp/metrics.py`（例如 `active_object_height_below_minimum`）
- 左手抓取策略：`method/pick_place_left_policy.py`（FSM 中的目标物位置计算）

---

## 3. 如何控制 / 调整随机化

### 3.1 调整生成物体的种类与数量

编辑 `RealmanSingleArmSortingEnvCfg.__post_init__` 中的 `object_randomization.asset_pool`：

- 增加新的 `TargetObjectSpec`，即可加入新的候选物种类。
- 修改 `num_instances`，控制物体池中每种物体的实例数。
- 删掉不需要的规格，例如：
  - 只保留 `red_cube`，则所有目标物都是红色立方体。

### 3.2 调整物体生成范围

#### 3.2.1 严格模式（推荐，当前默认）

当事件配置中提供了 `source_cfg` 和 `source_size_xy` 时，物体严格在源区域内部生成：

- **XY 位置**：自动绑定到源区域的范围，无需手动调整（除非你想修改源区域的尺寸）。
- **Z 位置**：通过 `pose_range["z"]` 控制高度范围。
- **Yaw 角度**：通过 `pose_range["yaw"]` 控制朝向随机度。

要调整源区域的尺寸，修改 `RealmanSingleArmSortingSceneCfg` 中的 `source_area`：

```python
source_area = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/SourceArea",
    init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.04, 1.8, 0.75], ...),
    spawn=sim_utils.CuboidCfg(
        size=(0.5, 0.35, 0.01),  # 修改这里的 X 和 Y 尺寸
        ...
    ),
)
```

**注意**：如果修改了源区域的 `size`，必须同步更新事件参数中的 `source_size_xy`：

```python
self.events.randomize_targets = EventTerm(
    func=randomize_target_objects,
    mode="reset",
    params={
        ...
        "source_size_xy": (0.5, 0.35),  # 必须与 source_area.spawn.size 的 XY 对应
    },
)
```

#### 3.2.2 通用模式（向后兼容）

如果注释掉 `source_cfg` 和 `source_size_xy` 参数，会退回到通用采样模式：

```python
self.events.randomize_targets = EventTerm(
    func=randomize_target_objects,
    mode="reset",
    params={
        "object_cfg": SceneEntityCfg("object"),
        "randomization_cfg": self.object_randomization,
        # "source_cfg": SceneEntityCfg("source_area"),  # 注释掉
        # "source_size_xy": (0.5, 0.35),  # 注释掉
    },
)
```

此时可以编辑 `object_randomization.pose_range`：

- 修改 `"x" / "y" / "z"` 范围，可以将物体生成在更近或更远的区域。
- 修改 `"yaw"` 范围可以控制旋转随机度。
- 已经将默认范围设置为**原本单物体坐标附近的小范围扰动**，便于迁移之前的策略与视觉。

### 3.3 控制“桌面上最多有几个物体”

编辑 `object_randomization.max_spawned_objects`：

- 例如设为 `1`：即使物体池很大，每次 reset 桌面上也只会出现 1 个物体（无干扰）。
- 设为 `3`：每次最多 3 个（1 个真实目标 + 若干干扰物）。

### 3.4 打开 / 关闭不同层级的随机化

- **使用严格模式（当前默认）**：
  - 保留 `RealmanEventsCfg.randomize_targets`，并确保传入了 `source_cfg` 和 `source_size_xy`。
  - 物体会严格在源区域内部生成。
  - 可选择性关掉 `EventCfg` 中的 `randomize_object_position / randomize_object_properties`（因为 `RigidObjectCollection` 会自动跳过这些通用随机化）。
- **使用通用模式（基于 pose_range）**：
  - 保留 `RealmanEventsCfg.randomize_targets`，但注释掉 `source_cfg` 和 `source_size_xy` 参数。
  - 物体会根据 `pose_range` 在世界坐标中采样，可能超出源区域的可见范围。
- **完全关闭随机化**：
  - 注释掉 `EventCfg` 中的通用事件。
  - 注释掉 `RealmanSingleArmSortingEnvCfg.__post_init__` 中的 `self.events.randomize_targets = EventTerm(...)`。
  - 此时 `scene.object` 将保持构建时的默认初始姿态。

---

## 4. 与左手抓取策略的关系

文件：`method/pick_place_left_policy.py`

- 策略在每个 step 中通过：

  ```python
  from isaaclab_logistics_vla.tasks.single_arm_sorting.object_randomization import (
      get_active_object_pose_w,
  )

  object_pos_w, _ = get_active_object_pose_w(env.unwrapped)
  ```

  获取当前目标物的世界坐标。

- 即便随机生成了多个物体，**策略始终只围绕 `get_active_object_pose_w` 返回的那一个“真目标物”进行抓取与放置**。

这保证了：

- 多物体随机化可以增加场景复杂度和干扰；
- 但任务定义依然清晰：在所有随机生成的物体中，始终只有一个被视为“当前任务的目标物”，并在观测、奖励和策略中保持一致。


