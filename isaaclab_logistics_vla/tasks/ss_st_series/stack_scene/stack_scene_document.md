# stack_scene 实现说明

基于当前代码说明 `stack_scene` 的复用关系、堆叠算法实现细节，以及如何在现有实现上新增物品。

---

## 1. 与 sparse_scene 的复用关系

| 层级 | 复用方式 |
|------|----------|
| **继承链** | 两者都继承 `AssignSSSTCommandTerm` → `OrderCommandTerm`。订单生成、订单完成率、错抓错放等指标及 reset 主流程由基类统一处理。 |
| **场景结构** | 同样继承 `BaseOrderSceneCfg`，通过 `SKU_DEFINITIONS` + 循环 `setattr` 动态注入物体，原料箱/目标箱命名一致（s_box_1/2/3, t_box_1/2/3）。 |
| **env_cfg / observation / reward / event** | 结构与 sparse 同构，仅替换为 stack 对应的 Scene/Commands/Reward/Event 配置类。 |

**主要差异点集中在 stack 的 CommandTerm：**

- `_assign_objects_boxes(env_ids)`：在基类分配逻辑基础上，增加了「随机原料箱 + 堆叠布局规划」；
- `_spawn_items_in_source_boxes(env_ids)`：完全改写为「按堆叠布局将物体摆放到 4 个槽位中」；
- `_update_spawn_metrics()`：新增针对堆叠质量的加权评分指标 `stack_weighted_score`。

其余如 `_reset_idx`、通用指标更新、命令接口等仍沿用基类逻辑。

---

## 2. stack_scene 具体实现

### 2.1 配置层

- **scene_cfg（`scene_cfg.py`）**
  - `SKU_DEFINITIONS` 为三元组：`(usd_path, count, scale)`，支持按 SKU 配置缩放（如 0.4），示例：
    - `"cracker_box": (CRACKER_BOX_PARAMS['USD_PATH'], 4, 0.4)` 等。
  - 当前仅使用方盒类 SKU：`cracker_box`, `sugar_box`, `plastic_package`, `sf_big`, `sf_small`。
  - 通过循环 `setattr(Spawn_ss_st_stack_SceneCfg, instance_name, obj_cfg)` 动态注入每个 SKU 的多个实例。

- **command_cfg / CommandTermCfg（`command_cfg.py` + `Spawn_ss_st_stack_CommandTermCfg.py`）**
  - `Spawn_ss_st_stack_CommandTermCfg` 继承 `OrderCommandTermCfg`，增加了堆叠相关参数：
    - `max_active_skus`: 单个环境最多激活的 SKU 种类数；
    - `max_stacks`: 单个原料箱最多允许的堆叠摞数；
    - `max_per_stack`: 每一摞允许的最大高度（物体数量）。
  - `Spawn_ss_st_stack_CommandsCfg.order_info` 中配置：
    - `objects=['cracker_box', 'sugar_box', 'plastic_package', 'sf_big', 'sf_small']`；
    - `source_boxes=['s_box_1', 's_box_2', 's_box_3']`；
    - `target_boxes=['t_box_1', 't_box_2', 't_box_3']`；
    - 以及上面的 `max_active_skus / max_stacks / max_per_stack`。

### 2.2 堆叠参数缓存（`Spawn_ss_st_stack_CommandTerm`）

- 顶部有一个 `_SKU_PARAMS_MAP` 映射：
  - `"cracker" → CRACKER_BOX_PARAMS`，`"sugar" → SUGER_BOX_PARAMS`，`"plastic_package" → PLASTIC_PACKAGE_PARAMS`，`"sf_big" → SFBIG_PARAMS`，`"sf_small" → SFSMALL_PARAMS`。
  - 新增 SKU 时只需在此字典里补一行即可挂到堆叠逻辑中。

- **`_build_stack_params_cache()`**
  - 对 `self.object_names` 中的每个实例调用 `_get_raw_params(obj_name)`，然后 `_compute_stack_params(...)`。
  - 得到的缓存结构大致为：
    - `base_area`: 作为底面的面积，用于排序；
    - `stack_height`: 作为单层堆叠高度；
    - `stack_orient`: 用于该 SKU 的堆叠朝向（来自 `constant.py` 中的 `STACK_ORIENT`）。

- **`_get_raw_params(obj_name)`**
  - 先从 `SKU_DEFINITIONS` 中读取该 SKU 的 `scale`，若无则用默认 `SCALE=1.0`；
  - 然后在 `_SKU_PARAMS_MAP` 中根据 sku 片段（如 `"cracker"`）查到对应的 `XXX_PARAMS`：
    - 将 `X/Y/Z_LENGTH` 乘以 `scale`，并带上 `STACK_ORIENT`，返回一个简化参数字典；
  - 若没有匹配到任何 key，则回退到 `CRACKER_BOX_PARAMS` 作为兜底。

- **`_compute_stack_params(params)`**
  - 对缩放后的 `(X_LENGTH, Y_LENGTH, Z_LENGTH)` 排序：
    - 最大两维相乘得到 `base_area`；
    - 最小一维作为 `stack_height`；
    - 保留 `STACK_ORIENT`。
  - 这一设计依赖当前 SKU 都是「方盒类」，可以通过长宽高直接推导出「最小维度朝上」的合理堆叠方式。

### 2.3 物品 / 箱子分配（`_assign_objects_boxes`）

在基类负责「哪个物体对应哪个 SKU、属于哪个 env」的基础上，本函数完成「**随机原料箱 + 堆叠布局规划**」：

1. **清空状态**
   - 将 `obj_to_target_id / obj_to_source_id / stack_layout` 在 `env_ids` 上全部置为 -1。
2. **读取堆叠参数**
   - 从 cfg 中读取 `max_stacks`, `max_per_stack`, `max_active_skus`。
3. **逐 env 处理**
   - 随机选一个原料箱：  
     `selected_source_box = torch.randint(0, self.num_sources, (1,)).item()`；
   - 在 \[1, `max_stacks`] 中随机出 `n_stacks`；
   - 根据 `max_active_skus`、总槽位上限 `max_per_stack * n_stacks` 以及当前场景可用实例数，确定激活的 SKU 数 `m_skus`；
   - 计算一个目标总数区间：
     - 下界：`min_total = max_per_stack * (n_stacks - 1) + 1`，保证必须用到第 `n_stacks` 摞；
     - 上界：`max_total = max_per_stack * n_stacks`，不超过可摆放总槽位；
     - 若可用实例不足，会减小 `n_stacks` 直到满足约束。
   - 在 \[min_total, max_total] 中随机一个 `target_total`，并在每种 SKU 上「先保证每个 SKU 至少 1 个，再随机分配剩余数量」。
4. **激活物体并写回映射**
   - 对每种选中的 SKU，根据 `counts[i]` 随机挑出若干个实例：
     - 将这些实例的 `obj_to_source_id` 设为 `selected_source_box`；
     - 将 `obj_to_target_id` 统一设为 0（**当前版本只有目标物，没有显式干扰物**）。
5. **构建堆叠布局 `stack_layout`**
   - 使用 `stacks: list[list[int]] = [[] for _ in range(n_stacks)]`；
   - 遍历所有激活的物体列表，按「**同种 SKU 优先填满当前摞，满了再换下一摞**」进行分配；
   - 每一摞内部再按 `base_area` 从大到小排序（底部最大，顶部最小）；
   - 最终将布局写入 `self.stack_layout[env_id, selected_source_box, stack_idx, pos]` 中。

最后，将 `is_active_mask` / `is_target_mask` 由 `obj_to_source_id` / `obj_to_target_id` 推导出来，供后续指标和奖励使用。

### 2.4 物品生成（`_spawn_items_in_source_boxes`）

该函数不再关心「选哪些物体/属于哪个 SKU」，而是**完全按照 `_assign_objects_boxes` 写好的 `stack_layout` 来摆放**：

1. 将 `env_ids` 中所有物体通过 `_move_all_objects_far` 先移到远处。
2. 通过 `_get_slot_anchors()` 计算原料箱中的 **4 个槽位锚点（2×2 网格）**：
   - 使用 `WORK_BOX_PARAMS['X_LENGTH']` / `Y_LENGTH`，在箱内划分 \(-x/4, +x/4) × (-y/4, +y/4) 的四个点。
3. 对每个 env：
   - 找到该 env 中被激活的原料箱索引 `box_idx`；
   - 取出对应的 `layout = self.stack_layout[env_id, box_idx]`；
   - 随机生成一个长为 4 的 `slot_perm`，决定「第 k 摞 → 第几个槽位」的映射；
   - 对每一摞：
     - 对应的槽位锚点为 `anchor = anchors[slot_perm[stack_idx % 4]]`；
     - 从底往上遍历当前摞中的所有物体：
       - 使用 `stack_height` 累加 `z_offset`，得到每一层的中心高度；
       - 在 XY 平面加入轻微抖动（±0.005m）；
       - 使用 `STACK_ORIENT` + 轻微 yaw 抖动（±5°）计算四元数；
       - 调用 `set_asset_relative_position(...)` 将物体相对原料箱放置到对应位置。

当前版本中**所有激活物都是目标物**，没有专门的干扰物堆叠/散放逻辑；若后续需要「目标摞 + 干扰摞/散放」的混合布局，可以在 `_assign_objects_boxes` 中区分目标 / 干扰，并在 spawn 时分别处理。

### 2.5 堆叠质量指标（`_update_spawn_metrics`）

为了衡量「堆叠中重要层次是否被成功放置」，实现了一个加权得分：

- 对每个 env、每个原料箱、每一摞：
  - 令 `stack_size` 为该摞物体数量；
  - 对第 `pos` 个物体（从底到顶），权重设置为 `weight = stack_size - pos`（底层权重最高）；
  - 若该物体 `current_states[env_id, obj_idx] == 3`（成功放置），则累计 `weighted_success += weight`；
  - 无论成功与否都累计 `total_weight += weight`。
- 最终 `stack_weighted_score = weighted_success / total_weight`，在 `total_weight > 0` 的 env 上生效。

该指标以 `metrics["stack_weighted_score"]` 暴露，后续可直接在 reward 或日志中使用。

---

## 3. 如何增加新物品（stack_scene）

### 3.1 在 `utils/constant.py` 中定义参数

1. 仿照现有方盒物体（如 `CRACKER_BOX_PARAMS`）增加一个 `XXX_PARAMS`：
   - 至少包含：`USD_PATH`, `X_LENGTH`, `Y_LENGTH`, `Z_LENGTH`, `STACK_ORIENT`；
   - 若要复用 sparse 逻辑，也可以额外提供 `SPARSE_ORIENT`。
2. `STACK_ORIENT` 决定「最小维度朝上」时的基准姿态，一般方盒可以直接设为 `(0, 0, 0)`，或按需要调整绕 XYZ 的旋转。

### 3.2 在 `scene_cfg.py` 中挂到 SKU_DEFINITIONS

在 `SKU_DEFINITIONS` 中增加一行，例如：

- 若需要缩放：`"xxx": (XXX_PARAMS['USD_PATH'], count, scale)`；
- 不需要缩放：`"xxx": (XXX_PARAMS['USD_PATH'], count, 1.0)`。

这样会自动在 `Spawn_ss_st_stack_SceneCfg` 中注入若干个 `RigidObjectCfg` 实例，并在 spawn 时根据 `scale` 对尺寸进行放大/缩小。

### 3.3 在 `command_cfg.py` 中加入 objects

在 `Spawn_ss_st_stack_CommandsCfg.order_info` 的 `objects` 列表中追加 `"xxx"`，保证订单/CommandTerm 能够识别该 SKU：

- 若只想在 stack_scene 中启用，可只在本场景的 `objects` 中添加；
- 若 sparse_scene 也要共用，请确保对应的 sparse config 也同步更新。

### 3.4 在 `_SKU_PARAMS_MAP` 中挂接新 SKU

在 `Spawn_ss_st_stack_CommandTerm` 顶部的 `_SKU_PARAMS_MAP` 中增加一行：

- 例如：`"xxx": XXX_PARAMS,`
- key 通常是 SKU 名的一部分（如 `"cracker"`、`"sugar"` 等），通过 `if key in obj_name` 进行匹配；
- 确保与 `scene_cfg.SKU_DEFINITIONS` 中使用的 SKU 名能够匹配上。

完成上述步骤后，`_get_raw_params` 就能基于 `XXX_PARAMS` 计算该 SKU 的缩放尺寸与堆叠参数，`_build_stack_params_cache` 会自动将其纳入堆叠布局与 spawn 逻辑中。

---

## 4. 文件一览

| 文件 | 作用 |
|------|------|
| `scene_cfg.py` | 定义 `SKU_DEFINITIONS`（含 count 和 scale），基于 `BaseOrderSceneCfg` 动态注入所有 SKU 实例，配置机器人、工作箱等场景元素。 |
| `command_cfg.py` | 定义 `Spawn_ss_st_stack_CommandsCfg`，挂载 `Spawn_ss_st_stack_CommandTermCfg`，配置 objects / source_boxes / target_boxes / max_active_skus / max_stacks / max_per_stack。 |
| `Spawn_ss_st_stack_CommandTermCfg.py` | 从 `OrderCommandTermCfg` 继承，设定本场景专用的 CommandTerm 类型及堆叠参数默认值。 |
| `Spawn_ss_st_stack_CommandTerm.py` | 负责 stack_scene 的核心逻辑：堆叠参数缓存、物品/箱子分配（含堆叠布局）、实际生成堆叠，以及堆叠质量指标 `stack_weighted_score`。 |
| `reward_cfg.py` 等 | 奖励/事件/观测配置整体结构与 sparse 场景一致，仅替换为 stack 对应的 Scene/Commands/Event/Reward 类。 |