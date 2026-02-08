# stack_scene 实现说明

基于现有代码对比 `stack_scene` 与 `sparse_scene`，说明复用关系、多出部分、设计思路及如何增加新物品。

---

## 1. 与 sparse_scene 的复用关系

| 层级 | 复用方式 |
|------|----------|
| **继承链** | 两者都继承 `AssignSSSTCommandTerm` → `OrderCommandTerm`。订单分配、指标（order_completion_rate、wrong_pick_rate 等）、reset 流程由基类统一处理。 |
| **场景结构** | 同样继承 `BaseOrderSceneCfg`，通过 `SKU_DEFINITIONS` + 循环 `setattr` 动态注入物体，原料箱/目标箱命名一致（s_box_1/2/3, t_box_1/2/3）。 |
| **env_cfg / observation / reward / event** | 与 sparse 同构，仅替换为 stack 的 Scene/Commands/Reward/Event 配置类。 |

**重写点**（仅 CommandTerm 内）：

- `_assign_objects_boxes(env_ids)`：分配「谁从哪个原料箱、谁是目标/干扰」；
- `_spawn_items_in_source_boxes(env_ids)`：在原料箱内如何摆放（sparse 用 6 槽位散放，stack 用两摞 + 可选散放）。

其余如 `_reset_idx`、`_update_assign_metrics`、command/obs 等均用基类逻辑。

---

## 2. stack_scene 多出来的部分

### 2.1 配置层

- **scene_cfg**
  - `SKU_DEFINITIONS` 为三元组：`(usd_path, count, scale)`，支持按 SKU 设缩放（如 0.5）；sparse 为二元组且 scale 固定 1.0。
  - 当前仅方盒类：cracker_box, sugar_box, plastic_package, sf_big, sf_small（无 tomato_soup_can）。

- **command_cfg / CommandTermCfg**
  - 新增参数：
    - `max_stack_height`：单摞最多放几个物品（默认 4），超出部分散放在箱内角落。
    - `distractor_mode`：`"stack"`（干扰物也摞一摞）或 `"scatter"`（干扰物散放）。
  - `objects` 列表与 scene 的 SKU 一致（仅方盒）。

### 2.2 逻辑层（Spawn_ss_st_stack_CommandTerm）

- **_assign_objects_boxes**
  - 与 sparse 的差异：原料箱不再固定为 0，而是每个 env 从 `num_sources` 个原料箱中**随机选一个**：`selected_source_box = torch.randint(0, self.num_sources, (1,)).item()`，目标物/干扰物都从该箱生成。

- **_spawn_items_in_source_boxes**（完全重写）
  1. `_move_all_objects_far`：先把本 env 所有物品移到远处。
  2. `_get_stack_anchors()`：箱内 2 个锚点（沿 Y 方向前后两区），分别给「目标摞」和「干扰摞」。
  3. `_split_objects(env_id)`：按 `obj_to_target_id` / `is_target_mask` 分成目标物列表和干扰物列表。
  4. 目标物：`_spawn_stack(..., anchor=目标锚点)`。
  5. 干扰物：若 `distractor_mode == "stack"` 则 `_spawn_stack(..., anchor=干扰锚点)`，否则 `_spawn_scattered`。

- **堆叠专用方法**
  - `_build_stack_params_cache`：按 SKU 预计算 `base_area`、`stack_height`、`stack_orient`（最小维度朝上的欧拉角）。
  - `_get_raw_params(obj_name)`：根据 obj_name 返回 constant 中的 PARAMS（含缩放后的尺寸）；**新增 SKU 时需在此加分支**。
  - `_compute_stack_params` / `_compute_orient_for_min_height`：由长宽高算底面、高度、朝向。
  - `_sort_by_base_area`：按底面积从大到小排序，保证大件在下。
  - `_spawn_stack`：先按底面积排序，前 `max_stack_height` 个摞在 anchor 上，其余用 `_get_scatter_anchors` 散放在箱内四角。
  - `_place_single_object`：在给定 anchor 和当前 z_offset 上放一个方盒，更新 z_offset。
  - `_spawn_scattered`：干扰物散放时使用，占箱内四角位置。
  - `_get_stack_anchors` / `_get_scatter_anchors`：箱内相对坐标，避免摞与摞、摞与散放重叠。

设计要点：**目标物和干扰物分成两摞，每摞内部按底面积从大到小叠放，单摞高度受 `max_stack_height` 限制，多余物品散放**；仅使用方盒类 SKU，保证「最小维度朝上」的堆叠朝向一致。

---

## 3. 如何增加新物品

### 3.1 sparse_scene

1. 在 `utils/constant.py` 增加 `XXX_PARAMS`（USD_PATH、X/Y/Z_LENGTH、可选 SPARSE_ORIENT）。
2. 在 `scene_cfg.py` 的 `SKU_DEFINITIONS` 中增加一项：`"xxx": (XXX_PARAMS['USD_PATH'], count)`。
3. 在 `command_cfg.py` 的 `objects` 列表中加入 `"xxx"`。
4. 若 sparse 的 `_spawn_items_in_source_boxes` 里有按名称取尺寸的逻辑（如 get_params_and_dims），需在那里加分支。

### 3.2 stack_scene（在 sparse 基础上）

1. **constant.py**：同上，增加 `XXX_PARAMS`（方盒需 X/Y/Z_LENGTH）。
2. **scene_cfg.py**：  
   - 若需缩放：`"xxx": (XXX_PARAMS['USD_PATH'], count, scale)`；  
   - 不缩放：`"xxx": (XXX_PARAMS['USD_PATH'], count, 1.0)`。
3. **command_cfg.py**：在 `order_info` 的 `objects` 列表中加入 `"xxx"`。
4. **Spawn_ss_st_stack_CommandTerm._get_raw_params(self, obj_name)**：  
   增加 `elif "xxx" in obj_name:`，返回该 SKU 的参数字典（若 scene 里用了 scale，这里要乘 scale 后的 X/Y/Z_LENGTH，与 plastic_package / sf_big / sf_small 写法一致）。

说明：stack 的摆放依赖 `_stack_params_cache`，而 cache 来自 `_get_raw_params`，因此**新 SKU 必须在 _get_raw_params 中有分支**，否则会落到默认的 CRACKER_BOX_PARAMS。当前实现仅支持方盒（用长宽高算底面与高度）；若将来支持圆柱等，需在 `_compute_stack_params` / `_compute_orient_for_min_height` 中扩展。

---

## 4. 文件一览

| 文件 | 作用 |
|------|------|
| `scene_cfg.py` | SKU 定义（含 scale）、动态注入物体；继承 BaseOrderSceneCfg。 |
| `command_cfg.py` | 挂载 Spawn_ss_st_stack_CommandTermCfg，配置 objects / source_boxes / target_boxes / max_stack_height / distractor_mode。 |
| `Spawn_ss_st_stack_CommandTermCfg.py` | 增加 max_stack_height、distractor_mode 默认值。 |
| `Spawn_ss_st_stack_CommandTerm.py` | 重写 _assign_objects_boxes（随机原料箱）、_spawn_items_in_source_boxes（两摞+散放），以及堆叠相关辅助方法；_get_raw_params 需随 SKU 扩展。 |
| `env_cfg.py` / `event_cfg.py` / `observation_cfg.py` / `reward_cfg.py` | 与 sparse 结构相同，仅替换为 stack 的 Scene/Commands/Event/Reward。 |

以上为 stack_scene 相对 sparse_scene 的复用方式、多出部分、设计与扩物步骤的简要说明。
