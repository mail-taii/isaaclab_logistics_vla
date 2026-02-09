# Dense Scene 实现说明

## 1. 与 sparse_scene 的复用关系

| 层级 | 复用方式 |
| :--- | :--- |
| **继承链** | 两者都继承 `AssignSSSTCommandTerm` → `OrderCommandTerm`。订单分配、指标计算（`order_completion_rate`, `wrong_pick_rate` 等）、以及 reset 流程均由基类统一处理。 |
| **场景结构** | 同样继承 `BaseOrderSceneCfg`，通过 `SKU_DEFINITIONS` + 循环 `setattr` 动态注入物体。原料箱/目标箱命名保持一致（`s_box`, `t_box`）。 |
| **Env 配置** | 与 sparse 同构，仅替换为 dense 版本的 Scene/Commands/Reward/Event 配置类。 |

**核心重写点（仅 CommandTerm 内）：**

* **`_assign_objects_boxes(env_ids)`**：
    * **Sparse**：仅分配逻辑（决定谁在哪个箱子），具体位置在生成时随机决定。
    * **Dense**：**预计算（Pre-calculation）**。在此阶段直接运行“装箱算法”（Box 模式）或“网格填充”（Tray 模式），计算出确切的相对位置和旋转，并存储在 `saved_relative_pos` 和 `saved_relative_quat` 中。
* **`_spawn_items_in_source_boxes(env_ids)`**：
    * **Sparse**：在 6 个固定槽位中随机抖动生成。
    * **Dense**：**渲染器（Renderer）**。它严格读取 `assign` 阶段存下的坐标进行生成，并处理坐标系转换（相对于 Box 还是相对于 Tray）。

---

## 2. dense_scene 多出来的部分

### 2.1 配置层 (`Spawn_ss_st_dense_CommandTermCfg`)

* **`max_instances_per_sku`**：默认值提升为 **6**（Sparse 为 2），以制造密集堆叠效果。
* **`tray_or_not`**：新增布尔列表（如 `[0, 0, 0]`），控制场景模式：
    * `True` (1)：**托盘模式 (Tray Mode)**，物体整齐摆放在小托盘内。
    * `False` (0)：**箱子模式 (Box Mode)**，物体在原料箱内进行密集平面拼图（Bin Packing）。

### 2.2 逻辑层 (`Spawn_ss_st_dense_CommandTerm`)

#### A. 双模式生成逻辑
1.  **Tray Mode**：
    * 基于托盘尺寸 (`TRAY_X/Y`) 和物体尺寸计算最大行列数 (`cols`, `rows`)。
    * 按网格索引填充，坐标系相对于 `tray_{id}`。
2.  **Box Mode (无托盘)**：
    * 使用 **2D 占用栅格地图 (`env_map`)**。将箱底划分为 `0.5cm` 分辨率的网格。
    * 尝试在空位放入物体，若放不下则标记为截断 (`source_id = -1`) 不生成。

#### B. 硬编码缩放与朝向锁定
* **内部缩放 (`get_real_dims_local`)**：
    * 代码内部硬编码了缩放逻辑。
    * `SCALED_OBJECTS1` (如 `CN_big`, `SF_small`) 缩放系数 **0.3**。
    * `SCALED_OBJECTS2` (如 `cracker_box`) 缩放系数 **0.8**。
* **朝向锁定 (`force_ori`)**：
    * constant.py中新增四个sku物品+托盘，同时每个物品新增DENSE_ORIENT用于控制紧密堆叠时物品朝向
        目前cracker_box, sugar_box,tomato_soup_can,sf_big支持侧立和平放，CN_big支持长边对着箱子的长宽两个方向，plastic_package, sf_small仅支持平放。新增物品均于scene_cfg中加入，同时设定了scale（注意与Spawn_ss_st_dense_CommandTerm中保持一致）
    * 为了保证计算的位置和最终生成一致，Dense 在计算前会先为该 SKU 随机选定一个朝向（或沿用配置），存入 `sku_orientation_map`，后续计算和生成都**强制**使用该朝向。

### 2.3 新增工具函数用于放托盘
**def update_tray_positions(env, env_ids, tray_or_not):
**在event_cfg中新增 setup_trays 用于在reset时根据trasy_or_not的值判断是否需要将托盘放置到箱子中

### 2.4 包裹随机化
** 在event_cfg中新增plastic_texture_randomizer用于改变空包裹纹理，可以指定上面的字等

---

## 3. 如何增加新物品

### 3.1 基础配置 (与 Sparse 相同)
1.  **`utils/constant.py`**：增加 `XXX_PARAMS`（需包含 `X/Y/Z_LENGTH`，以及可选的 `DENSE_ORIENT` 候选旋转列表）。
2.  **`scene_cfg.py`**：在 `SKU_DEFINITIONS` 中增加：`"xxx": (XXX_PARAMS['USD_PATH'], count)`。

### 3.2 关键步骤 (Dense 特有)
**注意：必须修改 `Spawn_ss_st_dense_CommandTerm.py`，否则物体可能因尺寸计算错误而重叠或悬空。**

1.  **修改 `get_real_dims_local` 函数**：
    * 找到内部函数 `get_real_dims_local`。
    * 在 `if/elif` 分支中增加该物体的名称匹配：
        ```python
        elif "xxx" in obj_name: p = XXX_PARAMS
        ```
    * **原因**：Dense 的装箱算法依赖此函数返回的精确尺寸（含缩放）来画占用地图。

2.  **配置缩放组 (如有必要)**：
    * 如果新物体属于需要缩放的类别（例如它是巨型资产需要缩小），将其名字加入到文件开头的列表：
        * `SCALED_OBJECTS1` (缩放 0.3)
        * `SCALED_OBJECTS2` (缩放 0.8)
    * 如果不加，默认按 `constant.py` 的原始尺寸处理。

---

## 4. 文件一览

| 文件 | 作用 |
| :--- | :--- |
| `scene_cfg.py` | SKU 定义、动态注入物体；继承 `BaseOrderSceneCfg`。 |
| `Spawn_ss_st_dense_CommandTerm.py` | **核心逻辑重写**。包含 `get_real_dims_local`（尺寸与缩放逻辑）、Occupancy Map 装箱算法、Tray 网格逻辑、以及预计算坐标存储。 |
| `Spawn_ss_st_dense_CommandTermCfg.py` | Dense 专用配置：`max_instances_per_sku` (6), `tray_or_not`。 |
| `constant.py` | 增加新物品的 `PARAMS`（尺寸、USD路径）。新增DENSE_ORIENT字段 |
| `env_cfg.py` `bservation_cfg.py` `eward_cfg.py`| 挂载 Dense 版的 CommandTerm 和 Event。 |
| `event_cfg.py` | 新增setup_trays、plastic_texture_randomizer |
| `isaaclab_logistics_vla/tasks/mdp/events.py` | 新增update_tray_positions用于放置托盘