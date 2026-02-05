from __future__ import annotations
import torch
from dataclasses import MISSING
from typing import Callable, TYPE_CHECKING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

from isaaclab_logistics_vla.utils.constant import *
from isaaclab_logistics_vla.utils.util import euler_to_quat_isaac
from isaaclab_logistics_vla.utils.object_position import set_asset_relative_position

# 直接在头部引入 Term 类
# 由于 Term 文件里使用了 TYPE_CHECKING 保护，这里不会发生循环引用报错
from isaaclab_logistics_vla.tasks.test_tasks.order_series.NewOrderCommandTerm import OrderCommand

def default_assign_objects_boxes(term: OrderCommand, env_ids: torch.Tensor):
    """
    钩子函数：负责在重置环境时，分配物品到原料箱，并计算目标箱。
    输入:
        term: OrderCommand 实例，包含任务的全部状态和配置。
        env_ids: 当前需要重置的环境索引 (1D Tensor)。
    """
    device = term.device
    num_envs = len(env_ids)      # 当前有多少个环境需要重置
    num_objects = term.num_objects # 物品总数 (例如 9)
    num_sources = term.num_sources # 原料箱总数 (例如 3)

    # === 2. 建立箱子映射关系 (Source Box <-> Target Box) ===
    # 生成随机噪声，形状为 (num_envs, num_sources)
    # argsort 会返回排序后的索引，这是一种生成随机排列(Permutation)的高效方法
    # 例如：[0.2, 0.9, 0.4] -> argsort -> [0, 2, 1]
    # 含义：原料箱0 -> 目标箱0, 原料箱1 -> 目标箱2, 原料箱2 -> 目标箱1
    rand_noise = torch.rand((num_envs, num_sources), device=device)
    source_to_target_map = torch.argsort(rand_noise, dim=1) 

    # === 3. 构造“槽位池” (The Slot Pool) ===
    # 这是算法的核心：物理约束保证
    MAX_ITEMS_PER_BOX = 4
    
    # 简单的安全检查：如果物品总数比总坑位还多，肯定会出问题
    if num_objects > num_sources * MAX_ITEMS_PER_BOX:
        print(f"[Warning] 物品数量({num_objects}) 超过容量! 将发生溢出。")

    # A. 创建基础池子
    # torch.arange(3) -> [0, 1, 2]
    # repeat_interleave(4) -> [0,0,0,0, 1,1,1,1, 2,2,2,2]
    # 这代表了所有可用的物理空位：箱子0有4个位，箱子1有4个位...
    slot_pool = torch.arange(num_sources, device=device).repeat_interleave(MAX_ITEMS_PER_BOX)
    
    # B. 扩展到 Batch 维度
    # shape: (num_envs, num_sources * MAX_ITEMS_PER_BOX)
    # 每个环境都拥有一套完整的、未打乱的槽位池
    batch_slot_pool = slot_pool.repeat(num_envs, 1)
    
    # === 4. 随机洗牌 (Parallel Shuffling) ===
    # PyTorch 没有直接对 Tensor 某维度进行 shuffle 的函数，所以使用 argsort 技巧
    # 生成与池子形状一样的随机浮点数
    rand_perms = torch.rand_like(batch_slot_pool, dtype=torch.float)
    
    # 获取排序索引：这相当于获得了随机打乱的下标
    perm_indices = torch.argsort(rand_perms, dim=1)
    
    # 使用 gather 根据随机下标重排 batch_slot_pool
    # 结果 shuffled_slots 可能是: [1, 0, 2, 0, 1, 2, 0, 2, ...] (乱序的箱子ID)
    shuffled_slots = torch.gather(batch_slot_pool, 1, perm_indices)
    
    # === 5. 分配物品 (Assignment) ===
    # 从打乱的池子中，截取前 num_objects 个
    # 例如物品有9个，就取前9个 ID。
    # 因为池子里每个 ID 最多出现 4 次，所以截取结果里每个 ID 也最多出现 4 次。
    # shape: (num_envs, num_objects)
    obj_source_idx = shuffled_slots[:, :num_objects]

    # === 6. 推导目标 (Derive Targets) ===
    # 现在我们知道每个物品在哪个原料箱 (obj_source_idx)，
    # 我们也知道原料箱对应哪个目标箱 (source_to_target_map)。
    # 使用 gather 进行“查表”操作。
    # 逻辑：对于每个物品，用它的 source_id 作为索引，去 map 里找对应的 target_id
    obj_target_idx = torch.gather(
        input=source_to_target_map, 
        dim=1, 
        index=obj_source_idx
    )

    # === 7. 写回状态 (State Update) ===
    # 将计算结果存入 term 的全局状态中，供后续的 spawn 钩子或 RL 观测使用
    term.obj_to_source_id[env_ids] = obj_source_idx
    term.obj_to_target_id[env_ids] = obj_target_idx


def default_spawn_objects_boxes(term: OrderCommand, env_ids: torch.Tensor):
    """
    负责执行物理层面的物品生成（放置）。
    输入:
        term: OrderCommand 实例，包含当前的分配状态 (obj_to_source_id)。
        env_ids: 需要重置的环境 ID。
    """
    device = term.device
    
    def get_params(obj_name):
        p = CRACKER_BOX_PARAMS
        x = p.get('X_LENGTH', 0.06)
        y = p.get('Y_LENGTH', 0.20)
        z = p.get('Z_LENGTH', 0.16)
        ori = p.get('STANDARD_ORI', (0,90,0))
        return x, y, z, ori

    # === 1. 计算箱子网格锚点 (Anchors) ===
    # 获取原料箱的物理尺寸 (长和宽)
    box_x = WORK_BOX_PARAMS['X_LENGTH']
    box_y = WORK_BOX_PARAMS['Y_LENGTH']
    
    # 将箱子底面划分为 2x2 的网格。
    # cell_x, cell_y 是每个小格子(Slot)的中心到箱子中心的偏移距离。
    # 箱子总宽 box_x，半宽 box_x/2。再把半宽分两半，就是 box_x/4。
    # 所以 cell_x = box_x / 2.0 实际上是“格子间距”。
    cell_x, cell_y = box_x / 2.0, box_y / 2.0

    # 定义 4 个锚点坐标 (相对于箱子中心)
    # 顺序对应：左下，左上，右下，右上 (或类似的四个象限)
    anchors = torch.tensor([
        [-cell_x/2, -cell_y/2], [-cell_x/2,  cell_y/2],
        [ cell_x/2, -cell_y/2], [ cell_x/2,  cell_y/2]
    ], device=device)

    # === 2. 遍历每个原料箱进行填充 ===
    # 我们以“箱子”为视角，把属于这个箱子的物品一个个放进去
    for box_idx, box_asset in enumerate(term.source_box_assets):
        # 查找当前批次中，哪些物品被分配到了当前箱子 (box_idx)
        # term.obj_to_source_id 形状: (num_envs, num_objects)
        # is_in_this_box 形状: (num_envs, num_objects)，布尔矩阵
        is_in_this_box = (term.obj_to_source_id[env_ids] == box_idx)

        # 如果没有任何环境在这个箱子里放东西，直接跳过，节省计算
        if not is_in_this_box.any(): continue

        # === 3. 计数器初始化 ===
        # count_tensor 用于记录：在每个环境中，当前箱子“已经”放了几个物品了
        # 形状: (num_envs_to_reset, )
        count_tensor = torch.zeros(len(env_ids), dtype=torch.long, device=device)

        # === 4. 遍历所有物品进行放置 ===
        for obj_idx, obj_asset in enumerate(term.object_assets):
            # 获取该物品在哪些环境中属于当前箱子
            # active_mask 形状: (num_envs, )，例如 [True, False, True...]
            active_mask = is_in_this_box[:, obj_idx]

            # 如果没有环境需要放这个物品，跳过
            if not active_mask.any(): continue
            
            # 获取需要操作的那些环境 ID
            active_env_ids = env_ids[active_mask]
            num_active = len(active_env_ids)

            # 获取当前物品的物理尺寸 (最大估算值)
            item_x, item_y, item_z, item_ori = get_params(term.object_names[obj_idx])

            # --- A. 确定坑位 (Slot Selection) ---
            # 根据计数器，决定当前物品放第几个格子 (0, 1, 2, 3)
            # 例如：如果是箱子里的第1个物品，放 Slot 0；第2个放 Slot 1
            # clamp(0, 3) 是为了防止万一分配了超过4个物品，防止索引越界报错
            current_slot_indices = torch.clamp(count_tensor[active_mask], 0, 3)

            # 从锚点列表中取出对应的坐标
            # batch_anchors 形状: (num_active, 2) -> [[x0, y0], [x1, y1]...]
            batch_anchors = anchors[current_slot_indices]

            # --- B. 计算随机抖动 (Jitter) ---
            # 计算物品在格子内还有多少空隙可以移动
            # cell_x 是格子的宽，item_x 是物品的宽
            # 减去 0.01 (1cm) 是为了留出安全边距，防止贴边
            margin_x = max(0.0, (cell_x - item_x)/2.0 - 0.01)
            margin_y = max(0.0, (cell_y - item_y)/2.0 - 0.01)
            
            # 生成随机偏移量
            # torch.rand 生成 [0, 1]，*2-1 变成 [-1, 1]
            # 最终 rand_x 范围是 [-margin_x, +margin_x]
            rand_x = (torch.rand(num_active, device=device) * 2 - 1) * margin_x
            rand_y = (torch.rand(num_active, device=device) * 2 - 1) * margin_y

            # --- C. 计算 Z 轴高度 ---
            # 物品中心高度 = 物品全高/2 + 垫高偏移
            # +0.02 (2cm) 是为了防止物品生成时一半嵌在箱子底板里，导致物理引擎将其弹飞
            z_pos = (item_z / 2.0) + 0.02

            # --- D. 组合最终相对坐标 ---
            # 最终位置 = 锚点中心 + 随机抖动
            # stack 将 x, y, z 组合成 (num_active, 3) 的坐标向量
            relative_pos = torch.stack([
                batch_anchors[:, 0] + rand_x,
                batch_anchors[:, 1] + rand_y,
                torch.full((num_active,), z_pos, device=device)
            ], dim=-1)

            # --- E. 计算旋转姿态 ---
            # 将欧拉角转换为四元数
            # repeat(num_active, 1) 是为了将单个四元数扩展给这一批次的所有环境
            relative_quat = euler_to_quat_isaac(item_ori[0], item_ori[1], item_ori[2]).repeat(num_active, 1)

            # --- F. 执行物理放置 ---
            # 调用底层接口，设置 asset 相对于 reference_asset (箱子) 的位置
            set_asset_relative_position(
                env=term.env,
                env_ids=active_env_ids,
                target_asset=obj_asset,
                reference_asset=box_asset,
                relative_pos=relative_pos,
                relative_quat=relative_quat
            )
            count_tensor[active_mask] += 1

# =============================================================================
# 配置类定义
# =============================================================================
@configclass
class OrderCommandCfg(CommandTermCfg):
    # [关键修复] 直接赋值，不再使用 MISSING，满足 configclass 校验
    class_type: type = OrderCommand 

    resampling_time_range = [1e5, 1e5]
    asset_name: str = MISSING
    body_name: str = MISSING
    
    objects: list[str] = MISSING
    source_boxes: list[str] = MISSING
    target_boxes: list[str] = MISSING    

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/body_pose")
    debug_vis = False

    assign_objects_hook: Callable = default_assign_objects_boxes
    spawn_objects_hook: Callable = default_spawn_objects_boxes

