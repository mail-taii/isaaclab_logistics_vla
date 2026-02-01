from isaaclab.utils import configclass

from isaaclab_logistics_vla.tasks import mdp
from isaaclab_logistics_vla.utils.register import register

# 引入具体的指令配置类定义
# 这个类定义了该任务需要哪些参数结构（如需要 objects 列表，需要 source_boxes 列表等）
# 它对应 OrderCommandTerm.py 里的 class OrderCommand(CommandTerm)
from isaaclab_logistics_vla.tasks.test_tasks.order_series.NewOrderCommandCfg import OrderCommandCfg

####################################################################################################################################
# 这是任务逻辑的配置文件。
#     CommandsCfg 类：定义了环境中所有的“指令项”。
#     order_info：实例化了你之前提到的 OrderCommandCfg。
#         连接点：注意看参数 objects=['o_cracker_box_1', ...]。这里的字符串列表，必须与 scene_cfg.py 中定义的变量名完全一致。
#         作用：它将物理场景中的物体（Scene Assets）绑定到了逻辑控制器（Command Term）上。
####################################################################################################################################
ASSET_NAME = "robot"
BODY_NAME = "base_link_underpan"

# 定义没用的占位符，保持原文件结构一致性
a1 = 'world_anchor'
b1 = 'WorldAnchor'
LEFT_START_POS =  [-1.35,-0.125,1.0] 
LEFT_END_POS = [-1.17,-0.125,1.09] 
RIGHT_START_POS = [-1.35,0.46,1.0] 
RIGHT_END_POS = [-1.17, 0.46, 1.09]

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    order_info = OrderCommandCfg(
        asset_name = ASSET_NAME,
        body_name=BODY_NAME,
        
        # 物品列表必须包含 9 个名字，与 scene_cfg.py 里的变量名一一对应
        objects=[f"o_item_{i}" for i in range(9)],
        source_boxes = ['s_box_1','s_box_2','s_box_3'],
        target_boxes = ['t_box_1','t_box_2','t_box_3'],
    )