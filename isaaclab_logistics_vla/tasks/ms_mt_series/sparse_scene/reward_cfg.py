from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab_logistics_vla.tasks import mdp

@configclass
class Spawn_ms_mt_sparse_RewardCfg:
    #---1. 核心进度奖励(Progress Rewards)---
    # 物品级订单完成率奖励 (Dense)
    # 引导模型每正确放置一个物品都能获得即时反馈
    completion_rate_reward = RewTerm(
        func=mdp.command_term_metric,
        weight=2.0,
        params={
            "command_name": "order_info",
            "metric_key": "order_completion_rate"
        },
    )

    #全订单成功额外奖励(Sparse Bonus)
    #当环境中所有目标物品均处于状态3时触发，鼓励模型彻底完成多目标任务
    whole_order_bonus = RewTerm(
        func=mdp.command_term_metric,
        weight=5.0,     #给予较大的稀疏奖励，用于强化最终成功行为
        params={
            "command_name": "order_info",
            "metric_key": "whole_order_success"
        },
    )

    #---2. 负向行为惩罚(Penalty Terms)---
    #错放率惩罚(Critical Penalty)
    #在MS-MT中极其重要：惩罚目标物进错箱子、或干扰物进入任何目标箱
    wrong_place_penalty = RewTerm(
        func=mdp.command_term_metric,
        weight=-2.5,    # 强惩罚，防止模型为了刷完成率而盲目投放
        params={
            "command_name": "order_info",
            "metric_key": "wrong_place_rate"
        },
    )

    #错抓率惩罚
    #惩罚移动target_id = -1的干扰物的行为，引导模型在多源箱中精准定位目标
    wrong_pick_penalty = RewTerm(
        func=mdp.command_term_metric,
        weight=-1.5,
        params={
            "command_name": "order_info",
            "metric_key": "wrong_pick_rate"
        },
    )

    #失败率惩罚
    #惩罚物品掉落到地面或不可恢复状态（状态4）
    failure_rate_penalty = RewTerm(
        func=mdp.command_term_metric,
        weight=-1.2,
        params={
            "command_name": "order_info",
            "metric_key": "failure_rate"
        },
    )

    #---3. 效率与平滑正则(Efficiency & Regularization)---
    #平均时间惩罚
    #鼓励模型以更高效的路径和决策完成分拣
    mean_time_penalty = RewTerm(
        func=mdp.command_term_metric,
        weight=-0.05,       #保持较小权重，避免模型为了省时而选择不行动
        params={
            "command_name": "order_info",
            "metric_key": "mean_action_time"
        },
    )