from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab_logistics_vla.tasks import mdp

@configclass
class Spawn_ss_st_sparse_RewardCfg:
    """Reward terms for the task."""
    
    # 1. 物品级订单完成率奖励 (Dense Reward)
    # 每放对一个需要的物品，都会获得正向激励
    completion_rate_reward = RewTerm(
        func=mdp.command_term_metric,
        weight=1.0,  
        params={
            "command_name": "order_info", 
            "metric_key": "order_completion_rate"
        },
    )

    # 2. 全订单成功大奖 (Sparse Bonus) - 【新增】
    # 只有当环境里的有效数量完美达到订单需求时，给予高额奖励，鼓励彻底完成长序列任务
    whole_order_bonus = RewTerm(
        func=mdp.command_term_metric,
        weight=3.0,  # 额外大奖
        params={
            "command_name": "order_info",
            "metric_key": "whole_order_success"
        },
    )

    # 3. 平均时间惩罚
    mean_time_penalty = RewTerm(
        func=mdp.command_term_metric,
        weight=-0.05, # 因为要搬运多个物品，时间惩罚系数建议适当调低
        params={
            "command_name": "order_info",
            "metric_key": "mean_action_time"
        },
    )

    # 4. 失败率惩罚 (物品掉落等)
    failure_rate_penalty = RewTerm(
        func=mdp.command_term_metric,
        weight=-1.0,  
        params={
            "command_name": "order_info",
            "metric_key": "failure_rate"
        },
    )

    # 5. 错抓率惩罚 (抓了干扰物)
    wrong_pick_penalty = RewTerm(
        func=mdp.command_term_metric,
        weight=-1.0,  
        params={
            "command_name": "order_info",
            "metric_key": "wrong_pick_rate"
        },
    )

    # 6. 错放率惩罚 (放了多余的物品或干扰物)
    # 由于现在的判定非常精准（超量也会被算作错放），这里维持强惩罚
    wrong_place_penalty = RewTerm(
        func=mdp.command_term_metric,
        weight=-1.5,  # 稍微提高惩罚，防止模型为了刷完成率往箱子里无脑倒物品
        params={
            "command_name": "order_info",
            "metric_key": "wrong_place_rate"
        },
    )