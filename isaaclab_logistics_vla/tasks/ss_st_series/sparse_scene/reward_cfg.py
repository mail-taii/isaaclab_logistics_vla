from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
# 引入上面写的函数
from isaaclab_logistics_vla.tasks import mdp

@configclass
class Spawn_ss_st_sparse_RewardCfg:
    """Reward terms for the task."""
    
    # 1. 订单完成率奖励
    # 直接给予当前完成率作为奖励 (注意：这是一个稠密奖励，如果一直保持100%每帧都会加分)
    completion_rate_reward = RewTerm(
        func=mdp.command_term_metric,
        weight=1.0,  # 正奖励
        params={
            "command_name": "order_info", # 必须对应你在 CommandsCfg 里的名字
            "metric_key": "order_completion_rate"
        },
    )

    # 2. 平均时间惩罚
    # 时间越长，惩罚越大 (Mean Time 是正数，乘以负 weight 变成惩罚)
    mean_time_penalty = RewTerm(
        func=mdp.command_term_metric,
        weight=-0.1, # 负奖励 (系数需要根据数值范围调整)
        params={
            "command_name": "order_info",
            "metric_key": "mean_action_time"
        },
    )