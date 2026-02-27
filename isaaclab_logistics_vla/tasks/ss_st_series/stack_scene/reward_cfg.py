from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab_logistics_vla.tasks import mdp

@configclass
class Spawn_ss_st_stack_RewardCfg:
    """Reward terms for the task."""
    
    # 1. 订单完成率奖励
    completion_rate_reward = RewTerm(
        func=mdp.command_term_metric,
        weight=1.0,
        params={
            "command_name": "order_info",
            "metric_key": "order_completion_rate"
        },
    )

    # 2. 平均时间惩罚
    mean_time_penalty = RewTerm(
        func=mdp.command_term_metric,
        weight=-0.1,
        params={
            "command_name": "order_info",
            "metric_key": "mean_action_time"
        },
    )

    # 3. 失败率惩罚
    failure_rate_penalty = RewTerm(
        func=mdp.command_term_metric,
        weight=-1.0,
        params={
            "command_name": "order_info",
            "metric_key": "failure_rate"
        },
    )

    # 4. 错抓率惩罚
    wrong_pick_penalty = RewTerm(
        func=mdp.command_term_metric,
        weight=-1.0,
        params={
            "command_name": "order_info",
            "metric_key": "wrong_pick_rate"
        },
    )

    # 5. 错放率惩罚
    wrong_place_penalty = RewTerm(
        func=mdp.command_term_metric,
        weight=-1.0,
        params={
            "command_name": "order_info",
            "metric_key": "wrong_place_rate"
        },
    )
