from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab_logistics_vla.tasks import mdp


@configclass
class Spawn_ms_mt_stack_RewardCfg:
    """Reward terms for the task."""

    completion_rate_reward = RewTerm(
        func=mdp.command_term_metric,
        weight=1.0,
        params={
            "command_name": "order_info",
            "metric_key": "order_completion_rate"
        },
    )

    mean_time_penalty = RewTerm(
        func=mdp.command_term_metric,
        weight=-0.1,
        params={
            "command_name": "order_info",
            "metric_key": "mean_action_time"
        },
    )

    failure_rate_penalty = RewTerm(
        func=mdp.command_term_metric,
        weight=-1.0,
        params={
            "command_name": "order_info",
            "metric_key": "failure_rate"
        },
    )

    wrong_pick_penalty = RewTerm(
        func=mdp.command_term_metric,
        weight=-1.0,
        params={
            "command_name": "order_info",
            "metric_key": "wrong_pick_rate"
        },
    )

    wrong_place_penalty = RewTerm(
        func=mdp.command_term_metric,
        weight=-1.0,
        params={
            "command_name": "order_info",
            "metric_key": "wrong_place_rate"
        },
    )

    stack_quality_reward = RewTerm(
        func=mdp.command_term_metric,
        weight=0.5,
        params={
            "command_name": "order_info",
            "metric_key": "stack_weighted_score"
        },
    )
