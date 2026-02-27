from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab_logistics_vla.tasks import mdp

@configclass
class Spawn_ss_mt_sparse_RewardCfg:
    """Reward terms for the SS-MT (Single Source - Multiple Targets) task."""
    
    # --- 1. 任务完成奖励 (核心引导) ---
    # 物品级完成率奖励：每完成一个目标物品的正确放置，都会获得正向激励
    completion_rate_reward = RewTerm(
        func=mdp.command_term_metric,
        weight=2.0,  # 提高权重，作为主引导信号
        params={
            "command_name": "order_info", 
            "metric_key": "order_completion_rate"
        },
    )

    # 全订单成功奖励：当一局中所有目标物品全部正确归位时给予的额外巨额奖励（稀疏信号）
    # 对应 AssignSSMTCommandTerm 中新增的 whole_order_success 指标
    whole_order_bonus = RewTerm(
        func=mdp.command_term_metric,
        weight=5.0,  # 额外大奖，鼓励彻底完成任务
        params={
            "command_name": "order_info",
            "metric_key": "whole_order_success"
        },
    )

    # --- 2. 负向行为惩罚 (约束条件) ---
    # 错放率惩罚：非常关键！惩罚将目标物放错箱子，或将干扰物放进任何目标箱的行为
    wrong_place_penalty = RewTerm(
        func=mdp.command_term_metric,
        weight=-2.0,  # 强惩罚，防止模型盲目乱放
        params={
            "command_name": "order_info",
            "metric_key": "wrong_place_rate"
        },
    )

    # 错抓率惩罚：惩罚移动 target_id = -1 的干扰物的行为
    wrong_pick_penalty = RewTerm(
        func=mdp.command_term_metric,
        weight=-1.5,  # 惩罚不该动的物体被移动
        params={
            "command_name": "order_info",
            "metric_key": "wrong_pick_rate"
        },
    )

    # 失败率惩罚：物品掉落地板或进入错误的非目标区域
    failure_rate_penalty = RewTerm(
        func=mdp.command_term_metric,
        weight=-1.0, 
        params={
            "command_name": "order_info",
            "metric_key": "failure_rate"
        },
    )

    # --- 3. 效率与正则化 ---
    # 动作平滑惩罚（可选）：惩罚过大的关节转矩或速度变化，使动作更自然
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # 平均时间惩罚：鼓励更短路径完成任务
    mean_time_penalty = RewTerm(
        func=mdp.command_term_metric,
        weight=-0.05, # 较小权重，避免模型为了速度而牺牲成功率
        params={
            "command_name": "order_info",
            "metric_key": "mean_action_time"
        },
    )