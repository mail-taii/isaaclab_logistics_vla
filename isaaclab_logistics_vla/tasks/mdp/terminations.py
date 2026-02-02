import torch
from isaaclab.envs import ManagerBasedRLEnv

def check_order_completion(env: ManagerBasedRLEnv, command_name: str, threshold: float = 0.999) -> torch.Tensor:
    """
    当 metrics['order_completion_rate'] >= threshold 时，重置环境。
    """
    cmd_manager = env.command_manager
    term = cmd_manager.get_term(command_name)
    
    if term is None:
        # 防御性编程：如果找不到名字，就不触发重置
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    completion_rate = term.metrics.get("order_completion_rate")
    
    if completion_rate is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    return (completion_rate >= threshold)