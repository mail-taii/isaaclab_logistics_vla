import torch
import omni.replicator.core as rep
from isaaclab.envs import ManagerBasedRLEnv
import uuid

def randomize_unified_visual_texture(
    env: ManagerBasedRLEnv, 
    env_ids: torch.Tensor, 
    target_asset_names: list[str], 
    texture_paths: list[str]
):
    """
    将环境中的资产随机化为指定纹理。
    target_asset_names：列出的资产，在同一个环境中纹理一致
    texture_paths：可选纹理范围
    """
    # 1. 默认处理所有环境
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    
    num_envs = len(env_ids)
    num_textures = len(texture_paths)

    # 2. 随机分配纹理索引
    # assigned_tex_indices[i] = 0 表示第 i 个环境使用第 0 张图
    assigned_tex_indices = torch.randint(0, num_textures, (num_envs,), device=env.device)

    event_name = f"randomize_tex_{uuid.uuid4().hex}"

    # 3. 构建 Replicator Graph (注册到事件系统)
    # 注意：这里我们不用 rep.new_layer()，而是直接挂载到事件触发器上
    # 这样 Graph 只会在收到信号时执行，不会干扰主循环
    with rep.trigger.on_custom_event(event_name):
        
        for tex_idx, tex_path in enumerate(texture_paths):
            
            # A. 筛选环境
            subset_indices = (assigned_tex_indices == tex_idx).nonzero().squeeze(-1)
            if len(subset_indices) == 0:
                continue

            subset_env_ids_list = env_ids[subset_indices].cpu().tolist()

            # B. 构建正则路径
            env_regex = "|".join(map(str, subset_env_ids_list))
            box_regex = "|".join(target_asset_names)
            path_pattern = f"/World/envs/env_({env_regex})/({box_regex})/.*"

            # C. 构建节点
            try:
                target_prims = rep.get.prims(path_pattern=path_pattern)
                with target_prims:
                    rep.randomizer.texture(
                        textures=[tex_path],
                        project_uvw=True
                    )
            except Exception as e:
                print(f"[Error] Graph build failed: {e}")

    # 4. 【关键修复】发送信号触发执行
    # 这就像按下一个开关，Replicator 会在当前帧或下一帧的安全时刻执行上述逻辑
    # 不会阻塞主线程，完美解决卡死问题
    print(f"[Info] Triggering texture randomization event: {event_name}")
    rep.utils.send_og_event(event_name)