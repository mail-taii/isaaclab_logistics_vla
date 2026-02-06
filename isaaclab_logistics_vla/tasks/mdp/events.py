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


from PIL import Image, ImageDraw, ImageFont
import os

def add_text_to_image(image_path, output_path, text, box_coords, font_path):

    """
    在指定图片的指定区域内添加文字，并保存为新图片。
    image_path: 原始图片路径
    output_path: 保存新图片的文件夹路径
    text: 要添加的文字内容
    box_coords: 文字区域的坐标，格式为 (左, 上, 右, 下)
    font_path: 字体文件路径 (.ttf)
    """

    # 地址
    from datetime import datetime  # 引入时间模块
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{timestamp_str}.png"
    output_path = os.path.join(output_path, file_name)

    # 1. 打开图片
    if not os.path.exists(image_path):
        print(f"Error: 找不到图片 {image_path}")
        return
    
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # 2. 加载字体 (增加错误处理)
    try:
        # 设置字体大小
        font_size = 30 
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        print(f"Error: 无法加载字体文件: {font_path}")
        print("请确保 .ttf 文件路径正确，或者将字体文件与脚本放在同一目录")
        return

    # 3. 计算居中位置
    # box_coords = (左x, 上y, 右x, 下y)
    box_x1, box_y1, box_x2, box_y2 = box_coords
    box_w = box_x2 - box_x1
    box_h = box_y2 - box_y1

    # 获取文字宽高
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    # 计算绘制的起始坐标 (绝对居中)
    draw_x = box_x1 + (box_w - text_w) / 2
    draw_y = box_y1 + (box_h - text_h) / 2 - (text_bbox[1] / 2)

    # 4. 绘制文字 (颜色设为深灰，显得更自然)
    draw.text((draw_x, draw_y), text, font=font, fill=(0, 0, 0))

    # 5. 保存
    img.save(output_path)
    print(f"成功！图片已保存至: {output_path}")
    return output_path


def set_package_visual_texture(
    env: ManagerBasedRLEnv, 
    env_ids: torch.Tensor, 
    target_asset_names: list[str], 
    texture_paths: list[str],
    words: str = None,
):
    """
    将环境中的资产随机化为指定纹理。
    target_asset_names：列出的资产，在同一个环境中纹理一致
    texture_paths：可选纹理范围
    """

    # --- 配置部分 ---
    MY_FONT_PATH = '/home/daniel/fff/model_files/benchmark/fonts/SIMHEI.TTF' 

    # 这里的坐标需要测量需要写字的位置
    # 格式: (左, 上, 右, 下)
    MY_PACKAGE_COORDS = (670, 170, 870, 360) 

    # 1. 如果有文字需求，先生成图片
    # 注意：这里会更新 texture_paths 列表为具体的文件路径
    if words is not None:
        # 生成图片并获取具体路径
        # 假设 texture_paths[0] 是文件夹路径
        new_paths = add_text_to_image(
            image_path='/home/daniel/fff/model_files/benchmark/props/Collected_empty_plastic_package/textures/new.png',
            output_path="/home/daniel/fff/model_files/benchmark/props/Collected_empty_plastic_package/textures",
            text=words, 
            box_coords=MY_PACKAGE_COORDS,
            font_path=MY_FONT_PATH
        )
        #generate_textures_with_words("/home/daniel/fff/model_files/benchmark/props/Collected_plastic_package/textures", words)
        # 将生成的路径作为纹理源
        valid_texture_paths = [new_paths] # 变成列表
        print(f"[Info] Generated texture: {new_paths}")
    else:
        valid_texture_paths = texture_paths

    # 1. 默认处理所有环境
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    
    num_envs = len(env_ids)
    num_textures = len(valid_texture_paths)

    # 2. 随机分配纹理索引
    # assigned_tex_indices[i] = 0 表示第 i 个环境使用第 0 张图
    assigned_tex_indices = torch.randint(0, num_textures, (num_envs,), device=env.device)

    event_name = f"randomize_tex_{uuid.uuid4().hex}"

    # 3. 构建 Replicator Graph (注册到事件系统)
    # 注意：这里我们不用 rep.new_layer()，而是直接挂载到事件触发器上
    # 这样 Graph 只会在收到信号时执行，不会干扰主循环
    with rep.trigger.on_custom_event(event_name):
        
        for tex_idx, tex_path in enumerate(valid_texture_paths):
            
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
                        project_uvw=False  # 强制关闭 UV 投影，使用默认 UV
                    )
            except Exception as e:
                print(f"[Error] Graph build failed: {e}")

    # 4. 【关键修复】发送信号触发执行
    # 这就像按下一个开关，Replicator 会在当前帧或下一帧的安全时刻执行上述逻辑
    # 不会阻塞主线程，完美解决卡死问题
    print(f"[Info] Triggering texture randomization event: {event_name}")
    rep.utils.send_og_event(event_name)
    