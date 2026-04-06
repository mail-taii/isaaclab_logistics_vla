import argparse
import os
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task_scene_name", type=str, default="Spawn_ss_st_sparse_EnvCfg")
    parser.add_argument(
        "--asset_root_path",
        type=str,
        default=os.environ.get("ASSET_ROOT_PATH", ""),
        help="资产根路径（必须）。例如 /home/junzhe/Benchmark",
    )
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--enable_cameras", action="store_true", default=True)
    parser.add_argument("--out", type=str, default="./topview.png")
    args_cli, _ = parser.parse_known_args()

    if not args_cli.asset_root_path or not os.path.exists(args_cli.asset_root_path):
        raise SystemExit("请设置 --asset_root_path 或环境变量 ASSET_ROOT_PATH 指向资产根路径。")
    os.environ["ASSET_ROOT_PATH"] = args_cli.asset_root_path

    # 必须启用相机
    args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import torch
    import imageio.v2 as imageio

    import isaaclab_tasks  # noqa: F401
    import isaaclab_logistics_vla  # noqa: F401
    from isaaclab_logistics_vla.utils.register import register

    register.auto_scan("isaaclab_logistics_vla.tasks")
    from isaaclab_logistics_vla.evaluation.evaluator.VLAIsaacEnv import VLAIsaacEnv
    from isaaclab_logistics_vla.agent.tools import ToolManager, build_get_topview_image_tool

    env_cfg = register.load_env_configs(args_cli.task_scene_name)()
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
        env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device

    env = VLAIsaacEnv(cfg=env_cfg)
    env.reset()

    # 推进一帧，让相机输出可用
    adim = env.unwrapped.action_manager.total_action_dim
    obs, rew, terminated, truncated, info = env.step(torch.zeros((env.num_envs, adim), device=env.device))

    tools = ToolManager()
    tools.register(build_get_topview_image_tool(env))
    out = tools.execute("get_topview_image")
    if not out.get("success", False):
        raise RuntimeError(out)

    img = out["image_rgb_uint8"]
    out_path = Path(args_cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path, img)
    print(f"[OK] saved: {out_path}  shape={img.shape} dtype={img.dtype}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

