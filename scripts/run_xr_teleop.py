"""
XR(handtracking) 遥操作入口：在 isaaclab_logistics_vla 的任务场景上启用 CloudXR/OpenXR。

工作流（先确保已按 Isaac Lab 文档启动 CloudXR Runtime，并在 Isaac Sim UI 中 Start AR）：

```bash
conda activate env_isaaclab
./isaaclab.sh -p scripts/run_xr_teleop.py --task_scene_name Spawn_ds_st_sparse_XRTeleop_EnvCfg --xr
```
"""

import argparse
import os
from collections.abc import Callable

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Apple Vision Pro (CloudXR/OpenXR) 遥操作驱动 Logistics VLA 场景")
parser.add_argument(
    "--task_scene_name",
    type=str,
    default="Spawn_ds_st_sparse_XRTeleop_EnvCfg",
    help="XR teleop 用的任务场景名（建议用 *_XRTeleop_EnvCfg 变体）",
)
parser.add_argument("--control_hz", type=float, default=45.0, help="控制循环频率（XR 下通常 45Hz 渲染）")
parser.add_argument("--num_envs", type=int, default=1, help="环境数量（XR 遥操作建议 1）")
parser.add_argument(
    "--asset_root_path",
    type=str,
    default="/home/junzhe/Benchmark",
    help="资产根路径",
)
parser.add_argument(
    "--teleop_device",
    type=str,
    default="handtracking",
    choices=["handtracking"],
    help="仅支持 handtracking（OpenXR/CloudXR）",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# handtracking 必须启用 XR experience
args_cli.xr = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    if not os.path.exists(args_cli.asset_root_path):
        raise FileNotFoundError(f"资产路径不存在: {args_cli.asset_root_path}")
    os.environ["ASSET_ROOT_PATH"] = args_cli.asset_root_path

    import time
    import torch
    import omni.log

    import isaaclab_tasks  # noqa: F401
    import isaaclab_logistics_vla  # noqa: F401
    from isaaclab_logistics_vla.evaluation.evaluator.VLAIsaacEnv import VLAIsaacEnv
    from isaaclab_logistics_vla.utils.register import register

    # 扫描注册表（包含 *_XRTeleop_EnvCfg）
    register.auto_scan("isaaclab_logistics_vla.tasks")

    env_cfg = register.load_env_configs(args_cli.task_scene_name)()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # 创建环境
    env = VLAIsaacEnv(cfg=env_cfg).unwrapped

    # XR 手部遥操作设备（从 env_cfg.teleop_devices 中构造）
    from isaaclab.devices.teleop_device_factory import create_teleop_device

    should_reset = False
    teleoperation_active = False  # XR 下默认 inactive，等待 AVP UI 的 START/Play

    def reset_env() -> None:
        nonlocal should_reset
        should_reset = True
        print("[XR Teleop] RESET")

    def start_teleop() -> None:
        nonlocal teleoperation_active
        teleoperation_active = True
        print("[XR Teleop] START")

    def stop_teleop() -> None:
        nonlocal teleoperation_active
        teleoperation_active = False
        print("[XR Teleop] STOP")

    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "START": start_teleop,
        "STOP": stop_teleop,
        "RESET": reset_env,
        "R": reset_env,
    }

    if not hasattr(env_cfg, "teleop_devices") or args_cli.teleop_device not in env_cfg.teleop_devices.devices:
        omni.log.error(
            f"[XR Teleop] env_cfg.teleop_devices 缺少 '{args_cli.teleop_device}' 配置。"
            "请使用 *_XRTeleop_EnvCfg，或在你的 EnvCfg 中添加 teleop_devices.handtracking。"
        )
        env.close()
        return

    teleop_interface = create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks)
    print(f"[XR Teleop] Using teleop device: {teleop_interface}")

    env.reset()
    teleop_interface.reset()

    # 主循环：注意 XR 下 OpenXR 的 rate limit 主要由 runtime 决定；这里只做一个 best-effort pacing
    dt = 1.0 / float(args_cli.control_hz)

    while simulation_app.is_running():
        with torch.inference_mode():
            action = teleop_interface.advance()

            if teleoperation_active:
                actions = action.repeat(env.num_envs, 1)
                env.step(actions)
            else:
                env.sim.render()

            if should_reset:
                env.reset()
                should_reset = False

        time.sleep(dt)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

