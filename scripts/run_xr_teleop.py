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

    # 调试：若 AVP 点 Play 后仍无 START 回调，可设 TELEOP_FORCE_ACTIVE=1 强制开启遥操作，测试手部数据是否有输出
    teleoperation_active = os.environ.get("TELEOP_FORCE_ACTIVE", "").strip() in ("1", "true", "yes")
    if teleoperation_active:
        print("[XR Teleop] TELEOP_FORCE_ACTIVE=1，遥操作已强制开启（用于排查 Play 未生效）")
    should_reset = False

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
    print("[XR Teleop] 请在 AVP 上点击 Play 后再动手指，否则机器人不会跟随。")

    env.reset()
    teleop_interface.reset()

    # realman 有 platform_joint（升降柱），teleop 只输出双臂+夹爪，需补平台目标（与 init_state 一致）
    try:
        action_dim = env.unwrapped.action_manager.total_action_dim
        print(f"[XR Teleop] env action_dim={action_dim}, teleop 输出 dim 将在此后首帧打印")
    except Exception:
        action_dim = None
    PLATFORM_JOINT_DEFAULT = 0.8  # 与 realman_config.py init_state platform_joint 一致

    _logged_action_shape = False
    _last_debug_time = [time.perf_counter()]  # 用 list 以便在 closure 里更新
    dt = 1.0 / float(args_cli.control_hz)

    while simulation_app.is_running():
        with torch.inference_mode():
            action = teleop_interface.advance()

            if teleoperation_active:
                if not _logged_action_shape:
                    print(f"[XR Teleop] teleop advance() shape={action.shape}, action_dim={action_dim}")
                    _logged_action_shape = True
                # 每约 2 秒打印一次 action 范数，若始终为 0 说明手部数据未传到本机
                t = time.perf_counter()
                if t - _last_debug_time[0] >= 2.0:
                    _last_debug_time[0] = t
                    an = float(action.abs().sum()) if action.numel() else 0.0
                    print(f"[XR Teleop] active=1 action_sum_abs={an:.4f} (若始终≈0 则手部数据未到)")
                actions = action.repeat(env.num_envs, 1)
                if action_dim is not None and actions.shape[1] < action_dim:
                    pad = torch.full(
                        (actions.shape[0], action_dim - actions.shape[1]),
                        PLATFORM_JOINT_DEFAULT,
                        device=actions.device,
                        dtype=actions.dtype,
                    )
                    actions = torch.cat([actions, pad], dim=1)
                if not (torch.isfinite(actions).all()):
                    pass  # 跳过含 NaN/Inf 的帧，避免仿真炸
                else:
                    try:
                        env.step(actions)
                    except Exception as e:
                        print(f"[XR Teleop] env.step 异常: {e}")
                        import traceback
                        traceback.print_exc()
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

