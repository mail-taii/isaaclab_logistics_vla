"""
Bunny 遥操作入口：与 evaluate_vla.py 类似，先启动 Isaac App，再加载任务场景并订阅 Bunny 的 qpos 话题，
实时驱动仿真。需先在 Bunny 端发布 /bunny_teleop/left_qpos 与 /bunny_teleop/right_qpos。
"""
import argparse
import os

# 在 App 启动前设置 ROS2 Bridge 所需环境变量（Isaac Sim 内置 rclpy，无需 source /opt/ros/humble）
def _setup_ros2_bridge_env():
    try:
        import isaacsim
        ext_dir = os.path.join(os.path.dirname(isaacsim.__file__), "exts", "isaacsim.ros2.bridge")
        humble_lib = os.path.join(ext_dir, "humble", "lib")
        if os.path.isdir(humble_lib):
            os.environ.setdefault("ROS_DISTRO", "humble")
            os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
            ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if humble_lib not in ld_path:
                # 将 humble/lib 置于最前，确保 rclpy 加载时优先找到 librcl_action.so 等
                os.environ["LD_LIBRARY_PATH"] = f"{humble_lib}:{ld_path}".strip(":")
    except Exception:
        pass  # 若失败，扩展会提示用户手动设置


_setup_ros2_bridge_env()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Bunny 遥操作驱动 Isaac Lab 仿真")
parser.add_argument(
    "--task_scene_name",
    type=str,
    default="Spawn_ds_st_sparse_EnvCfg",
    help="与 evaluate_vla 相同的任务场景名",
)
parser.add_argument(
    "--left_topic",
    type=str,
    default="/bunny_teleop/left_qpos",
    help="Bunny 左臂 qpos 话题 (std_msgs/Float64MultiArray)",
)
parser.add_argument(
    "--right_topic",
    type=str,
    default="/bunny_teleop/right_qpos",
    help="Bunny 右臂 qpos 话题 (std_msgs/Float64MultiArray)",
)
parser.add_argument(
    "--control_hz",
    type=float,
    default=60.0,
    help="控制循环频率",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="环境数量，默认用任务配置",
)
parser.add_argument(
    "--asset_root_path",
    type=str,
    default="/home/junzhe/Benchmark",
    help="资产根路径",
)
parser.add_argument(
    "--sim_device",
    type=str,
    default=None,
    help="仿真 GPU，如 cuda:0",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

if args_cli.sim_device is not None:
    args_cli.device = args_cli.sim_device
    gpu_id = int(args_cli.sim_device.split(":")[-1])
    renderer_arg = f"--/renderer/activeGpu={gpu_id}"
    args_cli.kit_args = (args_cli.kit_args or "").strip()
    args_cli.kit_args = f"{args_cli.kit_args} {renderer_arg}".strip() if args_cli.kit_args else renderer_arg
args_cli.enable_cameras = True

# 启用 Isaac Sim 内置 ROS2 Bridge，以便在 Python 3.11 下使用 rclpy（无需 source /opt/ros/humble）
_ros2_enable = "--enable isaacsim.ros2.bridge"
args_cli.kit_args = (args_cli.kit_args or "").strip()
args_cli.kit_args = f"{args_cli.kit_args} {_ros2_enable}".strip() if args_cli.kit_args else _ros2_enable

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

if not os.path.exists(args_cli.asset_root_path):
    print(f"资产路径不存在: {args_cli.asset_root_path}")
    exit()
os.environ["ASSET_ROOT_PATH"] = args_cli.asset_root_path

import isaaclab_tasks  # noqa: F401 - 与 evaluate_vla 一致，确保 Nucleus/USD 加载顺序
import isaaclab_logistics_vla  # noqa: E402
from isaaclab_logistics_vla.teleop.run_bunny_teleop import run_bunny_teleop  # noqa: E402
from isaaclab_logistics_vla.utils.register import register  # noqa: E402

register.auto_scan("isaaclab_logistics_vla.tasks")


def main():
    run_bunny_teleop(
        task_scene_name=args_cli.task_scene_name,
        left_topic=args_cli.left_topic,
        right_topic=args_cli.right_topic,
        control_hz=args_cli.control_hz,
        num_envs=args_cli.num_envs,
        sim_device=args_cli.sim_device or args_cli.device,
    )


if __name__ == "__main__":
    main()
