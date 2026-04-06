import argparse
import os


def _parse_cuda_ordinal(s: str) -> int | None:
    t = (s or "").strip()
    if not t.startswith("cuda:"):
        return None
    tail = t.split(":", 1)[-1].strip()
    if not tail or tail == "cuda":
        return None
    try:
        return int(tail)
    except ValueError:
        return None


def _apply_cuda_visible_before_isaac_import() -> bool:
    """在导入 AppLauncher 之前收敛可见 GPU，使 Warp wp.init() 使用的 cuda:0 即目标物理卡。

    Isaac 加载 isaaclab_tasks 时会触发 warp 默认设备；若仅传 --device cuda:4 而不改
    CUDA_VISIBLE_DEVICES，Warp 仍可能尝试物理 GPU0 建流并报
    RuntimeError: Failed to create stream on device cuda:0.0。

    若用户已设置 CUDA_VISIBLE_DEVICES，不覆盖。若 --curobo_device 与仿真卡为不同物理
    索引，不自动设置（避免把第二张规划卡从进程中隐藏）。
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        return False
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--device", type=str, default=None)
    pre.add_argument("--sim_device", type=str, default=None)
    pre.add_argument("--curobo_device", type=str, default=None)
    ns, _ = pre.parse_known_args()
    sim_ord = _parse_cuda_ordinal(ns.sim_device or ns.device or "")
    if sim_ord is None or sim_ord <= 0:
        return False
    curobo_ord = _parse_cuda_ordinal(ns.curobo_device or "")
    if curobo_ord is not None and curobo_ord != sim_ord:
        print(
            "[evaluate_vla][WARN] 未自动设置 CUDA_VISIBLE_DEVICES："
            "--curobo_device 与仿真用卡物理索引不同。若出现 Warp cuda:0 建流失败，"
            "请自行 export CUDA_VISIBLE_DEVICES 或改为单卡评估。"
        )
        return False
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sim_ord)
    print(
        f"[evaluate_vla] 已设置 CUDA_VISIBLE_DEVICES={sim_ord}，"
        "进程内 cuda:0 对应该物理 GPU；后续将把 --device/--sim_device 规范为 cuda:0。"
    )
    return True


_EVAL_CUDA_ISOLATED = _apply_cuda_visible_before_isaac_import()

from isaaclab.app import AppLauncher

# 先注册 AppLauncher 参数，再挂脚本自有参数
parser = argparse.ArgumentParser(description="VLA-benchmark for Isaac Lab environments.")
AppLauncher.add_app_launcher_args(parser)

parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--policy",
    type=str,
    default="random",
    help="策略名：random、curobo_plan（双臂 cuRobo 规划演示）等。",
)
parser.add_argument("--from_json", type=int, default=2, help="0: Record JSON, 1: Replay JSON, 2: Pure Random")

parser.add_argument("--asset_root_path", type=str, default="/home/junzhe/Benchmark")
parser.add_argument("--task_scene_name", type=str, default="Spawn_ms_st_dense_EnvCfg")
parser.add_argument(
    "--robot_id",
    type=str,
    default="realman_dual_left_arm",
    help="评估侧机器人 ID，对应 evaluation/robot_registry.py；curobo_plan 会传给 CuRoboPlanPolicy。",
)
parser.add_argument(
    "--sim_device",
    type=str,
    default=None,
    help="Isaac 仿真/渲染所用设备，如 cuda:4；未指定时用 --device。可设 cpu（极慢）。",
)
parser.add_argument(
    "--curobo_device",
    type=str,
    default=None,
    help="Curobo 规划所用 GPU，如 cuda:5；写入 CUROBO_DEVICE（供后续扩展）。",
)
parser.add_argument(
    "--use_mesh_obstacles",
    action="store_true",
    help="设置 CUROBO_USE_MESH_OBSTACLES=1。",
)
parser.add_argument(
    "--omit_base_scene_assets",
    type=str,
    default="",
    help="逗号分隔，跳过 BaseOrderSceneCfg 中的资源（仅 e_table、e_conveyer）。"
    "默认空串：加载全部 base USD。若个别资产仍损坏可传例如 e_table,e_conveyer。"
    "环境变量 EVALUATE_VLA_OMIT_BASE_SCENE_ASSETS 若设置非空则用作省略列表（不覆盖 --use_full_base_scene_usd）。",
)
parser.add_argument(
    "--use_full_base_scene_usd",
    action="store_true",
    help="强制加载全部 base 场景 USD；若设置了 EVALUATE_VLA_OMIT_BASE_SCENE_ASSETS，本开关可覆盖之。",
)

args_cli, _ = parser.parse_known_args()

if args_cli.use_mesh_obstacles:
    os.environ["CUROBO_USE_MESH_OBSTACLES"] = "1"

# 在 AppLauncher 启动前把 sim_device 同步到 device，并令 Vulkan 使用同一张卡（避免渲染仍绑 GPU0）
if args_cli.sim_device is not None and str(args_cli.sim_device).strip():
    args_cli.device = str(args_cli.sim_device).strip()

# 与 _apply_cuda_visible_before_isaac_import 一致：单物理卡可见时逻辑设备必须是 cuda:0
if _EVAL_CUDA_ISOLATED:
    args_cli.device = "cuda:0"
    if args_cli.sim_device is not None and str(args_cli.sim_device).strip():
        args_cli.sim_device = "cuda:0"

_dev_for_gpu = (args_cli.sim_device or args_cli.device or "").strip()
if _dev_for_gpu.startswith("cuda:"):
    try:
        _gpu_id = int(_dev_for_gpu.split(":")[-1])
        _renderer_arg = f"--/renderer/activeGpu={_gpu_id}"
        _kc = (getattr(args_cli, "kit_args", None) or "").strip()
        args_cli.kit_args = f"{_kc} {_renderer_arg}".strip() if _kc else _renderer_arg
    except ValueError:
        pass

# VLA 评估依赖相机观测
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

if not os.path.exists(args_cli.asset_root_path):
    print(f"资产路径{args_cli.asset_root_path}未配置！请检查")
    exit()
else:
    print(f"Asset Root Path: {args_cli.asset_root_path}")
    os.environ["ASSET_ROOT_PATH"] = args_cli.asset_root_path

if args_cli.curobo_device:
    os.environ["CUROBO_DEVICE"] = args_cli.curobo_device.strip()

import isaaclab_tasks
import isaaclab_logistics_vla
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_logistics_vla.evaluation.evaluator.vla_evaluator import VLA_Evaluator

from isaaclab_logistics_vla.utils.register import register

register.auto_scan("isaaclab_logistics_vla.tasks")

_OMITTABLE_BASE_SCENE_KEYS = frozenset({"e_conveyer", "e_table"})


def _omit_base_scene_assets(env_cfg, omit_csv: str) -> None:
    """omit_csv 为空则不省略任何项。"""
    raw = (omit_csv or "").strip()
    if not raw:
        return
    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        return
    for name in [x.strip() for x in raw.split(",") if x.strip()]:
        if name not in _OMITTABLE_BASE_SCENE_KEYS:
            print(f"[evaluate_vla][WARN] 忽略未知键 {name!r}（仅支持 {sorted(_OMITTABLE_BASE_SCENE_KEYS)}）")
            continue
        if not hasattr(scene, name):
            continue
        setattr(scene, name, None)
        print(f"[evaluate_vla] 已跳过 base 场景资源 {name!r}（不加载对应 USD）")


def _resolve_omit_base_scene_csv(args) -> str:
    if getattr(args, "use_full_base_scene_usd", False):
        return ""
    env_v = os.environ.get("EVALUATE_VLA_OMIT_BASE_SCENE_ASSETS")
    if env_v is not None and str(env_v).strip() != "":
        return str(env_v).strip()
    return (getattr(args, "omit_base_scene_assets", None) or "").strip()


def _resolve_sim_device_string(device_str: str, app_launcher: AppLauncher) -> str:
    d = device_str if device_str else "cuda:0"
    if d == "cuda":
        return f"cuda:{app_launcher.device_id}"
    return d


def main():
    print(f"正在加载任务配置: {args_cli.task_scene_name}")
    env_cfg = register.load_env_configs(f"{args_cli.task_scene_name}")()

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    _omit_base_scene_assets(env_cfg, _resolve_omit_base_scene_csv(args_cli))

    # sim_device > device > cuda:0；此处与 AppLauncher 已同步的 args_cli.device 一致
    sim_dev = (
        args_cli.sim_device.strip()
        if args_cli.sim_device and str(args_cli.sim_device).strip()
        else _resolve_sim_device_string(args_cli.device if args_cli.device else "cuda:0", app_launcher)
    )
    env_cfg.sim.device = sim_dev

    evaluator = VLA_Evaluator(
        env_cfg=env_cfg,
        policy=args_cli.policy,
        from_json=args_cli.from_json,
        device=sim_dev,
        robot_id=args_cli.robot_id,
    )
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
