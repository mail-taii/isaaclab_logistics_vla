"""
将 assets/robots/ur5e/ur5e.urdf 转为 ur5e.usd，供 Isaac Lab 加载。
需在 Isaac Sim / Isaac Lab 环境下运行（会启动一次 Sim 做转换后退出）。

用法（在仓库根目录）：
  python scripts/convert_ur5e_urdf_to_usd.py --headless
  # 可选：--force 强制重新转换（覆盖已有 ur5e.usd）
"""
import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR5e URDF -> USD for Isaac Lab.")
parser.add_argument("--force", action="store_true", help="强制重新转换（覆盖已有 ur5e.usd）")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 无头模式加快转换
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 仓库根 = scripts 的父目录
repo_root = Path(__file__).resolve().parent.parent
ur5e_dir = repo_root / "isaaclab_logistics_vla" / "assets" / "robots" / "ur5e"
urdf_path = ur5e_dir / "ur5e.urdf"
usd_dir = str(ur5e_dir)

if not urdf_path.exists():
    print(f"错误: 未找到 {urdf_path}，请先确保 URDF 与 meshes 已放在 assets/robots/ur5e/")
    simulation_app.close()
    exit(1)

try:
    from isaaclab.sim.converters import UrdfConverter
    from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg
except ImportError as e:
    print(f"错误: 无法导入 Isaac Lab 转换器: {e}")
    print("请确保在 Isaac Lab / Isaac Sim 环境中运行此脚本。")
    simulation_app.close()
    exit(1)

# UrdfConverterCfg 必填：asset_path, usd_dir, fix_base, joint_drive.gains.stiffness（Isaac Lab 2.2）
JointDriveCfg = UrdfConverterCfg.JointDriveCfg
PDGainsCfg = JointDriveCfg.PDGainsCfg
cfg_kw = dict(
    asset_path=str(urdf_path),
    usd_dir=usd_dir,
    force_usd_conversion=args.force,
    fix_base=True,
    joint_drive=JointDriveCfg(gains=PDGainsCfg(stiffness=80.0, damping=4.0)),
)
if hasattr(UrdfConverterCfg, "__dataclass_fields__") and "usd_file_name" in getattr(UrdfConverterCfg, "__dataclass_fields__", {}):
    cfg_kw["usd_file_name"] = "ur5e.usd"
cfg = UrdfConverterCfg(**cfg_kw)
converter = UrdfConverter(cfg)
# 访问 usd_path 会触发懒转换（若 USD 不存在或 force=True）
out_path = converter.usd_path
print(f"已生成: {out_path}")
if Path(out_path).exists():
    print("完成。之后可用 --robot_id ur5e 运行评估。")
else:
    print("警告: 未检测到输出文件，请检查上述路径。")

simulation_app.close()
