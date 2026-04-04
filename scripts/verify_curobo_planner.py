#!/usr/bin/env python3
"""
验证 ``isaaclab_logistics_vla.utils.curobo_planner`` 封装是否正常。

不经过 ``isaaclab_logistics_vla`` 包根 ``__init__.py``（避免拉取 Isaac Lab / 全量 tasks），
仅按文件路径加载 ``utils/curobo_planner`` 子模块。

用法（在仓库根目录 ``isaaclab_logistics_vla`` 下）::

    python scripts/verify_curobo_planner.py
    python scripts/verify_curobo_planner.py --device cuda:0 --urdf /path/to/robot.urdf
    python scripts/verify_curobo_planner.py --smoke-only
    python scripts/verify_curobo_planner.py --no-smoke

退出码:
    0 - 已执行的测试全部通过
    1 - 有测试失败
    2 - 本应跑冒烟但被跳过（无 CUDA / 无 curobo / URDF 不存在）
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
import traceback
import types
from types import SimpleNamespace
from typing import Callable, List, Tuple

# 仓库根（外层 isaaclab_logistics_vla，内含子目录 isaaclab_logistics_vla/）
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_INNER_PKG = os.path.join(_REPO_ROOT, "isaaclab_logistics_vla")
_CUROBO_PL_DIR = os.path.join(_INNER_PKG, "utils", "curobo_planner")
_PKG = "isaaclab_logistics_vla.utils.curobo_planner"

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _register_stub_packages() -> None:
    """注册父包桩，使相对导入 ``from .config_generator`` 可用，且不执行扩展根 __init__。"""
    if "isaaclab_logistics_vla" not in sys.modules:
        m = types.ModuleType("isaaclab_logistics_vla")
        m.__path__ = [_INNER_PKG]
        sys.modules["isaaclab_logistics_vla"] = m
    if "isaaclab_logistics_vla.utils" not in sys.modules:
        u = types.ModuleType("isaaclab_logistics_vla.utils")
        u.__path__ = [os.path.join(_INNER_PKG, "utils")]
        sys.modules["isaaclab_logistics_vla.utils"] = u
    if _PKG not in sys.modules:
        cp = types.ModuleType(_PKG)
        cp.__path__ = [_CUROBO_PL_DIR]
        sys.modules[_PKG] = cp


def _load_submodule(name: str, filename: str):
    path = os.path.join(_CUROBO_PL_DIR, filename)
    mod_name = f"{_PKG}.{name}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = _PKG
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_result_utils():
    _register_stub_packages()
    return _load_submodule("result_utils", "result_utils.py")


def load_full_curobo_planner_class():
    """加载 result_utils → config_generator → curobo_planner（会 import curobo/torch）。"""
    _register_stub_packages()
    _load_submodule("result_utils", "result_utils.py")
    _load_submodule("config_generator", "config_generator.py")
    m = _load_submodule("curobo_planner", "curobo_planner.py")
    return m.CuroboPlanner


def _ok(name: str) -> None:
    print(f"  [OK] {name}")


def _fail(name: str, err: Exception) -> None:
    print(f"  [FAIL] {name}: {err}")
    traceback.print_exc()


def test_plan_grippers_linear() -> None:
    ru = load_result_utils()
    g = ru.plan_grippers_linear(0.0, 1.0, num_step=5)
    assert g["num_step"] == 5
    assert g["result"].shape == (5,)
    assert abs(float(g["result"][0]) - 0.0) < 1e-5
    assert abs(float(g["result"][-1]) - 1.0) < 1e-5
    assert "per_step" in g


class _FakeSuccTensor:
    """模拟 ``result.success`` 的一维 batch，不依赖 torch。"""

    def __init__(self, value: bool):
        self._v = value

    def dim(self) -> int:
        return 1

    def __getitem__(self, _idx):
        return self

    def item(self) -> bool:
        return self._v


def test_motion_gen_result_to_plan_dict_fail_shape() -> None:
    ru = load_result_utils()
    r = SimpleNamespace(success=_FakeSuccTensor(False), status=None)
    out = ru.motion_gen_batch_result_to_plan_dict(r, batch_index=0)
    assert out["status"] == "Fail"
    assert out["position"] is None
    assert out["velocity"] is None


def test_curobo_planner_static_api() -> None:
    CuroboPlanner = load_full_curobo_planner_class()
    assert CuroboPlanner.dof_dual_arm == 14
    g = CuroboPlanner.plan_grippers(0.2, 0.8, num_step=10)
    assert g["num_step"] == 10
    assert len(g["result"]) == 10


def run_logic_tests(*, strict_planner_import: bool = False) -> Tuple[int, List[str]]:
    errors: List[str] = []
    tests: List[Tuple[str, Callable[[], None]]] = [
        ("plan_grippers_linear（仅 numpy，无 torch/curobo）", test_plan_grippers_linear),
    ]

    print("\n== 逻辑测试 1：plan_grippers_linear ==")
    for name, fn in tests:
        try:
            fn()
            _ok(name)
        except Exception as e:
            _fail(name, e)
            errors.append(name)

    print("\n== 逻辑测试 2：motion_gen_batch_result_to_plan_dict（mock，无 torch）==")
    try:
        test_motion_gen_result_to_plan_dict_fail_shape()
        _ok("mock Fail 分支 + dict 结构")
    except Exception as e:
        _fail("motion_gen_batch_result_to_plan_dict", e)
        errors.append("motion_gen_batch_result_to_plan_dict")

    print("\n== 逻辑测试 3：CuroboPlanner 静态 API（需 torch + curobo；无环境则跳过）==")
    try:
        test_curobo_planner_static_api()
        _ok("dof_dual_arm + plan_grippers")
    except Exception as e:
        if strict_planner_import:
            _fail("CuroboPlanner 静态 API (--strict-imports)", e)
            errors.append("CuroboPlanner 静态 API")
        else:
            print(f"  [SKIP] CuroboPlanner 静态 API: {e}")

    return len(errors), errors


def smoke_curobo_planner(urdf_path: str, device: str) -> Tuple[bool, str]:
    import numpy as np

    CuroboPlanner = load_full_curobo_planner_class()

    with tempfile.TemporaryDirectory(prefix="curobo_planner_verify_") as tmp:
        cache_path = os.path.join(tmp, "robot_cache.yaml")
        planner = CuroboPlanner(
            urdf_path=urdf_path,
            device=device,
            cache_path=cache_path,
            use_curobo_cache=True,
            interpolation_dt=0.05,
            use_cuda_graph=False,
        )
        planner.clear_world()
        planner.reset(reset_seed=True)

        start = np.array(
            [
                0.0,
                -0.5,
                0.0,
                -1.2,
                0.0,
                0.5,
                0.0,
                0.0,
                -0.5,
                0.0,
                -1.2,
                0.0,
                0.5,
                0.0,
            ],
            dtype=np.float32,
        )
        goals = {
            "left": {
                "position": np.array([-0.15, 0.45, 0.35], dtype=np.float64),
                "quaternion": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            },
            "right": {
                "position": np.array([0.15, 0.45, 0.35], dtype=np.float64),
                "quaternion": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            },
        }

        out = planner.plan_dual(
            start,
            goals,
            max_attempts=5,
            timeout=5.0,
            enable_graph=True,
            enable_opt=True,
        )

        for k in ("status", "position", "velocity"):
            if k not in out:
                return False, f"返回 dict 缺少键 {k!r}，实际 keys={list(out.keys())}"

        if out["status"] not in ("Success", "Fail"):
            return False, f"status 非法: {out['status']!r}"

        if out["status"] == "Success":
            pos = out["position"]
            if pos is None:
                return False, "Success 但 position 为 None"
            if pos.ndim != 2 or pos.shape[1] != 14:
                return False, f"Success 时期望 position.shape[1]==14，得到 {getattr(pos, 'shape', None)}"
            if out["velocity"] is not None and out["velocity"].shape != pos.shape:
                return (
                    False,
                    f"velocity 形状应与 position 一致: pos {pos.shape}, vel {out['velocity'].shape}",
                )
        else:
            if out["position"] is not None:
                return False, "Fail 时期望 position 为 None"

        return True, "smoke OK"


def can_run_smoke() -> Tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "未安装 torch"

    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() 为 False"

    try:
        import curobo  # noqa: F401
    except ImportError:
        return False, "未安装或无法导入 curobo"

    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description="验证 curobo_planner 封装")
    parser.add_argument(
        "--urdf",
        type=str,
        default="/home/junzhe/Benchmark/robot/realman/realman_franka_ee.urdf",
        help="双臂 URDF 路径",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="CuRobo CUDA 设备")
    parser.add_argument("--no-smoke", action="store_true", help="不运行需要 GPU 的 CuRobo 冒烟")
    parser.add_argument("--smoke-only", action="store_true", help="只运行 CuRobo 冒烟")
    parser.add_argument(
        "--strict-smoke",
        action="store_true",
        help="冒烟额外要求一次规划 Success（更严，易因目标不可达失败）",
    )
    parser.add_argument(
        "--strict-imports",
        action="store_true",
        help="逻辑测试 3 必须能导入 CuroboPlanner（无 curobo/torch 时失败而非跳过）",
    )
    args = parser.parse_args()

    if not args.smoke_only:
        n_err, err_names = run_logic_tests(strict_planner_import=args.strict_imports)
        if n_err:
            print(f"\n逻辑测试失败 {n_err} 项: {err_names}")
            return 1

    if args.no_smoke:
        print("\n已跳过 CuRobo 冒烟 (--no-smoke)。")
        return 0

    ok_env, reason = can_run_smoke()
    if not ok_env:
        print(f"\n[SKIP] CuRobo 冒烟: {reason}")
        return 2

    if not os.path.isfile(args.urdf):
        print(f"\n[SKIP] CuRobo 冒烟: URDF 不存在: {args.urdf}")
        return 2

    print("\n== CuRobo 冒烟（构造 MotionGen + plan_dual）==")
    try:
        good, msg = smoke_curobo_planner(args.urdf, args.device)
        if not good:
            print(f"  [FAIL] {msg}")
            return 1
        _ok(msg)
        if args.strict_smoke:
            import numpy as np

            CuroboPlanner = load_full_curobo_planner_class()
            with tempfile.TemporaryDirectory(prefix="curobo_strict_") as tmp:
                p = CuroboPlanner(
                    urdf_path=args.urdf,
                    device=args.device,
                    cache_path=os.path.join(tmp, "cache.yaml"),
                    use_cuda_graph=False,
                )
                p.clear_world()
                st = np.zeros(14, dtype=np.float32)
                g = {
                    "left": {
                        "position": np.array([0.0, 0.3, 0.3]),
                        "quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
                    },
                    "right": {
                        "position": np.array([0.0, 0.3, 0.3]),
                        "quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
                    },
                }
                out = p.plan_dual(st, g, max_attempts=20, timeout=15.0)
                if out["status"] != "Success":
                    print(
                        f"  [FAIL] --strict-smoke 要求规划成功，实际: {out.get('detail', out['status'])}"
                    )
                    return 1
                _ok("strict-smoke: plan Success")
    except Exception as e:
        print(f"  [FAIL] CuRobo 冒烟异常: {e}")
        traceback.print_exc()
        return 1

    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
