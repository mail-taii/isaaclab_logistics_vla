"""评估用策略（与 ``scripts/evaluate_vla.py`` 的 ``--policy`` 配合）。"""

__all__ = ["CuRoboPlanPolicy"]


def __getattr__(name: str):
    if name == "CuRoboPlanPolicy":
        from .curobo_plan_policy import CuRoboPlanPolicy

        return CuRoboPlanPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
