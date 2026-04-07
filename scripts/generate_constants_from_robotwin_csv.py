import argparse
import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class ModelIdParts:
    numeric_id: str
    object_name: str  # e.g. RUBIKSCUBE
    variant: str  # e.g. BASE0

    @property
    def object_name_lower(self) -> str:
        return self.object_name.lower()

    @property
    def variant_lower(self) -> str:
        return self.variant.lower()

    @property
    def var_name(self) -> str:
        # Python变量名：去掉前缀数字ID
        return f"{self.object_name}_{self.variant}"

    @property
    def sku_name(self) -> str:
        # constant.py里的 NAME 字段：全小写
        return self.var_name.lower()


def _parse_model_id(model_id: str) -> ModelIdParts:
    parts = model_id.strip().split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid Model ID: {model_id!r}")
    numeric_id = parts[0]
    variant = parts[-1]
    object_name = "_".join(parts[1:-1])
    if not numeric_id.isdigit():
        raise ValueError(f"Invalid numeric prefix in Model ID: {model_id!r}")
    if not object_name:
        raise ValueError(f"Invalid object name in Model ID: {model_id!r}")
    if not variant:
        raise ValueError(f"Invalid variant in Model ID: {model_id!r}")
    return ModelIdParts(numeric_id=numeric_id, object_name=object_name, variant=variant)


def _parse_scalar_or_tuple(raw: str) -> Any:
    s = (raw or "").strip()
    if s == "":
        return None
    # 常见情况："(0,0,0)"、"[...]"、"1"、"1.0"
    if s[0] in "([{" or s in {"None", "True", "False"}:
        try:
            return ast.literal_eval(s)
        except Exception:
            # 兜底：如果表格里用了不规范格式，就原样输出字符串
            return s
    # 数值
    try:
        if any(ch in s for ch in [".", "e", "E"]):
            return float(s)
        return int(s)
    except Exception:
        return s


def _py_literal(value: Any) -> str:
    if value is None:
        return "None"
    return repr(value)


def _build_usd_path(parts: ModelIdParts) -> str:
    # 按你描述的命名规律：
    # f"{ASSET_ROOT_PATH}/objects_adapted/073_rubikscube/visual/base0.usd"
    return (
        f'{{ASSET_ROOT_PATH}}/objects_adapted/{parts.numeric_id}_{parts.object_name_lower}/visual/{parts.variant_lower}.usd'
    )


def _iter_rows(csv_path: Path) -> Iterable[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {str(csv_path)!r}")
        yield from reader


def _row_to_const_dict(row: Dict[str, str]) -> Tuple[str, Dict[str, Any]]:
    model_id = row.get("Model ID", "")
    parts = _parse_model_id(model_id)

    const: Dict[str, Any] = {
        "NAME": parts.sku_name,
        "USD_PATH": _build_usd_path(parts),
        "X_LENGTH": float(row["X"]),
        "Y_LENGTH": float(row["Y"]),
        "Z_LENGTH": float(row["Z"]),
        "SPARSE_ORIENT": _parse_scalar_or_tuple(row.get("SPARSE_ORIENT", "")),
        "STACK_ORIENT": _parse_scalar_or_tuple(row.get("STACK_ORIENT", "")),
        "DENSE_ORIENT": _parse_scalar_or_tuple(row.get("DENSE_ORIENT", "")),
        "STACK_SCALE": _parse_scalar_or_tuple(row.get("STACK_SCALE", "")),
        "SPARSE_SCALE": _parse_scalar_or_tuple(row.get("SPARSE_SCALE", "")),
        "DENSE_SCALE": _parse_scalar_or_tuple(row.get("DENSE_SCALE", "")),
    }

    var_name = parts.var_name
    if not var_name[0].isalpha() and var_name[0] != "_":
        var_name = f"OBJ_{var_name}"

    return var_name, const


def _render_constant_py(items: List[Tuple[str, Dict[str, Any]]]) -> str:
    lines: List[str] = []
    # 保留手写常量区（用户要求：这些常量不从 CSV 生成）
    lines.extend(
        [
            "import os",
            'ASSET_ROOT_PATH = os.getenv("ASSET_ROOT_PATH", "")',
            "",
            "WORK_BOX_PARAMS = {",
            "    'USD_PATH':f\"{ASSET_ROOT_PATH}/env/Box.usd\",",
            "    'X_LENGTH':0.36,",
            "    'Y_LENGTH' :0.56,",
            "    'Z_LENGTH':0.23",
            "}",
            "",
            "#---障碍物参数(新增)---",
            "LARGE_OBSTACLE_PARAMS = {",
            "    'X_LENGTH': 0.30,",
            "    'Y_LENGTH': 0.15,",
            "    'Z_LENGTH': 0.30,",
            "    'COLOR': (0.8, 0.1, 0.1)",
            "}",
            "",
            "tray_PARAMS = {",
            "    'USD_PATH':f\"{ASSET_ROOT_PATH}/env/Collected_Blue_Tray/SM_Crate_A08_Blue_01.usd\",",
            "    'X_LENGTH':1.2,",
            "    'Y_LENGTH':0.76,",
            "    'Z_LENGTH':0.32,",
            "    #SPARSE_ORIENT':(0,0,0)    #相对于箱子的坐标",
            "}",
            "",
            "CRACKER_BOX_PARAMS = {",
            "    'NAME': \"cracker_box\",",
            "    'USD_PATH':f\"{ASSET_ROOT_PATH}/props/Collected_003_cracker_box/003_cracker_box.usd\",",
            "    'X_LENGTH':0.16,",
            "    'Y_LENGTH':0.20,",
            "    'Z_LENGTH':0.06,",
            "    'SPARSE_ORIENT':(0,90,0),   #相对于箱子的坐标",
            "    \"DENSE_ORIENT\":[(0,90,0),(0,0,0)],",
            "    'STACK_ORIENT':(0,0,0),     # Z最小，默认朝向即可",
            "    'STACK_SCALE': 0.6",
            "}",
            "",
            "SUGER_BOX_PARAMS = {",
            "    'NAME': \"sugar_box\",",
            "    'USD_PATH':f\"{ASSET_ROOT_PATH}/props/Collected_004_sugar_box/004_sugar_box.usd\",",
            "    'X_LENGTH':0.09,",
            "    'Y_LENGTH':0.17,",
            "    'Z_LENGTH':0.04,",
            "    'SPARSE_ORIENT':(0,90,0),",
            "    \"DENSE_ORIENT\":[(0,90,0),(0,0,0)],",
            "    'STACK_ORIENT':(0,0,0),        # Z最小，默认朝向即可",
            "    'STACK_SCALE': 0.6,",
            "}",
            "",
            "TOMATO_SOUP_CAN_PARAMS = {",
            "    'USD_PATH':f\"{ASSET_ROOT_PATH}/props/Collected_005_tomato_soup_can/005_tomato_soup_can.usd\",",
            "    'RADIUS':0.035,",
            "    'X_LENGTH':0.07,",
            "    'Y_LENGTH':0.10,",
            "    'Z_LENGTH':0.07,",
            "    'SPARSE_ORIENT':(90,0,0),",
            "    \"DENSE_ORIENT\":[(90,0,0),(0,0,0)],",
            "     'STACK_ORIENT':(0,0,0)         # X=Z 等大，默认朝向即可",
            "}",
            "",
            "CN_BIG_PARAMS = {",
            "    'USD_PATH':f\"{ASSET_ROOT_PATH}/props/Collected_CNBig/CNBig.usdc\",",
            "    \"RADIUS\":0.275,",
            "    'X_LENGTH':0.55,",
            "    'Y_LENGTH':0.14,",
            "    'Z_LENGTH':0.14,",
            "    'SPARSE_ORIENT':(0,0,90),#or (0,0,0)",
            "    \"DENSE_ORIENT\":[(0,0,90),(0,0,0)]",
            "}",
            "",
            "SF_SMALL_PARAMS = {",
            "    'NAME': \"sf_small\",",
            "    'USD_PATH':f\"{ASSET_ROOT_PATH}/props/Collected_SFSmall/SFSmall.usdc\",",
            "    \"RADIUS\":0.035,",
            "    'X_LENGTH':0.34,    ",
            "    'Y_LENGTH':0.43,",
            "    'Z_LENGTH':0.08,",
            "    'SPARSE_ORIENT':(0,0,0),#or (0,0,0)",
            "    'STACK_ORIENT':(0,0,0),",
            "    \"DENSE_ORIENT\":[(0,0,0)],",
            "    'STACK_SCALE': 0.3,",
            "}",
            "",
            "EMPTY_PLASTIC_PACKAGE_PARAMS = {",
            "    'USD_PATH':f\"{ASSET_ROOT_PATH}/props/Collected_empty_plastic_package/empty_plastic_package.usdc\",",
            "    \"RADIUS\":0.185,",
            "    'X_LENGTH':0.35,    ",
            "    'Y_LENGTH':0.37,",
            "    'Z_LENGTH':0.07,",
            "    'SPARSE_ORIENT':(0,0,0),#or (0,0,0)",
            "    \"DENSE_ORIENT\":[(0,0,0)]",
            "}",
            "",
            "SF_BIG_PARAMS = {",
            "    'NAME': \"sf_big\",",
            "    'USD_PATH':f\"{ASSET_ROOT_PATH}/props/Collected_SFBig/SFBig.usdc\",",
            "    \"RADIUS\":0.22,",
            "    'X_LENGTH':0.47,    ",
            "    'Z_LENGTH':0.34,",
            "    'Y_LENGTH':0.15,",
            "    'SPARSE_ORIENT':(90,0,0), # (0,0,0)",
            "    'STACK_ORIENT':(90,0,90),",
            "    \"DENSE_ORIENT\":[(90,0,0),(0,0,0)],",
            "    'STACK_SCALE': 0.3,",
            "}",
            "",
            "PLASTIC_PACKAGE_PARAMS = {",
            "    'NAME': \"plastic_package\",",
            "    'USD_PATH':f\"{ASSET_ROOT_PATH}/props/Collected_plastic_package/plastic_package.usdc\",",
            "    'X_LENGTH':0.34,",
            "    'Y_LENGTH':0.39,",
            "    'Z_LENGTH':0.07,",
            "    'SPARSE_ORIENT':(0,0,0),",
            "    'STACK_ORIENT':(0,0,0),        # Z最小，默认朝向即可",
            "    'STACK_SCALE': 0.4,",
            "}",
            "",
            "# NOTE: 该文件由脚本自动生成（包含手写常量前缀 + CSV 物品区），请勿手工修改 CSV 生成区。",
            "# 生成脚本：isaaclab_logistics_vla/scripts/generate_constants_from_robotwin_csv.py",
            "",
        ]
    )

    # 按变量名排序，保证 diff 稳定
    items_sorted = sorted(items, key=lambda x: x[0])
    for var_name, const in items_sorted:
        lines.append(f"{var_name} = {{")
        lines.append(f"    'NAME': {_py_literal(const['NAME'])},")
        lines.append(f"    'USD_PATH': f\"{const['USD_PATH']}\",")

        lines.append(f"    'X_LENGTH': {const['X_LENGTH']},")
        lines.append(f"    'Y_LENGTH': {const['Y_LENGTH']},")
        lines.append(f"    'Z_LENGTH': {const['Z_LENGTH']},")

        for k in ["SPARSE_ORIENT", "STACK_ORIENT", "DENSE_ORIENT"]:
            lines.append(f"    '{k}': {_py_literal(const[k])},")
        for k in ["STACK_SCALE", "SPARSE_SCALE", "DENSE_SCALE"]:
            lines.append(f"    '{k}': {_py_literal(const[k])},")
        # 去掉最后一个逗号，让格式更像你现有的 constant.py
        if lines[-1].endswith(","):
            lines[-1] = lines[-1].rstrip(",")
        lines.append("}")
        lines.append("")

    lines.append("# 统一的 SKU 配置表（每个变体独立挂在这里）")
    lines.append("SKU_CONFIG = {")
    # 手写 SKU（有 NAME 字段的）
    for var_name in [
        "CRACKER_BOX_PARAMS",
        "SUGER_BOX_PARAMS",
        "PLASTIC_PACKAGE_PARAMS",
        "SF_BIG_PARAMS",
        "SF_SMALL_PARAMS",
    ]:
        lines.append(f"    {var_name}['NAME']: {var_name},")
    for var_name, _ in items_sorted:
        lines.append(f"    {var_name}['NAME']: {var_name},")
    if lines[-1].endswith(","):
        lines[-1] = lines[-1].rstrip(",")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从 RoboTwin.csv 生成 isaaclab_logistics_vla/utils/constant.py"
    )
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).resolve().parents[1] / "RoboTwin.csv"),
        help="输入 CSV 路径（默认：仓库根目录 RoboTwin.csv）",
    )
    parser.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parents[1]
            / "isaaclab_logistics_vla"
            / "utils"
            / "constant.py"
        ),
        help="输出 constant.py 路径（默认：isaaclab_logistics_vla/utils/constant.py）",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="仅打印到 stdout，不写文件",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    items: List[Tuple[str, Dict[str, Any]]] = []
    for row in _iter_rows(csv_path):
        var_name, const = _row_to_const_dict(row)
        items.append((var_name, const))

    rendered = _render_constant_py(items)
    if args.stdout:
        print(rendered)
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
