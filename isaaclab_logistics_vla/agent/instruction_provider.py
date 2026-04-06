from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class InstructionSpec:
    """给 VLM 的任务指令（目前是占位接口，后续可替换为真实订单系统）。"""

    instruction: str
    task_type: str = "virtual"
    task_id: str = "virtual-0"
    metadata: Dict[str, Any] | None = None


class InstructionProvider(ABC):
    """指令提供器抽象接口。

    设计目的：
    - 未来接真实“订单系统/任务服务器”时只需要实现这个接口
    - runner/脚本层不依赖具体订单来源
    """

    @abstractmethod
    def next_instruction(self) -> InstructionSpec: ...


class VirtualInstructionProvider(InstructionProvider):
    """虚拟指令（用于当前没有真实订单系统的测试阶段）。"""

    def __init__(
        self,
        task_type: str = "move_all_between_targets",
        src_target_idx: int = 1,
        dst_target_idx: int = 2,
    ):
        self._task_type = task_type
        self._src = int(src_target_idx)
        self._dst = int(dst_target_idx)

    def next_instruction(self) -> InstructionSpec:
        if self._task_type == "move_all_between_targets":
            text = (
                f"虚拟指令：将订单箱{self._src}中的所有物品全部移动到订单箱{self._dst}中。"
                "完成后请用 JSON 输出一条 scene_description 描述最终画面，并 done 结束。"
            )
            meta = {"src_target_idx": self._src, "dst_target_idx": self._dst}
            return InstructionSpec(instruction=text, task_type=self._task_type, task_id=f"virtual-moveall-{self._src}-{self._dst}", metadata=meta)

        # 兜底：未知 task_type
        text = (
            "虚拟指令：请先调用 get_topview_image，然后输出 scene_description 描述你看到的场景，最后 done 结束。"
        )
        return InstructionSpec(instruction=text, task_type="virtual_unknown", task_id="virtual-unknown", metadata={"raw_task_type": self._task_type})

