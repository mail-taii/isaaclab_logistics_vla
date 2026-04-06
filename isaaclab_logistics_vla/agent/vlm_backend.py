from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


Message = Dict[str, Any]  # {"role": "system"|"user"|"assistant"|"tool", ...}


@dataclass(frozen=True)
class VlmToolSchema:
    """给 VLM 的 tools 描述（用于 system prompt / function list）。"""

    name: str
    description: str


class VlmBackend(ABC):
    """VLM 后端抽象接口。

    设计目标：
    - 同一个 agent 框架可评测不同 VLM（只替换 backend）
    - backend 只关心：输入图像 + 对话历史 + tools 描述，输出一段文本（通常是 JSON tool call）
    """

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def infer(
        self,
        image_rgb_uint8: np.ndarray,
        messages: List[Message],
        tools: List[VlmToolSchema],
    ) -> str:
        """返回 VLM 的原始输出文本（建议为 JSON）。"""


class DummyBackend(VlmBackend):
    """用于本地端到端验证：不调用外部 API。

    行为：
    - 默认：第一步 get_topview_image，第二步 done
    - test_describe=True：tool → scene_description → done（解说图测试）
    - test_plan=True：tool → task_create(DAG) → done（任务编排测试）
    """

    def __init__(self, test_describe: bool = False, test_plan: bool = False):
        self._test_describe = bool(test_describe)
        self._test_plan = bool(test_plan)
        self._step = 0

    @property
    def model_name(self) -> str:
        return "dummy"

    def infer(self, image_rgb_uint8: np.ndarray, messages: List[Message], tools: List[VlmToolSchema]) -> str:
        self._step += 1
        if self._test_plan:
            # 1) 看图 2) 创建 task 1 3) 创建 task 2 依赖 1 4) done
            if self._step == 1:
                return '{"tool_name":"get_topview_image","parameters":{}}'
            if self._step == 2:
                return '{"tool_name":"task_create","parameters":{"subject":"Scan scene","description":"Look at topview and identify boxes/objects"}}'
            if self._step == 3:
                return '{"tool_name":"task_create","parameters":{"subject":"Plan moves","description":"Create a high-level plan","blockedBy":[1]}}'
            return '{"done":true,"summary":"dummy plan finished"}'
        if self._test_describe:
            if self._step == 1:
                return '{"tool_name":"get_topview_image","parameters":{}}'
            if self._step == 2:
                return '{"scene_description":"[dummy] 顶视图中可见工作台、原料箱与订单箱等区域。"}'
            return '{"done":true,"summary":"dummy 测试结束","scene_description":"[dummy] 结束。"}'
        if self._step == 1:
            return '{"tool_name":"get_topview_image","parameters":{}}'
        return '{"done":true,"summary":"dummy finished"}'

