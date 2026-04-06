"""VLM-based agent framework (tool-use loop).

当前最小可用能力：
- 仅提供一个 tool：从场景顶视相机获取一张 RGB 图片（`get_topview_image`）。
"""

from .vlm_backend import VlmBackend, DummyBackend  # noqa: F401
from .runner import VlmToolUseRunner  # noqa: F401
from .anthropic_volc_backend import VolcArkAnthropicBackend  # noqa: F401
from .backend_registry import create_vlm_backend  # noqa: F401
from .instruction_provider import InstructionProvider, InstructionSpec, VirtualInstructionProvider  # noqa: F401
from .task_system import TaskManager, Task  # noqa: F401

