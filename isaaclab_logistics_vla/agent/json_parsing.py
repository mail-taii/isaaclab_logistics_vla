from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ParsedVlmOutput:
    ok: bool
    raw_text: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


_JSON_FENCE_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _try_json_load(s: str) -> Optional[Dict[str, Any]]:
    try:
        x = json.loads(s)
        return x if isinstance(x, dict) else None
    except Exception:
        return None


def parse_json_object_loose(text: str) -> ParsedVlmOutput:
    """尽量从任意文本中解析出一个 JSON object。

    依次尝试：
    - 整段就是 JSON
    - ```json ...``` 代码块
    - 从文本中找第一个 {...} 匹配（非严格，贪婪度有限）
    """
    raw = text
    t = text.strip()
    if not t:
        return ParsedVlmOutput(ok=False, raw_text=raw, error="empty_output")

    d = _try_json_load(t)
    if d is not None:
        return ParsedVlmOutput(ok=True, raw_text=raw, data=d)

    m = _JSON_FENCE_RE.search(t)
    if m:
        d = _try_json_load(m.group(1).strip())
        if d is not None:
            return ParsedVlmOutput(ok=True, raw_text=raw, data=d)

    # 兜底：找第一个大括号块（尽量短）
    # 注意：这里不做完美的括号匹配，只做最小闭环的容错。
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        d = _try_json_load(m.group(0))
        if d is not None:
            return ParsedVlmOutput(ok=True, raw_text=raw, data=d)

    return ParsedVlmOutput(ok=False, raw_text=raw, error="json_parse_failed")

