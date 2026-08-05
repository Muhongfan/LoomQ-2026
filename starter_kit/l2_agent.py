"""L2 agent core logic: unified prompt, QASM extraction, timeout budgeting.

adapter.py::agent_chat delegates here (Phase 4 of the L2 roadmap; not wired
yet — see ROADMAP.md). Phase 1 scope: a real LLM call through a single
unified prompt covering all three graded task shapes (generation,
correction, backend selection), with wall-clock budget tracking so a later
self-verification retry loop (Phase 2) has room to make more than one call
within the 120-second per-case limit. No retry loop or backend-selection
safety net yet — those are Phase 2/3.
"""

import contextlib
import json
import os
import re
import time
from typing import Optional

try:
    from . import llm_client
except ImportError:
    import llm_client

_QASM_RE = re.compile(r"OPENQASM\s+2\.0;.*?(?=^\s*```|\Z)", re.DOTALL | re.MULTILINE)


def extract_qasm(text: str) -> Optional[str]:
    """Mirrors evaluator.py's own extract_qasm exactly (duplicated, not
    imported, to avoid a circular import: evaluator.py itself imports
    adapter, which will import this module)."""
    if not isinstance(text, str):
        return None
    match = _QASM_RE.search(text)
    return match.group(0).strip() if match else None


def _backend_capabilities_text() -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend_capabilities.json")
    with open(path, encoding="utf-8") as handle:
        return handle.read()


SYSTEM_PROMPT = """你是 LoomQ 平台的智能体，帮助不熟悉量子计算的用户使用量子计算机。

用户的请求属于以下三类之一，请根据内容自行判断属于哪一类并按对应格式回复：

## 1. 生成电路 / 2. 修复电路
用户会描述一个目标量子态或需求，或提供一段有问题的代码要求修复。
两种情况都请输出一段完整、可执行的 OpenQASM 2.0 代码：
- 只能使用以下 12 个门：h, x, s, sdg, t, tdg, rz(theta), ry(theta), cx, cu1(theta), swap, ccx
- 必须包含 `include "qelib1.inc";`、`qreg`/`creg` 声明、以及 `measure`
- 如果是修复任务，必须保持用户声明的目标态不变，只修正语法或语义错误
- 代码必须放在一个 ``` 代码块中，且代码块内容**严格以 `OPENQASM 2.0;` 开头**（前面不要加任何说明文字），例如：

```
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
```

代码块之外可以用一两句话说明你做了什么。

## 3. 选择后端
用户会描述比特数、排队、真机、费用等约束。请依据下面这份官方后端能力表，选出**唯一**
满足用户所有约束的后端（如果有多个都满足，选其中一个即可），在回复中明确包含它的
`id` 字段值（例如 `spinq_taurus_simulator`），并用一两句话说明理由。如果没有任何
后端满足全部约束，如实说明，并给出最接近的替代方案。

后端能力表（JSON）：
{backend_table}
"""


class TimeBudget:
    """Tracks wall-clock time remaining within a fixed total budget, so a
    caller planning several sequential LLM calls (e.g. a self-verification
    retry loop) can divide what's left instead of any one call consuming
    the entire per-case allowance."""

    def __init__(self, total_seconds: float):
        self._deadline = time.monotonic() + total_seconds

    def remaining(self) -> float:
        return max(0.0, self._deadline - time.monotonic())

    def next_call_timeout(self, reserved_for_more_calls: int = 0, min_timeout: float = 5.0) -> float:
        divisor = reserved_for_more_calls + 1
        return max(min_timeout, self.remaining() / divisor)


@contextlib.contextmanager
def _temporary_timeout(seconds: float):
    """llm_client.chat_completion() has no per-call timeout parameter — it
    re-reads LOOMQ_LLM_TIMEOUT_SECONDS from the environment on every call.
    Overriding it for the duration of one call (single-threaded, synchronous
    here) lets TimeBudget actually shrink each call's timeout instead of
    every call using the same fixed value from the environment."""
    key = "LOOMQ_LLM_TIMEOUT_SECONDS"
    previous = os.environ.get(key)
    os.environ[key] = repr(seconds)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


# Stay comfortably under l2_policy.json's 120-second per-case limit so a
# slow network round trip doesn't blow through the grading timeout outright.
DEFAULT_CASE_BUDGET_SECONDS = 100.0


def agent_chat(prompt: str) -> str:
    system_prompt = SYSTEM_PROMPT.format(backend_table=_backend_capabilities_text())
    budget = TimeBudget(DEFAULT_CASE_BUDGET_SECONDS)
    with _temporary_timeout(budget.next_call_timeout()):
        response = llm_client.chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
    return response["choices"][0]["message"]["content"]
