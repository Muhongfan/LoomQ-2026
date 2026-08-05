"""L2 agent core logic: unified prompt, QASM extraction, timeout budgeting,
and a self-verification retry loop for generation/correction tasks.

adapter.py::agent_chat delegates here (Phase 4 of the L2 roadmap; not wired
yet — see ROADMAP.md).
"""

import contextlib
import json
import math
import os
import re
import tempfile
import time
from typing import Dict, List, Optional, Tuple

try:
    from . import llm_client
    from .circuit_ir import Circuit, parse_qasm2
    from .validator import validate_circuit
except ImportError:
    import llm_client
    from circuit_ir import Circuit, parse_qasm2
    from validator import validate_circuit

_QASM_RE = re.compile(r"OPENQASM\s+2\.0;.*?(?=^\s*```|\Z)", re.DOTALL | re.MULTILINE)


def extract_qasm(text: str) -> Optional[str]:
    """Mirrors evaluator.py's own extract_qasm exactly (duplicated, not
    imported, to avoid a circular import: evaluator.py itself imports
    adapter, which will import this module)."""
    if not isinstance(text, str):
        return None
    match = _QASM_RE.search(text)
    return match.group(0).strip() if match else None


def _hellinger_fidelity(observed: Dict[str, float], expected: Dict[str, float]) -> float:
    """Duplicates evaluator.py::calculate_hellinger_fidelity exactly, for
    the same circular-import reason as extract_qasm above."""
    states = set(observed) | set(expected)
    distance = math.sqrt(
        sum(
            (math.sqrt(observed.get(s, 0.0)) - math.sqrt(expected.get(s, 0.0))) ** 2
            for s in states
        )
    ) / math.sqrt(2.0)
    return max(0.0, min(1.0, 1.0 - distance))


def _execute_via_spinqit(qasm_text: str, shots: int) -> Dict[str, int]:
    """Small self-contained SpinQ execution for internal self-verification
    only — not the graded run() path (runner.py isn't on this branch; this
    doesn't need target-specific emission or bit-order normalization,
    just "does this circuit look like the right state" before returning)."""
    from spinqit import BasicSimulatorConfig, get_basic_simulator, get_compiler

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".qasm", delete=False, encoding="utf-8")
    try:
        tmp.write(qasm_text)
        tmp.close()
        ir = get_compiler("qasm").compile(tmp.name, 0)
        engine = get_basic_simulator()
        config = BasicSimulatorConfig()
        config.configure_shots(shots)
        result = engine.execute(ir, config)
        return {str(key): int(value) for key, value in result.counts.items()}
    finally:
        os.unlink(tmp.name)


def _ghz_ideal(n: int) -> Dict[str, float]:
    """Bell state is the n=2 member of this same family: (|00..0>+|11..1>)/sqrt(2)."""
    return {"0" * n: 0.5, "1" * n: 0.5}


def _uniform_ideal(n: int) -> Dict[str, float]:
    size = 2**n
    return {format(i, f"0{n}b"): 1.0 / size for i in range(size)}


def _infer_ideal_distribution(prompt: str, n_clbits: int) -> Optional[Dict[str, float]]:
    """Recognizes a bounded set of nameable target-state families from the
    ORIGINAL user prompt (not the generated circuit) — deliberately narrow:
    Bell/GHZ-n, uniform superposition-n, and an explicit computational basis
    state like "|101>". QFT-n/Grover-n are excluded on purpose: QFT measured
    right after only a forward transform is always uniform regardless of
    whether cu1's phases are correct (a structural DFT property, see
    ROADMAP.md Phase 7), and Grover's ideal distribution depends on the
    marked state and iteration count, not something inferable from a
    one-line prompt. Circuits we can't recognize a family for still get
    syntax/whitelist validation — this is an honest capability boundary,
    not a shortcut.
    """
    lowered = prompt.lower()
    if "ghz" in lowered or "最大纠缠" in prompt or "GHZ" in prompt:
        return _ghz_ideal(n_clbits)
    if "bell" in lowered or "贝尔" in prompt:
        return _ghz_ideal(n_clbits)
    if "uniform superposition" in lowered or "均匀叠加" in prompt:
        return _uniform_ideal(n_clbits)
    basis_match = re.search(r"[|｜]([01]+)[⟩>》]", prompt)
    if basis_match and len(basis_match.group(1)) == n_clbits:
        return {basis_match.group(1): 1.0}
    return None


def _verify_generated_circuit(prompt: str, qasm: str) -> Tuple[bool, str]:
    try:
        circuit = parse_qasm2(qasm)
        validate_circuit(circuit)
    except ValueError as exc:
        return False, f"电路解析或校验失败：{exc}"

    ideal = _infer_ideal_distribution(prompt, circuit.n_clbits)
    if ideal is None:
        return True, ""  # can't infer a target family; syntax validity is all we can check

    try:
        raw_counts = _execute_via_spinqit(qasm, shots=2048)
    except Exception as exc:  # noqa: BLE001 - any execution failure means verification fails
        return False, f"电路无法在模拟器上执行：{type(exc).__name__}: {exc}"

    total = sum(raw_counts.values())
    observed = {key: value / total for key, value in raw_counts.items()}
    fidelity = _hellinger_fidelity(observed, ideal)
    if fidelity < 0.97:
        return False, f"保真度不足（{fidelity:.3f} < 0.97），电路未能实现目标态"
    return True, ""


def _backend_capabilities_text() -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend_capabilities.json")
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _load_backend_capabilities() -> List[Dict]:
    return json.loads(_backend_capabilities_text())["backends"]


_BACKEND_ID_RE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")

_QUBIT_COUNT_RE = re.compile(r"(\d+)\s*(?:个)?\s*(?:量子)?比特|(\d+)\s*[- ]?qubits?", re.IGNORECASE)
_ZERO_QUEUE_RE = re.compile(r"零排队|不(?:想|要)?排队|无需排队|no\s*queue|zero\s*queue", re.IGNORECASE)
_REAL_HARDWARE_RE = re.compile(
    r"真机|真实(?:的)?(?:量子)?硬件|真实量子计算机|real\s*hardware|physical\s*qpu|actual\s*quantum\s*computer",
    re.IGNORECASE,
)
_COST_SENSITIVE_RE = re.compile(
    r"不想花钱|不花钱|零成本|免费|free\b|no\s*cost|without\s*paying", re.IGNORECASE
)


def _extract_backend_constraints(prompt: str) -> Dict[str, object]:
    """Deterministic (regex/keyword) extraction of the constraint dimensions
    backend_capabilities.json itself declares as the judging basis. Only a
    safety net over the LLM's own answer (see _apply_backend_selection_
    safety_net) — the LLM still does the primary natural-language
    understanding of the (possibly reworded) user request."""
    constraints: Dict[str, object] = {}
    qubit_match = _QUBIT_COUNT_RE.search(prompt)
    if qubit_match:
        constraints["min_qubits"] = int(qubit_match.group(1) or qubit_match.group(2))
    if _ZERO_QUEUE_RE.search(prompt):
        constraints["queue_none"] = True
    if _REAL_HARDWARE_RE.search(prompt):
        constraints["require_qpu"] = True
    if _COST_SENSITIVE_RE.search(prompt):
        constraints["cost_sensitive"] = True
    return constraints


def _matching_backends(constraints: Dict[str, object], backends: List[Dict]) -> List[Dict]:
    candidates = []
    for backend in backends:
        if "min_qubits" in constraints and backend["max_qubits"] < constraints["min_qubits"]:
            continue
        if constraints.get("queue_none") and backend["queue"] != "none":
            continue
        if constraints.get("require_qpu") and backend["kind"] != "qpu":
            continue
        if constraints.get("cost_sensitive") and backend["cost"] == "paid":
            continue
        candidates.append(backend)
    return candidates


def _apply_backend_selection_safety_net(prompt: str, reply: str) -> str:
    constraints = _extract_backend_constraints(prompt)
    if not constraints:
        return reply  # nothing extractable to verify against; trust the LLM

    backends = _load_backend_capabilities()
    candidates = _matching_backends(constraints, backends)
    valid_ids = {b["id"] for b in candidates}
    mentioned_ids = set(_BACKEND_ID_RE.findall(reply))

    if candidates:
        if mentioned_ids & valid_ids:
            return reply  # LLM already gave a satisfying id; leave it alone
        choice = candidates[0]
        return (
            f"推荐使用 `{choice['id']}`（{choice['name']}），"
            f"它满足你提出的约束条件（最多支持 {choice['max_qubits']} 比特，"
            f"排队情况：{choice['queue']}，费用：{choice['cost']}）。"
        )

    largest = max(backends, key=lambda b: b["max_qubits"])
    return (
        "抱歉，目前没有任何后端能同时满足你提出的全部约束条件。"
        f"最接近的替代方案是 `{largest['id']}`（{largest['name']}，"
        f"最多支持 {largest['max_qubits']} 比特），但可能需要放宽部分约束"
        "（例如比特数、排队时间或费用要求），或将电路拆分后分别运行。"
    )


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

# "生成 QASM -> 用自己的 L1 跑一遍自验 -> 不对就重试" per problem_statement.md.
MAX_GENERATION_ATTEMPTS = 3


def agent_chat(prompt: str) -> str:
    system_prompt = SYSTEM_PROMPT.format(backend_table=_backend_capabilities_text())
    budget = TimeBudget(DEFAULT_CASE_BUDGET_SECONDS)
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    reply = ""
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        remaining_attempts = MAX_GENERATION_ATTEMPTS - attempt - 1
        with _temporary_timeout(budget.next_call_timeout(reserved_for_more_calls=remaining_attempts)):
            response = llm_client.chat_completion(conversation)
        reply = response["choices"][0]["message"]["content"]
        conversation.append({"role": "assistant", "content": reply})

        qasm = extract_qasm(reply)
        if qasm is None:
            # No QASM present: treat this as a backend-selection-style reply.
            # Trust the LLM's natural-language understanding as the primary
            # path; the deterministic table lookup only overrides it when
            # extractable constraints show the LLM's answer doesn't satisfy
            # backend_capabilities.json (the organizers' own stated ground
            # truth for this task).
            return _apply_backend_selection_safety_net(prompt, reply)

        ok, feedback = _verify_generated_circuit(prompt, qasm)
        if ok or remaining_attempts == 0 or budget.remaining() <= 0:
            return reply

        conversation.append(
            {
                "role": "user",
                "content": (
                    f"这段电路没有通过校验：{feedback}。"
                    "请在保持用户原始目标不变的前提下修复电路，"
                    "仍然只输出一个完整的 OpenQASM 2.0 代码块。"
                ),
            }
        )

    return reply
