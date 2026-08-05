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
_TASK_TAG_RE = re.compile(r"^\s*\[TASK:\s*(generate|fix|select_backend|clarify)\]", re.IGNORECASE)


def _classify_task_tag(reply: str) -> Optional[str]:
    """Reads the explicit [TASK: ...] tag the SYSTEM_PROMPT requires as the
    first line of every reply. Returns None if the model didn't include a
    recognizable tag (older/malformed replies) — callers should fall back
    to inferring from content in that case, not treat it as an error, since
    a model occasionally not following the tag instruction shouldn't break
    the whole turn."""
    match = _TASK_TAG_RE.match(reply)
    return match.group(1).lower() if match else None


def _extract_reply_content(response: Dict) -> str:
    """Defensive parsing of the chat-completion response shape. Without
    this, a malformed/unexpected response (empty choices from a content
    filter, a provider returning a different shape, etc.) surfaces as a
    bare IndexError/KeyError deep in agent_chat with no indication of what
    actually went wrong."""
    if not isinstance(response, dict):
        raise RuntimeError(f"LLM response was not a JSON object: {type(response).__name__}")
    choices = response.get("choices")
    if not choices:
        raise RuntimeError(f"LLM response contained no choices: {response!r}")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict) or "content" not in message:
        raise RuntimeError(f"LLM response choice has no message content: {choices[0]!r}")
    content = message["content"]
    if not isinstance(content, str):
        raise RuntimeError(f"LLM response content was not a string: {type(content).__name__}")
    return content


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


_TARGET_TAG_RE = re.compile(r"\[TARGET:\s*(ghz|bell|uniform|basis:[01]+|other)\]", re.IGNORECASE)


def _classify_target_tag(reply: str) -> Optional[str]:
    """Reads the model's own [TARGET: ...] self-report (required by
    SYSTEM_PROMPT for generate/fix replies). Preferred over regexing the
    raw prompt: the model has already done the natural-language work of
    understanding a possibly reworded request, so trusting its own
    classification is more robust than our own keyword matching."""
    match = _TARGET_TAG_RE.search(reply)
    return match.group(1).lower() if match else None


def _infer_ideal_distribution_from_prompt(
    prompt: str, n_clbits: int
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """Fallback for when the model didn't include a [TARGET: ...] tag
    (older/malformed replies) — regexes the ORIGINAL user prompt instead.
    Deliberately narrow: Bell/GHZ-n, uniform superposition-n, and an
    explicit computational basis state like "|101>". QFT-n/Grover-n are
    excluded on purpose: QFT measured right after only a forward transform
    is always uniform regardless of whether cu1's phases are correct (a
    structural DFT property, see ROADMAP.md Phase 7), and Grover's ideal
    distribution depends on the marked state and iteration count, not
    something inferable from a one-line prompt.
    """
    lowered = prompt.lower()
    if "ghz" in lowered or "最大纠缠" in prompt or "GHZ" in prompt:
        return _ghz_ideal(n_clbits), "ghz"
    if "bell" in lowered or "贝尔" in prompt:
        return _ghz_ideal(n_clbits), "ghz"
    if "uniform superposition" in lowered or "均匀叠加" in prompt:
        return _uniform_ideal(n_clbits), "uniform"
    basis_match = re.search(r"[|｜]([01]+)[⟩>》]", prompt)
    if basis_match and len(basis_match.group(1)) == n_clbits:
        return {basis_match.group(1): 1.0}, "basis"
    return None, None


def _recognize_target(
    prompt: str, reply: str, n_clbits: int
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """Returns (ideal_distribution, family_label) if a target family the
    self-verifier knows how to compute a closed-form distribution for is
    recognized, else (None, None). Circuits we can't recognize a family for
    still get syntax/whitelist validation plus the structural sanity checks
    in _structural_sanity_check — an honest capability boundary, not a
    silent pass-through."""
    tag = _classify_target_tag(reply)
    if tag and tag != "other":
        if tag in ("ghz", "bell"):
            return _ghz_ideal(n_clbits), "ghz"
        if tag == "uniform":
            return _uniform_ideal(n_clbits), "uniform"
        if tag.startswith("basis:"):
            bits = tag.split(":", 1)[1]
            if len(bits) == n_clbits:
                return {bits: 1.0}, "basis"
        # tag present but doesn't line up with the circuit's actual clbit
        # count (e.g. basis:101 tagged on a 2-clbit circuit) -- fall through
        # to the prompt-based fallback rather than silently trusting a tag
        # that can't possibly be right.
    return _infer_ideal_distribution_from_prompt(prompt, n_clbits)


_ENTANGLING_GATES = frozenset({"cx", "cu1", "swap", "ccx"})


def _structural_sanity_check(prompt: str, circuit: Circuit) -> Optional[str]:
    """Cheap checks that don't require knowing the exact ideal distribution
    — catch obvious mismatches instead of silently passing anything
    syntactically valid when the target family isn't one we can compute a
    closed-form distribution for."""
    qubit_match = _QUBIT_COUNT_RE.search(prompt)
    if qubit_match:
        stated = int(qubit_match.group(1) or qubit_match.group(2))
        if circuit.n_qubits != stated:
            return f"你要求 {stated} 个比特，但电路声明了 {circuit.n_qubits} 个量子比特"

    if re.search(r"纠缠|entangle", prompt, re.IGNORECASE):
        if not any(gate.name in _ENTANGLING_GATES for gate in circuit.gates):
            return "描述中提到了纠缠，但电路里没有任何两比特门（cx/cu1/swap/ccx），无法产生纠缠"

    return None


_FAMILY_HINTS = {
    "ghz": "请检查是否漏加了纠缠门（如 cx），或纠缠链是否覆盖了全部比特。",
    "uniform": "请检查是否给每一个比特都加上了 h 门。",
    "basis": "请检查每个比特是否用 x 门正确翻转到了目标值。",
}


def _format_distribution(distribution: Dict[str, float], limit: int = 6) -> str:
    ordered = sorted(distribution.items(), key=lambda kv: -kv[1])[:limit]
    parts = [f"{key}: {value:.1%}" for key, value in ordered]
    suffix = ", ..." if len(distribution) > limit else ""
    return "{" + ", ".join(parts) + suffix + "}"


def _fidelity_failure_feedback(
    family: Optional[str], observed: Dict[str, float], ideal: Dict[str, float], fidelity: float
) -> str:
    hint = _FAMILY_HINTS.get(family, "")
    return (
        f"保真度不足（{fidelity:.3f} < 0.97）。"
        f"你的电路实际测量分布约为 {_format_distribution(observed)}，"
        f"但目标分布应为 {_format_distribution(ideal)}。"
        f"{hint}"
    )


def _verify_generated_circuit(prompt: str, reply: str, qasm: str) -> Tuple[bool, str]:
    try:
        circuit = parse_qasm2(qasm)
        validate_circuit(circuit)
    except ValueError as exc:
        return False, f"电路解析或校验失败：{exc}"

    ideal, family = _recognize_target(prompt, reply, circuit.n_clbits)
    if ideal is None:
        structural_issue = _structural_sanity_check(prompt, circuit)
        if structural_issue:
            return False, structural_issue
        return True, ""  # can't infer an exact target family; structural checks passed

    try:
        raw_counts = _execute_via_spinqit(qasm, shots=2048)
    except Exception as exc:  # noqa: BLE001 - any execution failure means verification fails
        return False, f"电路无法在模拟器上执行：{type(exc).__name__}: {exc}"

    total = sum(raw_counts.values())
    observed = {key: value / total for key, value in raw_counts.items()}
    fidelity = _hellinger_fidelity(observed, ideal)
    if fidelity < 0.97:
        return False, _fidelity_failure_feedback(family, observed, ideal, fidelity)
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


_NO_FIT_ACKNOWLEDGED_RE = re.compile(
    r"没有.*?(?:后端|平台).*?满足|无法(?:同时)?满足|不满足(?:全部|所有)?约束|无解|"
    r"no\s+backend.*?satisf|doesn'?t\s+(?:fully\s+)?satisf|cannot\s+satisf",
    re.IGNORECASE,
)


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

    if _NO_FIT_ACKNOWLEDGED_RE.search(reply):
        # Empirically observed (real DeepSeek calls, not a hypothetical):
        # the model can correctly conclude no backend fits and explain why
        # in more detail than our own template -- only fall back to the
        # generic honest-answer template when it hasn't already said this,
        # rather than unconditionally discarding a correct, more useful
        # explanation.
        return reply

    largest = max(backends, key=lambda b: b["max_qubits"])
    return (
        "抱歉，目前没有任何后端能同时满足你提出的全部约束条件。"
        f"最接近的替代方案是 `{largest['id']}`（{largest['name']}，"
        f"最多支持 {largest['max_qubits']} 比特），但可能需要放宽部分约束"
        "（例如比特数、排队时间或费用要求），或将电路拆分后分别运行。"
    )


SYSTEM_PROMPT = """你是 LoomQ 平台的智能体，帮助不熟悉量子计算的用户使用量子计算机。

用户的请求属于以下四类之一。**你的回复必须以下面四个标记之一开头，独占一行**，
帮助下游程序判断你识别出的任务类型（这不是给用户看的内容，只是一个路由标记）：

```
[TASK: generate]
[TASK: fix]
[TASK: select_backend]
[TASK: clarify]
```

**如果用户的一句话里同时包含了多种任务**（比如既要生成电路，又问该用哪个后端），
只能标记并完整回答其中最主要的一个（生成/修复电路优先于选后端），但**不要静默
忽略**另一部分请求——在正文末尾另起一段，明确说明"你的问题里还包含 XX 请求，
请单独再问一次，我可以专门回答"。

**如果用户的请求信息不足以生成或修复任何电路**（比如内容完全无关、没有提供要
修复的代码、也没有说明任何目标态），请使用 `[TASK: clarify]`，用一两句话说明
还需要用户补充什么信息，**不要**尝试勉强输出一个代码块，也不要因为找不到代码块
而反复重试——这种情况下你的判断就是最终答案。

标记之后，请根据内容按对应格式回复：

## [TASK: generate] 生成电路 / [TASK: fix] 修复电路
用户会描述一个目标量子态或需求，或提供一段有问题的代码要求修复。
在 `[TASK: ...]` 之后另起一行，加一个 `[TARGET: ...]` 标记，说明你识别出的目标
态类型（同样是路由标记，不是给用户看的内容）：

```
[TARGET: ghz]          GHZ 态或贝尔态（多比特最大纠缠态，全 0/全 1 各占一半概率）
[TARGET: uniform]      均匀叠加态（每个比特独立做 Hadamard，所有基态等概率）
[TARGET: basis:101]    制备一个确定的计算基态，冒号后是具体的比特串
[TARGET: other]        以上都不是
```

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

## [TASK: select_backend] 选择后端
用户会描述比特数、排队、真机、费用等约束。请依据下面这份官方后端能力表，选出**唯一**
满足用户所有约束的后端（如果有多个都满足，选其中一个即可），在回复中明确包含它的
`id` 字段值（例如 `spinq_taurus_simulator`），并用一两句话说明理由。如果没有任何
后端满足全部约束，如实说明，并给出最接近的替代方案。这类回复**不要**包含
OpenQASM 代码块。

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
        reply = _extract_reply_content(response)
        conversation.append({"role": "assistant", "content": reply})

        task = _classify_task_tag(reply)
        qasm = extract_qasm(reply)

        # task is the model's own explicit self-report of which of the three
        # task types it thinks this is (required as the reply's first line
        # by SYSTEM_PROMPT) -- trusted when present, since relying only on
        # "did we happen to find a QASM code block" misroutes a backend-
        # selection answer that happens to quote example code in its
        # explanation, and equally misroutes a generation answer that failed
        # to follow the code-block format (silently treated as if it were a
        # valid backend-selection reply instead of a failed attempt to fix).
        if task == "select_backend":
            return _apply_backend_selection_safety_net(prompt, reply)

        if task == "clarify":
            # The model has judged there isn't enough information to
            # generate/fix anything -- that judgment is the final answer.
            # Treating a missing QASM block here as "bad format, retry" (as
            # the generate/fix branch below does) would waste attempts
            # forcing the model to produce a circuit it has already said it
            # can't responsibly produce.
            return reply

        if task in ("generate", "fix"):
            if qasm is None:
                ok, feedback = False, "回复中没有找到符合格式要求的 OpenQASM 2.0 代码块"
            else:
                ok, feedback = _verify_generated_circuit(prompt, reply, qasm)
        elif qasm is not None:
            # No tag recognized (model didn't follow the instruction) but a
            # QASM block is present -- fall back to the original heuristic.
            ok, feedback = _verify_generated_circuit(prompt, reply, qasm)
        else:
            # No tag, no QASM -- fall back to the original heuristic: treat
            # as a backend-selection-style reply.
            return _apply_backend_selection_safety_net(prompt, reply)

        if ok or remaining_attempts == 0 or budget.remaining() <= 0:
            return reply

        conversation.append(
            {
                "role": "user",
                "content": (
                    f"这段电路没有通过校验：{feedback}。"
                    "请在保持用户原始目标不变的前提下修复电路，"
                    "仍然以 [TASK: generate] 或 [TASK: fix] 开头，"
                    "并只输出一个完整的 OpenQASM 2.0 代码块。"
                ),
            }
        )

    return reply
