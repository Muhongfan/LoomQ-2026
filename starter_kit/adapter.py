#!/usr/bin/env python3
"""LoomQ submission adapter contract v1.0."""

from typing import Any, Dict, List, Tuple

try:
    from . import runner as _runner
    from .circuit_ir import parse_qasm2
    from .emitters import EMITTERS
    from .lowering import lower
    from .validator import validate_circuit
    from . import l2_agent
    from . import hybrid_compiler
except ImportError:
    import runner as _runner
    from circuit_ir import parse_qasm2
    from emitters import EMITTERS
    from lowering import lower
    from validator import validate_circuit
    import l2_agent
    import hybrid_compiler


SUPPORTED_TARGETS = ("spinq", "originq", "braket")


def _check_target(target: str) -> None:
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"unsupported target {target!r}; expected one of {SUPPORTED_TARGETS}")


def transpile(qasm_str: str, target: str) -> str:
    """Translate OpenQASM 2.0 into the target backend's native representation."""
    _check_target(target)
    circuit = parse_qasm2(qasm_str)
    validate_circuit(circuit)
    lowered = lower(circuit, target)
    return EMITTERS[target](lowered)


def run(qasm_str: str, target: str, shots: int) -> Dict[str, Any]:
    """Execute a circuit and return the unified result schema from the rules."""
    _check_target(target)
    circuit = parse_qasm2(qasm_str)
    return _runner.run(circuit, target, shots)


def agent_chat(prompt: str) -> str:
    """Optional L2 entry point using the documented LOOMQ_LLM_* environment."""
    return l2_agent.agent_chat(prompt)


def compile_hybrid(hybrid_qasm_str: str) -> Tuple[List[str], str]:
    """Optional L3 entry point. Return quantum operations and RISC-V assembly."""
    return hybrid_compiler.compile_hybrid(hybrid_qasm_str)
