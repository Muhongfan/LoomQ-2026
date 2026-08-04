"""Per-target gate lowering.

tools/gate_audit.py confirmed all 12 whitelist gates run natively on SpinQ
(via its bundled qelib1.inc) and on Braket (via the 5 renames handled in its
emitter). target_ir_contract.md's OriginIR gate list also names all 12
directly. Lowering is therefore a documented no-op for every target today;
it stays in the pipeline as the fallback path if that ever changes (a hidden
circuit edge case, or OriginQ turning out to differ once its API token
arrives and it gets audited).
"""

from dataclasses import replace
from typing import Dict, FrozenSet, List

from .circuit_ir import Circuit, GateOp
from .gate_identities import DECOMPOSITIONS
from .validator import WHITELIST

SUPPORTED_GATES: Dict[str, FrozenSet[str]] = {
    "spinq": WHITELIST,
    "braket": WHITELIST,
    "originq": WHITELIST,
}


def lower(circuit: Circuit, target: str) -> Circuit:
    supported = SUPPORTED_GATES[target]
    lowered_gates: List[GateOp] = []
    for gate in circuit.gates:
        lowered_gates.extend(_expand(gate, supported))
    return replace(circuit, gates=lowered_gates)


def _expand(gate: GateOp, supported: FrozenSet[str]) -> List[GateOp]:
    if gate.name in supported:
        return [gate]
    if gate.name not in DECOMPOSITIONS:
        raise ValueError(f"no known decomposition for unsupported gate {gate.name!r}")
    expanded: List[GateOp] = []
    for sub_gate in DECOMPOSITIONS[gate.name](gate):
        expanded.extend(_expand(sub_gate, supported))
    return expanded
