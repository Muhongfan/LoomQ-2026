"""Gate decompositions from gate_identities.md, as functions on Circuit IR ops.

Each function takes one GateOp for a whitelist gate and returns the equivalent
sequence using more elementary gates. These are the fallback path lowering.py
reaches for only if a target's supported-gate set doesn't cover something
directly.
"""

import math
from typing import List

from .circuit_ir import GateOp


def decompose_s(gate: GateOp) -> List[GateOp]:
    (q,) = gate.qubits
    return [GateOp("u1", [q], [math.pi / 2])]


def decompose_sdg(gate: GateOp) -> List[GateOp]:
    (q,) = gate.qubits
    return [GateOp("u1", [q], [-math.pi / 2])]


def decompose_t(gate: GateOp) -> List[GateOp]:
    (q,) = gate.qubits
    return [GateOp("u1", [q], [math.pi / 4])]


def decompose_tdg(gate: GateOp) -> List[GateOp]:
    (q,) = gate.qubits
    return [GateOp("u1", [q], [-math.pi / 4])]


def decompose_swap(gate: GateOp) -> List[GateOp]:
    a, b = gate.qubits
    return [GateOp("cx", [a, b]), GateOp("cx", [b, a]), GateOp("cx", [a, b])]


def decompose_cu1(gate: GateOp) -> List[GateOp]:
    a, b = gate.qubits
    (theta,) = gate.params
    return [
        GateOp("u1", [a], [theta / 2]),
        GateOp("cx", [a, b]),
        GateOp("u1", [b], [-theta / 2]),
        GateOp("cx", [a, b]),
        GateOp("u1", [b], [theta / 2]),
    ]


def decompose_ccx(gate: GateOp) -> List[GateOp]:
    a, b, c = gate.qubits
    return [
        GateOp("h", [c]),
        GateOp("cx", [b, c]),
        GateOp("tdg", [c]),
        GateOp("cx", [a, c]),
        GateOp("t", [c]),
        GateOp("cx", [b, c]),
        GateOp("tdg", [c]),
        GateOp("cx", [a, c]),
        GateOp("t", [b]),
        GateOp("t", [c]),
        GateOp("h", [c]),
        GateOp("cx", [a, b]),
        GateOp("t", [a]),
        GateOp("tdg", [b]),
        GateOp("cx", [a, b]),
    ]


def decompose_ry_fallback(gate: GateOp) -> List[GateOp]:
    """Last-resort ry decomposition; ry is natively supported almost everywhere."""
    (q,) = gate.qubits
    (theta,) = gate.params
    return [
        GateOp("sdg", [q]),
        GateOp("h", [q]),
        GateOp("rz", [q], [theta]),
        GateOp("h", [q]),
        GateOp("s", [q]),
    ]


DECOMPOSITIONS = {
    "s": decompose_s,
    "sdg": decompose_sdg,
    "t": decompose_t,
    "tdg": decompose_tdg,
    "swap": decompose_swap,
    "cu1": decompose_cu1,
    "ccx": decompose_ccx,
    "ry": decompose_ry_fallback,
}
