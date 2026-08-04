"""Minimal OpenQASM 2.0 parser producing a flat Circuit IR.

Only the subset needed for the LoomQ gate whitelist (h, x, s, sdg, t, tdg,
rz, ry, cx, cu1, swap, ccx) plus qreg/creg/measure/barrier is supported.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class GateOp:
    name: str
    qubits: List[int]
    params: List[float] = field(default_factory=list)


@dataclass
class Circuit:
    n_qubits: int
    n_clbits: int
    qregs: Dict[str, Tuple[int, int]]  # name -> (offset, size)
    cregs: Dict[str, Tuple[int, int]]
    gates: List[GateOp]
    measurements: List[Tuple[int, int]]  # (qubit_index, clbit_index)


_PARAM_RE = re.compile(r"[0-9eE.+\-*/() \t]+")


def _eval_param(expr: str) -> float:
    expr = expr.strip()
    substituted = re.sub(r"\bpi\b", repr(math.pi), expr)
    if not _PARAM_RE.fullmatch(substituted):
        raise ValueError(f"unsupported parameter expression: {expr!r}")
    return float(eval(substituted, {"__builtins__": {}}, {}))


_QUBIT_REF_RE = re.compile(r"^(\w+)(?:\[(\d+)\])?$")


def _resolve_ref(token: str, registers: Dict[str, Tuple[int, int]]) -> Tuple[str, int, bool]:
    """Returns (register_name, index_or_-1, is_whole_register)."""
    match = _QUBIT_REF_RE.match(token.strip())
    if not match:
        raise ValueError(f"malformed register reference: {token!r}")
    name, index = match.group(1), match.group(2)
    if name not in registers:
        raise ValueError(f"reference to undeclared register: {name!r}")
    if index is None:
        return name, -1, True
    return name, int(index), False


def _expand_broadcast(
    args: List[str],
    registers: Dict[str, Tuple[int, int]],
) -> List[List[int]]:
    """Resolves each arg to a global index, broadcasting whole-register args."""
    parsed = [_resolve_ref(arg, registers) for arg in args]
    whole_sizes = {registers[name][1] for name, _, whole in parsed if whole}
    if not whole_sizes:
        return [[registers[name][0] + index for name, index, _ in parsed]]
    if len(whole_sizes) > 1:
        raise ValueError("broadcast register size mismatch: " + str(args))
    n = whole_sizes.pop()
    rows = []
    for k in range(n):
        row = []
        for name, index, whole in parsed:
            offset, size = registers[name]
            row.append(offset + (k if whole else index))
        rows.append(row)
    return rows


_GATE_LINE_RE = re.compile(r"^(\w+)\s*(?:\(([^)]*)\))?\s+(.+)$")


def parse_qasm2(qasm_str: str) -> Circuit:
    text = re.sub(r"//.*", "", qasm_str)
    statements = [s.strip() for s in text.split(";")]

    qregs: Dict[str, Tuple[int, int]] = {}
    cregs: Dict[str, Tuple[int, int]] = {}
    gates: List[GateOp] = []
    measurements: List[Tuple[int, int]] = []
    n_qubits = 0
    n_clbits = 0

    for stmt in statements:
        if not stmt:
            continue
        if stmt.startswith("OPENQASM"):
            if not re.match(r"OPENQASM\s+2\.0$", stmt):
                raise ValueError(f"unsupported QASM version declaration: {stmt!r}")
            continue
        if stmt.startswith("include"):
            continue
        if stmt.startswith("barrier"):
            continue

        m = re.match(r"^qreg\s+(\w+)\s*\[\s*(\d+)\s*\]$", stmt)
        if m:
            name, size = m.group(1), int(m.group(2))
            qregs[name] = (n_qubits, size)
            n_qubits += size
            continue

        m = re.match(r"^creg\s+(\w+)\s*\[\s*(\d+)\s*\]$", stmt)
        if m:
            name, size = m.group(1), int(m.group(2))
            cregs[name] = (n_clbits, size)
            n_clbits += size
            continue

        m = re.match(r"^measure\s+(.+?)\s*->\s*(.+)$", stmt)
        if m:
            q_token, c_token = m.group(1).strip(), m.group(2).strip()
            q_rows = _expand_broadcast([q_token], qregs)
            c_rows = _expand_broadcast([c_token], cregs)
            if len(q_rows) != len(c_rows):
                raise ValueError(f"measure register size mismatch: {stmt!r}")
            for (qidx,), (cidx,) in zip(q_rows, c_rows):
                measurements.append((qidx, cidx))
            continue

        m = _GATE_LINE_RE.match(stmt)
        if not m:
            raise ValueError(f"unrecognized statement: {stmt!r}")
        gate_name, param_str, arg_str = m.group(1), m.group(2), m.group(3)
        params = (
            [_eval_param(p) for p in param_str.split(",")] if param_str else []
        )
        args = [a.strip() for a in arg_str.split(",")]
        combined = dict(qregs)
        for row in _expand_broadcast(args, combined):
            gates.append(GateOp(name=gate_name, qubits=row, params=params))

    return Circuit(
        n_qubits=n_qubits,
        n_clbits=n_clbits,
        qregs=qregs,
        cregs=cregs,
        gates=gates,
        measurements=measurements,
    )


if __name__ == "__main__":
    import json
    import os
    from dataclasses import asdict

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "circuits")
    for filename in ("bell.qasm", "ghz3.qasm"):
        with open(os.path.join(base, filename), encoding="utf-8") as handle:
            circuit = parse_qasm2(handle.read())
        print(f"--- {filename} ---")
        print(json.dumps(asdict(circuit), indent=2))
