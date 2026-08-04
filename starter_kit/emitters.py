"""Emit target-specific IR text from a validated/lowered Circuit IR.

Output formats must match starter_kit/target_ir_contract.md exactly, since
the organizers parse and simulate transpile()'s return value independently
of run()'s counts. Each emitter assumes its input Circuit already passed
validator.validate_circuit() and lowering.lower() for its target.
"""

from .circuit_ir import Circuit

# cu1 has no name in the official OpenQASM3 stdgates.inc (only a controlled-
# phase under a different name depending on interpreter). Rather than rely on
# any particular interpreter's native alias, the Braket emitter defines cu1
# itself inline, using the exact qelib1 decomposition from gate_identities.md
# expressed in terms of stdgates.inc primitives (cx, p) — self-contained and
# interpreter-agnostic.
_BRAKET_CU1_DEFINITION = (
    "gate cu1(lambda) a, b {\n"
    "  p(lambda/2) a;\n"
    "  cx a, b;\n"
    "  p(-lambda/2) b;\n"
    "  cx a, b;\n"
    "  p(lambda/2) b;\n"
    "}"
)

_ORIGINQ_GATE_NAMES = {
    "h": "H",
    "x": "X",
    "s": "S",
    "sdg": "SDAG",
    "t": "T",
    "tdg": "TDAG",
    "rz": "RZ",
    "ry": "RY",
    "cx": "CNOT",
    "cu1": "CU1",
    "swap": "SWAP",
    "ccx": "TOFFOLI",
}


def _fmt_param(value: float) -> str:
    return repr(value)


def _is_full_identity_measurement(circuit: Circuit) -> bool:
    if circuit.n_qubits != circuit.n_clbits:
        return False
    expected = {(i, i) for i in range(circuit.n_qubits)}
    return set(circuit.measurements) == expected


def emit_spinq(circuit: Circuit) -> str:
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{circuit.n_qubits}];",
        f"creg c[{circuit.n_clbits}];",
    ]
    for gate in circuit.gates:
        params = f"({', '.join(_fmt_param(p) for p in gate.params)})" if gate.params else ""
        qargs = ", ".join(f"q[{i}]" for i in gate.qubits)
        lines.append(f"{gate.name}{params} {qargs};")

    if _is_full_identity_measurement(circuit):
        lines.append("measure q -> c;")
    else:
        for qubit, clbit in circuit.measurements:
            lines.append(f"measure q[{qubit}] -> c[{clbit}];")

    return "\n".join(lines) + "\n"


def emit_braket(circuit: Circuit) -> str:
    uses_cu1 = any(gate.name == "cu1" for gate in circuit.gates)

    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";']
    if uses_cu1:
        lines.append(_BRAKET_CU1_DEFINITION)
    lines.append(f"qubit[{circuit.n_qubits}] q;")
    lines.append(f"bit[{circuit.n_clbits}] c;")

    for gate in circuit.gates:
        params = f"({', '.join(_fmt_param(p) for p in gate.params)})" if gate.params else ""
        qargs = ", ".join(f"q[{i}]" for i in gate.qubits)
        lines.append(f"{gate.name}{params} {qargs};")

    if _is_full_identity_measurement(circuit):
        lines.append("c = measure q;")
    else:
        for qubit, clbit in circuit.measurements:
            lines.append(f"c[{clbit}] = measure q[{qubit}];")

    return "\n".join(lines) + "\n"


def emit_originq(circuit: Circuit) -> str:
    lines = [f"QINIT {circuit.n_qubits}", f"CREG {circuit.n_clbits}"]
    for gate in circuit.gates:
        name = _ORIGINQ_GATE_NAMES[gate.name]
        params = f"({', '.join(_fmt_param(p) for p in gate.params)})" if gate.params else ""
        qargs = ", ".join(f"q[{i}]" for i in gate.qubits)
        lines.append(f"{name}{params} {qargs}")

    for qubit, clbit in circuit.measurements:
        lines.append(f"MEASURE q[{qubit}], c[{clbit}]")

    return "\n".join(lines) + "\n"


EMITTERS = {
    "spinq": emit_spinq,
    "braket": emit_braket,
    "originq": emit_originq,
}
