"""L1 gate-whitelist and structural validation for parsed Circuit IR."""

try:
    from .circuit_ir import Circuit
except ImportError:
    from circuit_ir import Circuit

# (n_qubits, n_params) per problem_statement.md's 12-gate whitelist.
GATE_ARITY = {
    "h": (1, 0),
    "x": (1, 0),
    "s": (1, 0),
    "sdg": (1, 0),
    "t": (1, 0),
    "tdg": (1, 0),
    "rz": (1, 1),
    "ry": (1, 1),
    "cx": (2, 0),
    "cu1": (2, 1),
    "swap": (2, 0),
    "ccx": (3, 0),
}

WHITELIST = frozenset(GATE_ARITY)


def validate_circuit(circuit: Circuit) -> None:
    for gate in circuit.gates:
        if gate.name not in WHITELIST:
            raise ValueError(f"gate {gate.name!r} is outside the 12-gate LoomQ whitelist")

        expected_qubits, expected_params = GATE_ARITY[gate.name]
        if len(gate.qubits) != expected_qubits:
            raise ValueError(
                f"gate {gate.name!r} expects {expected_qubits} qubit(s), got {len(gate.qubits)}"
            )
        if len(gate.params) != expected_params:
            raise ValueError(
                f"gate {gate.name!r} expects {expected_params} parameter(s), got {len(gate.params)}"
            )
        if len(set(gate.qubits)) != len(gate.qubits):
            raise ValueError(f"gate {gate.name!r} reuses a qubit within one operation: {gate.qubits}")
        for qubit in gate.qubits:
            if not (0 <= qubit < circuit.n_qubits):
                raise ValueError(f"gate {gate.name!r} references out-of-range qubit {qubit}")

    for qubit, clbit in circuit.measurements:
        if not (0 <= qubit < circuit.n_qubits):
            raise ValueError(f"measurement references out-of-range qubit {qubit}")
        if not (0 <= clbit < circuit.n_clbits):
            raise ValueError(f"measurement references out-of-range clbit {clbit}")
