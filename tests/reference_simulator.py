"""Independent statevector reference simulator for Phase 7 stress tests.

Deliberately does not reuse spinqit or amazon-braket-sdk — the point is an
independent ground truth to compare both live backends against, not just
comparing them to each other. Implements the 12-gate whitelist exactly per
qelib1.inc's own definitions (notably rz(phi) = u1(phi) = diag(1, e^{i*phi}),
matching qelib1.inc, not the physics-convention rz with a global phase).

State layout: a length-2**n complex vector where qubit 0 is the least
significant bit of the basis-state index — i.e. reading a basis index's
binary digits MSB-to-LSB spells out qubit (n-1)...qubit 0, exactly matching
the contract's `key = c[n-1]...c[1]c[0]` convention. This means probability
dicts produced here are directly comparable to runner.py's normalized output
with no extra remapping.
"""

import cmath
from typing import Dict, List, Tuple

import numpy as np

from starter_kit.circuit_ir import Circuit


def _single_qubit_matrix(name: str, params: List[float]) -> np.ndarray:
    if name == "h":
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    if name == "x":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if name == "s":
        return np.array([[1, 0], [0, 1j]], dtype=complex)
    if name == "sdg":
        return np.array([[1, 0], [0, -1j]], dtype=complex)
    if name == "t":
        return np.array([[1, 0], [0, cmath.exp(1j * np.pi / 4)]], dtype=complex)
    if name == "tdg":
        return np.array([[1, 0], [0, cmath.exp(-1j * np.pi / 4)]], dtype=complex)
    if name == "rz":
        (phi,) = params
        return np.array([[1, 0], [0, cmath.exp(1j * phi)]], dtype=complex)
    if name == "ry":
        (theta,) = params
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    raise ValueError(f"not a single-qubit gate: {name}")


def _two_qubit_matrix(name: str, params: List[float]) -> np.ndarray:
    if name == "cx":
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
        )
    if name == "cu1":
        (theta,) = params
        return np.diag([1, 1, 1, cmath.exp(1j * theta)]).astype(complex)
    if name == "swap":
        return np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
        )
    raise ValueError(f"not a two-qubit gate: {name}")


def _ccx_matrix() -> np.ndarray:
    matrix = np.eye(8, dtype=complex)
    matrix[[6, 7]] = matrix[[7, 6]]
    return matrix


def _apply(state: np.ndarray, matrix: np.ndarray, qubits: Tuple[int, ...], n: int) -> np.ndarray:
    k = len(qubits)
    axes = [n - 1 - q for q in qubits]
    tensor = matrix.reshape([2] * k + [2] * k)
    state = np.tensordot(tensor, state, axes=(list(range(k, 2 * k)), axes))
    state = np.moveaxis(state, list(range(k)), axes)
    return state


def simulate(circuit: Circuit) -> Dict[str, float]:
    n = circuit.n_qubits
    state = np.zeros([2] * n, dtype=complex)
    state.flat[0] = 1.0

    for gate in circuit.gates:
        if len(gate.qubits) == 1:
            matrix = _single_qubit_matrix(gate.name, gate.params)
        elif gate.name == "ccx":
            matrix = _ccx_matrix()
        else:
            matrix = _two_qubit_matrix(gate.name, gate.params)
        state = _apply(state, matrix, tuple(gate.qubits), n)

    probabilities = np.abs(state.flatten()) ** 2
    qubit_probs: Dict[str, float] = {}
    for index, p in enumerate(probabilities):
        if p < 1e-12:
            continue
        bits = format(index, f"0{n}b")  # bits[0] = qubit (n-1) ... bits[-1] = qubit 0
        qubit_probs[bits] = qubit_probs.get(bits, 0.0) + float(p)

    clbit_probs: Dict[str, float] = {}
    for bits, p in qubit_probs.items():
        key_chars = ["0"] * circuit.n_clbits
        for qubit, clbit in circuit.measurements:
            key_chars[circuit.n_clbits - 1 - clbit] = bits[n - 1 - qubit]
        key = "".join(key_chars)
        clbit_probs[key] = clbit_probs.get(key, 0.0) + p
    return clbit_probs
