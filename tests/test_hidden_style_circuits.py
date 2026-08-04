"""Phase 7: stress-test the pipeline on circuit families like the private
hidden set (GHZ-5, QFT-4, Grover-3, random circuits) — we don't have the
actual private circuits (organizer-generated, private seeds), so these are
representative analogs built from the same 12-gate whitelist.

Ground truth is tests/reference_simulator.py, an independent NumPy
statevector simulator (does not reuse spinqit or amazon-braket-sdk), so
this isn't just comparing our two backends to each other.

QFT-4 specifically uses a forward+inverse round trip rather than measuring
right after a single forward QFT: QFT maps every computational-basis input
to a uniform-magnitude output regardless of whether cu1's phases are
correct (a structural property of the DFT, not something specific to a
correct implementation) — measuring immediately after only a forward QFT
therefore cannot detect a cu1 phase bug at all. A round trip that must land
back on the exact original input does exercise cu1's phase correctness
(multiple distinct angles, both signs) rigorously.
"""

import random
import unittest

from starter_kit.circuit_ir import parse_qasm2
from starter_kit.evaluator import calculate_hellinger_fidelity
from starter_kit.runner import run as backend_run
from tests.reference_simulator import simulate

GHZ5_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[5];\ncreg c[5];\n'
    "h q[0];\ncx q[0],q[1];\ncx q[1],q[2];\ncx q[2],q[3];\ncx q[3],q[4];\n"
    "measure q -> c;\n"
)

QFT4_ROUNDTRIP_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\ncreg c[4];\n'
    # nontrivial input: q0=1, q2=1, q1=q3=0
    "x q[0];\nx q[2];\n"
    # forward QFT
    "h q[0];\n"
    "cu1(1.5707963267948966) q[1],q[0];\n"
    "cu1(0.7853981633974483) q[2],q[0];\n"
    "cu1(0.39269908169872414) q[3],q[0];\n"
    "h q[1];\n"
    "cu1(1.5707963267948966) q[2],q[1];\n"
    "cu1(0.7853981633974483) q[3],q[1];\n"
    "h q[2];\n"
    "cu1(1.5707963267948966) q[3],q[2];\n"
    "h q[3];\n"
    "swap q[0],q[3];\nswap q[1],q[2];\n"
    # inverse QFT: exact reverse order, cu1 angles negated
    "swap q[1],q[2];\nswap q[0],q[3];\n"
    "h q[3];\n"
    "cu1(-1.5707963267948966) q[3],q[2];\n"
    "h q[2];\n"
    "cu1(-0.7853981633974483) q[3],q[1];\n"
    "cu1(-1.5707963267948966) q[2],q[1];\n"
    "h q[1];\n"
    "cu1(-0.39269908169872414) q[3],q[0];\n"
    "cu1(-0.7853981633974483) q[2],q[0];\n"
    "cu1(-1.5707963267948966) q[1],q[0];\n"
    "h q[0];\n"
    "measure q -> c;\n"
)

# Standard 3-qubit Grover, 1 iteration, oracle marks |q2 q1 q0> = |101>.
GROVER3_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\n'
    "h q[0];\nh q[1];\nh q[2];\n"
    # oracle: flip the '0' bit (q1) so the pattern to detect becomes all-1s
    "x q[1];\n"
    "h q[2];\nccx q[0],q[1],q[2];\nh q[2];\n"
    "x q[1];\n"
    # diffusion
    "h q[0];\nh q[1];\nh q[2];\n"
    "x q[0];\nx q[1];\nx q[2];\n"
    "h q[2];\nccx q[0],q[1],q[2];\nh q[2];\n"
    "x q[0];\nx q[1];\nx q[2];\n"
    "h q[0];\nh q[1];\nh q[2];\n"
    "measure q -> c;\n"
)


def _random_circuit_qasm(seed: int, n_qubits: int = 3, n_gates: int = 18) -> str:
    rng = random.Random(seed)
    zero_param_1q = ["h", "x", "s", "sdg", "t", "tdg"]
    param_1q = ["rz", "ry"]
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n_qubits}];", f"creg c[{n_qubits}];"]
    for _ in range(n_gates):
        choice = rng.random()
        if choice < 0.4:
            gate = rng.choice(zero_param_1q)
            q = rng.randrange(n_qubits)
            lines.append(f"{gate} q[{q}];")
        elif choice < 0.6:
            gate = rng.choice(param_1q)
            q = rng.randrange(n_qubits)
            theta = rng.uniform(-3.0, 3.0)
            lines.append(f"{gate}({theta!r}) q[{q}];")
        elif choice < 0.8:
            a, b = rng.sample(range(n_qubits), 2)
            lines.append(f"cx q[{a}],q[{b}];")
        elif choice < 0.9:
            a, b = rng.sample(range(n_qubits), 2)
            theta = rng.uniform(-3.0, 3.0)
            lines.append(f"cu1({theta!r}) q[{a}],q[{b}];")
        else:
            a, b = rng.sample(range(n_qubits), 2)
            lines.append(f"swap q[{a}],q[{b}];")
    lines.append("measure q -> c;")
    return "\n".join(lines) + "\n"


class HiddenStyleCircuitFidelityTests(unittest.TestCase):
    def _assert_matches_reference(self, qasm: str, target: str, shots: int = 8192):
        circuit = parse_qasm2(qasm)
        reference = simulate(circuit)
        result = backend_run(circuit, target, shots)
        observed = {k: v / shots for k, v in result["counts"].items()}
        fidelity = calculate_hellinger_fidelity(observed, reference)
        self.assertGreaterEqual(
            fidelity, 0.97, f"{target}: fidelity={fidelity} observed={observed} reference={reference}"
        )

    def test_ghz5_spinq(self):
        self._assert_matches_reference(GHZ5_QASM, "spinq")

    def test_ghz5_braket(self):
        self._assert_matches_reference(GHZ5_QASM, "braket")

    def test_qft4_roundtrip_spinq(self):
        # Deterministic (identity round trip): tight shots suffice, but keep
        # 8192 for consistency with the official fidelity methodology.
        self._assert_matches_reference(QFT4_ROUNDTRIP_QASM, "spinq")

    def test_qft4_roundtrip_braket(self):
        self._assert_matches_reference(QFT4_ROUNDTRIP_QASM, "braket")

    def test_grover3_spinq(self):
        self._assert_matches_reference(GROVER3_QASM, "spinq")

    def test_grover3_braket(self):
        self._assert_matches_reference(GROVER3_QASM, "braket")

    def test_random_circuit_spinq_seed_1(self):
        self._assert_matches_reference(_random_circuit_qasm(seed=1), "spinq")

    def test_random_circuit_braket_seed_1(self):
        self._assert_matches_reference(_random_circuit_qasm(seed=1), "braket")

    def test_random_circuit_spinq_seed_2(self):
        self._assert_matches_reference(_random_circuit_qasm(seed=2), "spinq")

    def test_random_circuit_braket_seed_2(self):
        self._assert_matches_reference(_random_circuit_qasm(seed=2), "braket")


class QFT4RoundTripIsDeterministicTest(unittest.TestCase):
    def test_returns_exactly_to_input_state(self):
        # x q[0]; x q[2]; forward+inverse QFT should be a mathematical
        # identity: single-shot outcome must always be exactly "0101"
        # (q3=0,q2=1,q1=0,q0=1 -> contract key c[3]c[2]c[1]c[0]).
        circuit = parse_qasm2(QFT4_ROUNDTRIP_QASM)
        reference = simulate(circuit)
        self.assertEqual(set(reference.keys()), {"0101"})
        self.assertAlmostEqual(reference["0101"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
