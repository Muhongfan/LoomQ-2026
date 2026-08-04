"""Correctness checks for gate_identities.py, not just transcription checks.

Phase/interference-sensitive gates (s, sdg, t, tdg, ry, cu1) are checked by
comparing measurement distributions between the native gate and its
decomposition under an interference-sensitive context (Hellinger fidelity).
Permutation-only gates (swap, ccx) are checked exactly via truth tables on
computational basis inputs (shots=1, deterministic), independent of
spinqit's internal bit-ordering convention.
"""

import math
import os
import tempfile
import unittest
from typing import Dict, List

from spinqit import BasicSimulatorConfig, get_basic_simulator, get_compiler

from starter_kit.circuit_ir import GateOp
from starter_kit.evaluator import calculate_hellinger_fidelity
from starter_kit.gate_identities import DECOMPOSITIONS


def _run(n_qubits: int, ops: List[GateOp], shots: int = 20000) -> Dict[str, float]:
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n_qubits}];", f"creg c[{n_qubits}];"]
    for op in ops:
        params = f"({', '.join(repr(p) for p in op.params)})" if op.params else ""
        qubits = ", ".join(f"q[{i}]" for i in op.qubits)
        lines.append(f"{op.name}{params} {qubits};")
    lines.append("measure q -> c;")
    qasm = "\n".join(lines)

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".qasm", delete=False, encoding="utf-8")
    try:
        tmp.write(qasm)
        tmp.close()
        compiler = get_compiler("qasm")
        ir = compiler.compile(tmp.name, 0)
        engine = get_basic_simulator()
        config = BasicSimulatorConfig()
        config.configure_shots(shots)
        result = engine.execute(ir, config)
        counts = dict(result.counts)
    finally:
        os.unlink(tmp.name)
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()}


class InterferenceSensitiveDecompositionTests(unittest.TestCase):
    def assertDistributionsMatch(self, native_ops, decomposed_ops, n_qubits, msg):
        native = _run(n_qubits, native_ops)
        decomposed = _run(n_qubits, decomposed_ops)
        fidelity = calculate_hellinger_fidelity(native, decomposed)
        self.assertGreaterEqual(fidelity, 0.99, f"{msg}: fidelity={fidelity} native={native} decomposed={decomposed}")

    def test_s_decomposition(self):
        gate = GateOp("s", [0])
        native = [GateOp("h", [0]), gate, GateOp("h", [0])]
        decomposed = [GateOp("h", [0]), *DECOMPOSITIONS["s"](gate), GateOp("h", [0])]
        self.assertDistributionsMatch(native, decomposed, 1, "s")

    def test_sdg_decomposition(self):
        gate = GateOp("sdg", [0])
        native = [GateOp("h", [0]), gate, GateOp("h", [0])]
        decomposed = [GateOp("h", [0]), *DECOMPOSITIONS["sdg"](gate), GateOp("h", [0])]
        self.assertDistributionsMatch(native, decomposed, 1, "sdg")

    def test_t_decomposition(self):
        gate = GateOp("t", [0])
        native = [GateOp("h", [0]), gate, GateOp("h", [0])]
        decomposed = [GateOp("h", [0]), *DECOMPOSITIONS["t"](gate), GateOp("h", [0])]
        self.assertDistributionsMatch(native, decomposed, 1, "t")

    def test_tdg_decomposition(self):
        gate = GateOp("tdg", [0])
        native = [GateOp("h", [0]), gate, GateOp("h", [0])]
        decomposed = [GateOp("h", [0]), *DECOMPOSITIONS["tdg"](gate), GateOp("h", [0])]
        self.assertDistributionsMatch(native, decomposed, 1, "tdg")

    def test_ry_fallback_decomposition_across_angles(self):
        for theta in (0.3, math.pi / 3, 2.5, -1.2):
            gate = GateOp("ry", [0], [theta])
            native = [gate]
            decomposed = DECOMPOSITIONS["ry"](gate)
            self.assertDistributionsMatch(native, decomposed, 1, f"ry theta={theta}")

    def test_cu1_decomposition_across_angles(self):
        for theta in (0.5, math.pi / 2, 2.1):
            gate = GateOp("cu1", [0, 1], [theta])
            native = [GateOp("h", [0]), GateOp("h", [1]), gate, GateOp("h", [0]), GateOp("h", [1])]
            decomposed = [
                GateOp("h", [0]),
                GateOp("h", [1]),
                *DECOMPOSITIONS["cu1"](gate),
                GateOp("h", [0]),
                GateOp("h", [1]),
            ]
            self.assertDistributionsMatch(native, decomposed, 2, f"cu1 theta={theta}")


class PermutationGateTruthTableTests(unittest.TestCase):
    def test_swap_truth_table_is_exact(self):
        for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
            prep = ([GateOp("x", [0])] if a else []) + ([GateOp("x", [1])] if b else [])
            gate = GateOp("swap", [0, 1])

            native = _run(2, prep + [gate], shots=1)
            decomposed = _run(2, prep + DECOMPOSITIONS["swap"](gate), shots=1)
            self.assertEqual(native, decomposed, f"swap({a},{b}) native vs decomposed")

            expected_prep = ([GateOp("x", [0])] if b else []) + ([GateOp("x", [1])] if a else [])
            expected = _run(2, expected_prep, shots=1)
            self.assertEqual(native, expected, f"swap({a},{b}) semantics")

    def test_ccx_truth_table_is_exact(self):
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    prep = []
                    if a:
                        prep.append(GateOp("x", [0]))
                    if b:
                        prep.append(GateOp("x", [1]))
                    if c:
                        prep.append(GateOp("x", [2]))
                    gate = GateOp("ccx", [0, 1, 2])

                    native = _run(3, prep + [gate], shots=1)
                    decomposed = _run(3, prep + DECOMPOSITIONS["ccx"](gate), shots=1)
                    self.assertEqual(native, decomposed, f"ccx({a},{b},{c}) native vs decomposed")

                    expected_c = c ^ (a & b)
                    expected_prep = []
                    if a:
                        expected_prep.append(GateOp("x", [0]))
                    if b:
                        expected_prep.append(GateOp("x", [1]))
                    if expected_c:
                        expected_prep.append(GateOp("x", [2]))
                    expected = _run(3, expected_prep, shots=1)
                    self.assertEqual(native, expected, f"ccx({a},{b},{c}) semantics")


if __name__ == "__main__":
    unittest.main()
