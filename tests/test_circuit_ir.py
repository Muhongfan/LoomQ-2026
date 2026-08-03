import unittest
from pathlib import Path

from starter_kit.circuit_ir import parse_qasm2

CIRCUITS = Path(__file__).resolve().parents[1] / "starter_kit" / "circuits"


class CircuitIRPublicCircuitTests(unittest.TestCase):
    def test_bell_parses_to_expected_ir(self):
        qasm = (CIRCUITS / "bell.qasm").read_text(encoding="utf-8")
        circuit = parse_qasm2(qasm)

        self.assertEqual(circuit.n_qubits, 2)
        self.assertEqual(circuit.n_clbits, 2)
        self.assertEqual([(g.name, g.qubits) for g in circuit.gates], [("h", [0]), ("cx", [0, 1])])
        self.assertEqual(circuit.measurements, [(0, 0), (1, 1)])

    def test_ghz3_parses_to_expected_ir(self):
        qasm = (CIRCUITS / "ghz3.qasm").read_text(encoding="utf-8")
        circuit = parse_qasm2(qasm)

        self.assertEqual(circuit.n_qubits, 3)
        self.assertEqual(circuit.n_clbits, 3)
        self.assertEqual(
            [(g.name, g.qubits) for g in circuit.gates],
            [("h", [0]), ("cx", [0, 1]), ("cx", [1, 2])],
        )
        self.assertEqual(circuit.measurements, [(0, 0), (1, 1), (2, 2)])


class CircuitIRWhitelistGateTests(unittest.TestCase):
    def parse(self, body: str):
        qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\n' + body
        return parse_qasm2(qasm)

    def test_all_twelve_whitelist_gates_parse(self):
        circuit = self.parse(
            "h q[0]; x q[0]; s q[0]; sdg q[0]; t q[0]; tdg q[0]; "
            "rz(pi/4) q[0]; ry(-pi/2) q[0]; "
            "cx q[0],q[1]; cu1(pi/2) q[0],q[1]; swap q[0],q[1]; "
            "ccx q[0],q[1],q[2];"
        )
        names = [g.name for g in circuit.gates]
        self.assertEqual(
            names,
            ["h", "x", "s", "sdg", "t", "tdg", "rz", "ry", "cx", "cu1", "swap", "ccx"],
        )

    def test_param_expressions_evaluate_correctly(self):
        import math

        circuit = self.parse("rz(pi/4) q[0]; ry(-pi/2) q[1];")
        self.assertAlmostEqual(circuit.gates[0].params[0], math.pi / 4)
        self.assertAlmostEqual(circuit.gates[1].params[0], -math.pi / 2)

    def test_two_and_three_qubit_gates_carry_correct_qubit_lists(self):
        circuit = self.parse("cx q[0],q[1]; ccx q[0],q[1],q[2];")
        self.assertEqual(circuit.gates[0].qubits, [0, 1])
        self.assertEqual(circuit.gates[1].qubits, [0, 1, 2])

    def test_whole_register_measure_broadcasts_by_index(self):
        circuit = self.parse("measure q -> c;")
        self.assertEqual(circuit.measurements, [(0, 0), (1, 1), (2, 2)])

    def test_indexed_measure_is_not_broadcast(self):
        circuit = self.parse("measure q[1] -> c[2];")
        self.assertEqual(circuit.measurements, [(1, 2)])


class CircuitIRErrorHandlingTests(unittest.TestCase):
    def test_rejects_unsupported_qasm_version(self):
        with self.assertRaises(ValueError):
            parse_qasm2('OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[1] q;')

    def test_rejects_reference_to_undeclared_register(self):
        with self.assertRaises(ValueError):
            parse_qasm2("OPENQASM 2.0;\nqreg q[1];\nh r[0];")

    def test_rejects_mismatched_broadcast_sizes(self):
        with self.assertRaises(ValueError):
            parse_qasm2("OPENQASM 2.0;\nqreg q[2];\ncreg c[1];\nmeasure q -> c;")


if __name__ == "__main__":
    unittest.main()
