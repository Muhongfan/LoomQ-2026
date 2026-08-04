import unittest

from starter_kit.circuit_ir import parse_qasm2
from starter_kit.validator import validate_circuit


class ValidatorPublicCircuitTests(unittest.TestCase):
    def test_bell_and_ghz3_are_valid(self):
        for body in (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
            "h q[0];\ncx q[0],q[1];\nmeasure q -> c;\n",
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\n'
            "h q[0];\ncx q[0],q[1];\ncx q[1],q[2];\nmeasure q -> c;\n",
        ):
            validate_circuit(parse_qasm2(body))  # must not raise

    def test_all_twelve_whitelist_gates_are_valid(self):
        body = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\n'
            "h q[0]; x q[0]; s q[0]; sdg q[0]; t q[0]; tdg q[0]; "
            "rz(0.5) q[0]; ry(0.5) q[0]; "
            "cx q[0],q[1]; cu1(0.5) q[0],q[1]; swap q[0],q[1]; "
            "ccx q[0],q[1],q[2];\n"
        )
        validate_circuit(parse_qasm2(body))  # must not raise


class ValidatorRejectionTests(unittest.TestCase):
    def parse(self, body: str):
        return parse_qasm2('OPENQASM 2.0;\ninclude "qelib1.inc";\n' + body)

    def test_rejects_gate_outside_whitelist(self):
        circuit = self.parse("qreg q[1];\ncreg c[1];\nz q[0];\n")
        with self.assertRaisesRegex(ValueError, "whitelist"):
            validate_circuit(circuit)

    def test_rejects_wrong_qubit_count(self):
        from starter_kit.circuit_ir import GateOp

        circuit = self.parse("qreg q[2];\ncreg c[2];\n")
        circuit.gates.append(GateOp("h", [0, 1]))
        with self.assertRaisesRegex(ValueError, "expects 1 qubit"):
            validate_circuit(circuit)

    def test_rejects_wrong_param_count(self):
        from starter_kit.circuit_ir import GateOp

        circuit = self.parse("qreg q[1];\ncreg c[1];\n")
        circuit.gates.append(GateOp("rz", [0], []))
        with self.assertRaisesRegex(ValueError, "expects 1 parameter"):
            validate_circuit(circuit)

    def test_rejects_repeated_qubit_in_one_gate(self):
        from starter_kit.circuit_ir import GateOp

        circuit = self.parse("qreg q[2];\ncreg c[2];\n")
        circuit.gates.append(GateOp("cx", [0, 0]))
        with self.assertRaisesRegex(ValueError, "reuses a qubit"):
            validate_circuit(circuit)

    def test_rejects_out_of_range_qubit(self):
        from starter_kit.circuit_ir import GateOp

        circuit = self.parse("qreg q[1];\ncreg c[1];\n")
        circuit.gates.append(GateOp("h", [5]))
        with self.assertRaisesRegex(ValueError, "out-of-range qubit"):
            validate_circuit(circuit)

    def test_rejects_out_of_range_measurement(self):
        circuit = self.parse("qreg q[1];\ncreg c[1];\n")
        circuit.measurements.append((0, 9))
        with self.assertRaisesRegex(ValueError, "out-of-range clbit"):
            validate_circuit(circuit)


if __name__ == "__main__":
    unittest.main()
