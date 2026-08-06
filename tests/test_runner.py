import unittest
from pathlib import Path

from starter_kit.circuit_ir import parse_qasm2
from starter_kit.evaluator import calculate_hellinger_fidelity, validate_schema
from starter_kit.runner import run

CIRCUITS = Path(__file__).resolve().parents[1] / "starter_kit" / "circuits"

BELL_IDEAL = {"00": 0.5, "11": 0.5}
GHZ3_IDEAL = {"000": 0.5, "111": 0.5}

SCRAMBLED_MAPPING_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
    "x q[0];\nmeasure q[0] -> c[1];\nmeasure q[1] -> c[0];\n"
)


class RunnerPublicCircuitFidelityTests(unittest.TestCase):
    def test_spinq_bell_meets_fidelity_threshold(self):
        circuit = parse_qasm2((CIRCUITS / "bell.qasm").read_text(encoding="utf-8"))
        result = run(circuit, "spinq", shots=8192)
        valid, reason = validate_schema(result)
        self.assertTrue(valid, reason)
        observed = {k: v / result["shots"] for k, v in result["counts"].items()}
        self.assertGreaterEqual(calculate_hellinger_fidelity(observed, BELL_IDEAL), 0.97)

    def test_spinq_ghz3_meets_fidelity_threshold(self):
        circuit = parse_qasm2((CIRCUITS / "ghz3.qasm").read_text(encoding="utf-8"))
        result = run(circuit, "spinq", shots=8192)
        valid, reason = validate_schema(result)
        self.assertTrue(valid, reason)
        observed = {k: v / result["shots"] for k, v in result["counts"].items()}
        self.assertGreaterEqual(calculate_hellinger_fidelity(observed, GHZ3_IDEAL), 0.97)

    def test_braket_bell_meets_fidelity_threshold(self):
        circuit = parse_qasm2((CIRCUITS / "bell.qasm").read_text(encoding="utf-8"))
        result = run(circuit, "braket", shots=8192)
        valid, reason = validate_schema(result)
        self.assertTrue(valid, reason)
        observed = {k: v / result["shots"] for k, v in result["counts"].items()}
        self.assertGreaterEqual(calculate_hellinger_fidelity(observed, BELL_IDEAL), 0.97)

    def test_braket_ghz3_meets_fidelity_threshold(self):
        circuit = parse_qasm2((CIRCUITS / "ghz3.qasm").read_text(encoding="utf-8"))
        result = run(circuit, "braket", shots=8192)
        valid, reason = validate_schema(result)
        self.assertTrue(valid, reason)
        observed = {k: v / result["shots"] for k, v in result["counts"].items()}
        self.assertGreaterEqual(calculate_hellinger_fidelity(observed, GHZ3_IDEAL), 0.97)

    def test_originq_bell_meets_fidelity_threshold(self):
        circuit = parse_qasm2((CIRCUITS / "bell.qasm").read_text(encoding="utf-8"))
        result = run(circuit, "originq", shots=8192)
        valid, reason = validate_schema(result)
        self.assertTrue(valid, reason)
        observed = {k: v / result["shots"] for k, v in result["counts"].items()}
        self.assertGreaterEqual(calculate_hellinger_fidelity(observed, BELL_IDEAL), 0.97)

    def test_originq_ghz3_meets_fidelity_threshold(self):
        circuit = parse_qasm2((CIRCUITS / "ghz3.qasm").read_text(encoding="utf-8"))
        result = run(circuit, "originq", shots=8192)
        valid, reason = validate_schema(result)
        self.assertTrue(valid, reason)
        observed = {k: v / result["shots"] for k, v in result["counts"].items()}
        self.assertGreaterEqual(calculate_hellinger_fidelity(observed, GHZ3_IDEAL), 0.97)


class RunnerBitOrderNormalizationTests(unittest.TestCase):
    """x q[0]; measure q[0]->c[1]; measure q[1]->c[0]; is deterministic:
    q[0]=1, q[1]=0 -> c[1]=1, c[0]=0 -> contract key "c[1]c[0]" = "10".
    This directly exercises the spinqit/braket quirk compensation from
    runner.py's _normalize_*_counts, not just the identity-mapping path.
    """

    def test_spinq_compensates_for_qubit_index_quirk(self):
        circuit = parse_qasm2(SCRAMBLED_MAPPING_QASM)
        result = run(circuit, "spinq", shots=16)
        self.assertEqual(result["counts"], {"10": 16})

    def test_braket_compensates_for_statement_order_quirk(self):
        circuit = parse_qasm2(SCRAMBLED_MAPPING_QASM)
        result = run(circuit, "braket", shots=16)
        self.assertEqual(result["counts"], {"10": 16})

    def test_originq_native_output_already_matches_contract(self):
        """Unlike spinqit/braket, pyqpanda's OriginIR MEASURE respects the
        target index directly and its raw string already matches the
        contract's convention -- _normalize_originq_counts is a pass-through,
        so this confirms no remapping bug slipped in, not that one was fixed.
        """
        circuit = parse_qasm2(SCRAMBLED_MAPPING_QASM)
        result = run(circuit, "originq", shots=16)
        self.assertEqual(result["counts"], {"10": 16})


class RunnerValidationAndDispatchTests(unittest.TestCase):
    def test_invalid_gate_raises_before_touching_a_backend(self):
        circuit = parse_qasm2(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\nz q[0];\n'
        )
        with self.assertRaisesRegex(ValueError, "whitelist"):
            run(circuit, "spinq", shots=100)


if __name__ == "__main__":
    unittest.main()
