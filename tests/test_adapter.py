import unittest
from pathlib import Path

from starter_kit import adapter
from starter_kit.evaluator import calculate_hellinger_fidelity, validate_schema

CIRCUITS = Path(__file__).resolve().parents[1] / "starter_kit" / "circuits"
BELL_QASM = (CIRCUITS / "bell.qasm").read_text(encoding="utf-8")
BELL_IDEAL = {"00": 0.5, "11": 0.5}


class AdapterTargetValidationTests(unittest.TestCase):
    def test_transpile_rejects_unknown_target(self):
        with self.assertRaisesRegex(ValueError, "unsupported target"):
            adapter.transpile(BELL_QASM, "ibmq")

    def test_run_rejects_unknown_target(self):
        with self.assertRaisesRegex(ValueError, "unsupported target"):
            adapter.run(BELL_QASM, "ibmq", 100)


class AdapterTranspileTests(unittest.TestCase):
    def test_transpile_spinq_returns_openqasm2(self):
        text = adapter.transpile(BELL_QASM, "spinq")
        self.assertIn("OPENQASM 2.0;", text)
        self.assertIn("qreg q[2];", text)

    def test_transpile_braket_returns_openqasm3(self):
        text = adapter.transpile(BELL_QASM, "braket")
        self.assertIn("OPENQASM 3.0;", text)
        self.assertIn('include "stdgates.inc";', text)

    def test_transpile_originq_returns_originir_even_without_pyqpanda(self):
        # Text generation has no pyqpanda dependency; only run() needs the SDK.
        text = adapter.transpile(BELL_QASM, "originq")
        self.assertIn("QINIT 2", text)
        self.assertIn("CNOT q[0], q[1]", text)

    def test_transpile_rejects_invalid_qasm(self):
        with self.assertRaises(ValueError):
            adapter.transpile(
                'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\nz q[0];\n',
                "spinq",
            )


class AdapterRunTests(unittest.TestCase):
    def test_run_spinq_meets_fidelity_threshold(self):
        result = adapter.run(BELL_QASM, "spinq", shots=8192)
        valid, reason = validate_schema(result)
        self.assertTrue(valid, reason)
        observed = {k: v / result["shots"] for k, v in result["counts"].items()}
        self.assertGreaterEqual(calculate_hellinger_fidelity(observed, BELL_IDEAL), 0.97)

    def test_run_braket_meets_fidelity_threshold(self):
        result = adapter.run(BELL_QASM, "braket", shots=8192)
        valid, reason = validate_schema(result)
        self.assertTrue(valid, reason)
        observed = {k: v / result["shots"] for k, v in result["counts"].items()}
        self.assertGreaterEqual(calculate_hellinger_fidelity(observed, BELL_IDEAL), 0.97)

    def test_run_originq_meets_fidelity_threshold(self):
        result = adapter.run(BELL_QASM, "originq", shots=8192)
        valid, reason = validate_schema(result)
        self.assertTrue(valid, reason)
        observed = {k: v / result["shots"] for k, v in result["counts"].items()}
        self.assertGreaterEqual(calculate_hellinger_fidelity(observed, BELL_IDEAL), 0.97)


if __name__ == "__main__":
    unittest.main()
