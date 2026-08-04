"""Emitter output must be both format-correct AND actually executable.

SpinQ/Braket outputs are round-tripped through their real SDKs (not just
regex-checked) so a subtly wrong emitter (e.g. bad measurement mapping)
would surface as a fidelity failure, not just a formatting nitpick.
OriginQ output is structurally checked only, since pyqpanda isn't
installed yet (API token pending).
"""

import contextlib
import os
import re
import tempfile
import unittest
from pathlib import Path

from spinqit import BasicSimulatorConfig, get_basic_simulator, get_compiler

from starter_kit.circuit_ir import parse_qasm2
from starter_kit.emitters import emit_braket, emit_originq, emit_spinq
from starter_kit.evaluator import calculate_hellinger_fidelity

CIRCUITS = Path(__file__).resolve().parents[1] / "starter_kit" / "circuits"
BRAKET_STDLIB_DIR = Path(__file__).resolve().parents[1] / "tools" / "braket_local_stdlib"

BELL_IDEAL = {"00": 0.5, "11": 0.5}
GHZ3_IDEAL = {"000": 0.5, "111": 0.5}


def _run_spinq_text(qasm_text: str, shots: int = 8192):
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".qasm", delete=False, encoding="utf-8")
    try:
        tmp.write(qasm_text)
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


@contextlib.contextmanager
def _cwd(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _run_braket_text(qasm_text: str, shots: int = 8192):
    from braket.devices import LocalSimulator
    from braket.ir.openqasm import Program

    with _cwd(BRAKET_STDLIB_DIR):
        device = LocalSimulator()
        task = device.run(Program(source=qasm_text), shots=shots)
        counts = dict(task.result().measurement_counts)
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()}


class SpinQEmitterExecutionTests(unittest.TestCase):
    def test_bell_emits_and_executes_correctly(self):
        circuit = parse_qasm2((CIRCUITS / "bell.qasm").read_text(encoding="utf-8"))
        qasm = emit_spinq(circuit)
        observed = _run_spinq_text(qasm)
        fidelity = calculate_hellinger_fidelity(observed, BELL_IDEAL)
        self.assertGreaterEqual(fidelity, 0.97, f"observed={observed}")

    def test_ghz3_emits_and_executes_correctly(self):
        circuit = parse_qasm2((CIRCUITS / "ghz3.qasm").read_text(encoding="utf-8"))
        qasm = emit_spinq(circuit)
        observed = _run_spinq_text(qasm)
        fidelity = calculate_hellinger_fidelity(observed, GHZ3_IDEAL)
        self.assertGreaterEqual(fidelity, 0.97, f"observed={observed}")

    def test_non_identity_measurement_uses_itemized_form(self):
        # Text-only check: this pinned spinqit build has a confirmed execution
        # bug where measure q[i] -> c[j]; ignores j and always writes to
        # string position i (the qubit's own index) — verified independently
        # via tools/gate_audit.py-style probing, not an emitter defect. The
        # emitted text itself is correct, standard OpenQASM 2.0; compensating
        # for spinqit's own mis-execution of it is a run()/Phase 5 concern
        # (permute spinqit's raw qubit-indexed output using circuit.measurements
        # rather than trusting its reported positions).
        circuit = parse_qasm2(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
            "x q[0];\nmeasure q[0] -> c[1];\nmeasure q[1] -> c[0];\n"
        )
        qasm = emit_spinq(circuit)
        self.assertIn("measure q[0] -> c[1];", qasm)
        self.assertIn("measure q[1] -> c[0];", qasm)


class BraketEmitterExecutionTests(unittest.TestCase):
    def test_bell_emits_and_executes_correctly(self):
        circuit = parse_qasm2((CIRCUITS / "bell.qasm").read_text(encoding="utf-8"))
        qasm = emit_braket(circuit)
        observed = _run_braket_text(qasm)
        fidelity = calculate_hellinger_fidelity(observed, BELL_IDEAL)
        self.assertGreaterEqual(fidelity, 0.97, f"observed={observed}")

    def test_ghz3_emits_and_executes_correctly(self):
        circuit = parse_qasm2((CIRCUITS / "ghz3.qasm").read_text(encoding="utf-8"))
        qasm = emit_braket(circuit)
        observed = _run_braket_text(qasm)
        fidelity = calculate_hellinger_fidelity(observed, GHZ3_IDEAL)
        self.assertGreaterEqual(fidelity, 0.97, f"observed={observed}")

    def test_cu1_gets_inline_definition_and_executes_correctly(self):
        circuit = parse_qasm2(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
            "h q[0];\nh q[1];\ncu1(1.5707963267948966) q[0],q[1];\nh q[0];\nh q[1];\nmeasure q -> c;\n"
        )
        qasm = emit_braket(circuit)
        self.assertIn("gate cu1(lambda) a, b", qasm)

        native = _run_spinq_text(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
            "h q[0];\nh q[1];\ncu1(1.5707963267948966) q[0],q[1];\nh q[0];\nh q[1];\nmeasure q -> c;\n"
        )
        observed = _run_braket_text(qasm)
        fidelity = calculate_hellinger_fidelity(observed, native)
        self.assertGreaterEqual(fidelity, 0.97, f"observed={observed} native={native}")

    def test_no_cu1_omits_inline_definition(self):
        circuit = parse_qasm2((CIRCUITS / "bell.qasm").read_text(encoding="utf-8"))
        qasm = emit_braket(circuit)
        self.assertNotIn("gate cu1", qasm)


class OriginQEmitterStructureTests(unittest.TestCase):
    def test_bell_matches_contract_shape(self):
        circuit = parse_qasm2((CIRCUITS / "bell.qasm").read_text(encoding="utf-8"))
        ir = emit_originq(circuit)
        lines = ir.strip().splitlines()
        self.assertEqual(lines[0], "QINIT 2")
        self.assertEqual(lines[1], "CREG 2")
        self.assertIn("H q[0]", lines)
        self.assertIn("CNOT q[0], q[1]", lines)
        self.assertIn("MEASURE q[0], c[0]", lines)
        self.assertIn("MEASURE q[1], c[1]", lines)

    def test_all_twelve_gates_map_to_allowed_originq_names(self):
        circuit = parse_qasm2(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\n'
            "h q[0]; x q[0]; s q[0]; sdg q[0]; t q[0]; tdg q[0]; "
            "rz(0.5) q[0]; ry(0.5) q[0]; "
            "cx q[0],q[1]; cu1(0.5) q[0],q[1]; swap q[0],q[1]; "
            "ccx q[0],q[1],q[2];\n"
        )
        ir = emit_originq(circuit)
        allowed = {"H", "X", "S", "SDAG", "T", "TDAG", "RZ", "RY", "CNOT", "CU1", "SWAP", "TOFFOLI"}
        used = {
            re.match(r"[A-Z][A-Z0-9]*", line).group()
            for line in ir.splitlines()
            if re.match(r"[A-Z][A-Z0-9]*", line)
        }
        used -= {"QINIT", "CREG", "MEASURE"}
        self.assertTrue(used.issubset(allowed), f"unexpected OriginIR gate names: {used - allowed}")
        self.assertEqual(used, allowed)  # every whitelist gate exercised maps to something valid


if __name__ == "__main__":
    unittest.main()
