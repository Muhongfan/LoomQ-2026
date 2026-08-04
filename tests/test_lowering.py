import unittest

from starter_kit.circuit_ir import parse_qasm2
from starter_kit.lowering import SUPPORTED_GATES, lower
from starter_kit.validator import WHITELIST, validate_circuit

FULL_WHITELIST_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\n'
    "h q[0]; x q[0]; s q[0]; sdg q[0]; t q[0]; tdg q[0]; "
    "rz(0.5) q[0]; ry(0.5) q[0]; "
    "cx q[0],q[1]; cu1(0.5) q[0],q[1]; swap q[0],q[1]; "
    "ccx q[0],q[1],q[2];\n"
)


class LoweringIsNoOpForKnownTargetsTests(unittest.TestCase):
    def test_spinq_and_braket_and_originq_pass_through_unchanged(self):
        circuit = parse_qasm2(FULL_WHITELIST_QASM)
        for target in ("spinq", "braket", "originq"):
            lowered = lower(circuit, target)
            self.assertEqual(
                [(g.name, g.qubits, g.params) for g in lowered.gates],
                [(g.name, g.qubits, g.params) for g in circuit.gates],
            )


class LoweringFallbackPathTests(unittest.TestCase):
    def test_expands_unsupported_gate_using_gate_identities(self):
        circuit = parse_qasm2(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nswap q[0],q[1];\n'
        )
        restricted = frozenset(WHITELIST - {"swap"})
        lowered = lower(circuit, "spinq")  # baseline: full support, no expansion
        self.assertEqual([g.name for g in lowered.gates], ["swap"])

        from starter_kit.lowering import _expand

        expanded = [op for gate in circuit.gates for op in _expand(gate, restricted)]
        self.assertEqual([g.name for g in expanded], ["cx", "cx", "cx"])

    def test_expanded_output_still_only_uses_whitelist_or_helper_gates(self):
        from starter_kit.lowering import _expand

        circuit = parse_qasm2(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\nccx q[0],q[1],q[2];\n'
        )
        restricted = frozenset(WHITELIST - {"ccx"})
        expanded = [op for gate in circuit.gates for op in _expand(gate, restricted)]
        used_names = {op.name for op in expanded}
        self.assertTrue(used_names.issubset(WHITELIST), f"unexpected gate names: {used_names}")

    def test_raises_when_no_decomposition_exists(self):
        from starter_kit.circuit_ir import GateOp
        from starter_kit.lowering import _expand

        with self.assertRaises(ValueError):
            _expand(GateOp("h", [0]), frozenset({"x"}))


if __name__ == "__main__":
    unittest.main()
