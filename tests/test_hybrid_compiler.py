"""L3 tests: Hybrid-QASM splitting, classical mini-language compiler, and
RISC-V codegen correctness.

Correctness is checked the same way the organizers describe grading it:
compile -> load into TinyRISCVEmulator -> inject every combination of
measured bits -> compare final register state against an independent
reference (a plain tree-walking interpreter over the same AST, not the
RISC-V path) -- mirrors tests/reference_simulator.py's role for L1.
"""

import itertools
import random
import unittest

from starter_kit import adapter
from starter_kit.hybrid_compiler import (
    Assign,
    BinOp,
    Const,
    If,
    Var,
    _extract_quantum_ops,
    _parse_classical,
    _split_classical_block,
    _tokenize,
    compile_hybrid,
)
from starter_kit.riscv_emulator import TinyRISCVEmulator


# ---------------------------------------------------------------------------
# Independent reference interpreter for the classical mini-language: a plain
# tree-walk over the same AST, not touching RISC-V at all.
# ---------------------------------------------------------------------------


def _interpret(block, creg_values):
    """creg_values: dict mapping clbit index k -> 0/1. Returns a dict of
    r-register name ("x1".."x9") -> final integer value (0 if never
    assigned)."""
    state = {f"x{n}": 0 for n in range(1, 10)}
    for k, v in creg_values.items():
        state[f"x{10 + k}"] = v

    def eval_expr(node):
        if isinstance(node, Const):
            return node.value
        if isinstance(node, Var):
            return state.get(node.register, 0)
        if isinstance(node, BinOp):
            left, right = eval_expr(node.left), eval_expr(node.right)
            if node.op == "+":
                return left + right
            if node.op == "-":
                return left - right
            if node.op == "==":
                return 1 if left == right else 0
            if node.op == "!=":
                return 1 if left != right else 0
        raise TypeError(node)

    def exec_block(stmts):
        for stmt in stmts:
            if isinstance(stmt, Assign):
                state[stmt.target] = eval_expr(stmt.expr)
            elif isinstance(stmt, If):
                if eval_expr(stmt.cond):
                    exec_block(stmt.then_body)
                else:
                    exec_block(stmt.else_body)
            else:
                raise TypeError(stmt)

    exec_block(block)
    # Only r1..r9 are real outputs; c[k]-mapped keys are inputs the harness
    # injected, not something compile_hybrid's assembly is expected to
    # reproduce in its own output comparison.
    return {name: value for name, value in state.items() if int(name[1:]) <= 9}


def _run_on_emulator(assembly, creg_values):
    emu = TinyRISCVEmulator()
    emu.load_program(assembly)
    for k, v in creg_values.items():
        emu.set_register(f"x{10 + k}", v)
    final = emu.execute()
    return {f"x{n}": final.get(f"x{n}", 0) for n in range(1, 10)}


class OfficialPublicExampleTests(unittest.TestCase):
    """The exact example evaluator.py::evaluate_l3 uses -- must pass first."""

    SOURCE = (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "qreg q[1];\n"
        "creg c[1];\n"
        "measure q[0] -> c[0];\n"
        "classical { if (c[0] == 1) { r1 = 7; } else { r1 = 3; } }\n"
    )

    def test_public_branch_semantics(self):
        quantum_ops, assembly = adapter.compile_hybrid(self.SOURCE)
        self.assertIsInstance(quantum_ops, list)
        self.assertIsInstance(assembly, str)
        self.assertTrue(assembly.strip())
        for measured, expected in ((0, 3), (1, 7)):
            emulator = TinyRISCVEmulator()
            emulator.load_program(assembly)
            emulator.set_register("x10", measured)
            self.assertEqual(emulator.execute().get("x1", 0), expected)


class SplitClassicalBlockTests(unittest.TestCase):
    def test_splits_middle_placement(self):
        source = (
            "h q[0];\n"
            "measure q[0] -> c[0];\n"
            "classical { r1 = 1; }\n"
            "cx q[0], q[1];\n"
        )
        quantum_source, classical_source = _split_classical_block(source)
        self.assertNotIn("classical", quantum_source)
        self.assertIn("h q[0];", quantum_source)
        self.assertIn("cx q[0], q[1];", quantum_source)
        self.assertEqual(classical_source.strip(), "r1 = 1;")

    def test_nested_braces_do_not_confuse_the_split(self):
        source = "classical { if (r1 == 1) { r2 = 1; } else { r2 = 2; } }\nh q[0];\n"
        quantum_source, classical_source = _split_classical_block(source)
        self.assertIn("h q[0];", quantum_source)
        self.assertIn("if (r1 == 1)", classical_source)

    def test_rejects_missing_classical_block(self):
        with self.assertRaises(ValueError):
            _split_classical_block("h q[0];\n")

    def test_rejects_multiple_classical_blocks(self):
        source = "classical { r1 = 1; }\nclassical { r2 = 2; }\n"
        with self.assertRaises(ValueError):
            _split_classical_block(source)

    def test_rejects_unterminated_block(self):
        with self.assertRaises(ValueError):
            _split_classical_block("classical { r1 = 1;\n")


class ExtractQuantumOpsOrderTests(unittest.TestCase):
    def test_preserves_original_interleaving_including_measure_then_gate(self):
        # Regression guard for the specific failure mode the plan calls out:
        # naively concatenating Circuit.gates + Circuit.measurements would
        # put the measurement AFTER cx, reversing what the source says.
        quantum_source = (
            "OPENQASM 2.0;\n"
            'include "qelib1.inc";\n'
            "qreg q[2];\n"
            "creg c[2];\n"
            "h q[0];\n"
            "measure q[0] -> c[0];\n"
            "cx q[0], q[1];\n"
        )
        ops = _extract_quantum_ops(quantum_source)
        self.assertEqual(ops, ["h q[0];", "measure q[0] -> c[0];", "cx q[0], q[1];"])

    def test_classical_block_at_start_middle_end_all_preserve_order(self):
        for template in (
            "classical {{ r1 = 1; }}\nh q[0];\nmeasure q[0] -> c[0];\n",
            "h q[0];\nclassical {{ r1 = 1; }}\nmeasure q[0] -> c[0];\n",
            "h q[0];\nmeasure q[0] -> c[0];\nclassical {{ r1 = 1; }}\n",
        ):
            source = (
                "OPENQASM 2.0;\n" 'include "qelib1.inc";\n' "qreg q[1];\ncreg c[1];\n"
            ) + template.format()
            quantum_ops, _assembly = compile_hybrid(source)
            self.assertEqual(quantum_ops, ["h q[0];", "measure q[0] -> c[0];"])

    def test_quantum_ops_reparse_to_an_equivalent_circuit(self):
        from starter_kit.circuit_ir import parse_qasm2

        source = (
            "OPENQASM 2.0;\n"
            'include "qelib1.inc";\n'
            "qreg q[2];\n"
            "creg c[2];\n"
            "h q[0];\n"
            "classical { r1 = 1; }\n"
            "cx q[0], q[1];\n"
            "measure q -> c;\n"
        )
        quantum_ops, _assembly = compile_hybrid(source)
        rebuilt = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
            + "\n".join(quantum_ops)
        )
        circuit = parse_qasm2(rebuilt)
        self.assertEqual([g.name for g in circuit.gates], ["h", "cx"])
        self.assertEqual(circuit.measurements, [(0, 0), (1, 1)])


class TokenizerTests(unittest.TestCase):
    def test_tokenizes_public_example_body(self):
        tokens = _tokenize("if (c[0] == 1) { r1 = 7; } else { r1 = 3; }")
        kinds = [kind for kind, _ in tokens]
        self.assertEqual(
            kinds,
            [
                "IF", "LPAREN", "CREG", "EQ", "INT", "RPAREN", "LBRACE",
                "REG", "ASSIGN", "INT", "SEMI", "RBRACE", "ELSE", "LBRACE",
                "REG", "ASSIGN", "INT", "SEMI", "RBRACE", "EOF",
            ],
        )

    def test_minus_and_negative_literal_do_not_merge(self):
        # "r1 - 5" must tokenize as REG MINUS INT, not REG then a bogus
        # merged "-5" INT token that silently drops the operator.
        tokens = _tokenize("r1 = r1 - 5;")
        kinds = [kind for kind, _ in tokens]
        self.assertEqual(kinds, ["REG", "ASSIGN", "REG", "MINUS", "INT", "SEMI", "EOF"])

    def test_rejects_unknown_character(self):
        with self.assertRaises(ValueError):
            _tokenize("r1 = 1 & 2;")


class ParserTests(unittest.TestCase):
    def test_parses_assignment(self):
        block = _parse_classical("r1 = 5;")
        self.assertEqual(block, [Assign(target="x1", expr=Const(5))])

    def test_parses_unary_minus_as_zero_minus_operand(self):
        block = _parse_classical("r1 = -5;")
        self.assertEqual(block, [Assign(target="x1", expr=BinOp("-", Const(0), Const(5)))])

    def test_parses_if_without_else(self):
        block = _parse_classical("if (r1 == 1) { r2 = 1; }")
        self.assertEqual(len(block), 1)
        node = block[0]
        self.assertIsInstance(node, If)
        self.assertEqual(node.else_body, [])

    def test_creg_maps_to_offset_ten_plus_index(self):
        block = _parse_classical("r1 = c[2];")
        self.assertEqual(block, [Assign(target="x1", expr=Var(register="x12"))])


class CodegenDifferentialTests(unittest.TestCase):
    """Compile -> run on TinyRISCVEmulator, compare against the independent
    tree-walking reference interpreter, exhaustively over every measured-bit
    combination -- same method the graders describe using."""

    def _check(self, classical_source, n_measured_bits):
        block = _parse_classical(classical_source)
        from starter_kit.hybrid_compiler import _generate_riscv

        assembly = _generate_riscv(block)
        for bits in itertools.product((0, 1), repeat=n_measured_bits):
            creg_values = dict(enumerate(bits))
            expected = _interpret(block, creg_values)
            actual = _run_on_emulator(assembly, creg_values)
            self.assertEqual(actual, expected, msg=f"mismatch for creg_values={creg_values}")

    def test_simple_if_else(self):
        self._check("if (c[0] == 1) { r1 = 7; } else { r1 = 3; }", 1)

    def test_sequential_assignment_after_if(self):
        self._check(
            "if (c[0] == 1) { r1 = 100; } else { r1 = 10; }\nr1 = r1 + 5;",
            1,
        )

    def test_no_else_branch(self):
        self._check("if (c[0] == 1) { r1 = 42; }", 1)

    def test_nested_if_else(self):
        self._check(
            "if (c[0] == 1) {"
            "  if (c[1] == 1) { r1 = 1; } else { r1 = 2; }"
            "} else {"
            "  if (c[1] == 1) { r1 = 3; } else { r1 = 4; }"
            "}",
            2,
        )

    def test_not_equal_operator(self):
        self._check("if (c[0] != 1) { r1 = 9; } else { r1 = 8; }", 1)

    def test_multiple_measured_bits_and_registers(self):
        self._check(
            "r1 = c[0] + c[1];\n"
            "if (r1 == 2) { r2 = 100; } else { r2 = 0; }\n"
            "r3 = r1 - 5;",
            2,
        )

    def test_subtraction_with_literal_on_left(self):
        self._check("r1 = 5 - c[0];", 1)

    def test_unary_minus(self):
        self._check("r1 = -c[0];\nr2 = r1 + 10;", 1)

    def test_uninitialized_register_reads_as_zero(self):
        self._check("r2 = r1 + 1;", 0)

    def test_empty_classical_block_produces_runnable_assembly(self):
        self._check("", 0)


class RandomizedDifferentialTests(unittest.TestCase):
    """Random small programs within the grammar, checked the same way as
    CodegenDifferentialTests -- a cheap approximation of the organizers'
    private random-case generation."""

    def _random_expr(self, rng, depth):
        if depth <= 0 or rng.random() < 0.4:
            choice = rng.choice(["const", "reg", "creg"])
            if choice == "const":
                return str(rng.randint(-20, 20))
            if choice == "reg":
                return f"r{rng.randint(1, 9)}"
            return f"c[{rng.randint(0, 1)}]"
        op = rng.choice(["+", "-", "==", "!="])
        return f"{self._random_expr(rng, depth - 1)} {op} {self._random_expr(rng, depth - 1)}"

    def _random_block(self, rng, depth, n_stmts):
        lines = []
        for _ in range(n_stmts):
            if depth > 0 and rng.random() < 0.5:
                cond = self._random_expr(rng, 1)
                then_body = "\n".join(self._random_block(rng, depth - 1, 1))
                if rng.random() < 0.7:
                    else_body = "\n".join(self._random_block(rng, depth - 1, 1))
                    lines.append(f"if ({cond}) {{ {then_body} }} else {{ {else_body} }}")
                else:
                    lines.append(f"if ({cond}) {{ {then_body} }}")
            else:
                target = rng.randint(1, 9)
                expr = self._random_expr(rng, 1)
                lines.append(f"r{target} = {expr};")
        return lines

    def test_random_programs_match_reference_interpreter(self):
        rng = random.Random(1234)
        for trial in range(30):
            source = "\n".join(self._random_block(rng, depth=2, n_stmts=3))
            try:
                block = _parse_classical(source)
            except ValueError:
                continue  # generator occasionally emits something the grammar-adjacent generator itself botches; skip
            from starter_kit.hybrid_compiler import _generate_riscv

            assembly = _generate_riscv(block)
            for bits in itertools.product((0, 1), repeat=2):
                creg_values = dict(enumerate(bits))
                expected = _interpret(block, creg_values)
                actual = _run_on_emulator(assembly, creg_values)
                self.assertEqual(
                    actual, expected,
                    msg=f"trial {trial} mismatch for creg_values={creg_values}\nsource:\n{source}\nassembly:\n{assembly}",
                )


if __name__ == "__main__":
    unittest.main()
