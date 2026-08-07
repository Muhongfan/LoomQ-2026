"""L3: Hybrid-QASM -> (quantum operation sequence, RISC-V assembly).

Hybrid-QASM is standard OpenQASM 2.0 with exactly one classical control block
spliced in anywhere among the gate/measure statements:

    measure q[0] -> c[0];
    classical {
      if (c[0] == 1) { r1 = 100; } else { r1 = 10; }
      r1 = r1 + 5;
    }
    cx q[0], q[1];

compile_hybrid() splits the two halves, reuses circuit_ir.parse_qasm2 +
validator.validate_circuit to check the quantum half against the same
12-gate whitelist as L1, and compiles the classical block's mini-language
(int literals, r1..r9 registers, `+ - == !=`, if/else, sequential
assignment) into RISC-V assembly runnable on riscv_emulator.TinyRISCVEmulator
(li, add, sub, addi, beq, bne, j only).

Register convention (from problem_statement.md): r1..r9 -> x1..x9;
measured bit c[k] -> x(10+k), read-only (injected by the grading harness
after compile_hybrid runs, never written here).
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Union

try:
    from .circuit_ir import parse_qasm2
    from .validator import validate_circuit
except ImportError:
    from circuit_ir import parse_qasm2
    from validator import validate_circuit


# ---------------------------------------------------------------------------
# Hybrid-QASM splitting: locate the single `classical { ... }` block by brace
# counting (so nested if/else braces inside it don't confuse the split), and
# extract the quantum-only statements in their ORIGINAL relative order.
#
# This must not go through Circuit.gates/Circuit.measurements to rebuild the
# quantum operation list: those are two separate lists in circuit_ir.py and
# concatenating them loses the original interleaving between measure
# statements and any gates that come after them in the source (e.g. the
# official example's `measure q[0] -> c[0]; classical {...} cx q[0], q[1];`
# has a gate AFTER a measurement on a qubit already measured -- reordering
# that changes the circuit's semantics, not just its presentation).
# ---------------------------------------------------------------------------

_CLASSICAL_KEYWORD_RE = re.compile(r"\bclassical\b")
_HEADER_PREFIXES = ("OPENQASM", "include", "qreg", "creg")


def _split_classical_block(source: str) -> Tuple[str, str]:
    matches = list(_CLASSICAL_KEYWORD_RE.finditer(source))
    if len(matches) != 1:
        raise ValueError(
            "Hybrid-QASM must contain exactly one 'classical' block, found "
            f"{len(matches)}"
        )
    keyword_start, keyword_end = matches[0].span()

    brace_open = source.find("{", keyword_end)
    if brace_open == -1:
        raise ValueError("'classical' keyword is not followed by a '{' block")
    if source[keyword_end:brace_open].strip():
        raise ValueError("unexpected tokens between 'classical' and '{'")

    depth = 0
    brace_close = -1
    for i in range(brace_open, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                brace_close = i
                break
    if brace_close == -1:
        raise ValueError("unterminated 'classical' block: missing closing '}'")

    classical_source = source[brace_open + 1 : brace_close]
    quantum_source = source[:keyword_start] + source[brace_close + 1 :]
    return quantum_source, classical_source


def _extract_quantum_ops(quantum_source: str) -> List[str]:
    """Ordered list of the surviving gate/measure statement strings, header
    declarations (OPENQASM/include/qreg/creg) excluded. Mirrors
    circuit_ir.parse_qasm2's own comment-stripping + statement-splitting so
    the two stay in agreement about what a "statement" is."""
    text = re.sub(r"//.*", "", quantum_source)
    ops = []
    for raw_stmt in text.split(";"):
        stmt = raw_stmt.strip()
        if not stmt:
            continue
        if any(stmt.startswith(prefix) for prefix in _HEADER_PREFIXES):
            continue
        ops.append(stmt + ";")
    return ops


# ---------------------------------------------------------------------------
# Classical mini-language: tokenizer
# ---------------------------------------------------------------------------

_TOKEN_SPEC = [
    ("WS", r"[ \t\r\n]+"),
    ("EQ", r"=="),
    ("NE", r"!="),
    ("ASSIGN", r"="),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("SEMI", r";"),
    ("IF", r"if\b"),
    ("ELSE", r"else\b"),
    ("CREG", r"c\[\s*\d+\s*\]"),
    ("REG", r"r[1-9]\b"),
    ("INT", r"\d+"),
]
# Each alternative is a flat (non-nested-named-group) pattern so `m.lastgroup`
# is unambiguous; the CREG/REG index is pulled out separately below rather
# than via a nested named group (nesting named groups inside an alternation
# makes `lastgroup` resolution genuinely ambiguous).
_MASTER_RE = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in _TOKEN_SPEC))
_DIGITS_RE = re.compile(r"\d+")

Token = Tuple[str, object]


def _tokenize(source: str) -> List[Token]:
    tokens: List[Token] = []
    pos, length = 0, len(source)
    while pos < length:
        match = _MASTER_RE.match(source, pos)
        if not match:
            raise ValueError(
                f"unexpected character in classical block at position {pos}: "
                f"{source[pos]!r}"
            )
        kind = match.lastgroup
        text = match.group(kind)
        pos = match.end()
        if kind == "WS":
            continue
        if kind == "CREG":
            tokens.append(("CREG", int(_DIGITS_RE.search(text).group())))
        elif kind == "REG":
            tokens.append(("REG", int(_DIGITS_RE.search(text).group())))
        else:
            tokens.append((kind, text))
    tokens.append(("EOF", ""))
    return tokens


# ---------------------------------------------------------------------------
# Classical mini-language: AST
# ---------------------------------------------------------------------------


@dataclass
class Const:
    value: int


@dataclass
class Var:
    register: str  # canonical "x1".."x9" (r-registers) or "x10".. (c[k])


@dataclass
class BinOp:
    op: str  # one of '+', '-', '==', '!='
    left: "Expr"
    right: "Expr"


Expr = Union[Const, Var, BinOp]


@dataclass
class Assign:
    target: str  # "x1".."x9"
    expr: Expr


@dataclass
class If:
    cond: Expr
    then_body: List["Stmt"]
    else_body: List["Stmt"]


Stmt = Union[Assign, If]


# ---------------------------------------------------------------------------
# Classical mini-language: recursive-descent parser
#
# Grammar (per problem_statement.md's "迷你文法"):
#   block  := stmt*
#   stmt   := assign | if_stmt
#   assign := REG '=' expr ';'
#   if_stmt:= 'if' '(' expr ')' '{' block '}' ('else' '{' block '}')?
#   expr   := term ((PLUS|MINUS|EQ|NE) term)*   -- left-associative chain;
#             the spec only guarantees a single operator, chaining is a
#             strict superset kept for robustness, not required complexity.
#   term   := '-' term | INT | REG | CREG
# No parenthesized sub-expressions beyond the mandatory `if (...)` wrapper,
# no assignment to CREG (c[k] is a read-only value injected by the grading
# harness) -- both deliberately out of scope, matching the spec's grammar.
# ---------------------------------------------------------------------------

_BIN_OP_TEXT = {"PLUS": "+", "MINUS": "-", "EQ": "==", "NE": "!="}


class _Parser:
    def __init__(self, tokens: List[Token]):
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, kind: str) -> Token:
        tok = self._advance()
        if tok[0] != kind:
            raise ValueError(f"expected {kind} but got {tok[0]} ({tok[1]!r})")
        return tok

    def parse_program(self) -> List[Stmt]:
        block = self._parse_block_until("EOF")
        self._expect("EOF")
        return block

    def _parse_block_until(self, *stop_kinds: str) -> List[Stmt]:
        stmts: List[Stmt] = []
        while self._peek()[0] not in stop_kinds:
            stmts.append(self._parse_stmt())
        return stmts

    def _parse_stmt(self) -> Stmt:
        kind = self._peek()[0]
        if kind == "IF":
            return self._parse_if()
        if kind == "REG":
            return self._parse_assign()
        raise ValueError(f"unexpected token starting statement: {kind}")

    def _parse_assign(self) -> Assign:
        _, ridx = self._expect("REG")
        self._expect("ASSIGN")
        expr = self._parse_expr()
        self._expect("SEMI")
        return Assign(target=f"x{ridx}", expr=expr)

    def _parse_if(self) -> If:
        self._expect("IF")
        self._expect("LPAREN")
        cond = self._parse_expr()
        self._expect("RPAREN")
        self._expect("LBRACE")
        then_body = self._parse_block_until("RBRACE")
        self._expect("RBRACE")
        else_body: List[Stmt] = []
        if self._peek()[0] == "ELSE":
            self._advance()
            self._expect("LBRACE")
            else_body = self._parse_block_until("RBRACE")
            self._expect("RBRACE")
        return If(cond=cond, then_body=then_body, else_body=else_body)

    def _parse_expr(self) -> Expr:
        node = self._parse_term()
        while self._peek()[0] in _BIN_OP_TEXT:
            op_kind, _ = self._advance()
            rhs = self._parse_term()
            node = BinOp(op=_BIN_OP_TEXT[op_kind], left=node, right=rhs)
        return node

    def _parse_term(self) -> Expr:
        kind, value = self._peek()
        if kind == "MINUS":
            self._advance()
            operand = self._parse_term()
            return BinOp(op="-", left=Const(0), right=operand)
        if kind == "INT":
            self._advance()
            return Const(int(value))
        if kind == "REG":
            self._advance()
            return Var(register=f"x{value}")
        if kind == "CREG":
            self._advance()
            return Var(register=f"x{10 + value}")
        raise ValueError(f"unexpected token in expression: {kind}")


def _parse_classical(classical_source: str) -> List[Stmt]:
    return _Parser(_tokenize(classical_source)).parse_program()


# ---------------------------------------------------------------------------
# Classical mini-language: RISC-V codegen
#
# Only `li, add, sub, addi, beq, bne, j` are available (riscv_emulator.py's
# supported subset). Consequences that shape the codegen below:
#   - add/sub/beq/bne take two REGISTER operands; only addi takes one
#     immediate -- integer literals must be li'd into a scratch register
#     before they can take part in add/sub/beq/bne (addi covers the common
#     "register +/- literal" case without needing a scratch register).
#   - `==`/`!=` inside an `if` condition compile straight to beq/bne against
#     the THEN label, no intermediate 0/1 value needed. A comparison used
#     outside an `if` condition (not reachable through the documented
#     grammar, but handled defensively) falls back to materializing 0/1 via
#     a three-instruction beq/bne + li + j sequence.
# Scratch registers x20..x31 are reserved for expression evaluation; a
# stack-discipline allocator (matching the natural LIFO lifetime of
# recursive expression compilation) keeps peak usage proportional to
# expression *depth*, not the number of literals in a chain, so the pool
# comfortably covers anything the documented grammar can produce.
# ---------------------------------------------------------------------------

_TEMP_BASE = 20
_TEMP_MAX = 31


class _Codegen:
    def __init__(self) -> None:
        self.lines: List[str] = []
        self._temp_top = _TEMP_BASE
        self._label_counter = 0

    def emit(self, line: str) -> None:
        self.lines.append(line)

    def _alloc_temp(self) -> str:
        if self._temp_top > _TEMP_MAX:
            raise ValueError("classical expression too complex: ran out of scratch registers")
        reg = f"x{self._temp_top}"
        self._temp_top += 1
        return reg

    def _free_temp(self, reg: str) -> None:
        if int(reg[1:]) == self._temp_top - 1:
            self._temp_top -= 1

    def _new_label(self, prefix: str) -> str:
        label = f"{prefix}_{self._label_counter}"
        self._label_counter += 1
        return label

    def _resolve_operand(self, node: Expr) -> Tuple[str, bool]:
        """Returns (register, is_temp) -- is_temp tells the caller whether
        it's safe/expected to free the register once done with it."""
        if isinstance(node, Const):
            tmp = self._alloc_temp()
            self.emit(f"li {tmp}, {node.value}")
            return tmp, True
        if isinstance(node, Var):
            return node.register, False
        if isinstance(node, BinOp):
            return self._compile_value_binop(node), True
        raise TypeError(f"unexpected expression node: {node!r}")

    def _compile_bool_value(self, node: BinOp, dest_reg: str) -> None:
        lreg, ltmp = self._resolve_operand(node.left)
        rreg, rtmp = self._resolve_operand(node.right)
        true_label = self._new_label("BOOLT")
        end_label = self._new_label("BOOLE")
        branch_op = "beq" if node.op == "==" else "bne"
        self.emit(f"{branch_op} {lreg}, {rreg}, {true_label}")
        self.emit(f"li {dest_reg}, 0")
        self.emit(f"j {end_label}")
        self.emit(f"{true_label}:")
        self.emit(f"li {dest_reg}, 1")
        self.emit(f"{end_label}:")
        if rtmp:
            self._free_temp(rreg)
        if ltmp:
            self._free_temp(lreg)

    def _compile_value_binop(self, node: BinOp) -> str:
        if node.op in ("==", "!="):
            tmp = self._alloc_temp()
            self._compile_bool_value(node, tmp)
            return tmp

        # Constant-fold literal + literal.
        if isinstance(node.left, Const) and isinstance(node.right, Const):
            value = (
                node.left.value + node.right.value
                if node.op == "+"
                else node.left.value - node.right.value
            )
            tmp = self._alloc_temp()
            self.emit(f"li {tmp}, {value}")
            return tmp

        # register (op) literal -> addi, no scratch register needed for the
        # literal side.
        if isinstance(node.right, Const) and not isinstance(node.left, Const):
            lreg, ltmp = self._resolve_operand(node.left)
            tmp = self._alloc_temp()
            imm = node.right.value if node.op == "+" else -node.right.value
            self.emit(f"addi {tmp}, {lreg}, {imm}")
            if ltmp:
                self._free_temp(lreg)
            return tmp

        # literal + register (commutative case only -- literal - register
        # falls through to the general path below since order matters).
        if node.op == "+" and isinstance(node.left, Const) and not isinstance(node.right, Const):
            rreg, rtmp = self._resolve_operand(node.right)
            tmp = self._alloc_temp()
            self.emit(f"addi {tmp}, {rreg}, {node.left.value}")
            if rtmp:
                self._free_temp(rreg)
            return tmp

        # General case: materialize both sides into registers.
        lreg, ltmp = self._resolve_operand(node.left)
        rreg, rtmp = self._resolve_operand(node.right)
        tmp = self._alloc_temp()
        instr = "add" if node.op == "+" else "sub"
        self.emit(f"{instr} {tmp}, {lreg}, {rreg}")
        if rtmp:
            self._free_temp(rreg)
        if ltmp:
            self._free_temp(lreg)
        return tmp

    def _compile_assign(self, node: Assign) -> None:
        if isinstance(node.expr, Const):
            self.emit(f"li {node.target}, {node.expr.value}")
            return
        if isinstance(node.expr, Var):
            self.emit(f"addi {node.target}, {node.expr.register}, 0")
            return
        srcreg = self._compile_value_binop(node.expr)
        self.emit(f"addi {node.target}, {srcreg}, 0")
        self._free_temp(srcreg)

    def _compile_if(self, node: If) -> None:
        then_label = self._new_label("THEN")
        end_label = self._new_label("END")
        if isinstance(node.cond, BinOp) and node.cond.op in ("==", "!="):
            lreg, ltmp = self._resolve_operand(node.cond.left)
            rreg, rtmp = self._resolve_operand(node.cond.right)
            branch_op = "beq" if node.cond.op == "==" else "bne"
            self.emit(f"{branch_op} {lreg}, {rreg}, {then_label}")
            if rtmp:
                self._free_temp(rreg)
            if ltmp:
                self._free_temp(lreg)
        else:
            # Defensive fallback for a bare value used as a condition
            # (not reachable via the documented grammar): nonzero is true.
            reg, tmp = self._resolve_operand(node.cond)
            self.emit(f"bne {reg}, x0, {then_label}")
            if tmp:
                self._free_temp(reg)

        # Condition false falls through to here.
        self._compile_block(node.else_body)
        self.emit(f"j {end_label}")
        self.emit(f"{then_label}:")
        self._compile_block(node.then_body)
        self.emit(f"{end_label}:")

    def _compile_stmt(self, node: Stmt) -> None:
        # Scratch registers never need to live across statements -- r1..r9
        # and c[k] are the only real state -- so reset the pool per
        # statement rather than tracking cross-statement liveness.
        self._temp_top = _TEMP_BASE
        if isinstance(node, Assign):
            self._compile_assign(node)
        elif isinstance(node, If):
            self._compile_if(node)
        else:
            raise TypeError(f"unexpected statement node: {node!r}")

    def _compile_block(self, stmts: List[Stmt]) -> None:
        for stmt in stmts:
            self._compile_stmt(stmt)


def _generate_riscv(block: List[Stmt]) -> str:
    codegen = _Codegen()
    codegen._compile_block(block)
    if not codegen.lines:
        # An empty classical block still needs non-empty, harmless assembly
        # text (riscv_emulator.py's load_program handles it fine; x0 writes
        # are no-ops per its own set_register).
        codegen.emit("li x0, 0")
    return "\n".join(codegen.lines) + "\n"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compile_hybrid(hybrid_qasm_str: str) -> Tuple[List[str], str]:
    quantum_source, classical_source = _split_classical_block(hybrid_qasm_str)

    # Reuse the L1 parser/validator to catch malformed/out-of-whitelist
    # quantum statements early, with the same error style as L1.
    circuit = parse_qasm2(quantum_source)
    validate_circuit(circuit)

    quantum_ops = _extract_quantum_ops(quantum_source)
    block = _parse_classical(classical_source)
    assembly = _generate_riscv(block)
    return quantum_ops, assembly
