// Local-execution fixture only — NOT part of the transpile() contract output.
//
// amazon-braket-sdk's LocalSimulator does a bare open("stdgates.inc") relative
// to cwd when interpreting raw OpenQASM 3 text; it does not bundle a copy of
// its own (confirmed on both the antlr-pinned 1.99.0 and the latest 1.110.1
// releases). Its interpreter also does not implement the OpenQASM3 core-
// language CX primitive — gates must resolve to its own native registry
// (braket.default_simulator.gate_operations.BRAKET_GATES).
//
// h, x, s, t, rz, ry, swap already match that native registry under the same
// qelib1 name — no wrapper needed. These need a rename:
gate sdg a { si a; }
gate tdg a { ti a; }
gate cx a, b { cnot a, b; }
gate ccx a, b, c { ccnot a, b, c; }
// p is needed because starter_kit/emitters.py's Braket emitter defines cu1
// inline (via cx + p, see gate_identities.md) rather than relying on any
// interpreter-specific native alias — it is NOT redefined here to avoid
// colliding with that inline definition when both are loaded together.
gate p(lambda) a { phaseshift(lambda) a; }
