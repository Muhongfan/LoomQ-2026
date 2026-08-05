"""Backend execution + result normalization to the unified LoomQ JSON schema.

Both live local SDKs were found (via direct empirical probing, not assumption)
to silently ignore the classical target index in `measure q[i] -> c[j];` /
`c[j] = measure q[i];` and instead position each measured bit in their raw
output string using a DIFFERENT rule each:

  - spinqit (pinned 0.2.4): raw_string[i] = value of qubit i. The `-> c[j]`
    target is ignored outright; position is always the qubit's own index,
    and the string is always padded to n_qubits characters.
  - amazon-braket-sdk (pinned 1.99.0, LocalSimulator + raw OpenQASM3 text):
    raw_string position = the ORDER measure statements were written in the
    program (first statement executed -> leftmost character), independent
    of both the qubit index and the `c[j] =` target. Only explicitly
    measured qubits appear at all (no padding).

Neither convention matches target_ir_contract.md's required schema (`key`
must be `c[n-1]...c[1]c[0]`, rightmost char = c[0]). Since we control both
the emitted text and know the true qubit->clbit mapping from Circuit IR,
_normalize_spinq_counts/_normalize_braket_counts reconstruct the correct
key ourselves rather than trusting either SDK's raw positions.
"""

import contextlib
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from .circuit_ir import Circuit
    from .emitters import _is_full_identity_measurement, emit_braket, emit_originq, emit_spinq
    from .lowering import lower
    from .validator import validate_circuit
except ImportError:
    from circuit_ir import Circuit
    from emitters import _is_full_identity_measurement, emit_braket, emit_originq, emit_spinq
    from lowering import lower
    from validator import validate_circuit

REPO_ROOT = Path(__file__).resolve().parent
BRAKET_STDLIB_DIR = REPO_ROOT / "braket_local_stdlib"


def _braket_measurement_emission_order(circuit: Circuit) -> List[Tuple[int, int]]:
    """Must match emit_braket's own statement emission order exactly."""
    if _is_full_identity_measurement(circuit):
        return [(i, i) for i in range(circuit.n_qubits)]
    return list(circuit.measurements)


def _normalize_spinq_counts(raw_counts: Dict[str, int], circuit: Circuit) -> Dict[str, int]:
    normalized: Dict[str, int] = {}
    for raw_key, count in raw_counts.items():
        bits = ["0"] * circuit.n_clbits
        for qubit, clbit in circuit.measurements:
            bits[clbit] = raw_key[qubit]
        key = "".join(reversed(bits))
        normalized[key] = normalized.get(key, 0) + count
    return normalized


def _normalize_braket_counts(raw_counts: Dict[str, int], circuit: Circuit) -> Dict[str, int]:
    order = _braket_measurement_emission_order(circuit)
    normalized: Dict[str, int] = {}
    for raw_key, count in raw_counts.items():
        bits = ["0"] * circuit.n_clbits
        for position, (_qubit, clbit) in enumerate(order):
            bits[clbit] = raw_key[position]
        key = "".join(reversed(bits))
        normalized[key] = normalized.get(key, 0) + count
    return normalized


def _execute_spinq(qasm_text: str, shots: int) -> Dict[str, int]:
    from spinqit import BasicSimulatorConfig, get_basic_simulator, get_compiler

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
        return {str(key): int(value) for key, value in result.counts.items()}
    finally:
        os.unlink(tmp.name)


@contextlib.contextmanager
def _cwd(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _execute_braket(qasm_text: str, shots: int) -> Dict[str, int]:
    from braket.devices import LocalSimulator
    from braket.ir.openqasm import Program

    with _cwd(BRAKET_STDLIB_DIR):
        device = LocalSimulator()
        task = device.run(Program(source=qasm_text), shots=shots)
        return {str(key): int(value) for key, value in task.result().measurement_counts.items()}


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_spinq(circuit: Circuit, shots: int) -> Dict:
    lowered = lower(circuit, "spinq")
    qasm_text = emit_spinq(lowered)
    raw_counts = _execute_spinq(qasm_text, shots)
    counts = _normalize_spinq_counts(raw_counts, circuit)
    return {
        "backend": "spinq_basic_simulator",
        "job_id": f"spinq-local-{uuid.uuid4().hex[:12]}",
        "shots": shots,
        "counts": counts,
        "bit_order": "little",
        "timestamp": _timestamp(),
        "meta": {"transpiled_gates": len(lowered.gates), "qubits": circuit.n_qubits},
    }


def run_braket(circuit: Circuit, shots: int) -> Dict:
    lowered = lower(circuit, "braket")
    qasm_text = emit_braket(lowered)
    raw_counts = _execute_braket(qasm_text, shots)
    counts = _normalize_braket_counts(raw_counts, circuit)
    return {
        "backend": "braket_local_simulator",
        "job_id": f"braket-local-{uuid.uuid4().hex[:12]}",
        "shots": shots,
        "counts": counts,
        "bit_order": "little",
        "timestamp": _timestamp(),
        "meta": {"transpiled_gates": len(lowered.gates), "qubits": circuit.n_qubits},
    }


def run_originq(circuit: Circuit, shots: int) -> Dict:
    raise NotImplementedError("OriginQ execution is pending API token issuance")


RUNNERS = {
    "spinq": run_spinq,
    "braket": run_braket,
    "originq": run_originq,
}


def run(circuit: Circuit, target: str, shots: int) -> Dict:
    validate_circuit(circuit)
    return RUNNERS[target](circuit, shots)
