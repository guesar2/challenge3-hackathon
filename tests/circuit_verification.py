"""
circuit_verification.py

Shared statevector-verification helpers for the Iceberg fault-tolerance
test suite (test_iceberg_fault_tolerance.py). Not part of the production
pipeline -- test-time only.

Approach: rather than a full branching mid-circuit-measurement simulator,
every circuit under test here (init, syndrome measurement) is built and
faulted as a PURELY UNITARY circuit (ancilla/flag qubits included as
qubits, never measured inside the circuit itself -- Measure ops are
omitted). This is exact for our purpose because none of these circuits'
own gates are conditioned on an earlier measurement outcome (accept/
discard is decided entirely in classical post-processing after the whole
circuit finishes, per the paper's post-selection protocol -- see
iceberg_decode.py) -- so nothing is lost by deferring "measurement" to a
final amplitude-projection step on the completed unitary evolution.

Confirmed empirically (see the numeric check that produced this file):
pytket's statevector/unitary convention places qubit index 0 as the most
significant bit -- Circuit(2).X(0).get_statevector() puts the amplitude at
index 2 (binary '10'), not index 1.
"""
import numpy as np

_PAULI_1Q = {
    'i': np.eye(2, dtype=complex),
    'x': np.array([[0, 1], [1, 0]], dtype=complex),
    'y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'z': np.array([[1, 0], [0, -1]], dtype=complex),
}

# All two-qubit Pauli faults except the trivial 'ii' (no-op) -- 15 total,
# matching the paper's "every two-qubit Pauli error placed after each of
# the CNOT gates".
TWO_QUBIT_PAULI_FAULTS = [
    (a, b) for a in _PAULI_1Q for b in _PAULI_1Q if not (a == 'i' and b == 'i')
]
# Single-qubit Pauli faults except identity -- 3 total, for H-gate/single-
# qubit-gate locations.
ONE_QUBIT_PAULI_FAULTS = ['x', 'y', 'z']


def project(state: np.ndarray, n_qubits: int, fixed_values: dict) -> tuple:
    """Project `state` (a 2**n_qubits statevector, qubit 0 = MSB) onto the
    computational-basis values in fixed_values = {qubit_index: 0 or 1}.

    Returns (probability, reduced_state), where reduced_state is the
    (renormalized) statevector over the remaining qubits, in their
    original relative order, or None if probability is ~0.
    """
    remaining_qubits = [q for q in range(n_qubits) if q not in fixed_values]
    dim_remaining = 2 ** len(remaining_qubits)
    reduced = np.zeros(dim_remaining, dtype=complex)
    for idx in range(len(state)):
        bits = [(idx >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        if all(bits[q] == v for q, v in fixed_values.items()):
            remaining_bits = [bits[q] for q in remaining_qubits]
            remaining_idx = 0
            for b in remaining_bits:
                remaining_idx = (remaining_idx << 1) | b
            reduced[remaining_idx] = state[idx]
    prob = float(np.vdot(reduced, reduced).real)
    if prob < 1e-12:
        return prob, None
    return prob, reduced / np.sqrt(prob)


def _embedded_pauli_1q(qubit: int, n: int, pauli_type: str) -> np.ndarray:
    op = np.array([[1.0]], dtype=complex)
    for i in range(n):
        op = np.kron(op, _PAULI_1Q[pauli_type] if i == qubit else _PAULI_1Q['i'])
    return op


def matches_ideal_up_to_weight1_pauli(actual: np.ndarray, ideal: np.ndarray, n_qubits: int,
                                       atol: float = 1e-6):
    """Check whether `actual` equals (up to global phase) `ideal` with at
    most one single-qubit Pauli error applied.

    NOTE: this is a *stricter* check than the paper's own fault-tolerance
    definition actually requires, and is kept here only for diagnostics/
    comparison -- see eventually_detected_or_correct for the criterion
    tests/test_iceberg_fault_tolerance.py actually asserts against. A
    weight>1 error is not automatically a problem: it only matters whether
    it's *ever* detectable, not whether it happens to be small.

    Returns (matched: bool, description: str) for diagnostics.
    """
    candidates = [("I", np.eye(2 ** n_qubits, dtype=complex))]
    for q in range(n_qubits):
        for p in ONE_QUBIT_PAULI_FAULTS:
            candidates.append((f"{p.upper()}{q}", _embedded_pauli_1q(q, n_qubits, p)))

    for name, P in candidates:
        corrected = P @ actual
        # compare up to global phase
        overlap = np.vdot(ideal, corrected)
        if abs(abs(overlap) - 1.0) < atol:
            return True, name
    return False, "no weight<=1 Pauli matches"


def eventually_detected_or_correct(actual: np.ndarray, ideal: np.ndarray, sx_op, sz_op,
                                    atol: float = 1e-6):
    """The actual Gottesman fault-tolerance criterion (confirmed against
    the paper's own author): a single fault is fine as long as, whenever
    *this round's* flags don't fire, the result is either (a) exactly the
    ideal state (no error at all), or (b) guaranteed to flip S_X or S_Z on
    the *next* check (a future syndrome round, or the final measurement's
    reconstructed S_Z) -- it does NOT need to be caught by this specific
    round's own ancilla readings, and it does NOT need to be small (a
    weight-3 error that's guaranteed detectable later is fine; a weight-2
    error that stays silently within the code space forever is not).

    Since every fault considered here is a Pauli-type operator applied to
    an exact stabilizer eigenstate, <actual|S|actual> is always exactly
    +1 or -1 (never in between) for S in {S_X, S_Z} -- an in-between value
    indicates a bug in the simulation, not partial detectability, and is
    reported as a failure via the returned description.

    sx_op, sz_op: iceberg_code.stabilizer_sx(k)/stabilizer_sz(k) (sparse
    matrices over the same n_qubits as `actual`/`ideal`).

    Returns (ok: bool, description: str) for diagnostics.
    """
    overlap = np.vdot(ideal, actual)
    if abs(abs(overlap) - 1.0) < atol:
        return True, "no error"

    exp_sz = (np.vdot(actual, sz_op @ actual)).real
    exp_sx = (np.vdot(actual, sx_op @ actual)).real
    if exp_sz < -0.5:
        return True, f"guaranteed to flip S_Z on next check (<S_Z>={exp_sz:.3f})"
    if exp_sx < -0.5:
        return True, f"guaranteed to flip S_X on next check (<S_X>={exp_sx:.3f})"
    if abs(exp_sz - 1.0) > atol or abs(exp_sx - 1.0) > atol:
        return False, f"unexpected in-between eigenvalue -- check simulation (<S_Z>={exp_sz:.3f}, <S_X>={exp_sx:.3f})"
    return False, f"undetectable logical error: stays in code space, wrong value (<S_Z>={exp_sz:.3f}, <S_X>={exp_sx:.3f})"
