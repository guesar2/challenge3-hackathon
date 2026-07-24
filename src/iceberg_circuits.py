"""
iceberg_circuits.py

pytket circuit builders for the Iceberg [[k+2, k, 2]] quantum
error-detection code: fault-tolerant initialisation and syndrome
measurement, and the universal logical-gate compilers (single- and
two-qubit logical rotations, compiled as one MS/ZZPhase gate plus up to
four single-qubit Cliffords per the paper's Eqs. 1-12).

Physical qubit indexing matches iceberg_code.py: 0..k-1 for [k], k for t,
k+1 for b (n = k+2 total "code register" qubits). Circuits in this module
that need ancillas append them after index n (index n = flag/ancilla a1,
index n+1 = ancilla a2 where both are used).

On fault tolerance and where these circuits come from: the paper's Fig. 1
diagrams (the literal gate-by-gate init/syndrome-measurement circuits) are
images this implementation never had access to (see docs/ICEBERG_QEC_PLAN.md
Phase 0) -- Chao & Reichardt (arXiv:1705.02329) independently confirms this
code's stabilizer/logical-operator structure but its own flagged-circuit
figures are likewise images, not text. The circuits below are built from
first principles (the standard "flag qubit catches back-propagating faults"
technique) and verified computationally against the paper's own
fault-tolerance definition -- see tests/test_iceberg_fault_tolerance.py,
which exhaustively injects every single/two-qubit Pauli fault at every gate
location and checks no undetected logical error results. That test suite is
the actual correctness authority for these circuits, not a diagram match.
"""
import math

from pytket import Circuit

from iceberg_code import b_index, t_index, validate_k

# --- Universal logical gate set -------------------------------------------
#
# Every logical rotation exp(-i*theta*P/2) for a two-qubit physical Pauli
# string P = sigma_i sigma_j compiles to one ZZPhase (MS) gate plus a basis
# change on each qubit before/after -- verified numerically (phase-exact,
# not just up to global phase) against scipy.linalg.expm of the intended
# generator:
#   zz: no basis change (ZZPhase *is* exp(-i*theta*Z_iZ_j/2))
#   xx: H before and after each qubit (H Z H = X)          -- 4 Cliffords
#   yy: Rx(-0.5) before, Rx(+0.5) after each qubit (in      -- 4 Cliffords
#       pytket's half-turn convention, Rx(-pi/2) Z Rx(-pi/2)^dag = Y)
# matching the paper's "up to four single-qubit Clifford gates" bound.


def _basis_change(circuit, sigma, qubits, inverse, condition_kwargs):
    if sigma == 'zz':
        return
    if sigma == 'xx':
        for q in qubits:
            circuit.H(q, **condition_kwargs)
    elif sigma == 'yy':
        angle = 0.5 if inverse else -0.5
        for q in qubits:
            circuit.Rx(angle, q, **condition_kwargs)
    else:
        raise ValueError(f"sigma must be 'xx', 'yy', or 'zz', got {sigma!r}")


def compile_pairwise_rotation(circuit, sigma, qubit_i, qubit_j, theta, condition=None):
    """Append exp(-i*theta*sigma_i sigma_j/2) to `circuit`, sigma in
    {'xx','yy','zz'}, compiled as basis-change + ZZPhase + inverse
    basis-change (see module docstring for the verified identities).
    theta is in radians; ZZPhase takes pytket half-turns internally.

    condition: optional pytket classical condition (e.g. if_not_bit(...))
    passed through to every gate -- used for native early-exit gating
    (skip this rotation entirely once a discard flag has fired), see
    iceberg_tfim_circuit.py. pytket's gate methods reject condition=None
    outright (it must be omitted, not passed as None), so it's only
    included in the kwargs when actually given.
    """
    condition_kwargs = {} if condition is None else {"condition": condition}
    _basis_change(circuit, sigma, (qubit_i, qubit_j), inverse=False, condition_kwargs=condition_kwargs)
    circuit.ZZPhase(theta / math.pi, qubit_i, qubit_j, **condition_kwargs)
    _basis_change(circuit, sigma, (qubit_i, qubit_j), inverse=True, condition_kwargs=condition_kwargs)


def compile_logical_rx(circuit, k, i, theta, condition=None):
    """exp(-i*theta*Xbar_i/2), Xbar_i = X_i X_t (Eq. 1) -- one MS gate on
    (i, t) plus 4 Cliffords (xx-type compilation)."""
    validate_k(k)
    compile_pairwise_rotation(circuit, 'xx', i, t_index(k), theta, condition=condition)


def compile_logical_rz(circuit, k, i, theta, condition=None):
    """exp(-i*theta*Zbar_i/2), Zbar_i = Z_i Z_b (Eq. 5) -- one bare
    ZZPhase gate on (i, b), no extra Cliffords needed."""
    validate_k(k)
    compile_pairwise_rotation(circuit, 'zz', i, b_index(k), theta, condition=condition)


def compile_logical_rxx(circuit, k, i, j, theta, condition=None):
    """exp(-i*theta*Xbar_i Xbar_j/2) = exp(-i*theta*X_iX_j/2) (Eq. 2) --
    direct physical XX rotation on (i, j), no t/b involvement."""
    validate_k(k)
    compile_pairwise_rotation(circuit, 'xx', i, j, theta, condition=condition)


def compile_logical_rzz(circuit, k, i, j, theta, condition=None):
    """exp(-i*theta*Zbar_i Zbar_j/2) = exp(-i*theta*Z_iZ_j/2) (Eq. 6) --
    direct physical ZZ rotation on (i, j), no t/b involvement."""
    validate_k(k)
    compile_pairwise_rotation(circuit, 'zz', i, j, theta, condition=condition)


def compile_logical_ryy(circuit, k, i, j, theta, condition=None):
    """exp(-i*theta*Ybar_i Ybar_j/2) = exp(-i*theta*Y_iY_j/2) (Eq. 9) --
    direct physical YY rotation on (i, j), no t/b involvement."""
    validate_k(k)
    compile_pairwise_rotation(circuit, 'yy', i, j, theta, condition=condition)


# --- Fault-tolerant initialisation -----------------------------------------


def build_iceberg_init(k, flag_bit=None):
    """Fault-tolerant preparation of the encoded all-zero state |0bar>^[k]
    -- the n=k+2 qubit GHZ state (|0...0> + |1...1>)/sqrt(2) -- using one
    flag ancilla (index n, appended after the n code-register qubits).

    Construction: a linear CNOT ladder builds the GHZ state (H on qubit 0,
    then CX(q_i, q_{i+1}) for i=0..n-2); the flag is CX'd from qubit 0
    right after the Hadamard (before qubit 0 is touched again) and CX'd
    from qubit n-1 at the very end. In the fault-free case qubit 0 and
    qubit n-1 always carry the same bit value in every branch of the GHZ
    superposition, so flag = q0 XOR q_{n-1} = 0 deterministically and
    factors out as a clean |0> -- any single fault that breaks that
    equality flips the flag with nonzero probability. Verified fault-
    tolerant (no single fault produces an undetected logical error) by
    tests/test_iceberg_fault_tolerance.py.

    If flag_bit is given (a pytket Bit), appends a Measure of the flag
    qubit into it; otherwise the flag is left as an unmeasured qubit (used
    by the fault-tolerance test suite, which projects on it directly
    rather than sampling a measurement).

    Returns (circuit, flag_qubit_index). circuit has n+1 qubits: 0..n-1 is
    the code register (matching iceberg_code.py's indexing), n is the flag.
    """
    n = validate_k(k)
    flag = n
    circuit = Circuit(n + 1)

    circuit.H(0)
    circuit.CX(0, flag)
    for i in range(n - 1):
        circuit.CX(i, i + 1)
    circuit.CX(n - 1, flag)

    if flag_bit is not None:
        circuit.Measure(circuit.qubits[flag], flag_bit)

    return circuit, flag


# --- Fault-tolerant syndrome measurement ------------------------------------


_BLOCK_A_ORDER = ('a2', 'a1', 'a1', 'a2')  # palindromic
_BLOCK_B_ORDER = ('a2', 'a1', 'a2', 'a1')  # alternating


def _syndrome_pairs(k):
    """Partition the n = k+2 physical qubits into k/2+1 pairs: (t, qubit_0)
    at one end, (qubit_{k-1}, b) at the other, and consecutive pairs of
    the remaining numbered qubits in between -- e.g. for k=6: (t,0), (1,2),
    (3,4), (5,b)."""
    t, b = t_index(k), b_index(k)
    pairs = [(t, 0)]
    for i in range(1, k - 2, 2):
        pairs.append((i, i + 1))
    pairs.append((k - 1, b))
    return pairs


def _apply_flag_block(circuit, pair, a1, a2, order):
    """Apply the 4 CNOTs of one ABBB...BA block: two couple a1 (target)
    to the pair's two qubits, two couple a2 (control) to them, in the
    relative order given by `order` (a1's first occurrence always targets
    pair[0] before pair[1], same for a2 -- only the *interleaving* between
    the two ancillas' gates differs between the 'A' and 'B' orderings)."""
    p, q = pair
    next_qubit = {'a1': iter((p, q)), 'a2': iter((p, q))}
    for label in order:
        qubit = next(next_qubit[label])
        if label == 'a2':
            circuit.CX(a2, qubit)
        else:
            circuit.CX(qubit, a1)


def build_iceberg_syndrome_measurement(k, sz_bit=None, sx_bit=None):
    """Fault-tolerant measurement of both stabilisers S_Z and S_X, using
    exactly two ancillas -- a1 (index n, accumulates S_Z parity, target of
    every CX) and a2 (index n+1, accumulates S_X parity, control of every
    CX) -- structured as the paper's own "ABBB...BA" sequence (Fig. 1(d)):
    the n physical qubits are covered in k/2+1 pairs (the two end pairs
    touch t and b respectively; see _syndrome_pairs), each pair processed
    by a 4-CNOT block coupling both ancillas to both qubits of the pair,
    with the two *end* blocks using a palindromic ancilla interleaving
    (a2,a1,a1,a2) and every block in between using a strictly alternating
    one (a2,a1,a2,a1) -- see _apply_flag_block.

    A bare, unflagged version of this circuit (each ancilla doing a plain
    CX ladder with no interleaving at all) was tried first and confirmed
    -- via exhaustive fault injection against the actual Gottesman
    fault-tolerance criterion (see eventually_detected_or_correct in
    tests/test_iceberg_fault_tolerance.py: a single fault must never
    produce an error that stays silently within the code space, i.e. keeps
    S_X=S_Z=+1, while being logically wrong -- propagating to a large
    *odd*-weight error is fine, since that's guaranteed caught on a future
    check) -- to have a real gap: a lone X fault on a2, since a2 is always
    a CX *control*, can ride forward onto every later-coupled data qubit
    and land as an *even*-weight X error, which commutes with both
    stabilizers forever. The ABBB...BA interleaving is what fixes this,
    by controlling which data qubit a1 has already sampled relative to
    when such a leak could reach it. Verified fault-tolerant by
    tests/test_iceberg_fault_tolerance.py.

    Does not disturb the encoded logical state when no fault occurs (S_Z,
    S_X commute with every logical operator, so this is a standard
    stabilizer measurement).

    Returns (circuit, a1_index, a2_index). circuit has n+2 qubits: 0..n-1
    code register (same indexing as build_iceberg_init), n=a1, n+1=a2.
    """
    n = validate_k(k)
    a1, a2 = n, n + 1
    circuit = Circuit(n + 2)

    circuit.H(a2)
    pairs = _syndrome_pairs(k)
    for idx, pair in enumerate(pairs):
        order = _BLOCK_A_ORDER if idx in (0, len(pairs) - 1) else _BLOCK_B_ORDER
        _apply_flag_block(circuit, pair, a1, a2, order)
    circuit.H(a2)

    if sz_bit is not None:
        circuit.Measure(circuit.qubits[a1], sz_bit)
    if sx_bit is not None:
        circuit.Measure(circuit.qubits[a2], sx_bit)

    return circuit, a1, a2


def build_iceberg_measurement(k):
    """Final measurement: destructively measure every code-register qubit
    in the Z basis. S_Z and every logical Zbar_i are reconstructed from
    these bits in post-processing (iceberg_decode.py); S_X must be
    extracted beforehand (build_iceberg_syndrome_measurement) since it
    can't be recovered from a Z-basis measurement.

    Returns (circuit, creg) where creg is the n-bit classical register
    holding the measurement outcomes, code-register qubit i -> creg[i].
    """
    n = validate_k(k)
    circuit = Circuit(n, n)
    circuit.measure_all()
    return circuit, circuit.bits
