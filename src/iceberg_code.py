"""
iceberg_code.py

Pure linear-algebra definitions for the Iceberg [[k+2, k, 2]] quantum
error-detection code (Self, Benedetti & Amaro, "Protecting Expressive
Circuits with a Quantum Error Detection Code", arXiv:2211.06703).

No pytket dependency here on purpose -- this module only builds sparse
matrices over the n = k+2 "code register" qubits (numpy/scipy), so its
correctness (stabilizer algebra, logical-operator identities) can be
tested for free, without ever touching a circuit or qnexus. See
iceberg_circuits.py for the pytket circuit builders that use this module's
definitions, and tests/test_iceberg_code.py for the identities checked
against Eqs. (1)-(12) of the paper's Supplementary Information.

Physical qubit indexing (code register only, ancillas are not part of
this module): indices 0..k-1 are the k "numbered" logical-carrier qubits
[k] = {1,...,k} (0-indexed here), index k is qubit t, index k+1 is qubit
b -- n = k+2 total, matching the paper's [n] = [k] u {t, b}.

Stabilizers: S_X = X^{ox n}, S_Z = Z^{ox n} (both acting on all n qubits).
Logical single-qubit operators: Xbar_i = X_i X_t, Zbar_i = Z_i Z_b, for
i in [k]. These two definitions are Eqs. (1) and (5) of the paper; every
other logical-operator identity (Eqs. 2-4, 6-12) is a consequence of them
plus the stabilizer structure, not a separate encoding choice -- so this
module defines only Xbar_i/Zbar_i as primitives, and the test suite
verifies the rest algebraically (matrix multiplication) rather than
hardcoding each equation as its own constructor.

This same code family is Chao & Reichardt's [[n, n-2, 2]] error-detecting
code (arXiv:1705.02329, Sec. V), which the Iceberg paper cites (their
Ref. [31] is Chao & Reichardt's companion two-ancilla flagged-circuit
paper) for its fault-tolerant init/syndrome-measurement construction.
Chao & Reichardt's own logical operators, "Xbar_j = X_1 X_{j+1}, Zbar_j =
Z_{j+1} Z_n", match exactly under the renaming qubit-1 -> t, qubit-n -> b
-- confirming this module's definitions against an independent source.
"""
from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix, identity, kron

_PAULI = {
    'x': csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex)),
    'y': csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex)),
    'z': csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex)),
    'i': csr_matrix(np.eye(2, dtype=complex)),
}


def validate_k(k: int) -> int:
    """Validate k (number of logical qubits) and return n = k+2.

    k must be even -- the paper's global-operator sign convention (Eq. 12,
    (-1)^(1+k/2)) is stated for even k, and every k used elsewhere in this
    repo's config.py (H2_N, H2_ADIABATIC_N, H2_VQE_N, N) already is.
    """
    if k < 2 or k % 2 != 0:
        raise ValueError(f"k must be an even integer >= 2, got {k}")
    return k + 2


def t_index(k: int) -> int:
    """Physical index of qubit t (0-indexed, code register only)."""
    return k


def b_index(k: int) -> int:
    """Physical index of qubit b (0-indexed, code register only)."""
    return k + 1


@lru_cache(maxsize=None)
def _embedded_pauli(qubit: int, n: int, pauli_type: str) -> csr_matrix:
    """Single Pauli operator on `qubit`, identity elsewhere, over n qubits."""
    op = csr_matrix(np.array([[1.0]]))
    for i in range(n):
        op = kron(op, _PAULI[pauli_type] if i == qubit else _PAULI['i'], format='csr')
    return op


def physical_pauli_string(k: int, pauli_map: dict) -> csr_matrix:
    """Build the physical Pauli-string operator on the n = k+2 qubit code
    register specified by pauli_map = {qubit_index: 'x'|'y'|'z'}.

    Qubits not present in pauli_map get identity. qubit_index uses this
    module's 0-indexed convention (0..k-1 for [k], k for t, k+1 for b).
    """
    n = validate_k(k)
    op = identity(2 ** n, format='csr', dtype=complex)
    for qubit, pauli_type in pauli_map.items():
        if not (0 <= qubit < n):
            raise ValueError(f"qubit index {qubit} out of range for n={n}")
        op = op @ _embedded_pauli(qubit, n, pauli_type)
    return op


def stabilizer_sx(k: int) -> csr_matrix:
    """S_X = X^{ox n}, product of X on every physical qubit."""
    n = validate_k(k)
    return physical_pauli_string(k, {i: 'x' for i in range(n)})


def stabilizer_sz(k: int) -> csr_matrix:
    """S_Z = Z^{ox n}, product of Z on every physical qubit."""
    n = validate_k(k)
    return physical_pauli_string(k, {i: 'z' for i in range(n)})


def logical_x(k: int, i: int) -> csr_matrix:
    """Xbar_i = X_i X_t (Eq. 1), for logical qubit i in [k] (0-indexed)."""
    if not (0 <= i < k):
        raise ValueError(f"logical qubit index {i} out of range for k={k}")
    return physical_pauli_string(k, {i: 'x', t_index(k): 'x'})


def logical_z(k: int, i: int) -> csr_matrix:
    """Zbar_i = Z_i Z_b (Eq. 5), for logical qubit i in [k] (0-indexed)."""
    if not (0 <= i < k):
        raise ValueError(f"logical qubit index {i} out of range for k={k}")
    return physical_pauli_string(k, {i: 'z', b_index(k): 'z'})


def logical_y(k: int, i: int) -> csr_matrix:
    """Ybar_i = i * Xbar_i * Zbar_i = Y_i (Y_t Y_b)... derived, not a
    paper primitive -- provided for completeness/tests since Y_i = i X_i Z_i
    and Xbar_i Zbar_i = (X_i X_t)(Z_i Z_b) = (X_i Z_i)(X_t Z_b) = (-i Y_i)(X_t Z_b),
    so Ybar_i := i * Xbar_i @ Zbar_i = Y_i X_t Z_b (self-consistent single-
    logical-qubit Pauli, commutes with Xbar_i/Zbar_i as expected of a
    logical Y i.e. {Xbar_i, Ybar_i} != 0, [Ybar_i, Xbar_j]=0 for j!=i etc.
    -- checked in tests, not assumed).
    """
    return 1j * (logical_x(k, i) @ logical_z(k, i))


def code_space_projector(k: int) -> csr_matrix:
    """Projector onto the +1 joint eigenspace of S_X and S_Z (the code
    space), P = (I + S_X)/2 @ (I + S_Z)/2. Dense-friendly only for small k
    (2^n x 2^n) -- used by tests, not by any circuit-building code.
    """
    n = validate_k(k)
    ident = identity(2 ** n, format='csr', dtype=complex)
    return ((ident + stabilizer_sx(k)) / 2) @ ((ident + stabilizer_sz(k)) / 2)


def ghz_all_zero_statevector(k: int) -> np.ndarray:
    """The logical all-zero state |0bar>^{ox[k]}, i.e. the GHZ state
    (|0>^{ox n} + |1>^{ox n}) / sqrt(2) over the n = k+2 physical qubits
    (Supp. Info: "From the definition of the stabilisers and logical
    operators we see that this state is the ... GHZ state").
    """
    n = validate_k(k)
    dim = 2 ** n
    sv = np.zeros(dim, dtype=complex)
    sv[0] = 1 / np.sqrt(2)       # |0...0>
    sv[dim - 1] = 1 / np.sqrt(2)  # |1...1>
    return sv
