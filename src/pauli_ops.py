"""
pauli_ops.py

Core linear-algebra building blocks: single-qubit Pauli operators embedded
in an N-qubit Hilbert space, the TFIM Hamiltonian, and the collective
observables (total Z, total X, nearest-neighbor ZZ correlator) reused
throughout the project.

Caching note: get_pauli() and build_collective_observables() are the most
called functions in the whole pipeline (every ED call and every time-step
of the exact time-evolution rebuilds them). They're cached here so callers
don't have to think about it.
"""
from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix

_PAULI = {
    'x': np.array([[0, 1], [1, 0]], dtype=complex),
    'y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'z': np.array([[1, 0], [0, -1]], dtype=complex),
}


@lru_cache(maxsize=None)
def get_pauli(n: int, size: int, pauli_type: str) -> np.ndarray:
    """Return the full 2^size x 2^size Pauli matrix acting on qubit n.

    Cached because the same (n, size, pauli_type) triple is requested many
    times across ED, observable construction, and time evolution.
    """
    if pauli_type not in _PAULI:
        raise ValueError("pauli_type must be 'x', 'y', or 'z'")
    op = _PAULI[pauli_type]
    full_op = np.array([[1.0]])
    for i in range(size):
        full_op = np.kron(full_op, op if i == n else np.eye(2))
    return full_op


def build_tfim_hamiltonian(N: int, J: float = 1.0, h: float = 1.0) -> csr_matrix:
    """Build H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i (periodic chain).

    Note: get_pauli() returns dense ndarrays. Subtracting a dense ndarray
    from a scipy sparse matrix in-place (H -= ...) silently degrades the
    result to a numpy.matrix rather than raising an error or staying
    sparse - a bug present in the original notebook. Wrapping each term
    in csr_matrix(...) keeps the arithmetic (and the return type) sparse.
    """
    H = csr_matrix((2 ** N, 2 ** N), dtype=complex)
    for i in range(N):
        j = (i + 1) % N
        H = H - J * csr_matrix(get_pauli(i, N, 'z') @ get_pauli(j, N, 'z'))
    for i in range(N):
        H = H - h * csr_matrix(get_pauli(i, N, 'x'))
    return H.real


def build_collective_observables(N: int):
    """Return (Mz, Mz^2, Mx, Mzz) collective operators for an N-site chain.

    Mz  = sum_i Z_i
    Mx  = sum_i X_i
    Mzz = sum_i Z_i Z_{i+1}  (periodic)
    """
    Z_ops = [get_pauli(i, N, 'z') for i in range(N)]
    X_ops = [get_pauli(i, N, 'x') for i in range(N)]
    Mz_op = sum(Z_ops)
    Mx_op = sum(X_ops)
    Mzz_op = sum(Z_ops[i] @ Z_ops[(i + 1) % N] for i in range(N))
    return Mz_op, Mz_op @ Mz_op, Mx_op, Mzz_op


def expectation_values(vec: np.ndarray, N: int, Mz_sq_op, Mx_op, Mzz_op):
    """Compute (<Z>_rms per site, <ZZ> per bond, <X> per site) for a statevector."""
    mz_sq = np.real(vec.conj() @ (Mz_sq_op @ vec))
    z_rms = np.sqrt(max(mz_sq, 0.0)) / N
    mzz = np.real(vec.conj() @ (Mzz_op @ vec)) / N
    x_exp = np.real(vec.conj() @ (Mx_op @ vec)) / N
    return z_rms, mzz, x_exp
