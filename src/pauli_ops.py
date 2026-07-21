"""
pauli_ops.py

Core linear-algebra building blocks: single-qubit Pauli operators embedded
in an N-qubit Hilbert space, the TFIM Hamiltonian, and the collective
observables (total Z, total X, nearest-neighbor ZZ correlator) reused
throughout the project.

Sparse throughout: get_pauli() builds each embedded Pauli operator via a
*sparse* Kronecker product (scipy.sparse.kron), not np.kron. A dense
2^N x 2^N matrix is what actually makes this pipeline break down well
before N=20 -- e.g. at N=20 a dense complex128 matrix is 2^20 x 2^20 x
16 bytes ~= 17.6 TB, vs. a sparse single-Pauli operator's 2^N nonzeros
(~16.8 MB of data at N=20). This was previously dense (np.kron / np.eye),
which OOM'd the pipeline around N=12-14 -- well short of the 4-20 spin
range the Trotter simulation is supposed to cover -- even though the
*quantum circuit itself* (circuits.py) stays cheap at those sizes. Keeping
every operator here sparse is what lets the classical bookkeeping
(observable expectation values, and the ED reference where it's still
used) actually reach N=20 instead of being the thing that "breaks down".
This is also what run_ed.py's wall-clock runtime-scaling benchmark
measures -- the sparse implementation pushes that classical wall further
out than a dense np.kron implementation would.

Caching note: get_pauli() and build_collective_observables() are the most
called functions in the whole pipeline (every ED call and every time-step
of the exact time-evolution rebuilds them). They're cached here so callers
don't have to think about it -- the cache now holds sparse matrices, so
its memory footprint scales with N (via 2^N nonzeros per operator) rather
than N (via 4^N dense entries).
"""
from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix, identity, kron

_PAULI = {
    'x': csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex)),
    'y': csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex)),
    'z': csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex)),
}
_IDENTITY_2 = identity(2, format='csr', dtype=complex)


@lru_cache(maxsize=None)
def get_pauli(n: int, size: int, pauli_type: str) -> csr_matrix:
    """Return the full 2^size x 2^size Pauli operator acting on qubit n, as
    a sparse (csr) matrix -- built via sparse Kronecker products so memory
    scales with the operator's 2^size nonzeros, not a dense 4^size entries.

    Cached because the same (n, size, pauli_type) triple is requested many
    times across ED, observable construction, and time evolution.
    """
    if pauli_type not in _PAULI:
        raise ValueError("pauli_type must be 'x', 'y', or 'z'")
    op = _PAULI[pauli_type]
    full_op = csr_matrix(np.array([[1.0]]))
    for i in range(size):
        full_op = kron(full_op, op if i == n else _IDENTITY_2, format='csr')
    return full_op


def build_tfim_hamiltonian(N: int, J: float = 1.0, h: float = 1.0) -> csr_matrix:
    """Build H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i (periodic chain).

    get_pauli() already returns sparse (csr) operators, and scipy sparse
    arithmetic (+, -, @, scalar multiply) stays sparse throughout, so H is
    built and returned as a sparse matrix without ever materializing a
    dense 2^N x 2^N array.
    """
    H = csr_matrix((2 ** N, 2 ** N), dtype=complex)
    for i in range(N):
        j = (i + 1) % N
        H = H - J * (get_pauli(i, N, 'z') @ get_pauli(j, N, 'z'))
    for i in range(N):
        H = H - h * get_pauli(i, N, 'x')
    return H.real


def build_collective_observables(N: int):
    """Return (Mz, Mz^2, Mx, Mzz) collective operators for an N-site chain,
    all sparse.

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
    """Compute (<Z>_rms per site, <ZZ> per bond, <X> per site) for a
    statevector. Mz_sq_op/Mx_op/Mzz_op are sparse, so each `op @ vec` is a
    sparse matrix-vector product (dense ndarray in, dense ndarray out) --
    O(nonzeros) per observable per time step instead of O(4^N).
    """
    mz_sq = np.real(vec.conj() @ (Mz_sq_op @ vec))
    z_rms = np.sqrt(max(mz_sq, 0.0)) / N
    mzz = np.real(vec.conj() @ (Mzz_op @ vec)) / N
    x_exp = np.real(vec.conj() @ (Mx_op @ vec)) / N
    return z_rms, mzz, x_exp