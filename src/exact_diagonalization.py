"""
exact_diagonalization.py

Classical-baseline calculations: ground-state observables via sparse
diagonalization, and exact (dense) time evolution from a product state.
"""
import numpy as np
from scipy.sparse.linalg import eigsh

from pauli_ops import build_tfim_hamiltonian, build_collective_observables, expectation_values


def ed_baseline(N=6, h_values=(0.5, 1.0, 1.5), J=1.0, verbose=True):
    """Compute exact ground-state expectation values for the TFIM.

    Returns a list of dicts: {'h', 'mz_rms', 'mx', 'mzz', 'energy'}.
    """
    Mz_op, Mz_sq_op, Mx_op, Mzz_op = build_collective_observables(N)
    results = []

    if verbose:
        print("\n" + "=" * 82)
        print(f"EXACT DIAGONALIZATION BASELINE (N={N})")
        print("=" * 82)
        print(f"{'h/J':^8} | {'<Z>':^12} | {'<X>':^12} | {'<Zi Zi+1>':^14} | {'E/N':^12}")
        print("-" * 82)

    for h in h_values:
        H = build_tfim_hamiltonian(N, J=J, h=h)
        eigvals, eigvecs = eigsh(H, k=1, which='SA')
        gs = eigvecs[:, 0]

        z_rms, mzz, mx = expectation_values(gs, N, Mz_sq_op, Mx_op, Mzz_op)
        energy = eigvals[0].real / N
        results.append({'h': h, 'mz_rms': z_rms, 'mx': mx, 'mzz': mzz, 'energy': energy})

        if verbose:
            print(f"{h:^8.1f} | {z_rms:^12.6f} | {mx:^12.6f} | {mzz:^14.6f} | {energy:^12.6f}")

    if verbose:
        print("=" * 82 + "\n")
    return results


def ed_time_evolution_exact(N, h, J, dt, steps, initial_state_label=None):
    """Exact ED time evolution from a product state (bitstring label).

    Diagonalizes H once, then evolves psi(t) = sum_k exp(-i*E_k*t) <k|psi0> |k>.
    Returns: times, <Z>_rms per site, <Zi Zi+1> per bond, <X> per site.
    """
    if initial_state_label is None:
        initial_state_label = '0' * N

    H = build_tfim_hamiltonian(N, J=J, h=h).toarray()
    eigvals, eigvecs = np.linalg.eigh(H)  # H is Hermitian (real symmetric here)

    idx = int(initial_state_label, 2)
    psi0 = np.zeros(2 ** N, dtype=complex)
    psi0[idx] = 1.0
    coeffs = eigvecs.conj().T @ psi0  # overlap of psi0 with each eigenstate

    Mz_op, Mz_sq_op, Mx_op, Mzz_op = build_collective_observables(N)

    times = np.arange(1, steps + 1) * dt
    z_rms = np.zeros(steps)
    mzz = np.zeros(steps)
    x_exp = np.zeros(steps)

    for t_idx, t in enumerate(times):
        phase = np.exp(-1j * eigvals * t)
        vec = eigvecs @ (phase * coeffs)
        z_rms[t_idx], mzz[t_idx], x_exp[t_idx] = expectation_values(
            vec, N, Mz_sq_op, Mx_op, Mzz_op
        )

    return times, z_rms, mzz, x_exp
