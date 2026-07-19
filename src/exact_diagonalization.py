"""
exact_diagonalization.py

Exact Diagonalization (ED) baseline for the Transverse-Field Ising Model (TFIM).

Implements:
  1. Pauli matrix construction for arbitrary qubit count
  2. Sparse Hamiltonian assembly: H = -J Σ Z_i Z_j - h Σ X_i
  3. Ground-state properties (energy, magnetization, correlations)
  4. Time evolution from a product initial state |0...0>

Uses scipy.sparse for memory efficiency (scales to N ~ 20 on standard hardware).

References:
  - Sachdev, Quantum Phase Transitions, Chap. 1
  - Suzuki (1976), Trotter decomposition
"""

import numpy as np
from scipy.sparse import csr_matrix, kron, eye
from scipy.sparse.linalg import eigsh, expm_multiply


# ---------------------------------------------------------------------------
# Pauli operators (sparse)
# ---------------------------------------------------------------------------

PAULI_X = csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
PAULI_Y = csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
PAULI_Z = csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
PAULI_I = csr_matrix(np.eye(2, dtype=complex))

PAULI_MAP = {'x': PAULI_X, 'y': PAULI_Y, 'z': PAULI_Z, 'i': PAULI_I}


def pauli_on_qubit(n, N, pauli_type):
    """
    Return the sparse Pauli operator acting on qubit n in an N-qubit system.

    Parameters
    ----------
    n : int
        Qubit index (0-based).
    N : int
        Total number of qubits.
    pauli_type : str
        One of 'x', 'y', 'z', 'i'.

    Returns
    -------
    csr_matrix
        2^N x 2^N sparse matrix.
    """
    if pauli_type not in PAULI_MAP:
        raise ValueError(f"pauli_type must be one of {list(PAULI_MAP.keys())}")

    op = PAULI_MAP[pauli_type]

    # Build I ⊗ I ⊗ ... ⊗ op ⊗ ... ⊗ I
    factors = [PAULI_I] * N
    factors[n] = op

    result = factors[0]
    for f in factors[1:]:
        result = kron(result, f, format='csr')

    return result


def two_qubit_pauli(n, m, N, p1, p2):
    """
    Return the sparse two-qubit Pauli operator p1_n ⊗ p2_m in an N-qubit system.

    Parameters
    ----------
    n, m : int
        Qubit indices.
    N : int
        Total number of qubits.
    p1, p2 : str
        Pauli types for qubits n and m.

    Returns
    -------
    csr_matrix
        2^N x 2^N sparse matrix.
    """
    if p1 not in PAULI_MAP or p2 not in PAULI_MAP:
        raise ValueError("Pauli types must be 'x', 'y', 'z', or 'i'")

    factors = [PAULI_I] * N
    factors[n] = PAULI_MAP[p1]
    factors[m] = PAULI_MAP[p2]

    result = factors[0]
    for f in factors[1:]:
        result = kron(result, f, format='csr')

    return result


# ---------------------------------------------------------------------------
# Hamiltonian construction
# ---------------------------------------------------------------------------

def build_tfim_hamiltonian(N, J=1.0, h=1.0, boundary='open'):
    """
    Build the TFIM Hamiltonian: H = -J Σ Z_i Z_j - h Σ X_i

    Parameters
    ----------
    N : int
        Number of spins (qubits).
    J : float
        ZZ coupling strength (ferromagnetic when J > 0).
    h : float
        Transverse field strength.
    boundary : str
        'open' or 'periodic'. Open: i couples to i+1 for i=0,...,N-2.
        Periodic: i couples to (i+1) % N.

    Returns
    -------
    csr_matrix
        Sparse Hamiltonian matrix (2^N x 2^N).
    """
    dim = 2 ** N
    H = csr_matrix((dim, dim), dtype=complex)

    # ZZ interactions
    num_bonds = N if boundary == 'periodic' else N - 1
    for i in range(num_bonds):
        j = (i + 1) % N
        H -= J * two_qubit_pauli(i, j, N, 'z', 'z')

    # Transverse field X
    for i in range(N):
        H -= h * pauli_on_qubit(i, N, 'x')

    # Hamiltonian is real and symmetric
    return H.real


# ---------------------------------------------------------------------------
# Ground state properties
# ---------------------------------------------------------------------------

def ground_state_properties(N, J=1.0, h=1.0, boundary='open'):
    """
    Compute ground-state expectation values for the TFIM.

    Returns
    -------
    dict with keys:
        'energy' : float — ground state energy per site
        'psi'    : ndarray — ground state wavefunction
        'mz'     : float — <Z> / N (average magnetization in Z)
        'mx'     : float — <X> / N (average magnetization in X)
        'mzz'    : float — average nearest-neighbor ZZ correlation
        'xi'     : float — correlation length (from ZZ decay)
    """
    H = build_tfim_hamiltonian(N, J, h, boundary)

    # Compute ground state (k=1 lowest eigenvalue)
    eigvals, eigvecs = eigsh(H, k=1, which='SA')
    E0 = eigvals[0].real
    psi = eigvecs[:, 0]

    # Ensure real ground state (degeneracy may give complex phases)
    psi = psi / np.linalg.norm(psi)

    # Build observable operators
    Mz_op = csr_matrix((2**N, 2**N), dtype=complex)
    Mx_op = csr_matrix((2**N, 2**N), dtype=complex)
    Mzz_op = csr_matrix((2**N, 2**N), dtype=complex)

    for i in range(N):
        Mz_op += pauli_on_qubit(i, N, 'z')
        Mx_op += pauli_on_qubit(i, N, 'x')

    num_bonds = N if boundary == 'periodic' else N - 1
    for i in range(num_bonds):
        j = (i + 1) % N
        Mzz_op += two_qubit_pauli(i, j, N, 'z', 'z')

    # Expectation values
    mz = (psi.conj().T @ (Mz_op @ psi)).real / N
    mx = (psi.conj().T @ (Mx_op @ psi)).real / N
    mzz = (psi.conj().T @ (Mzz_op @ psi)).real / num_bonds

    # Correlation length from ZZ correlations at all distances
    zz_corrs = []
    for d in range(1, N):
        corr_sum = 0.0
        count = 0
        for i in range(N):
            j = (i + d) % N
            if boundary == 'open' and j >= N:
                continue
            op = two_qubit_pauli(i, j, N, 'z', 'z')
            corr_sum += (psi.conj().T @ (op @ psi)).real
            count += 1
        if count > 0:
            zz_corrs.append(corr_sum / count)

    # Estimate correlation length from exponential fit
    xi = estimate_correlation_length(zz_corrs)

    return {
        'energy': E0 / N,
        'psi': psi,
        'mz': mz,
        'mx': mx,
        'mzz': mzz,
        'xi': xi,
        'zz_corrs': zz_corrs,
    }


def estimate_correlation_length(zz_corrs):
    """
    Estimate correlation length from ZZ correlation decay.
    Fit log(|corr|) vs distance to a line: slope = -1/xi.
    """
    if len(zz_corrs) < 2:
        return 0.0

    distances = np.arange(1, len(zz_corrs) + 1)
    # Use absolute value and avoid log(0)
    log_corrs = np.log(np.maximum(np.abs(zz_corrs), 1e-15))

    # Linear fit
    slope, _ = np.polyfit(distances, log_corrs, 1)
    xi = -1.0 / slope if slope < 0 else np.inf

    return xi


# ---------------------------------------------------------------------------
# Time evolution 
# ---------------------------------------------------------------------------

def evolve_state_ed(H, psi0, t):
    """
    Evolve a state under Hamiltonian H for time t using exact matrix exponential.

    psi(t) = exp(-i * H * t) * psi0

    Uses scipy.sparse.linalg.expm_multiply for efficiency with sparse H.

    Parameters
    ----------
    H : csr_matrix
        Hamiltonian (sparse).
    psi0 : ndarray
        Initial state vector.
    t : float
        Evolution time.

    Returns
    -------
    ndarray
        Time-evolved state vector (normalized).
    """
    # expm_multiply computes exp(A) @ b efficiently
    # We need exp(-i * H * t) @ psi0
    psi_t = expm_multiply(-1j * H * t, psi0)

    # Normalize (numerical safety)
    norm = np.linalg.norm(psi_t)
    if norm > 1e-15:
        psi_t = psi_t / norm

    return psi_t


def time_evolution_tfim(N, J=1.0, h=1.0, t_max=5.0, num_points=100,
                        initial_state='zeros', boundary='open'):
    """
    Compute time evolution of TFIM from a product initial state.

    Parameters
    ----------
    N : int
        Number of spins.
    J, h : float
        Hamiltonian parameters.
    t_max : float
        Maximum evolution time.
    num_points : int
        Number of time points.
    initial_state : str
        'zeros' -> |0...0> (all spins down, ferromagnetic)
        'plus'  -> |+...+> (all spins in X eigenstate, paramagnetic)
    boundary : str
        'open' or 'periodic'.

    Returns
    -------
    dict with keys:
        'times' : ndarray — time points
        'mz_t'  : ndarray — <Z(t)> / N at each time
        'mx_t'  : ndarray — <X(t)> / N at each time
        'mzz_t' : ndarray — <Z_i Z_{i+1}(t)> at each time
        'entropy_t' : ndarray — von Neumann entropy of half-chain
    """
    H = build_tfim_hamiltonian(N, J, h, boundary)
    times = np.linspace(0, t_max, num_points)

    # Initial state
    if initial_state == 'zeros':
        psi0 = np.zeros(2**N, dtype=complex)
        psi0[0] = 1.0  # |0...0>
    elif initial_state == 'plus':
        # |+> = (|0> + |1>)/sqrt(2), so |+...+> = tensor product
        psi0 = np.ones(2**N, dtype=complex) / np.sqrt(2**N)
    else:
        raise ValueError("initial_state must be 'zeros' or 'plus'")

    # Pre-build observable operators
    Mz_ops = [pauli_on_qubit(i, N, 'z') for i in range(N)]
    Mx_ops = [pauli_on_qubit(i, N, 'x') for i in range(N)]

    num_bonds = N if boundary == 'periodic' else N - 1
    Mzz_ops = []
    for i in range(num_bonds):
        j = (i + 1) % N
        Mzz_ops.append(two_qubit_pauli(i, j, N, 'z', 'z'))

    # Time evolution
    mz_t = np.zeros(num_points)
    mx_t = np.zeros(num_points)
    mzz_t = np.zeros(num_points)
    entropy_t = np.zeros(num_points)

    for idx, t in enumerate(times):
        psi_t = evolve_state_ed(H, psi0, t)

        # Magnetizations
        mz_val = sum((psi_t.conj().T @ (op @ psi_t)).real for op in Mz_ops) / N
        mx_val = sum((psi_t.conj().T @ (op @ psi_t)).real for op in Mx_ops) / N

        # Nearest-neighbor correlations
        mzz_val = sum((psi_t.conj().T @ (op @ psi_t)).real for op in Mzz_ops) / num_bonds

        # Half-chain entanglement entropy
        S = half_chain_entropy(psi_t, N)

        mz_t[idx] = mz_val
        mx_t[idx] = mx_val
        mzz_t[idx] = mzz_val
        entropy_t[idx] = S

    return {
        'times': times,
        'mz_t': mz_t,
        'mx_t': mx_t,
        'mzz_t': mzz_t,
        'entropy_t': entropy_t,
        'initial_state': initial_state,
        'N': N, 'J': J, 'h': h,
    }


def half_chain_entropy(psi, N):
    """
    Compute the von Neumann entropy of the reduced density matrix
    for the first N//2 qubits.

    S = -Tr(ρ_A log ρ_A) where A = first N//2 qubits.
    """
    n_A = N // 2
    n_B = N - n_A

    # Reshape psi into a matrix: psi_{i_A, i_B}
    psi_matrix = psi.reshape((2**n_A, 2**n_B))

    # SVD: psi = U @ diag(s) @ Vh
    # Reduced density matrix eigenvalues are s^2
    _, s, _ = np.linalg.svd(psi_matrix, full_matrices=False)

    # Entropy
    lambdas = s**2
    lambdas = lambdas[lambdas > 1e-15]  # avoid log(0)
    entropy = -np.sum(lambdas * np.log2(lambdas))

    return entropy



# ---------------------------------------------------------------------------
# Batch computation for phase transition analysis
# ---------------------------------------------------------------------------

def ed_phase_transition_scan(N=8, h_values=None, J=1.0, boundary='open'):
    """
    Scan h/J values to map the quantum phase transition.

    Parameters
    ----------
    N : int
        System size.
    h_values : list or ndarray
        Values of h/J to scan. Default: np.linspace(0.1, 2.0, 20).
    J : float
        Coupling (fixed to 1.0 by convention).
    boundary : str
        'open' or 'periodic'.

    Returns
    -------
    dict with arrays of results vs h/J.
    """
    if h_values is None:
        h_values = np.linspace(0.1, 2.0, 20)

    results = {
        'h_over_J': [],
        'energy': [],
        'mz': [],
        'mx': [],
        'mzz': [],
        'xi': [],
    }

    print("\n" + "=" * 70)
    print(f"ED PHASE TRANSITION SCAN (N={N}, boundary={boundary})")
    print("=" * 70)
    print(f"{'h/J':^8} | {'E/N':^12} | {'<Z>':^12} | {'<X>':^12} | {'<ZZ>':^12} | {'ξ':^10}")
    print("-" * 70)

    for h in h_values:
        props = ground_state_properties(N, J, h, boundary)

        results['h_over_J'].append(h / J)
        results['energy'].append(props['energy'])
        results['mz'].append(props['mz'])
        results['mx'].append(props['mx'])
        results['mzz'].append(props['mzz'])
        results['xi'].append(props['xi'])

        print(f"{h/J:^8.2f} | {props['energy']:^12.6f} | {props['mz']:^12.6f} | "
              f"{props['mx']:^12.6f} | {props['mzz']:^12.6f} | {props['xi']:^10.3f}")

    print("=" * 70)

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    return results


# ---------------------------------------------------------------------------
# Convenience function: full ED baseline for challenge requirements
# ---------------------------------------------------------------------------

def ed_baseline(N=6, h_values=None, J=1.0, boundary='open', t_max=5.0):
    """
    Full ED baseline matching the challenge requirements.

    Computes:
      - Ground-state properties at each h/J
      - Time evolution from |0...0> at h/J = 1.0 (critical point)

    Parameters
    ----------
    N : int
        Number of spins.
    h_values : list
        h/J values. Default: [0.5, 1.0, 2.0] as per challenge.
    J : float
        Coupling strength.
    boundary : str
        'open' or 'periodic'.
    t_max : float
        Maximum time for evolution.

    Returns
    -------
    dict with 'ground_state' and 'time_evolution' sub-dicts.
    """
    if h_values is None:
        h_values = [0.5, 1.0, 2.0]

    print("\n" + "█" * 70)
    print("  EXACT DIAGONALIZATION BASELINE")
    print(f"  N={N}, J={J}, boundary='{boundary}'")
    print("█" * 70)

    # Ground state properties
    gs_results = {}
    print("\n--- Ground-State Properties ---")
    print(f"{'h/J':^8} | {'E0/N':^12} | {'<Z>':^12} | {'<X>':^12} | {'<ZiZi+1>':^12}")
    print("-" * 60)

    for h in h_values:
        props = ground_state_properties(N, J, h, boundary)
        gs_results[h] = props
        print(f"{h/J:^8.1f} | {props['energy']:^12.6f} | {props['mz']:^12.6f} | "
              f"{props['mx']:^12.6f} | {props['mzz']:^12.6f}")

    # Time evolution at critical point h/J = 1.0
    print("\n--- Time Evolution (h/J = 1.0, initial |0...0>) ---")
    te_results = time_evolution_tfim(
        N=N, J=J, h=1.0, t_max=t_max, num_points=100,
        initial_state='zeros', boundary=boundary
    )
    print(f"  t ∈ [0, {t_max}], {len(te_results['times'])} points")
    print(f"  <Z(0)> = {te_results['mz_t'][0]:.6f}")
    print(f"  <Z({t_max})> = {te_results['mz_t'][-1]:.6f}")

    return {
        'ground_state': gs_results,
        'time_evolution': te_results,
        'N': N, 'J': J, 'boundary': boundary,
    }


# ---------------------------------------------------------------------------
# Main execution (for testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick test
    results = ed_baseline(N=6, h_values=[0.5, 1.0, 2.0], t_max=5.0)

    # Phase transition scan
    scan = ed_phase_transition_scan(N=8, h_values=np.linspace(0.1, 2.5, 25))