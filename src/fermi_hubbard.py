"""
fermi_hubbard.py

2D Fermi-Hubbard model: Jordan-Wigner-mapped fermionic operators, the
Hamiltonian, and collective observables (double occupancy, particle number,
total Sz), all as sparse (csr) matrices. Plays the same role for the
Fermi-Hubbard model that pauli_ops.py plays for the TFIM.

Mode ordering (interleaved): the qubit index of spin-orbital (site s, spin
sigma) is  mode_index(s, sigma) = 2*s + sigma,  with sigma in {0=up, 1=down}.
The two spin-orbitals of a site therefore land on adjacent qubits, so the
on-site  U n_up n_down  term carries no Jordan-Wigner string. Sites are
numbered row-major on the Lx x Ly lattice: site_index(row, col) = row*Lx + col.

Sparse throughout for the same reason as pauli_ops.py: a Fermi-Hubbard state
lives on 2*L qubits (2 spin-orbitals per site), so the dense 2^(2L) x 2^(2L)
operator is what actually breaks the classical baseline early. The single-
qubit building block get_pauli() is reused from pauli_ops, so the JW
operators inherit its sparse Kronecker construction and lru_cache.
"""
from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix, identity

from pauli_ops import get_pauli


def mode_index(site: int, spin: int) -> int:
    """Qubit index of spin-orbital (site, spin), interleaved ordering.
    spin: 0 = up, 1 = down."""
    return 2 * site + spin


def site_index(row: int, col: int, Lx: int) -> int:
    """Row-major flattening of a 2D lattice coordinate to a site index."""
    return row * Lx + col



@lru_cache(maxsize=None)
def jw_annihilation(p: int, num_modes: int) -> csr_matrix:
    """Fermionic annihilation operator c_p under Jordan-Wigner, sparse (csr).

        c_p = (prod_{q<p} Z_q) * (X_p + i Y_p)/2

    The Z-string enforces the fermionic anticommutation relations; because
    every factor is sparse, the strings cancel automatically when operators
    are multiplied together to assemble H. Cached like get_pauli() since the
    same (p, num_modes) is requested many times across H and observables.
    """
    string = identity(2 ** num_modes, format='csr', dtype=complex)
    for q in range(p):
        string = string @ get_pauli(q, num_modes, 'z')
    lowering = (get_pauli(p, num_modes, 'x') + 1j * get_pauli(p, num_modes, 'y')) / 2
    return (string @ lowering).tocsr()


@lru_cache(maxsize=None)
def jw_creation(p: int, num_modes: int) -> csr_matrix:
    """Creation operator c_p^dag = (c_p)^dagger, sparse (csr)."""
    return jw_annihilation(p, num_modes).getH().tocsr()


@lru_cache(maxsize=None)
def jw_number(p: int, num_modes: int) -> csr_matrix:
    """Number operator n_p = c_p^dag c_p (diagonal, entries 0/1), sparse."""
    c = jw_annihilation(p, num_modes)
    return (c.getH() @ c).real.tocsr()


def lattice_bonds(Lx: int, Ly: int, periodic: bool = False):
    """Nearest-neighbour bonds of an Lx x Ly square lattice as a set of
    sorted (i, j) site-index tuples (deduplicated).

    Each interior site registers only its right/down neighbour, so every
    bond is recorded exactly once. periodic=True adds wrap-around bonds,
    skipped on a length-2 direction where the wrap would duplicate the
    existing bond.
    """
    bonds = set()
    for row in range(Ly):
        for col in range(Lx):
            s = site_index(row, col, Lx)
            if col + 1 < Lx:
                bonds.add(tuple(sorted((s, site_index(row, col + 1, Lx)))))
            elif periodic and Lx > 2:
                bonds.add(tuple(sorted((s, site_index(row, 0, Lx)))))
            if row + 1 < Ly:
                bonds.add(tuple(sorted((s, site_index(row + 1, col, Lx)))))
            elif periodic and Ly > 2:
                bonds.add(tuple(sorted((s, site_index(0, col, Lx)))))
    return bonds


def build_fermi_hubbard_hamiltonian(Lx: int, Ly: int, t: float = 1.0,
                                    U: float = 0.0, periodic: bool = False) -> csr_matrix:
    """Build the 2D Fermi-Hubbard Hamiltonian, sparse (csr), on 2*Lx*Ly qubits:

        H = -t sum_<i,j>,sigma (c_{i,sigma}^dag c_{j,sigma} + h.c.)
            + U sum_s n_{s,up} n_{s,down}

    Hermitian by construction: each hopping bond is added as hop + hop.getH(),
    which equals c_i^dag c_j + c_j^dag c_i. Scipy sparse arithmetic stays
    sparse throughout, so H is never densified here.
    """
    L = Lx * Ly
    num_modes = 2 * L
    H = csr_matrix((2 ** num_modes, 2 ** num_modes), dtype=complex)

    # hopping term
    for (i, j) in lattice_bonds(Lx, Ly, periodic):
        for spin in (0, 1):
            hop = jw_creation(mode_index(i, spin), num_modes) @ \
                  jw_annihilation(mode_index(j, spin), num_modes)
            H = H - t * (hop + hop.getH())

    # on-site interaction term
    for s in range(L):
        n_up = jw_number(mode_index(s, 0), num_modes)
        n_dn = jw_number(mode_index(s, 1), num_modes)
        H = H + U * (n_up @ n_dn)

    return H.tocsr()


def build_fh_observables(Lx: int, Ly: int):
    """Return a dict of collective (extensive) Fermi-Hubbard observables,
    all sparse:

        'N'    total particle number     sum_p n_p
        'Docc' total double occupancy     sum_s n_{s,up} n_{s,down}
        'Sz'   total z-spin               (1/2) sum_s (n_{s,up} - n_{s,down})

    Divide 'Docc' by L for double occupancy per site, the standard signal of
    the Mott transition (it drops toward 0 as U/t grows at half filling).
    """
    L = Lx * Ly
    num_modes = 2 * L
    N_op = sum(jw_number(p, num_modes) for p in range(num_modes))
    Docc = sum(jw_number(mode_index(s, 0), num_modes) @ jw_number(mode_index(s, 1), num_modes)
               for s in range(L))
    Sz = 0.5 * sum(jw_number(mode_index(s, 0), num_modes) - jw_number(mode_index(s, 1), num_modes)
                   for s in range(L))
    return {'N': N_op, 'Docc': Docc, 'Sz': Sz}


def sector_indices(Lx: int, Ly: int, n_particles: int, sz2: int = None):
    """Computational-basis indices with a given total particle number
    (and, if sz2 is not None, a given 2*Sz = n_up - n_down).

    Restricting to a sector is what makes the double-occupancy-vs-U baseline
    physically meaningful: without a chemical potential the global ground
    state of a repulsive Hubbard model drifts off half filling as U grows,
    so the Mott signal only shows cleanly at fixed filling.
    """
    L = Lx * Ly
    num_modes = 2 * L
    idx = []
    for b in range(2 ** num_modes):
        # mode q is occupied iff bit (num_modes-1-q) of b is set
        occ = [(b >> (num_modes - 1 - q)) & 1 for q in range(num_modes)]
        if sum(occ) != n_particles:
            continue
        if sz2 is not None:
            n_up = sum(occ[mode_index(s, 0)] for s in range(L))
            n_dn = sum(occ[mode_index(s, 1)] for s in range(L))
            if n_up - n_dn != sz2:
                continue
        idx.append(b)
    return np.array(idx, dtype=int)



# --------------------- state preparation & dynamics obs ---------------------
def neel_state(Lx: int, Ly: int):
    """Half-filling Néel (checkerboard) product state as a dense statevector.

    Bipartite sublattices: sites with (row+col) even carry a spin-up electron,
    those with (row+col) odd carry a spin-down electron -> N = L particles,
    Sz = 0. This is the canonical initial state for Fermi-Hubbard dynamics
    (magnetic-polaron / staggered-magnetisation studies).
    """
    L = Lx * Ly
    num_modes = 2 * L
    occupied = []
    for row in range(Ly):
        for col in range(Lx):
            s = site_index(row, col, Lx)
            spin = 0 if (row + col) % 2 == 0 else 1   # A -> up, B -> down
            occupied.append(mode_index(s, spin))
    idx = sum(1 << (num_modes - 1 - q) for q in occupied)
    psi = np.zeros(2 ** num_modes, dtype=complex)
    psi[idx] = 1.0
    return psi


def build_staggered_magnetisation(Lx: int, Ly: int) -> csr_matrix:
    """Staggered magnetisation operator, sparse:

        Ms = (1/L) sum_s (-1)^(row_s + col_s) * (1/2)(n_{s,up} - n_{s,down})

    Its expectation starts at 1/2 on the Néel state and decays as the state
    thermalises -- the observable tracked in Fig. 4 of arXiv:2510.26845.
    """
    L = Lx * Ly
    num_modes = 2 * L
    Ms = csr_matrix((2 ** num_modes, 2 ** num_modes), dtype=complex)
    for row in range(Ly):
        for col in range(Lx):
            s = site_index(row, col, Lx)
            sign = 1 if (row + col) % 2 == 0 else -1
            sz_s = 0.5 * (jw_number(mode_index(s, 0), num_modes)
                          - jw_number(mode_index(s, 1), num_modes))
            Ms = Ms + sign * sz_s
    return (Ms / L).tocsr()
