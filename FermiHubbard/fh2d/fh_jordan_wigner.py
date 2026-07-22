"""
fh_jordan_wigner.py

The Jordan-Wigner (JW) layer: it is the single source of truth for the
Fermi-Hubbard Hamiltonian, in two equivalent representations that this module
guarantees agree (see verify_consistency / the test in run_fh_selfcheck):

  1. Sparse matrices (scipy.sparse) on the 2*n_sites-qubit Hilbert space, used
     by the classical baseline (exact diagonalisation and exact time evolution)
     and to evaluate observables on statevectors.

  2. A list of (Pauli string, real coefficient) terms, used to build the
     Trotter quantum circuit (fh_tket_circuit.py) and the qiskit statevector
     Trotter reference (fh_trotter_simulation.py).

Both come from the SAME fermionic definitions, so the circuit and the ED
Hamiltonian can never silently drift apart. Everything here is sparse (like the
TFIM project's pauli_ops.py) so memory scales with 2^n nonzeros, not 4^n.

Conventions
-----------
- Occupied orbital = qubit state |1>; number operator n = (1 - Z)/2.
- Annihilation operator (with JW string):  a_p = (prod_{q<p} Z_q) (X_p + i Y_p)/2.
- For p < q, the hopping term is exactly
      c_p^dag c_q + c_q^dag c_p = (1/2)(X_p X_q + Y_p Y_q) * prod_{p<k<q} Z_k .
  We build it both by multiplying the sparse a_p operators (strings automatic)
  AND from this explicit Pauli form, and assert the two match.
- On-site interaction  U n_{i,up} n_{i,down}
      = (U/4)(1 - Z_up - Z_down + Z_up Z_down).
- Hamiltonian:  H = -t sum_<ij>,sigma (hop) + U sum_i n_up n_down
                    - mu sum_i (n_up + n_down).
  mu is a chemical potential. Setting mu = U/2 makes the model particle-hole
  symmetric, so the GLOBAL ground state sits at half-filling -- letting a
  full-space eigensolver land on the half-filled sector without explicit
  sector projection. Reported energies subtract the mu term back out so the
  quoted number is <H_Hubbard> (t/U only), and <N> is reported as a self-check.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix, identity, kron

from fh_lattice import HubbardLattice, UP, DOWN

# Local single-qubit ops as sparse 2x2.
_I2 = identity(2, format="csr", dtype=complex)
_Z = csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
_X = csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
_Y = csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
_ANNIH = csr_matrix(np.array([[0, 1], [0, 0]], dtype=complex))  # (X+iY)/2
_PAULI = {"I": _I2, "X": _X, "Y": _Y, "Z": _Z}


@lru_cache(maxsize=None)
def _embed_single(q: int, n_qubits: int, pauli: str) -> csr_matrix:
    """Embed a single-qubit Pauli 'pauli' acting on qubit q into n_qubits, as a
    sparse Kronecker product (qubit 0 is the leftmost/most-significant factor)."""
    op = _PAULI[pauli]
    full = csr_matrix(np.array([[1.0 + 0j]]))
    for i in range(n_qubits):
        full = kron(full, op if i == q else _I2, format="csr")
    return full


def _pauli_string_op(term, n_qubits: int) -> csr_matrix:
    """Sparse operator for a Pauli string given as a dict {qubit: 'X'/'Y'/'Z'}."""
    full = csr_matrix(np.array([[1.0 + 0j]]))
    for i in range(n_qubits):
        full = kron(full, _PAULI.get(term.get(i, "I"), _I2), format="csr")
    return full


@lru_cache(maxsize=None)
def _annihilation(p: int, n_qubits: int) -> csr_matrix:
    """JW annihilation operator a_p on n_qubits (sparse)."""
    full = csr_matrix(np.array([[1.0 + 0j]]))
    for i in range(n_qubits):
        if i < p:
            local = _Z
        elif i == p:
            local = _ANNIH
        else:
            local = _I2
        full = kron(full, local, format="csr")
    return full


def number_op(q: int, n_qubits: int) -> csr_matrix:
    """n_q = a_q^dag a_q = (1 - Z_q)/2 (sparse)."""
    Zq = _embed_single(q, n_qubits, "Z")
    return 0.5 * (identity(2 ** n_qubits, format="csr", dtype=complex) - Zq)


# ---------------------------------------------------------------------------
#  Pauli-term representation (single source of truth for circuit + statevector)
# ---------------------------------------------------------------------------

def hopping_pauli_terms(a: int, b: int):
    """Pauli terms for  c_a^dag c_b + c_b^dag c_a  (a < b), as a list of
    (dict{qubit:pauli}, coeff). Returns the (XX + YY)/2 * Z-string form."""
    if a >= b:
        raise ValueError("require a < b")
    zstring = {k: "Z" for k in range(a + 1, b)}
    xx = dict(zstring); xx[a] = "X"; xx[b] = "X"
    yy = dict(zstring); yy[a] = "Y"; yy[b] = "Y"
    return [(xx, 0.5), (yy, 0.5)]


def interaction_pauli_terms(up_q: int, down_q: int, include_constant=False):
    """Pauli terms for  n_up n_down = (1 - Z_up)(1 - Z_down)/4
    = 1/4 - Z_up/4 - Z_down/4 + Z_up Z_down/4. The constant 1/4 is dropped by
    default (it is a global energy shift irrelevant to dynamics and circuits)."""
    terms = [
        ({up_q: "Z"}, -0.25),
        ({down_q: "Z"}, -0.25),
        ({up_q: "Z", down_q: "Z"}, 0.25),
    ]
    if include_constant:
        terms.append(({}, 0.25))
    return terms


def hubbard_pauli_terms(lat: HubbardLattice, t: float, U: float, mu: float = 0.0,
                        include_constant=False):
    """Full Hamiltonian as a list of (pauli-dict, coeff) terms:
        H = -t sum hop + U sum n_up n_down - mu sum (n_up + n_down).

    Returns (hopping_terms, interaction_terms, onsite_terms) so callers can
    Trotter-split by group. onsite_terms carries the interaction Z/ZZ pieces and
    the chemical-potential single-Z pieces together (all mutually commuting and
    diagonal), which is the natural "e^{-i dt H_int}" block.
    """
    hopping = []
    for spin in (UP, DOWN):
        for (a, b) in lat.spin_bonds(spin):
            for pauli, c in hopping_pauli_terms(a, b):
                hopping.append((pauli, -t * c))

    onsite = []
    const = 0.0
    for site in lat.sites:
        up_q = lat.qubit(site, UP)
        dn_q = lat.qubit(site, DOWN)
        for pauli, c in interaction_pauli_terms(up_q, dn_q, include_constant=include_constant):
            if pauli == {}:
                const += U * c
            else:
                onsite.append((pauli, U * c))
        # chemical potential: -mu (n_up + n_down) = -mu( (1-Z_up)/2 + (1-Z_down)/2 )
        if mu != 0.0:
            onsite.append(({up_q: "Z"}, 0.5 * mu))
            onsite.append(({dn_q: "Z"}, 0.5 * mu))
            const += -mu  # two * (-mu/2) constant per site
    const_terms = [({}, const)] if (include_constant and const != 0.0) else []
    return hopping, onsite, const_terms


# ---------------------------------------------------------------------------
#  Sparse Hamiltonian and observables
# ---------------------------------------------------------------------------

def build_hopping_sparse(lat: HubbardLattice, t: float) -> csr_matrix:
    """-t sum_<ij>,sigma (a_i^dag a_j + h.c.), built by MULTIPLYING JW a-operators
    so the Z-strings are automatic and guaranteed correct."""
    n = lat.n_qubits
    H = csr_matrix((2 ** n, 2 ** n), dtype=complex)
    for spin in (UP, DOWN):
        for (a, b) in lat.spin_bonds(spin):
            aa = _annihilation(a, n)
            ab = _annihilation(b, n)
            hop = aa.conj().T @ ab + ab.conj().T @ aa
            H = H - t * hop
    return H


def build_hubbard_sparse(lat: HubbardLattice, t: float, U: float, mu: float = 0.0) -> csr_matrix:
    """Full sparse Hubbard Hamiltonian H = -t*hop + U*interaction - mu*N."""
    n = lat.n_qubits
    H = build_hopping_sparse(lat, t)
    for site in lat.sites:
        up_q = lat.qubit(site, UP)
        dn_q = lat.qubit(site, DOWN)
        n_up = number_op(up_q, n)
        n_dn = number_op(dn_q, n)
        H = H + U * (n_up @ n_dn)
        if mu != 0.0:
            H = H - mu * (n_up + n_dn)
    return H


def build_hubbard_from_terms(lat: HubbardLattice, t: float, U: float, mu: float = 0.0) -> csr_matrix:
    """Same Hamiltonian, but assembled from the Pauli-term list -- used only to
    cross-check that the term list (which feeds the circuits) matches the
    operator-product Hamiltonian (which feeds ED)."""
    n = lat.n_qubits
    hop, onsite, const = hubbard_pauli_terms(lat, t, U, mu=mu, include_constant=True)
    H = csr_matrix((2 ** n, 2 ** n), dtype=complex)
    for pauli, c in hop + onsite + const:
        H = H + c * _pauli_string_op(pauli, n)
    return H


class Observables:
    """Cheap, cached sparse observable operators for a lattice."""

    def __init__(self, lat: HubbardLattice):
        self.lat = lat
        n = lat.n_qubits
        self.n_up = {s: number_op(lat.qubit(s, UP), n) for s in lat.sites}
        self.n_dn = {s: number_op(lat.qubit(s, DOWN), n) for s in lat.sites}
        self.double = {s: self.n_up[s] @ self.n_dn[s] for s in lat.sites}
        self.N_total = sum(self.n_up.values()) + sum(self.n_dn.values())

    @staticmethod
    def _exp(op, vec):
        return float(np.real(np.vdot(vec, op @ vec)))

    def density_per_site(self, vec):
        """<n_i> = <n_i_up + n_i_down> for each site (dict site -> value)."""
        return {s: self._exp(self.n_up[s] + self.n_dn[s], vec) for s in self.lat.sites}

    def double_occupancy_per_site(self, vec):
        """<n_i_up n_i_down> for each site."""
        return {s: self._exp(self.double[s], vec) for s in self.lat.sites}

    def sz_per_site(self, vec):
        """<S^z_i> = <n_i_up - n_i_down>/2 for each site."""
        return {s: 0.5 * self._exp(self.n_up[s] - self.n_dn[s], vec) for s in self.lat.sites}

    def total_particles(self, vec):
        return self._exp(self.N_total, vec)

    def avg_double_occupancy(self, vec):
        d = self.double_occupancy_per_site(vec)
        return sum(d.values()) / self.lat.n_sites

    def staggered_magnetization(self, vec):
        """m_s = (1/N) sum_i (-1)^(x+y) <S^z_i>  (order-parameter for AFM)."""
        sz = self.sz_per_site(vec)
        tot = 0.0
        for (x, y), val in sz.items():
            tot += ((-1) ** (x + y)) * val
        return tot / self.lat.n_sites


def verify_consistency(lat: HubbardLattice, t=1.0, U=4.0, mu=1.3, atol=1e-9) -> float:
    """Assert the Pauli-term Hamiltonian equals the operator-product one.
    Returns the max abs difference (0 within tolerance = consistent)."""
    H1 = build_hubbard_sparse(lat, t, U, mu=mu)
    H2 = build_hubbard_from_terms(lat, t, U, mu=mu)
    diff = (H1 - H2)
    maxdiff = abs(diff).max() if diff.nnz else 0.0
    assert maxdiff < atol, f"JW term/operator mismatch: {maxdiff}"
    return float(maxdiff)


# ---------------------------------------------------------------------------
#  Hamiltonian-Variational-Ansatz generators (single source of truth for VQE)
# ---------------------------------------------------------------------------

def hva_generators(lat: HubbardLattice, t: float, U: float):
    """Return (hop_groups, int_groups) for a NUMBER-CONSERVING, bond/site-resolved
    Hamiltonian-Variational Ansatz.

    Each *group* is a list of (pauli-dict, coeff) terms that MUST share a single
    variational angle so that particle number is conserved:

      - hop_groups: one group per (spin, bond), each holding the two terms
        (XX + YY)/2 * Z-string of that hop. XX and YY commute, so exponentiating
        them with the SAME angle realises exp(-i theta (c^dag c + h.c.)) exactly
        -> number conserving. Giving XX and YY *different* angles would break
        particle-number conservation (a real pitfall; see the project notes).
      - int_groups: one group per site, holding that site's (-Z_up, -Z_dn, +Z_up
        Z_dn)*U/4 terms (all diagonal, trivially number conserving).

    Different groups get INDEPENDENT angles, which breaks the global translational
    symmetry of the Neel reference -- necessary because global-angle generators
    cannot rotate a symmetry-broken product state back onto the symmetric ground
    state. n_params_per_layer = len(hop_groups) + len(int_groups).
    """
    hop_groups = []
    for spin in (UP, DOWN):
        for (a, b) in lat.spin_bonds(spin):
            group = [(pauli, -t * c) for pauli, c in hopping_pauli_terms(a, b)]
            hop_groups.append(group)
    int_groups = []
    for site in lat.sites:
        up_q = lat.qubit(site, UP)
        dn_q = lat.qubit(site, DOWN)
        group = [(pauli, U * c) for pauli, c in interaction_pauli_terms(up_q, dn_q)]
        int_groups.append(group)
    return hop_groups, int_groups