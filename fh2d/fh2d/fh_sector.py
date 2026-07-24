"""
fh_sector.py

Half-filling, Sz=0 symmetry-sector engine for the 2D Fermi-Hubbard model.

WHY THIS MODULE EXISTS
----------------------
fh_jordan_wigner builds the Hamiltonian on the FULL 2^(2*N_sites) Hilbert space.
That is fine for 2x2 (8 qubits, dim 256) but impossible for 3x4:

    3x4 -> 12 sites -> 24 qubits -> dim 2^24 = 16 777 216
    sparse H would need ~32 GB, and the Trotter term list ~44 GB.

The Hubbard Hamiltonian conserves N_up and N_dn separately, so the half-filled,
Sz=0 sector is all we ever need for ground states and for quenches started from
a half-filled product state. That sector has dimension

    D^2  with  D = C(N_sites, N_sites/2)
    2x2 -> D=6,   dim 36
    3x4 -> D=924, dim 853 776          <- entirely comfortable

and it factorises: because the up-modes occupy JW indices 0..N-1 and the
down-modes N..2N-1, the JW string of an up-hop never touches a down mode and
vice versa. Hence

    H = T (x) I  +  I (x) T  +  D_int ,

where T is the D x D single-spin hopping matrix and D_int is diagonal. Storing
the state as a D x D matrix Psi[i_up, i_dn] turns a matrix-vector product into

    T @ Psi + Psi @ T.T + D_int * Psi

which costs a few tens of milliseconds at 3x4 and needs ~14 MB, not 32 GB.

A second benefit: working inside the sector makes half-filling EXACT by
construction, so the particle-hole chemical-potential trick (mu = U/2) used by
the full-space solver is no longer needed. That trick silently fails at U=0
(where mu=0 and the global ground state is NOT half-filled), which is precisely
one of the U values we now scan.

CONVENTIONS (identical to fh_lattice / fh_jordan_wigner)
--------------------------------------------------------
- A single-spin configuration is a bitmask over SNAKE POSITIONS: bit j is set
  iff snake position j carries a fermion of that spin.
- Fermionic sign for c^dag_p c_q inside one spin block is (-1)^(number of
  occupied modes strictly between p and q) -- the Jordan-Wigner string.
- Observables here are all diagonal in the occupation basis, so they only ever
  need the probability matrix P[i_up, i_dn] = |Psi[i_up, i_dn]|^2.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, eigsh

from fh_lattice import HubbardLattice, UP, DOWN


# ---------------------------------------------------------------------------
#  Single-spin configuration basis
# ---------------------------------------------------------------------------

def _configs(n_sites: int, n_part: int):
    """All bitmasks over n_sites snake positions with exactly n_part bits set,
    in increasing integer order. Returns (masks, index_lookup, occ).

    occ : (D, n_sites) float array, occ[c, j] = 1 if config c occupies snake
          position j.
    """
    masks = [m for m in range(1 << n_sites) if bin(m).count("1") == n_part]
    masks = np.asarray(masks, dtype=np.int64)
    lookup = -np.ones(1 << n_sites, dtype=np.int64)
    lookup[masks] = np.arange(len(masks))
    occ = np.zeros((len(masks), n_sites))
    for j in range(n_sites):
        occ[:, j] = (masks >> j) & 1
    return masks, lookup, occ


def _hop_sign(mask: int, p: int, q: int) -> float:
    """JW sign of c^dag_p c_q on `mask`: (-1)^(occupied modes strictly between)."""
    lo, hi = (p, q) if p < q else (q, p)
    between = mask & (((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1))
    return -1.0 if (bin(between).count("1") & 1) else 1.0


class HalfFillingSector:
    """The N_up = N_dn = N_sites/2 sector of one lattice, with tensor-structured
    operators. Build once per lattice, reuse for every U."""

    def __init__(self, lat: HubbardLattice, t: float = 1.0):
        if lat.n_sites % 2 != 0:
            raise ValueError("half filling with Sz=0 needs an even site count")
        self.lat = lat
        self.t = t
        self.n_sites = lat.n_sites
        self.n_part = lat.n_sites // 2

        self.masks, self.lookup, self.occ = _configs(self.n_sites, self.n_part)
        self.D = len(self.masks)
        self.dim = self.D * self.D

        self.T = self._hopping_matrix(t)          # D x D, real symmetric
        # per-site double occupancy weight matrix: W[i_up, i_dn] = # of doubly
        # occupied sites. U * W is the interaction energy (diagonal).
        self.W = self.occ @ self.occ.T            # (D, D)

        signs = np.array([(-1.0) ** (x + y) for (x, y) in lat.sites])
        self.stag_signs = signs

    # ---- operators -------------------------------------------------------
    def _hopping_matrix(self, t: float) -> csr_matrix:
        """Single-spin -t * sum_<ij> (c^dag_i c_j + h.c.) as a D x D sparse
        matrix, with Jordan-Wigner signs. Bonds come from lat.bonds(), so this
        is the SAME geometry the full-space Hamiltonian uses."""
        pos = {s: self.lat.site_pos(s) for s in self.lat.sites}
        rows, cols, vals = [], [], []
        for (s1, s2) in self.lat.bonds():
            p, q = pos[s1], pos[s2]
            for (a, b) in ((p, q), (q, p)):      # c^dag_a c_b
                for ci, m in enumerate(self.masks):
                    if not ((m >> b) & 1):       # need b occupied
                        continue
                    if (m >> a) & 1:             # need a empty
                        continue
                    m2 = (m & ~(1 << b)) | (1 << a)
                    rows.append(self.lookup[m2])
                    cols.append(ci)
                    vals.append(-t * _hop_sign(m, a, b))
        return csr_matrix((vals, (rows, cols)), shape=(self.D, self.D))

    def hamiltonian_operator(self, U: float, dtype=np.float64) -> LinearOperator:
        """H as a matrix-free LinearOperator on the flattened (D*D,) sector."""
        T, W, D = self.T, U * self.W, self.D

        def matvec(v):
            Psi = v.reshape(D, D)
            out = T @ Psi + Psi @ T.T + W * Psi
            return out.reshape(-1)

        return LinearOperator((self.dim, self.dim), matvec=matvec,
                              rmatvec=matvec, dtype=dtype)

    # ---- states ----------------------------------------------------------
    def occupation_to_state(self, occ_qubits):
        """Turn a length-2*N_sites per-qubit occupation list (fh_lattice
        convention) into a normalised Psi[i_up, i_dn] product state."""
        m_up = m_dn = 0
        for j, site in enumerate(self.lat.sites):
            if occ_qubits[self.lat.qubit(site, UP)]:
                m_up |= (1 << j)
            if occ_qubits[self.lat.qubit(site, DOWN)]:
                m_dn |= (1 << j)
        iu, idn = self.lookup[m_up], self.lookup[m_dn]
        if iu < 0 or idn < 0:
            raise ValueError(
                f"initial state is not in the half-filling Sz=0 sector "
                f"(N_up={bin(m_up).count('1')}, N_dn={bin(m_dn).count('1')}, "
                f"expected {self.n_part} each)")
        Psi = np.zeros((self.D, self.D), dtype=complex)
        Psi[iu, idn] = 1.0
        return Psi

    def initial_state(self, initial_state="neel"):
        """'neel' | 'stripe' | an explicit 2*N_sites occupation bitstring."""
        if initial_state == "neel":
            occ = self.lat.neel_occupation()
        elif initial_state == "stripe":
            occ = self.lat.stripe_occupation()
        else:
            occ = [int(b) for b in initial_state]
        return self.occupation_to_state(occ)

    # ---- observables (all diagonal in the occupation basis) --------------
    def _prob(self, Psi):
        P = np.abs(Psi) ** 2
        return P / P.sum()

    def observables(self, Psi_or_P, is_prob=False):
        """All requested observables from one state (or from an ensemble
        probability matrix, e.g. an averaged degenerate manifold)."""
        P = Psi_or_P if is_prob else self._prob(Psi_or_P)
        p_up = P.sum(axis=1)                 # marginal over down configs
        p_dn = P.sum(axis=0)
        n_up_site = p_up @ self.occ          # (n_sites,)
        n_dn_site = p_dn @ self.occ
        dens = n_up_site + n_dn_site
        sz = 0.5 * (n_up_site - n_dn_site)
        # <n_up_i n_dn_i> = sum_{a,b} P[a,b] occ[a,i] occ[b,i]
        A = self.occ.T @ P                   # (n_sites, D)
        dbl = np.einsum("id,di->i", A, self.occ)
        sites = self.lat.sites
        return {
            "total_particles": float(dens.sum()),
            "avg_double_occupancy": float(dbl.mean()),
            "staggered_magnetization": float((self.stag_signs * sz).mean()),
            "density_per_site": {s: float(dens[i]) for i, s in enumerate(sites)},
            "double_per_site": {s: float(dbl[i]) for i, s in enumerate(sites)},
            "sz_per_site": {s: float(sz[i]) for i, s in enumerate(sites)},
            "density_array": dens,
        }

    def energy(self, Psi, U):
        H = self.hamiltonian_operator(U, dtype=complex)
        v = Psi.reshape(-1)
        return float(np.real(np.vdot(v, H @ v)))


@lru_cache(maxsize=8)
def get_sector(Lx: int, Ly: int, t: float = 1.0) -> HalfFillingSector:
    """Cached sector builder (the basis + hopping matrix are U-independent)."""
    return HalfFillingSector(HubbardLattice(Lx, Ly), t=t)


# ---------------------------------------------------------------------------
#  Ground state
# ---------------------------------------------------------------------------

def ground_state_sector(lat: HubbardLattice, t: float, U: float,
                        k: int = 6, degeneracy_tol: float = 1e-8,
                        verbose: bool = True):
    """Half-filling ground state and its observables, solved inside the sector.

    Degeneracy is handled honestly: at U=0 the half-filled shell can be open
    (e.g. 2x2, whose single-particle levels are -2t, 0, 0, +2t), so <D> and
    m_stag are NOT properties of a single eigenvector. When the lowest level is
    g-fold degenerate we report the ENSEMBLE average over that manifold, which
    is basis independent, and return the degeneracy alongside.
    """
    sec = get_sector(lat.Lx, lat.Ly, t)
    H = sec.hamiltonian_operator(U)
    k = min(k, sec.dim - 2) if sec.dim > 4 else 1

    if sec.dim <= 200:                       # tiny: dense is faster and safer
        Hd = np.zeros((sec.dim, sec.dim))
        for i in range(sec.dim):
            e = np.zeros(sec.dim); e[i] = 1.0
            Hd[:, i] = H @ e
        w, v = np.linalg.eigh(Hd)
        w, v = w[:k], v[:, :k]
    else:
        w, v = eigsh(H, k=k, which="SA")
        order = np.argsort(w)
        w, v = w[order], v[:, order]

    e0 = float(w[0])
    manifold = [i for i in range(len(w)) if abs(w[i] - e0) < degeneracy_tol
                * max(1.0, abs(e0))]
    g = len(manifold)

    P = np.zeros((sec.D, sec.D))
    for i in manifold:
        Psi = v[:, i].reshape(sec.D, sec.D)
        P += np.abs(Psi) ** 2
    P /= P.sum()
    obs = sec.observables(P, is_prob=True)

    result = {
        "Lx": lat.Lx, "Ly": lat.Ly, "t": t, "U": U,
        "n_sites": lat.n_sites, "n_qubits": lat.n_qubits,
        "energy": e0,
        "energy_per_site": e0 / lat.n_sites,
        "degeneracy": g,
        "sector_dim": sec.dim,
        **{key: obs[key] for key in ("total_particles", "avg_double_occupancy",
                                     "staggered_magnetization",
                                     "density_per_site", "double_per_site",
                                     "sz_per_site")},
    }
    if verbose:
        note = "" if g == 1 else f"  [GS is {g}-fold degenerate: ensemble average]"
        print(f"  {lat.Lx}x{lat.Ly}  U/t={U/t:>4.1f}: "
              f"E={e0:+.6f} (E/N={e0/lat.n_sites:+.6f}), "
              f"<N>={result['total_particles']:.3f}, "
              f"<D>={result['avg_double_occupancy']:.4f}, "
              f"m_stag={result['staggered_magnetization']:+.4f}{note}")
    return result


# ---------------------------------------------------------------------------
#  Exact real-time evolution inside the sector (Lanczos / Krylov)
# ---------------------------------------------------------------------------

def _lanczos_expm_step(matvec, psi, dt, m=24):
    """psi <- exp(-i dt H) psi via an m-step Lanczos (Krylov) approximation.

    Standard short-iteration Lanczos: build an orthonormal Krylov basis V and
    the tridiagonal projection Tm, then psi ~ ||psi|| V exp(-i dt Tm) e_1.
    Cheap, unitary to working precision for small dt, and needs only m state
    vectors of memory (m * 14 MB at 3x4).
    """
    beta0 = np.linalg.norm(psi)
    if beta0 == 0:
        return psi
    V = [psi / beta0]
    alphas, betas = [], []
    for j in range(m):
        w = matvec(V[j])
        a = np.vdot(V[j], w).real
        alphas.append(a)
        w = w - a * V[j]
        if j > 0:
            w = w - betas[j - 1] * V[j - 1]
        # one round of re-orthogonalisation keeps the basis clean
        for u in V:
            w = w - np.vdot(u, w) * u
        b = np.linalg.norm(w)
        if b < 1e-12:
            break
        betas.append(b)
        V.append(w / b)
    mm = len(alphas)
    Tm = np.diag(alphas).astype(complex)
    for j in range(mm - 1):
        Tm[j, j + 1] = Tm[j + 1, j] = betas[j]
    ew, ev = np.linalg.eigh(Tm)
    e1 = np.zeros(mm, dtype=complex); e1[0] = 1.0
    coef = ev @ (np.exp(-1j * dt * ew) * (ev.conj().T @ e1))
    out = np.zeros_like(psi)
    for j in range(mm):
        out += coef[j] * V[j]
    return beta0 * out


def ed_time_evolution_sector(lat: HubbardLattice, t: float, U: float, dt: float,
                             steps: int, initial_state="neel", krylov_dim=24):
    """EXACT quench dynamics inside the half-filling sector.

    Same return shape as fh_exact_diagonalization.ed_time_evolution, so the
    plotting layer does not care which engine produced the data.
    """
    sec = get_sector(lat.Lx, lat.Ly, t)
    H = sec.hamiltonian_operator(U, dtype=complex)
    psi = sec.initial_state(initial_state).reshape(-1)

    times = np.arange(1, steps + 1) * dt
    dbl = np.zeros(steps); mstag = np.zeros(steps)
    dens = np.zeros((lat.n_sites, steps))

    matvec = lambda v: H @ v
    for k in range(steps):
        psi = _lanczos_expm_step(matvec, psi, dt, m=krylov_dim)
        psi = psi / np.linalg.norm(psi)
        o = sec.observables(psi.reshape(sec.D, sec.D))
        dbl[k] = o["avg_double_occupancy"]
        mstag[k] = o["staggered_magnetization"]
        dens[:, k] = o["density_array"]

    return {
        "times": times,
        "avg_double_occupancy": dbl,
        "staggered_magnetization": mstag,
        "density_per_site": dens,
        "sites": lat.sites,
    }


if __name__ == "__main__":
    for (Lx, Ly) in ((2, 2), (3, 4)):
        lat = HubbardLattice(Lx, Ly)
        sec = get_sector(Lx, Ly)
        print(f"{lat}  sector dim = {sec.dim} (vs full 2^{lat.n_qubits})")
        for U in (0.0, 1.0, 4.0, 8.0):
            ground_state_sector(lat, 1.0, U)
