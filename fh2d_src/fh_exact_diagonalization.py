"""
fh_exact_diagonalization.py

Classical baseline for the 2D Fermi-Hubbard model:

  - ed_ground_state:   sparse lowest eigenpair (eigsh) at half-filling, plus the
                       ground-state observables (double occupancy, per-site
                       density, magnetization) the challenge asks for.
  - ed_time_evolution: EXACT quench dynamics from a product state via
                       scipy.sparse.linalg.expm_multiply -- e^{-iHt}|psi0>
                       without ever forming a dense matrix, so it reaches larger
                       sizes than a dense diagonalisation would. This is the
                       reference the Trotter circuit is checked against (<5%).

Half-filling is enforced with a particle-hole chemical potential mu=U/2 so the
global ground state sits in the half-filled sector; the REPORTED energy is
<H_Hubbard> with the mu term removed, and <N> is returned as a self-check.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigsh, expm_multiply

from fh_lattice import HubbardLattice
import fh_jordan_wigner as jw
from fh_sector import ground_state_sector


def _initial_statevector(lat: HubbardLattice, initial_state):
    """Return a dense statevector for the requested product initial state."""
    if initial_state == "neel":
        occ = lat.neel_occupation()
    elif initial_state == "stripe":
        occ = lat.stripe_occupation()
    else:
        occ = [int(b) for b in initial_state]
        if len(occ) != lat.n_qubits:
            raise ValueError(f"initial bitstring must have length {lat.n_qubits}")
    # qubit 0 is the most-significant bit in the Kronecker convention used by
    # fh_jordan_wigner._embed_single, so the integer index is sum(bit<<(n-1-q)).
    n = lat.n_qubits
    idx = 0
    for q, b in enumerate(occ):
        if b:
            idx |= (1 << (n - 1 - q))
    psi = np.zeros(2 ** n, dtype=complex)
    psi[idx] = 1.0
    return psi


def ed_ground_state(lat: HubbardLattice, t: float, U: float, verbose=True):
    """Ground-state observables at half-filling for one (t, U).

    Thin wrapper around fh_sector.ground_state_sector. The solve happens inside
    the half-filling Sz=0 symmetry sector rather than on the full 2^(2N) space,
    for two reasons:

      1. SIZE. 3x4 is 24 qubits: the full space is 2^24 and its sparse
         Hamiltonian would need ~32 GB. The sector is 853 776 states, ~14 MB.

      2. CORRECTNESS AT U=0. The old full-space route enforced half filling with
         a particle-hole chemical potential mu = U/2. At U=0 that gives mu = 0,
         and the global ground state is then NOT half-filled -- on 2x2 it comes
         out at <N> = 2 (or a degenerate mixture), not 4. Since U=0 is now one of
         the scanned points, the sector solve (where half filling is exact by
         construction) is the only correct option.

    The two engines were checked against each other on 2x2 for U = 1, 4, 8 and
    agree to machine precision; see run_fh_selfcheck.check_sector_vs_full_space.

    Returns a dict with energy, energy_per_site, total_particles,
    avg_double_occupancy, staggered_magnetization, the per-site maps, and the
    ground-state degeneracy.
    """
    return ground_state_sector(lat, t, U, verbose=verbose)


def ed_time_evolution(lat: HubbardLattice, t: float, U: float, dt: float, steps: int,
                      initial_state="neel"):
    """Exact quench dynamics from a product state. Returns a dict of arrays:
    times, avg_double_occupancy, staggered_magnetization, and per-site density
    (n_sites x steps). Uses expm_multiply step-by-step (no mu here: dynamics is
    in a fixed particle sector, chemical potential only shifts a global phase).
    """
    H = jw.build_hubbard_sparse(lat, t=t, U=U, mu=0.0)
    obs = jw.Observables(lat)
    psi = _initial_statevector(lat, initial_state)

    times = np.arange(1, steps + 1) * dt
    dbl = np.zeros(steps)
    mstag = np.zeros(steps)
    dens = np.zeros((lat.n_sites, steps))
    szs = np.zeros((lat.n_sites, steps))
    dbls = np.zeros((lat.n_sites, steps))

    # Precompute the propagator action per fixed dt with expm_multiply.
    A = -1j * dt * H
    for k in range(steps):
        psi = expm_multiply(A, psi)
        psi = psi / np.linalg.norm(psi)
        dbl[k] = obs.avg_double_occupancy(psi)
        mstag[k] = obs.staggered_magnetization(psi)
        d = obs.density_per_site(psi)
        z = obs.sz_per_site(psi)
        dd = obs.double_occupancy_per_site(psi)
        for si, s in enumerate(lat.sites):
            dens[si, k] = d[s]
            szs[si, k] = z[s]
            dbls[si, k] = dd[s]

    return {
        "times": times,
        "avg_double_occupancy": dbl,
        "staggered_magnetization": mstag,
        "density_per_site": dens,
        "sz_per_site": szs,
        "double_per_site": dbls,
        "sites": lat.sites,
    }


if __name__ == "__main__":
    lat = HubbardLattice(2, 2)
    print("ED ground state (half-filling):")
    for U in (0.0, 1.0, 4.0, 8.0):
        ed_ground_state(lat, 1.0, U)
    print("\nED quench dynamics (first 3 steps):")
    dyn = ed_time_evolution(lat, 1.0, 8.0, 0.1, 3)
    print("  times:", dyn["times"])
    print("  <D>(t):", dyn["avg_double_occupancy"])
    print("  m_stag(t):", dyn["staggered_magnetization"])