"""
fh_trotter_simulation.py

Noiseless statevector Trotter evolution for the 2D Fermi-Hubbard model.

It applies the SAME Pauli-term decomposition that the pytket circuit
(fh_tket_circuit.py) uses, so this "statevector Trotter" is exactly the
operation the quantum circuit implements (up to gate compilation), and it is
directly comparable to the exact ED reference. Each single-Pauli exponential is
applied via the identity

    exp(-i theta P) = cos(theta) I - i sin(theta) P        (since P^2 = I),

so no dense matrices and no qiskit qubit-ordering conversion are needed -- the
statevector lives in the same Kronecker convention as fh_jordan_wigner.

Trotter orders:
  - order 1: one forward sweep of all terms with angle coeff*dt.
  - order 2: symmetric (Strang) sweep -- forward half (coeff*dt/2) then backward
    half (coeff*dt/2). Second-order accurate in dt for the full Hamiltonian.
The term ORDER puts the on-site (diagonal, mutually commuting) block adjacent in
the middle of the symmetric sweep, so the two half-applications of the on-site
block merge exactly -- the natural U_int U_hop U_int structure used in the paper.
"""
from __future__ import annotations

import numpy as np

from fh_lattice import HubbardLattice
import fh_jordan_wigner as jw
from fh_exact_diagonalization import _initial_statevector


def _compiled_terms(lat, t, U):
    """Return an ordered list of (sparse P, coeff) with the hopping block first
    and the on-site block last, so the symmetric 2nd-order sweep merges the
    on-site halves in the middle (hop | onsite onsite | hop ...)."""
    hop, onsite, _ = jw.hubbard_pauli_terms(lat, t, U, mu=0.0, include_constant=False)
    n = lat.n_qubits
    ordered = hop + onsite
    return [(jw._pauli_string_op(p, n), float(c)) for p, c in ordered]


def _apply_pauli_exp(psi, P, theta):
    """psi <- exp(-i theta P) psi = cos(theta) psi - i sin(theta) (P psi)."""
    if theta == 0.0:
        return psi
    return np.cos(theta) * psi - 1j * np.sin(theta) * (P @ psi)


def _trotter_step(psi, terms, dt, order):
    if order == 1:
        for P, c in terms:
            psi = _apply_pauli_exp(psi, P, c * dt)
        return psi
    if order == 2:
        for P, c in terms:
            psi = _apply_pauli_exp(psi, P, c * dt / 2)
        for P, c in reversed(terms):
            psi = _apply_pauli_exp(psi, P, c * dt / 2)
        return psi
    raise ValueError("order must be 1 or 2")


def trotter_time_evolution(lat: HubbardLattice, t: float, U: float, dt: float, steps: int,
                           initial_state="neel", order=2):
    """Trotterised quench dynamics from a product state. Returns the same dict
    shape as fh_exact_diagonalization.ed_time_evolution, for direct comparison.
    """
    terms = _compiled_terms(lat, t, U)
    obs = jw.Observables(lat)
    psi = _initial_statevector(lat, initial_state)

    times = np.arange(1, steps + 1) * dt
    dbl = np.zeros(steps)
    mstag = np.zeros(steps)
    dens = np.zeros((lat.n_sites, steps))

    for k in range(steps):
        psi = _trotter_step(psi, terms, dt, order)
        psi = psi / np.linalg.norm(psi)
        dbl[k] = obs.avg_double_occupancy(psi)
        mstag[k] = obs.staggered_magnetization(psi)
        d = obs.density_per_site(psi)
        for si, s in enumerate(lat.sites):
            dens[si, k] = d[s]

    return {
        "times": times,
        "avg_double_occupancy": dbl,
        "staggered_magnetization": mstag,
        "density_per_site": dens,
        "sites": lat.sites,
        "order": order, "dt": dt,
    }


def max_percent_deviation(trot, exact, key):
    """Max |trot - exact| / max|exact| * 100 over the time series for `key`."""
    a = np.asarray(trot[key], dtype=float)
    b = np.asarray(exact[key], dtype=float)
    denom = np.max(np.abs(b))
    if denom == 0:
        return 0.0
    return float(np.max(np.abs(a - b)) / denom * 100)


if __name__ == "__main__":
    from fh_exact_diagonalization import ed_time_evolution
    lat = HubbardLattice(2, 2)
    for order in (1, 2):
        for dt in (0.2, 0.1, 0.05):
            steps = int(round(1.0 / dt))
            ex = ed_time_evolution(lat, 1.0, 8.0, dt, steps)
            tr = trotter_time_evolution(lat, 1.0, 8.0, dt, steps, order=order)
            dD = max_percent_deviation(tr, ex, "avg_double_occupancy")
            dM = max_percent_deviation(tr, ex, "staggered_magnetization")
            print(f"order={order} dt={dt:<5}: max dev <D>={dD:6.2f}%  m_stag={dM:6.2f}%")