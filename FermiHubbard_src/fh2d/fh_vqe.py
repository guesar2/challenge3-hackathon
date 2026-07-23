"""
fh_vqe.py

Optional extension: a Variational Quantum Eigensolver for the Fermi-Hubbard
ground-state energy at half-filling, using the number-conserving,
bond/site-resolved Hamiltonian-Variational Ansatz (HVA) whose generators are
defined once in fh_jordan_wigner.hva_generators and consumed identically by the
pytket circuit (fh_tket_circuit.build_hva_ansatz_circuit) and by the fast sparse
evaluator here (validated equivalent in run_fh_circuit_check).

Energy is evaluated on the NOISELESS statevector: E(theta) = <psi(theta)|H|psi>
with H the sparse Hubbard Hamiltonian (mu=0; the ansatz fixes the filling). This
is exact and is the appropriate "noiseless VQE" baseline; the sparse path is
~100x faster per optimiser step than rebuilding the pytket box circuit, so COBYLA
with restarts is cheap.

Honest scope (reported transparently by run_vqe_local), p=3, 2x2:
  * U/t=0: ~0.2% of ED. (This case used to return E=0 against an exact -4.000;
    see the comment in fh_jordan_wigner.hva_generators for why.)
  * U/t=1: ~3-4% of ED -- the variational circuit demonstrably works.
  * U/t=4: ~15-20%.
  * U/t=8: the Neel-referenced HVA PLATEAUS at roughly 25-30% above ED, and the
    exact figure moves by a few percent between runs because the optimiser is
    landing on different local minima, not because it is still converging. More
    restarts and more iterations do not fix it. This is a genuine limitation of
    the ansatz, not of the optimiser: strong-coupling Hubbard ground states are
    best approached from the Heisenberg limit with a dedicated state-preparation
    circuit (exactly what arXiv:2511.02125 does), which is beyond a basic
    hackathon VQE. We do NOT paper over this.

A hardware (H2) VQE additionally needs H grouped into commuting Pauli sets, each
measured in its own basis (Z for the diagonal on-site terms, rotated bases for
the hopping XX/YY strings); that heavier extension is described in the README.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from fh_lattice import HubbardLattice
import fh_jordan_wigner as jw
from fh_tket_circuit import build_hva_ansatz_circuit
from fh_trotter_simulation import _apply_pauli_exp
from fh_exact_diagonalization import _initial_statevector, ed_ground_state


def _sparse_generators(lat, t, U):
    """Sparse version of hva_generators: lists of groups, each a list of
    (sparse P, coeff). Same structure/order as the pytket ansatz."""
    hop_groups, int_groups = jw.hva_generators(lat, t, U)
    n = lat.n_qubits
    hop_s = [[(jw._pauli_string_op(p, n), float(c)) for p, c in g] for g in hop_groups]
    int_s = [[(jw._pauli_string_op(p, n), float(c)) for p, c in g] for g in int_groups]
    return hop_s, int_s


def hva_statevector_sparse(lat, params, hop_s, int_s, initial_state="neel"):
    """Fast HVA statevector via sparse Pauli exponentials -- the SAME unitary as
    fh_tket_circuit.build_hva_ansatz_circuit (checked in run_fh_circuit_check)."""
    per_layer = len(hop_s) + len(int_s)
    n_layers = len(params) // per_layer
    psi = _initial_statevector(lat, initial_state)
    k = 0
    for _ in range(n_layers):
        for group in hop_s:
            theta = params[k]; k += 1
            for P, c in group:
                psi = _apply_pauli_exp(psi, P, theta * c)
        for group in int_s:
            theta = params[k]; k += 1
            for P, c in group:
                psi = _apply_pauli_exp(psi, P, theta * c)
    return psi


def n_params(lat: HubbardLattice, layers: int) -> int:
    hop_groups, int_groups = jw.hva_generators(lat, 1.0, 1.0)
    return layers * (len(hop_groups) + len(int_groups))


def energy_statevector(lat, t, U, params, H_sparse=None, hop_s=None, int_s=None,
                       use_circuit=False):
    """Exact <H> for the HVA state with the given angles."""
    if H_sparse is None:
        H_sparse = jw.build_hubbard_sparse(lat, t=t, U=U, mu=0.0)
    if use_circuit:
        psi = np.asarray(build_hva_ansatz_circuit(lat, t, U, list(params)).get_statevector())
    else:
        if hop_s is None or int_s is None:
            hop_s, int_s = _sparse_generators(lat, t, U)
        psi = hva_statevector_sparse(lat, list(params), hop_s, int_s)
    return float(np.real(np.vdot(psi, H_sparse @ psi)))


def run_vqe_local(lat: HubbardLattice, t, U, layers=2, maxiter=1500, restarts=8,
                  seed=7, verbose=True):
    """Optimise the HVA energy with COBYLA (multi-restart) on the noiseless
    statevector; compare to the ED half-filling ground state. Returns a dict."""
    H_sparse = jw.build_hubbard_sparse(lat, t=t, U=U, mu=0.0)
    hop_s, int_s = _sparse_generators(lat, t, U)
    obs = jw.Observables(lat)
    ndim = n_params(lat, layers)

    best_fun = np.inf
    best_x = None
    best_history = None
    for r in range(restarts):
        rng = np.random.default_rng(seed + r)
        # Angle range. Every generator here is bounded: the diagonal n_up n_dn
        # generator has eigenvalues 0/1, so exp(-i theta n n) is 2*pi periodic and
        # [-pi, pi] already covers the whole group. Sampling from [-1, 1] (the old
        # default) explored only a third of it. This matters now that the U factor
        # has been taken out of the interaction generators -- with U folded in, an
        # angle of 1 became 8 at U/t=8 and wrapped past the period by accident.
        x0 = rng.uniform(-np.pi, np.pi, ndim)
        history = []

        def cost(x):
            psi = hva_statevector_sparse(lat, list(x), hop_s, int_s)
            e = float(np.real(np.vdot(psi, H_sparse @ psi)))
            history.append(e)
            return e

        res = minimize(cost, x0, method="COBYLA",
                       options={"maxiter": maxiter, "rhobeg": 1.0, "tol": 1e-7})
        if res.fun < best_fun:
            best_fun, best_x, best_history = res.fun, res.x, history

    psi = hva_statevector_sparse(lat, list(best_x), hop_s, int_s)
    N = obs.total_particles(psi)
    D = obs.avg_double_occupancy(psi)
    M = obs.staggered_magnetization(psi)

    ed = ed_ground_state(lat, t, U, verbose=False)
    e_vqe = float(best_fun)
    e_ed = ed["energy"]
    err_pct = abs(e_vqe - e_ed) / abs(e_ed) * 100

    result = {
        "Lx": lat.Lx, "Ly": lat.Ly, "t": t, "U": U, "layers": layers, "n_params": ndim,
        "energy_vqe": e_vqe, "energy_ed": e_ed, "error_percent": err_pct,
        "vqe_particles": N, "vqe_double_occ": D, "vqe_m_stag": M,
        "ed_double_occ": ed["avg_double_occupancy"],
        "ed_particles": ed["total_particles"],
        "ed_m_stag": ed["staggered_magnetization"],
        "ed_degeneracy": ed.get("degeneracy", 1),
        "history": best_history, "opt_params": best_x.tolist(),
    }
    if verbose:
        print(f"  VQE {lat.Lx}x{lat.Ly} U/t={U/t:.1f} p={layers} ({ndim} params): "
              f"E_vqe={e_vqe:+.5f}  E_ed={e_ed:+.5f}  err={err_pct:.2f}%  "
              f"<N>={N:.3f} (target {lat.n_sites})  "
              f"<D>_vqe={D:.4f} vs <D>_ed={ed['avg_double_occupancy']:.4f}")
    return result


if __name__ == "__main__":
    lat = HubbardLattice(2, 2)
    print("Weak coupling (VQE works well):")
    run_vqe_local(lat, 1.0, 1.0, layers=2, restarts=4)
    run_vqe_local(lat, 1.0, 1.0, layers=3, restarts=4)
    print("Strong coupling (documented HVA limitation):")
    run_vqe_local(lat, 1.0, 8.0, layers=3, restarts=4)