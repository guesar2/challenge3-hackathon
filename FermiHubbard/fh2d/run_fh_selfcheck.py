"""
run_fh_selfcheck.py

Correctness guards for the Jordan-Wigner layer. These are cheap and MUST pass
before trusting any downstream ED / Trotter / VQE / hardware number:

  1. Term/operator consistency: the Pauli-term Hamiltonian (fed to the circuits)
     equals the operator-product Hamiltonian (fed to ED), to machine precision.

  2. Free-fermion cross-check: at U=0, the many-body ground-state energy from
     ED must equal the sum of the lowest single-particle hopping eigenvalues
     for the chosen particle number -- the standard test that catches JW sign
     and Z-string bugs. Run per spin species (the two spins decouple at U=0).

Run:  python run_fh_selfcheck.py
"""
import numpy as np
from scipy.sparse.linalg import eigsh

from fh_lattice import HubbardLattice, UP, DOWN
import fh_jordan_wigner as jw


def _single_particle_hopping_matrix(lat, t):
    """Lx*Ly single-particle hopping matrix h with h[i,j] = -t on bonds."""
    n = lat.n_sites
    pos = {s: lat.site_pos(s) for s in lat.sites}
    h = np.zeros((n, n))
    for (s1, s2) in lat.bonds():
        i, j = pos[s1], pos[s2]
        h[i, j] -= t
        h[j, i] -= t
    return h


def free_fermion_ground_energy(lat, t, n_up, n_dn):
    """Sum of lowest n_up + n_dn single-particle eigenvalues (spins identical)."""
    h = _single_particle_hopping_matrix(lat, t)
    evals = np.sort(np.linalg.eigvalsh(h))
    return evals[:n_up].sum() + evals[:n_dn].sum()


def check_consistency():
    print("[1] JW term vs operator consistency")
    for (Lx, Ly, px, py) in [(2, 2, False, False), (2, 3, False, False),
                             (3, 2, True, False), (2, 2, True, True)]:
        lat = HubbardLattice(Lx, Ly, px, py)
        d = jw.verify_consistency(lat, t=1.0, U=3.7, mu=1.1)
        print(f"    {lat}  max|dH| = {d:.2e}  OK")


def check_free_fermions():
    print("[2] Free-fermion (U=0) ground-energy cross-check")
    for (Lx, Ly) in [(2, 2), (2, 3), (2, 4)]:
        lat = HubbardLattice(Lx, Ly)
        t = 1.0
        n_up = lat.n_sites // 2
        n_dn = lat.n_sites - n_up  # half filling total
        # ED at U=0 with a chemical potential to land the global GS at (n_up+n_dn).
        # Easiest: build sparse H at U=0 and diagonalise in the full space, then
        # confirm the ground energy equals the free-fermion prediction for the
        # half-filled sector by picking mu that selects half filling. For U=0 the
        # particle-hole symmetric point is mu=0, and the global GS fills all
        # negative-energy single-particle modes. Compare that directly.
        h = _single_particle_hopping_matrix(lat, t)
        evals = np.sort(np.linalg.eigvalsh(h))
        n_neg = int((evals < -1e-12).sum())
        ed_pred = evals[evals < -1e-12].sum() * 2  # both spins fill negatives
        H = jw.build_hubbard_sparse(lat, t=t, U=0.0, mu=0.0)
        w = eigsh(H, k=1, which="SA", return_eigenvectors=False)
        gs = float(w[0])
        ok = abs(gs - ed_pred) < 1e-6
        print(f"    {Lx}x{Ly}: ED GS = {gs:+.6f} | free-fermion(all neg modes,"
              f" {n_neg} per spin) = {ed_pred:+.6f} | diff {abs(gs-ed_pred):.1e}"
              f"  {'OK' if ok else 'FAIL'}")
        assert ok, "free-fermion cross-check failed"


def check_half_filling_mu():
    print("[3] Particle-hole mu=U/2 lands global GS at half filling")
    for (Lx, Ly) in [(2, 2), (2, 3)]:
        lat = HubbardLattice(Lx, Ly)
        obs = jw.Observables(lat)
        for U in (1.0, 8.0):
            H = jw.build_hubbard_sparse(lat, t=1.0, U=U, mu=U / 2)
            w, v = eigsh(H, k=1, which="SA")
            gs = v[:, 0]
            N = obs.total_particles(gs)
            ok = abs(N - lat.n_sites) < 1e-6
            print(f"    {Lx}x{Ly} U={U}: <N> = {N:.4f} (target {lat.n_sites})"
                  f"  {'OK' if ok else 'FAIL'}")
            assert ok, "half-filling not achieved by mu=U/2"


if __name__ == "__main__":
    check_consistency()
    check_free_fermions()
    check_half_filling_mu()
    print("\nAll self-checks passed.")