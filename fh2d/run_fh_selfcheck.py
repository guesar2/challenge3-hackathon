"""
run_fh_selfcheck.py

Correctness guards. These are cheap and MUST pass before trusting any downstream
ED / Trotter / VQE / hardware number:

  1. Term/operator consistency: the Pauli-term Hamiltonian (fed to the circuits)
     equals the operator-product Hamiltonian (fed to the full-space solver), to
     machine precision.

  2. Free-fermion cross-check: at U=0 the many-body ground-state energy must
     equal the sum of the occupied single-particle hopping eigenvalues -- the
     standard test that catches Jordan-Wigner sign and Z-string bugs.

  3. Sector vs full space: the half-filling symmetry-sector engine
     (fh_sector.py, which is the only way 3x4 = 24 qubits is reachable) must
     reproduce the full-space solver exactly on a lattice small enough for both.
     This is the check that lets the 3x4 numbers be trusted, since there is no
     full-space solve to compare them against.

Run:  python run_fh_selfcheck.py
"""
import numpy as np
from scipy.sparse.linalg import eigsh

from fh_lattice import HubbardLattice, UP, DOWN
import fh_jordan_wigner as jw
from fh_sector import get_sector, ground_state_sector, ed_time_evolution_sector
from fh_exact_diagonalization import ed_time_evolution


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


def check_consistency():
    print("[1] JW term vs operator consistency")
    for (Lx, Ly) in [(2, 2), (2, 3), (3, 2), (2, 4)]:
        lat = HubbardLattice(Lx, Ly)
        d = jw.verify_consistency(lat, t=1.0, U=3.7, mu=1.1)
        print(f"    {lat}  max|dH| = {d:.2e}  OK")


def check_free_fermions():
    """U=0: the full-space ground state at mu=0 fills every negative-energy
    single-particle mode, for both spins. Compare directly."""
    print("[2] Free-fermion (U=0) ground-energy cross-check")
    for (Lx, Ly) in [(2, 2), (2, 3), (2, 4)]:
        lat = HubbardLattice(Lx, Ly)
        t = 1.0
        h = _single_particle_hopping_matrix(lat, t)
        evals = np.sort(np.linalg.eigvalsh(h))
        n_neg = int((evals < -1e-12).sum())
        pred = evals[evals < -1e-12].sum() * 2      # both spins fill negatives
        H = jw.build_hubbard_sparse(lat, t=t, U=0.0, mu=0.0)
        w = eigsh(H, k=1, which="SA", return_eigenvectors=False)
        gs = float(w[0])
        ok = abs(gs - pred) < 1e-6
        print(f"    {Lx}x{Ly}: ED GS = {gs:+.6f} | free-fermion({n_neg} neg modes"
              f" per spin) = {pred:+.6f} | diff {abs(gs-pred):.1e}"
              f"  {'OK' if ok else 'FAIL'}")
        assert ok, "free-fermion cross-check failed"


def check_sector_vs_full_space():
    """The sector engine must agree with the full-space engine wherever both can
    run. 2x2 is the overlap point (8 qubits: full dim 256, sector dim 36)."""
    print("[3] Half-filling sector engine vs full-space engine (2x2)")
    lat = HubbardLattice(2, 2)
    obs = jw.Observables(lat)

    # -- ground state. U=0 is skipped for the full-space comparison on purpose:
    #    the mu=U/2 particle-hole trick degenerates to mu=0 there and no longer
    #    selects half filling, which is exactly why the sector engine replaced it.
    for U in (1.0, 4.0, 8.0):
        r = ground_state_sector(lat, 1.0, U, verbose=False)
        H = jw.build_hubbard_sparse(lat, t=1.0, U=U, mu=U / 2)
        w, v = eigsh(H, k=1, which="SA")
        gs = v[:, 0]
        N = obs.total_particles(gs)
        e_full = float(w[0]) + (U / 2) * N
        dE = abs(e_full - r["energy"])
        dD = abs(obs.avg_double_occupancy(gs) - r["avg_double_occupancy"])
        ok = dE < 1e-8 and dD < 1e-8
        print(f"    U/t={U:>3.0f}: |dE| = {dE:.1e}, |d<D>| = {dD:.1e}"
              f"  {'OK' if ok else 'FAIL'}")
        assert ok, "sector vs full-space ground state mismatch"

    # -- quench dynamics, both initial states
    for U in (0.0, 8.0):
        for init in ("neel", "stripe"):
            a = ed_time_evolution(lat, 1.0, U, 0.05, 20, initial_state=init)
            b = ed_time_evolution_sector(lat, 1.0, U, 0.05, 20, initial_state=init)
            dD = np.max(np.abs(a["avg_double_occupancy"] - b["avg_double_occupancy"]))
            dM = np.max(np.abs(a["staggered_magnetization"] - b["staggered_magnetization"]))
            dn = np.max(np.abs(a["density_per_site"] - b["density_per_site"]))
            ok = max(dD, dM, dn) < 1e-10
            print(f"    quench U/t={U:>3.0f} init={init:<6}: max|d<D>|={dD:.1e} "
                  f"max|dm_s|={dM:.1e} max|dn_i|={dn:.1e}  {'OK' if ok else 'FAIL'}")
            assert ok, "sector vs full-space dynamics mismatch"


def check_sector_sizes():
    """Report why the sector engine is not optional."""
    print("[4] Sector dimensions vs full Hilbert space")
    for (Lx, Ly) in [(2, 2), (3, 4)]:
        lat = HubbardLattice(Lx, Ly)
        sec = get_sector(Lx, Ly)
        print(f"    {Lx}x{Ly}: {lat.n_qubits} qubits | full 2^{lat.n_qubits} = "
              f"{2**lat.n_qubits:,} | half-filling sector = {sec.dim:,} "
              f"({sec.dim / 2**lat.n_qubits * 100:.3f} %)")


if __name__ == "__main__":
    check_consistency()
    check_free_fermions()
    check_sector_vs_full_space()
    check_sector_sizes()
    print("\nAll self-checks passed.")
