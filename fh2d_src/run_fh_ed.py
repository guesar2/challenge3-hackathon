"""
run_fh_ed.py  --  classical exact-diagonalisation baseline.

Half-filling ground-state observables (total energy, average double occupancy,
staggered magnetization) across U/t in fh_config.U_VALUES, for the two lattices
in fh_config.GS_LATTICES:

    2x2 periodic   ->  4 sites,  8 qubits,  sector dim         36
    3x4 periodic   -> 12 sites, 24 qubits,  sector dim    853 776
                      (the full Hilbert space would be 2^24 = 16 777 216)

Both solves run inside the half-filling Sz=0 symmetry sector (fh_sector.py).
That is what makes 3x4 possible at all, and it also makes U=0 correct -- the old
particle-hole chemical-potential trick silently leaves the half-filled sector
when U=0, because it sets mu = U/2 = 0.

Writes fig1_groundstate_vs_U.png.
"""
from __future__ import annotations

import fh_config as cfg
from fh_lattice import HubbardLattice
from fh_exact_diagonalization import ed_ground_state
import fh_plotting as plotting
import fh_persistence as persistence


def run(U_values=None, lattices=None, save=True, plot=True):
    U_values = list(U_values or cfg.U_VALUES)
    lattices = list(lattices or cfg.GS_LATTICES)
    gs_by_lattice = {}
    all_rows = []
    print("Exact diagonalisation, half-filling ground state (periodic):")
    for (Lx, Ly) in lattices:
        lat = HubbardLattice(Lx, Ly)
        label = f"{Lx}x{Ly}"
        print(f"  {label}: {lat.n_sites} sites, {lat.n_qubits} qubits")
        rows = []
        for U in U_values:
            r = ed_ground_state(lat, cfg.T_HOP, U)
            # drop only the bulky per-site MAPS for JSON compactness
            _drop = {"density_per_site", "double_per_site", "sz_per_site"}
            slim = {k: v for k, v in r.items() if k not in _drop}
            rows.append(slim)
            all_rows.append(slim)
        gs_by_lattice[label] = rows

    if save:
        persistence.save_stage_results("ed_ground_state", {"rows": all_rows})
    if plot:
        path = plotting.plot_ground_state_vs_U(gs_by_lattice)
        print(f"  wrote {path}")
    return gs_by_lattice


if __name__ == "__main__":
    run()
