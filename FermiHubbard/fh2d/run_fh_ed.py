"""
run_fh_ed.py  --  classical exact-diagonalisation baseline.

Ground-state observables (energy density, average double occupancy, staggered
magnetization) at half-filling for the weak (U/t=1) and strong (U/t=8) regimes,
across the ED-feasible lattices in fh_config.SCALING_LATTICES. Saves results and
draws fig1_groundstate_vs_U.png.
"""
from __future__ import annotations

import numpy as np

import fh_config as cfg
from fh_lattice import HubbardLattice
from fh_exact_diagonalization import ed_ground_state
import fh_plotting as plotting
import fh_persistence as persistence


def run(U_values=None, lattices=None, save=True, plot=True):
    U_values = U_values or list(cfg.U_VALUES)
    lattices = lattices or list(cfg.SCALING_LATTICES)
    gs_by_lattice = {}
    all_rows = []
    print("Exact diagonalisation, half-filling ground state:")
    for (Lx, Ly) in lattices:
        lat = HubbardLattice(Lx, Ly, cfg.PERIODIC_X, cfg.PERIODIC_Y)
        label = f"{Lx}x{Ly}"
        rows = []
        for U in U_values:
            r = ed_ground_state(lat, cfg.T_HOP, U)
            # drop only the bulky per-site MAPS for JSON compactness (keep the
            # scalar energy_per_site!)
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