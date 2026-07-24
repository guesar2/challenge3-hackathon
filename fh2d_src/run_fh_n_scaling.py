"""
run_fh_n_scaling.py  --  scaling across lattice sizes.

Reports ED energy density at half-filling for weak/strong coupling across the
SCALING_LATTICES, and the Jordan-Wigner qubit budget (2 qubits/site) against the
26-qubit H2 exact-emulator limit -- making explicit WHY 4x4 (32 qubits) is out of
reach here. Writes fig5_n_scaling.png.
"""
from __future__ import annotations

import fh_config as cfg
from fh_lattice import HubbardLattice
from fh_exact_diagonalization import ed_ground_state
import fh_plotting as plotting
import fh_persistence as persistence


def run(lattices=None, save=True, plot=True):
    lattices = lattices or list(cfg.SCALING_LATTICES)
    rows = []
    print("Scaling: ED energy density and qubit budget (half-filling):")
    for (Lx, Ly) in lattices:
        lat = HubbardLattice(Lx, Ly, cfg.PERIODIC_X, cfg.PERIODIC_Y)
        weak = ed_ground_state(lat, cfg.T_HOP, 1.0, verbose=False)
        strong = ed_ground_state(lat, cfg.T_HOP, 8.0, verbose=False)
        row = {"lattice": f"{Lx}x{Ly}", "n_sites": lat.n_sites, "n_qubits": lat.n_qubits,
               "epn_weak": weak["energy_per_site"], "epn_strong": strong["energy_per_site"]}
        rows.append(row)
        print(f"  {Lx}x{Ly}: {lat.n_sites} sites, {lat.n_qubits} qubits | "
              f"E/N(U=1)={row['epn_weak']:+.4f}  E/N(U=8)={row['epn_strong']:+.4f}")

    if save:
        persistence.save_stage_results("n_scaling", {"rows": rows})
    if plot:
        path = plotting.plot_n_scaling(rows)
        print(f"  wrote {path}")
    return rows


if __name__ == "__main__":
    run()