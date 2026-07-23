"""
run_fh_heatmap.py  --  per-site particle density during a quench (fig3).

WHY THIS IS ITS OWN STAGE NOW
-----------------------------
The old heatmap was drawn from the Neel quench and came out as a flat field of
1.00 at every time and every site. That was not a bug: the Neel product state
has exactly one fermion per site, and on a translationally invariant lattice the
density stays uniform, so there is nothing for a density map to show.

To make the map informative we quench a CHARGE-IMBALANCED state instead
(fh_lattice.stripe_occupation): columns are loaded left to right as
doubly-occupied / singly-occupied / empty, which on 3x4 gives an initial density
profile of 2 / 1 / 0 per column while keeping half filling and N_up = N_dn. The
quench then shows charge actually flowing into the empty region.

Lattice is 3x4 periodic (24 qubits = the largest that fits the 26-qubit H2
emulator), solved inside the half-filling sector (fh_sector.py) since the full
2^24 space is out of reach classically.

The U value is the knob worth playing with (cfg.HEATMAP_U):
    U/t = 0  free fermions: coherent ballistic spreading, and on a finite
             periodic cluster the stripe REVIVES almost perfectly around t~2/t.
    U/t = 4  (default) the stripe visibly melts toward uniform density.
    U/t = 8  doublons are energetically blocked, so relaxation is slower and
             the residual structure survives longer.
"""
from __future__ import annotations

import numpy as np

import fh_config as cfg
from fh_lattice import HubbardLattice
from fh_sector import ed_time_evolution_sector, get_sector
import fh_plotting as plotting
import fh_persistence as persistence


def run(save=True, plot=True):
    Lx, Ly = cfg.HEATMAP_LATTICE
    lat = HubbardLattice(Lx, Ly)
    U, dt, steps = cfg.HEATMAP_U, cfg.HEATMAP_DT, cfg.HEATMAP_STEPS
    init = cfg.HEATMAP_INITIAL_STATE
    sec = get_sector(Lx, Ly, cfg.T_HOP)

    print(f"Density heatmap quench on {lat}")
    print(f"  U/t={U:.0f}, dt={dt}, {steps} steps (t_final={dt*steps:.2f}/t), "
          f"initial state = {init}")
    print(f"  half-filling sector dim = {sec.dim:,} "
          f"(full Hilbert space would be 2^{lat.n_qubits} = {2**lat.n_qubits:,})")

    dyn = ed_time_evolution_sector(lat, cfg.T_HOP, U, dt, steps,
                                   initial_state=init)

    dens = np.asarray(dyn["density_per_site"])
    # column-averaged profile is the clearest one-line summary of the melting
    cols = {}
    for si, (x, y) in enumerate(dyn["sites"]):
        cols.setdefault(x, []).append(si)
    print("  time   column-averaged density profile        <D>")
    for k in range(steps):
        prof = [np.mean(dens[cols[x], k]) for x in sorted(cols)]
        print(f"  {dyn['times'][k]:4.2f}   "
              f"[{'  '.join(f'{p:5.3f}' for p in prof)}]        "
              f"{dyn['avg_double_occupancy'][k]:.4f}")

    if save:
        persistence.save_stage_results("density_heatmap", {
            "lattice": [Lx, Ly], "n_qubits": lat.n_qubits,
            "sector_dim": sec.dim, "U": U, "dt": dt, "steps": steps,
            "initial_state": init,
            "times": dyn["times"].tolist(),
            "density_per_site": dens.tolist(),
            "sites": [list(s) for s in dyn["sites"]],
            "avg_double_occupancy": dyn["avg_double_occupancy"].tolist(),
        })
    if plot:
        path = plotting.plot_density_heatmaps(
            dyn, Lx, Ly, n_snapshots=cfg.HEATMAP_SNAPSHOTS, U=U,
            initial_state=init)
        print(f"  wrote {path}")
    return dyn


if __name__ == "__main__":
    run()
