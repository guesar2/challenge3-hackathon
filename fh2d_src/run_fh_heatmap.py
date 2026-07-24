"""
run_fh_heatmap.py  --  per-site quench map (fig3).

Evolves a half-filling product state on cfg.HEATMAP_LATTICE and draws the
per-site field selected by cfg.HEATMAP_QUANTITY ("density" -> <n_i>,
"sz" -> <S^z_i>, "double" -> <D_i>) at cfg.HEATMAP_SNAPSHOTS time slices.

The solve runs inside the half-filling Sz=0 sector (fh_sector.py), which is what
keeps larger lattices such as 3x4 (24 qubits) reachable classically.

Writes fig3_density_heatmaps.png.
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
    quantity = getattr(cfg, "HEATMAP_QUANTITY", "density")
    spec = plotting.HEATMAP_QUANTITIES[quantity]
    sec = get_sector(Lx, Ly, cfg.T_HOP)

    print(f"Per-site {quantity} map on {lat}")
    print(f"  U/t={U:.0f}, dt={dt}, {steps} steps (t_final={dt*steps:.2f}/t), "
          f"initial state = {init}")
    print(f"  half-filling sector dim = {sec.dim:,} "
          f"(full Hilbert space would be 2^{lat.n_qubits} = {2**lat.n_qubits:,})")

    dyn = ed_time_evolution_sector(lat, cfg.T_HOP, U, dt, steps,
                                   initial_state=init)

    field = np.asarray(dyn[spec["key"]])

    # Column profile, one compact line per time slice. A signed field (sz) is
    # averaged with the checkerboard sign (-1)^(x+y); a plain column mean would
    # cancel to zero on any staggered pattern and show nothing.
    staggered = spec["symmetric"]
    weight = np.array([(-1.0) ** (x + y) if staggered else 1.0
                       for (x, y) in dyn["sites"]])
    cols = {}
    for si, (x, y) in enumerate(dyn["sites"]):
        cols.setdefault(x, []).append(si)
    tag = f"staggered {quantity}" if staggered else quantity
    print(f"  time   column-averaged {tag} profile        <D>")
    for k in range(steps):
        prof = [np.mean(weight[cols[x]] * field[cols[x], k]) for x in sorted(cols)]
        print(f"  {dyn['times'][k]:4.2f}   "
              f"[{'  '.join(f'{p:+6.3f}' for p in prof)}]        "
              f"{dyn['avg_double_occupancy'][k]:.4f}")

    if save:
        persistence.save_stage_results("density_heatmap", {
            "lattice": [Lx, Ly], "n_qubits": lat.n_qubits,
            "sector_dim": sec.dim, "U": U, "dt": dt, "steps": steps,
            "initial_state": init, "quantity": quantity,
            "times": dyn["times"].tolist(),
            "density_per_site": np.asarray(dyn["density_per_site"]).tolist(),
            "sz_per_site": np.asarray(dyn["sz_per_site"]).tolist(),
            "double_per_site": np.asarray(dyn["double_per_site"]).tolist(),
            "sites": [list(s) for s in dyn["sites"]],
            "avg_double_occupancy": dyn["avg_double_occupancy"].tolist(),
        })
    if plot:
        path = plotting.plot_density_heatmaps(
            dyn, Lx, Ly, n_snapshots=cfg.HEATMAP_SNAPSHOTS, U=U,
            initial_state=init, quantity=quantity)
        print(f"  wrote {path}")
    return dyn


if __name__ == "__main__":
    run()
