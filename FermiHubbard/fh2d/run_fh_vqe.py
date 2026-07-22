"""
run_fh_vqe.py  --  optional VQE ground-state energy.

Runs the number-conserving Hamiltonian-Variational-Ansatz VQE at weak coupling
(where it reaches ~1-2% of ED) and at strong coupling (where the Neel-referenced
HVA plateaus -- reported honestly, not hidden). Writes fig6_vqe.png.
"""
from __future__ import annotations

import fh_config as cfg
from fh_lattice import HubbardLattice
from fh_vqe import run_vqe_local
import fh_plotting as plotting
import fh_persistence as persistence


def run(save=True, plot=True, restarts=4):
    Lx, Ly = cfg.VQE_LATTICE
    lat = HubbardLattice(Lx, Ly, cfg.PERIODIC_X, cfg.PERIODIC_Y)
    print("VQE (Hamiltonian-Variational Ansatz, noiseless statevector):")
    rows = []
    # weak coupling: two depths to show improvement
    rows.append(run_vqe_local(lat, cfg.T_HOP, 1.0, layers=2, restarts=restarts,
                              maxiter=cfg.VQE_MAXITER_LOCAL, seed=cfg.VQE_SEED))
    rows.append(run_vqe_local(lat, cfg.T_HOP, 1.0, layers=cfg.VQE_LAYERS + 1, restarts=restarts,
                              maxiter=cfg.VQE_MAXITER_LOCAL, seed=cfg.VQE_SEED))
    # strong coupling: documented limitation
    rows.append(run_vqe_local(lat, cfg.T_HOP, cfg.VQE_U, layers=cfg.VQE_LAYERS + 1,
                              restarts=restarts, maxiter=cfg.VQE_MAXITER_LOCAL, seed=cfg.VQE_SEED))

    if save:
        slim = [{k: v for k, v in r.items() if k not in ("history", "opt_params")} for r in rows]
        persistence.save_stage_results("vqe", {"rows": slim})
    if plot:
        path = plotting.plot_vqe(rows)
        print(f"  wrote {path}")
    return rows


if __name__ == "__main__":
    run()