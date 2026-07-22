"""
run_fh_dt_convergence.py  --  Trotter error analysis.

Halve the Trotter step at fixed total evolution time and record the max
%-deviation from ED for both observables and both Trotter orders -- the
"reduce dt and confirm convergence" analysis the rubric asks for, and the check
that distinguishes Trotter error from (later) hardware noise. Writes
fig4_dt_convergence.png.
"""
from __future__ import annotations

import numpy as np

import fh_config as cfg
from fh_lattice import HubbardLattice
from fh_exact_diagonalization import ed_time_evolution
from fh_trotter_simulation import trotter_time_evolution, max_percent_deviation
import fh_plotting as plotting
import fh_persistence as persistence


def run(save=True, plot=True):
    Lx, Ly = cfg.LX, cfg.LY
    lat = HubbardLattice(Lx, Ly, cfg.PERIODIC_X, cfg.PERIODIC_Y)
    t, U = cfg.T_HOP, cfg.QUENCH_U
    T = cfg.DT_CONVERGENCE_TOTAL_TIME
    init = cfg.QUENCH_INITIAL_STATE

    conv = {}
    print(f"Trotter dt-convergence on {lat}, U/t={U/t:.0f}, total time {T}/t:")
    for order in (1, 2):
        dts, devD, devM = [], [], []
        for dt in cfg.DT_CONVERGENCE_VALUES:
            steps = int(round(T / dt))
            ed = ed_time_evolution(lat, t, U, dt, steps, initial_state=init)
            tr = trotter_time_evolution(lat, t, U, dt, steps, initial_state=init, order=order)
            dD = max_percent_deviation(tr, ed, "avg_double_occupancy")
            dM = max_percent_deviation(tr, ed, "staggered_magnetization")
            dts.append(dt); devD.append(dD); devM.append(dM)
            print(f"  order {order} dt={dt:<5}: <D> {dD:6.2f}%  m_stag {dM:6.2f}%")
        conv[order] = {"dt": dts, "dev_D": devD, "dev_M": devM}

    if save:
        persistence.save_stage_results("dt_convergence", {"conv": {str(k): v for k, v in conv.items()}})
    if plot:
        path = plotting.plot_dt_convergence(conv)
        print(f"  wrote {path}")
    return conv


if __name__ == "__main__":
    run()