"""
run_fh_dt_convergence.py  --  Trotter error analysis (second order only).

Halve the Trotter step at fixed total evolution time and record the max
%-deviation from ED for both observables -- the "reduce dt and confirm
convergence" analysis the rubric asks for, and the check that distinguishes
Trotter error from (later) hardware noise.

Only the SECOND-ORDER (symmetric Strang) splitting is run; the first-order
comparison was removed. The figure instead draws the expected O(dt^2) reference
slope, which is the more direct statement that the splitting is behaving as
advertised. Writes fig4_dt_convergence.png.
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
    lat = HubbardLattice(Lx, Ly)
    t, U = cfg.T_HOP, cfg.QUENCH_U
    T = cfg.DT_CONVERGENCE_TOTAL_TIME
    init = cfg.QUENCH_INITIAL_STATE
    order = cfg.TROTTER_ORDER

    dts, devD, devM = [], [], []
    print(f"Trotter dt-convergence on {lat}, U/t={U/t:.0f}, total time {T}/t, "
          f"order {order}:")
    for dt in cfg.DT_CONVERGENCE_VALUES:
        steps = int(round(T / dt))
        ed = ed_time_evolution(lat, t, U, dt, steps, initial_state=init)
        tr = trotter_time_evolution(lat, t, U, dt, steps, initial_state=init,
                                    order=order)
        dD = max_percent_deviation(tr, ed, "avg_double_occupancy")
        dM = max_percent_deviation(tr, ed, "staggered_magnetization")
        dts.append(dt); devD.append(dD); devM.append(dM)
        print(f"  dt={dt:<5}: <D> {dD:6.2f}%  m_stag {dM:6.2f}%   "
              f"{'PASS' if max(dD, dM) < 5 else 'above'} 5% bar")
    conv = {"dt": dts, "dev_D": devD, "dev_M": devM, "order": order}

    if save:
        persistence.save_stage_results("dt_convergence", {"conv": conv})
    if plot:
        path = plotting.plot_dt_convergence(conv, order=order)
        print(f"  wrote {path}")
    return conv


if __name__ == "__main__":
    run()
