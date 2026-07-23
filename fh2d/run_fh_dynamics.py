"""
run_fh_dynamics.py  --  quench time dynamics (fig2).

From the half-filling Neel product state, evolve under the Hubbard Hamiltonian
and track average double occupancy and staggered magnetization. Compares three
levels on the SAME instance:
  ED (exact, expm_multiply)  vs  second-order Trotter (statevector)  vs  shots
(via the free local statevector sampler by default, i.e. the H2-1LE noiseless
distribution; swap in the real H2 backend by setting cfg.RUN_ON_H2_EMULATOR).

Writes fig2_quench_dynamics.png and prints the max %-deviation vs ED (the <5%
pass check). The per-site density map moved to run_fh_heatmap.py, which quenches
a charge-imbalanced state on 3x4 -- a Neel quench has uniform density and so
produced a featureless map here. The ED-vs-shots bar chart (fig7) was removed;
the same information is already in fig2, with error bars and time resolution.
"""
from __future__ import annotations

import numpy as np

import fh_config as cfg
from fh_lattice import HubbardLattice
from fh_exact_diagonalization import ed_time_evolution
from fh_trotter_simulation import trotter_time_evolution, max_percent_deviation
from fh_tket_circuit import build_quench_circuit
from fh_local_sampler import sample_bitstrings
from fh_shot_observables import bitstrings_to_observables, bootstrap_errors
import fh_plotting as plotting
import fh_persistence as persistence


def _shot_trajectory(lat, t, U, dt, steps, order, initial_state, shots, seed=0):
    """Build a quench circuit at each time point, sample, and collect observables
    into the same dict shape as the ED/Trotter trajectories, with error bars."""
    times = np.arange(1, steps + 1) * dt
    D = np.zeros(steps); M = np.zeros(steps)
    eD = np.zeros(steps); eM = np.zeros(steps)
    dens = np.zeros((lat.n_sites, steps))
    for k in range(steps):
        circ = build_quench_circuit(lat, t, U, dt, k + 1, initial_state=initial_state,
                                    order=order)
        bits = sample_bitstrings(circ, shots, seed=seed + k)
        obs = bitstrings_to_observables(bits, lat)
        err = bootstrap_errors(bits, lat, n_boot=200, seed=seed + 1000 + k)
        D[k] = obs["avg_double_occupancy"]; M[k] = obs["staggered_magnetization"]
        eD[k] = err["avg_double_occupancy"]; eM[k] = err["staggered_magnetization"]
        dens[:, k] = obs["density_per_site"]
    return ({"times": times, "avg_double_occupancy": D, "staggered_magnetization": M,
             "density_per_site": dens, "sites": lat.sites},
            {"avg_double_occupancy": eD, "staggered_magnetization": eM})


def run(save=True, plot=True, with_shots=True, shots=2000):
    Lx, Ly = cfg.LX, cfg.LY
    lat = HubbardLattice(Lx, Ly)
    t, U, dt, steps = cfg.T_HOP, cfg.QUENCH_U, cfg.QUENCH_DT, cfg.QUENCH_STEPS
    order, init = cfg.TROTTER_ORDER, cfg.QUENCH_INITIAL_STATE

    print(f"Quench dynamics on {lat} at U/t={U/t:.0f}, dt={dt}, {steps} steps, "
          f"Trotter order {order}, init={init}")
    ed = ed_time_evolution(lat, t, U, dt, steps, initial_state=init)
    trot = trotter_time_evolution(lat, t, U, dt, steps, initial_state=init, order=order)

    devD = max_percent_deviation(trot, ed, "avg_double_occupancy")
    devM = max_percent_deviation(trot, ed, "staggered_magnetization")
    print(f"  Trotter vs ED max deviation: <D> {devD:.2f}% | m_stag {devM:.2f}%  "
          f"({'PASS' if max(devD, devM) < 5 else 'ABOVE'} 5% bar)")

    shots_traj = shot_err = None
    if with_shots:
        shots_traj, shot_err = _shot_trajectory(lat, t, U, dt, steps, order, init, shots)
        sD = max_percent_deviation(shots_traj, ed, "avg_double_occupancy")
        sM = max_percent_deviation(shots_traj, ed, "staggered_magnetization")
        print(f"  Emulator shots ({shots}/pt) vs ED: <D> {sD:.2f}% | m_stag {sM:.2f}%")

    if save:
        persistence.save_stage_results("dynamics", {
            "lattice": [Lx, Ly], "t": t, "U": U, "dt": dt, "steps": steps, "order": order,
            "ed": {k: np.asarray(v).tolist() for k, v in ed.items() if k != "sites"},
            "trotter": {k: np.asarray(v).tolist() for k, v in trot.items() if k != "sites"},
            "max_dev_percent": {"double_occ": devD, "m_stag": devM},
        })
    if plot:
        p1 = plotting.plot_quench_dynamics(ed, trot, shots_traj, shot_err, order=order)
        print(f"  wrote {p1}")
    return ed, trot, shots_traj


if __name__ == "__main__":
    run()
