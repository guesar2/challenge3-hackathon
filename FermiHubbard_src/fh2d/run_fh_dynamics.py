"""
run_fh_dynamics.py  --  quench time dynamics (fig2).

From the half-filling Neel product state, evolve under the Hubbard Hamiltonian
and track average double occupancy and staggered magnetization. Compares FIVE
levels on the SAME instance, each one adding a single error source to the one
above it:

  1. ED (exact, expm_multiply)          -- the reference
  2. second-order Trotter (statevector) -- + algorithmic (Trotter) error
  3. NOISELESS emulator shots           -- + shot noise (free local statevector
                                           sampler = the H2-1LE distribution)
  4. raw noisy emulator shots           -- + device noise (H2-Emulator's
                                           published noise model)
  5. ZNE-mitigated noisy shots          -- 4 extrapolated back to zero noise

Levels 4 and 5 come from fh_zne.py; they are evaluated on a subset of the time
grid (cfg.NOISY_TIME_STRIDE) to keep the circuit count down, and they run
against the free local noisy stand-in unless cfg.RUN_ON_H2_EMULATOR is True, in
which case they go to the real H2-Emulator through qnexus and cost quota.

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
from fh_zne import noisy_zne_trajectories, compare_to_ed, zne_improvement
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


def run(save=True, plot=True, with_shots=True, shots=2000, with_noise=None,
        noisy_shots=None, _noisy_override=None):
    Lx, Ly = cfg.LX, cfg.LY
    lat = HubbardLattice(Lx, Ly)
    t, U, dt, steps = cfg.T_HOP, cfg.QUENCH_U, cfg.QUENCH_DT, cfg.QUENCH_STEPS
    order, init = cfg.TROTTER_ORDER, cfg.QUENCH_INITIAL_STATE
    with_noise = cfg.RUN_NOISY_DYNAMICS if with_noise is None else with_noise

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
        print(f"  Noiseless shots ({shots}/pt) vs ED: <D> {sD:.2f}% | m_stag {sM:.2f}%")

    # ---- noisy device + zero-noise extrapolation ----
    noisy = noisy_err = zne = zne_err = None
    if _noisy_override is not None:
        # Series recovered from an already-executed Nexus job -- see
        # fh_zne.resume_noisy_zne(), which redraws the figure without resubmitting.
        noisy, noisy_err, zne, zne_err = _noisy_override
    elif with_noise:
        noisy, noisy_err, zne, zne_err = noisy_zne_trajectories(
            lat, t, U, dt, steps, order=order, initial_state=init, shots=noisy_shots)
    if noisy is not None:
        for key, label in (("avg_double_occupancy", "<D>    "),
                           ("staggered_magnetization", "m_stag ")):
            rawdev = compare_to_ed(ed, noisy, key)
            znedev = compare_to_ed(ed, zne, key)
            better, total = zne_improvement(ed, noisy, zne, key)
            print(f"  {label} vs ED:  raw noisy {rawdev:6.2f}%  ->  ZNE {znedev:6.2f}%   "
                  f"(ZNE closer to ED at {better}/{total} time points)")

    if save:
        payload = {
            "lattice": [Lx, Ly], "t": t, "U": U, "dt": dt, "steps": steps, "order": order,
            "ed": {k: np.asarray(v).tolist() for k, v in ed.items() if k != "sites"},
            "trotter": {k: np.asarray(v).tolist() for k, v in trot.items() if k != "sites"},
            "max_dev_percent": {"double_occ": devD, "m_stag": devM},
        }
        if noisy is not None:
            # Persist the noisy/ZNE series too -- on the real backend these cost
            # quota, so they must survive a plotting crash.
            payload["noisy"] = {
                "backend": noisy["backend"], "shots": noisy["shots"],
                "fold_factors": list(noisy["fold_factors"]),
                "noise_scale": noisy["noise_scale"],
                "step_counts": list(noisy["step_counts"]),
                "times": np.asarray(noisy["times"]).tolist(),
                "avg_double_occupancy": np.asarray(noisy["avg_double_occupancy"]).tolist(),
                "staggered_magnetization": np.asarray(noisy["staggered_magnetization"]).tolist(),
                "avg_double_occupancy_err": np.asarray(noisy_err["avg_double_occupancy"]).tolist(),
                "staggered_magnetization_err": np.asarray(noisy_err["staggered_magnetization"]).tolist(),
            }
            payload["zne"] = {
                "fit_deg": zne["fit_deg"],
                "times": np.asarray(zne["times"]).tolist(),
                "avg_double_occupancy": np.asarray(zne["avg_double_occupancy"]).tolist(),
                "staggered_magnetization": np.asarray(zne["staggered_magnetization"]).tolist(),
                "avg_double_occupancy_err": np.asarray(zne_err["avg_double_occupancy"]).tolist(),
                "staggered_magnetization_err": np.asarray(zne_err["staggered_magnetization"]).tolist(),
            }
        persistence.save_stage_results("dynamics", payload)
    if plot:
        p1 = plotting.plot_quench_dynamics(ed, trot, shots_traj, shot_err, order=order,
                                           noisy=noisy, noisy_err=noisy_err,
                                           zne=zne, zne_err=zne_err)
        print(f"  wrote {p1}")
    return ed, trot, shots_traj, noisy, zne


if __name__ == "__main__":
    run()
