"""
run_fh_h2.py  --  Quantinuum H2 emulator run (or free local fallback).

Builds the shallow Trotter quench circuit(s) for the small H2 lattice and submits
them for shots. Backend selection:

  * cfg.RUN_ON_H2_EMULATOR = False (default): uses fh_local_sampler, which draws
    Z-basis shots from the exact statevector -- i.e. the SAME distribution as
    Quantinuum's noiseless H2-1LE emulator, for free and instantly. Good for
    reproducibility and CI.
  * cfg.RUN_ON_H2_EMULATOR = True: imports submit_vqe_batch_job from a real
    backend module (qnexus_backend for the cloud H2 emulators, or
    local_emulator_backend for a local pytket-quantinuum+pecos install). Those
    modules are shared with the TFIM project and expose the same generic
    (circuits, n_shots, device_name, project_name, job_name) -> bitstrings API.

All requested observables are Z-diagonal, so a single Z-basis measurement per
time point suffices. Results are compared to ED and saved.
"""
from __future__ import annotations

import numpy as np

import fh_config as cfg
from fh_lattice import HubbardLattice
from fh_exact_diagonalization import ed_time_evolution
from fh_tket_circuit import build_quench_circuit
from fh_shot_observables import bitstrings_to_observables, bootstrap_errors
import fh_plotting as plotting
import fh_persistence as persistence


def _get_submit_fn():
    """Return a submit_vqe_batch_job(circuits, n_shots, device_name, project_name,
    job_name) callable for the configured backend."""
    if not cfg.RUN_ON_H2_EMULATOR:
        from fh_local_sampler import submit_vqe_batch_job
        return submit_vqe_batch_job, "local-sampler (H2-1LE-equivalent, noiseless)"
    # Real backends (shared with the TFIM project). Prefer qnexus, fall back to
    # the local pytket-quantinuum backend.
    try:
        from fh_qnexus_backend import submit_vqe_batch_job
        return submit_vqe_batch_job, f"qnexus:{cfg.H2_DEVICE_NAME}"
    except Exception:
        # local_emulator_backend lives in src/, not this package -- only
        # importable when src/ is also on sys.path (e.g. when this module
        # runs via the repo root's main.py, which puts both on sys.path).
        from local_emulator_backend import submit_vqe_batch_job
        return submit_vqe_batch_job, f"local-emulator:{cfg.H2_DEVICE_NAME}"


def run(save=True, plot=True):
    Lx, Ly = cfg.H2_LATTICE
    lat = HubbardLattice(Lx, Ly)
    t, U, dt, steps = cfg.T_HOP, cfg.H2_U, cfg.H2_DT, cfg.H2_STEPS
    order, init, shots = cfg.H2_TROTTER_ORDER, cfg.H2_INITIAL_STATE, cfg.H2_SHOTS

    submit_fn, backend_desc = _get_submit_fn()
    print(f"H2 run on {lat} ({lat.n_qubits} qubits) via {backend_desc}")
    print(f"  U/t={U/t:.0f}, dt={dt}, steps={steps}, order={order}, shots={shots}")

    # one measured circuit per time point (1..steps)
    circuits = [build_quench_circuit(lat, t, U, dt, k + 1, initial_state=init, order=order)
                for k in range(steps)]
    results = submit_fn(circuits, shots, device_name=cfg.H2_DEVICE_NAME,
                        project_name=cfg.H2_PROJECT_NAME, job_name="fh-quench")

    ed = ed_time_evolution(lat, t, U, dt, steps, initial_state=init)
    times = np.arange(1, steps + 1) * dt
    D = np.zeros(steps); M = np.zeros(steps); eD = np.zeros(steps); eM = np.zeros(steps)
    N = np.zeros(steps); dens = np.zeros((lat.n_sites, steps))
    for k, bits in enumerate(results):
        obs = bitstrings_to_observables(bits, lat)
        err = bootstrap_errors(bits, lat, n_boot=200, seed=k)
        D[k] = obs["avg_double_occupancy"]; M[k] = obs["staggered_magnetization"]
        eD[k] = err["avg_double_occupancy"]; eM[k] = err["staggered_magnetization"]
        N[k] = obs["total_particles"]; dens[:, k] = obs["density_per_site"]

    print("  time    <D>_shots        m_stag_shots      <D>_ED   m_stag_ED   <N>")
    for k in range(steps):
        print(f"  {times[k]:4.2f}  {D[k]:.3f}+/-{eD[k]:.3f}   "
              f"{M[k]:+.3f}+/-{eM[k]:.3f}   {ed['avg_double_occupancy'][k]:.3f}   "
              f"{ed['staggered_magnetization'][k]:+.3f}   {N[k]:.2f}")

    shots_traj = {"times": times, "avg_double_occupancy": D,
                  "staggered_magnetization": M, "density_per_site": dens, "sites": lat.sites}
    shot_err = {"avg_double_occupancy": eD, "staggered_magnetization": eM}

    if save:
        persistence.save_stage_results("h2_run", {
            "backend": backend_desc, "lattice": [Lx, Ly], "n_qubits": lat.n_qubits,
            "U": U, "dt": dt, "steps": steps, "shots": shots,
            "times": times.tolist(), "D": D.tolist(), "M": M.tolist(),
            "D_err": eD.tolist(), "M_err": eM.tolist(), "N": N.tolist(),
            "ed_D": ed["avg_double_occupancy"].tolist(),
            "ed_M": ed["staggered_magnetization"].tolist(),
        })
    if plot:
        path = plotting.plot_quench_dynamics(ed, ed, shots_traj, shot_err, order=order,
                                             save_dir=cfg.PLOT_SAVE_DIR,
                                             fname="fig8_h2_run.png")
        # (passing ED as the "trotter" curve too keeps the reference line clean;
        #  the shots are what we care about here)
        print(f"  wrote {path}")
    return shots_traj, ed


if __name__ == "__main__":
    run()