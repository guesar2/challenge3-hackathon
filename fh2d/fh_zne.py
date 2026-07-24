"""
fh_zne.py

Noisy and error-mitigated quench trajectories for the Fermi-Hubbard quench
figure -- the FH counterpart of the TFIM project's run_zne.py, reduced to the
piece the quench figure actually needs.

Pipeline, one point per requested time:

  1. Build the Trotter quench ansatz at that step count (fh_tket_circuit).
  2. Fold it by every factor in cfg.FH_ZNE_FOLD_FACTORS to amplify noise
     (qermit Folding.circuit on the real backend; an equivalent gate-count
     scaling in the local stand-in), measure in the Z basis, and take shots.
     Folding.circuit at fold_factor=1 performs zero fold iterations, so that
     point IS the raw-noisy circuit -- no separate raw submission is needed.
  3. Convert each fold's bitstrings to observables with real bootstrap standard
     errors (fh_shot_observables), exactly as the noiseless shot path does.
  4. Fit each observable's (fold_factor, value, error) series and extrapolate to
     fold_factor = 0, the zero-noise limit, propagating the bootstrap errors
     into an error bar on the mitigated point (zne_fit.zne_extrapolate, reused
     unchanged from the TFIM project -- it is pure numpy and model-agnostic).

If the client gives up waiting before Nexus finishes (TimeoutError), the job
keeps running and its name is already on disk, so resume_noisy_zne() collects
the results afterwards for free -- resubmitting would spend the quota twice.

Backend selection follows cfg.RUN_ON_H2_EMULATOR, the same switch the rest of
the project uses:
  True  -> fh_qnexus_backend.submit_zne_batch on cfg.H2_DEVICE_NAME_NOISY
           ("H2-Emulator", Quantinuum's published noise model). Costs quota.
  False -> fh_local_noisy_sampler.submit_zne_batch, a free local stand-in.
           Good for testing the plumbing and producing the figure offline; NOT
           a device-accurate noise model (see that module's docstring).
"""
from __future__ import annotations

import numpy as np

import fh_config as cfg
import fh_persistence as persistence
from fh_shot_observables import bitstrings_to_observables, bootstrap_errors
from zne_fit import zne_extrapolate

# The two scalar observables the quench figure tracks.
OBS_KEYS = ("avg_double_occupancy", "staggered_magnetization")


def _get_zne_backend():
    """Return (submit_zne_batch, device_name, description) for the configured
    backend. Same shape as run_fh_h2._get_submit_fn."""
    if not cfg.RUN_ON_H2_EMULATOR:
        from fh_local_noisy_sampler import submit_zne_batch
        return (submit_zne_batch, "local-noisy-sampler",
                "local noisy sampler (depolarizing + readout stand-in, free)")
    from fh_qnexus_backend import submit_zne_batch
    return (submit_zne_batch, cfg.H2_DEVICE_NAME_NOISY,
            f"qnexus:{cfg.H2_DEVICE_NAME_NOISY} (published device noise model)")


def zne_step_counts(steps, stride=None, max_steps=None):
    """Trotter step counts at which the noisy/ZNE curves are evaluated: every
    `stride`-th step, up to `max_steps` (cfg.NOISY_MAX_STEPS; None = all of
    them), always including the last one kept.

    These are a SUBSET of the ED/Trotter grid (time = step_count * dt), so every
    curve in the figure lands on the same time axis. Two separate reasons to
    trim: `stride` cuts the number of circuits, `max_steps` cuts their DEPTH,
    which is what dominates both emulator runtime and how much signal survives
    the noise (see the cost model in fh_config.py).
    """
    stride = cfg.NOISY_TIME_STRIDE if stride is None else int(stride)
    stride = max(1, stride)
    if max_steps is None:
        max_steps = getattr(cfg, "NOISY_MAX_STEPS", None)
    last = steps if max_steps is None else min(int(max_steps), steps)
    counts = list(range(stride, last + 1, stride))
    if not counts:
        counts = [last]
    if counts[-1] != last:
        counts.append(last)
    return counts


def estimate_cost(lat, step_counts, fold_factors, shots, order=2):
    """(n_circuits, total two-qubit gate-shots) for a ZNE grid -- the number that
    actually predicts emulator runtime. Uses the real per-step CX count of the
    decomposed circuit (112 for 2x2 at order 2), so it stays right if the lattice
    or Trotter order changes."""
    from fh_tket_circuit import build_quench_ansatz_circuit
    from fh_local_noisy_sampler import count_entangling_gates
    per_step = count_entangling_gates(
        build_quench_ansatz_circuit(lat, 1.0, 1.0, 0.1, 1, order=order))
    n_circuits = len(step_counts) * len(fold_factors)
    gate_shots = per_step * sum(step_counts) * sum(fold_factors) * shots
    return n_circuits, int(gate_shots)


def _postprocess(batch, lat, step_counts, fold_factors, dt, fit_deg, raw_idx,
                 seed=0, n_boot=200, meta=None):
    """Bitstrings -> (noisy, noisy_err, zne, zne_err). Shared by the fresh-run and
    the resume path so both produce byte-identical trajectories."""
    meta = meta or {}
    vals = {key: np.zeros((len(step_counts), len(fold_factors))) for key in OBS_KEYS}
    errs = {key: np.zeros((len(step_counts), len(fold_factors))) for key in OBS_KEYS}
    dens = np.zeros((lat.n_sites, len(step_counts)))

    for i, sc in enumerate(step_counts):
        for j, fold in enumerate(fold_factors):
            bits = batch[sc][fold]
            obs = bitstrings_to_observables(bits, lat)
            err = bootstrap_errors(bits, lat, n_boot=n_boot, seed=seed + 7919 * i + j)
            for key in OBS_KEYS:
                vals[key][i, j] = obs[key]
                errs[key][i, j] = err[key]
            if fold == fold_factors[raw_idx]:
                dens[:, i] = obs["density_per_site"]

    times = np.asarray(step_counts, dtype=float) * dt
    noisy = {"times": times, "step_counts": list(step_counts), "density_per_site": dens,
             "sites": lat.sites, "fold_factors": tuple(fold_factors), **meta}
    zne = {"times": times, "step_counts": list(step_counts),
           "fold_factors": tuple(fold_factors), "fit_deg": fit_deg,
           "backend": meta.get("backend", "?")}
    noisy_err, zne_err = {}, {}

    for key in OBS_KEYS:
        noisy[key] = vals[key][:, raw_idx].copy()
        noisy_err[key] = errs[key][:, raw_idx].copy()
        z = np.zeros(len(step_counts))
        ze = np.zeros(len(step_counts))
        for i in range(len(step_counts)):
            z[i], ze[i] = zne_extrapolate(fold_factors, vals[key][i], errs[key][i],
                                          deg=fit_deg)
        zne[key] = z
        zne_err[key] = ze

    return noisy, noisy_err, zne, zne_err


def _save_raw(batch, step_counts, fold_factors, meta):
    """Archive the raw shots BEFORE any postprocessing. On the real backend these
    cost quota, so a crash in the observable/fit code must not force a resubmit.
    JSON keys are strings, hence the "{steps}_{fold}" flattening."""
    payload = dict(meta)
    payload.update({
        "step_counts": list(step_counts), "fold_factors": list(fold_factors),
        "bitstrings": {f"{sc}_{fold}": batch[sc][fold]
                       for sc in step_counts for fold in fold_factors},
    })
    return persistence.save_stage_results("dynamics_zne_raw", payload)


def noisy_zne_trajectories(lat, t, U, dt, steps, order=2, initial_state="neel",
                           shots=None, fold_factors=None, fit_deg=None, stride=None,
                           max_steps=None, noise_scale=None, timeout=None,
                           seed=0, n_boot=200, verbose=True):
    """Run the noisy quench and its zero-noise extrapolation.

    Returns (noisy, noisy_err, zne, zne_err), four dicts whose 'times' and the
    two OBS_KEYS entries have the same shape as the ED/Trotter/noiseless-shot
    trajectories, so fh_plotting.plot_quench_dynamics can overlay them directly.
    Returns (None, None, None, None) if the backend raises -- the rest of the
    figure still gets made. In particular, a client-side TimeoutError leaves the
    Nexus job running and its name saved, so resume_noisy_zne() can pick the
    (already paid for) results up afterwards.
    """
    shots = cfg.NOISY_SHOTS if shots is None else shots
    fold_factors = tuple(cfg.FH_ZNE_FOLD_FACTORS if fold_factors is None else fold_factors)
    fit_deg = cfg.FH_ZNE_FIT_DEG if fit_deg is None else fit_deg
    noise_scale = cfg.NOISY_NOISE_SCALE if noise_scale is None else noise_scale
    timeout = getattr(cfg, "NOISY_TIMEOUT", 1800.0) if timeout is None else timeout

    if 1 not in fold_factors:
        raise ValueError("fold_factors must contain 1 -- fold 1 IS the raw-noisy "
                         "circuit (Folding.circuit does zero fold iterations there)")
    if fit_deg >= len(fold_factors):
        raise ValueError(f"fit_deg={fit_deg} needs more than {len(fold_factors)} fold factors")
    raw_idx = list(fold_factors).index(1)

    step_counts = zne_step_counts(steps, stride, max_steps)
    submit_fn, device_name, desc = _get_zne_backend()
    n_circuits, gate_shots = estimate_cost(lat, step_counts, fold_factors, shots, order)

    if verbose:
        print(f"  Noisy + ZNE via {desc}")
        print(f"    {len(step_counts)} time points (t = {step_counts[0] * dt:.2f} .. "
              f"{step_counts[-1] * dt:.2f}) x {len(fold_factors)} folds = {n_circuits} "
              f"circuits, {shots} shots each")
        print(f"    folds={fold_factors}, fit deg={fit_deg}, noise_scale={noise_scale}, "
              f"cost ~ {gate_shots:.2e} two-qubit gate-shots, client timeout {timeout:g}s")

    meta = {"backend": desc, "device": device_name, "shots": shots,
            "noise_scale": noise_scale, "lattice": [lat.Lx, lat.Ly],
            "t": t, "U": U, "dt": dt, "order": order, "initial_state": initial_state}

    # Start and collect as two steps on the real backend, so the job name is on
    # disk before any waiting begins and a timeout is recoverable.
    if cfg.RUN_ON_H2_EMULATOR:
        from fh_qnexus_backend import start_zne_batch, collect_zne_batch
        try:
            pending = start_zne_batch(
                lat, t, U, dt, step_counts, fold_factors, shots,
                initial_state=initial_state, order=order, device_name=device_name,
                project_name=cfg.H2_PROJECT_NAME, noise_scale=noise_scale)
        except Exception as exc:                  # noqa: BLE001
            print(f"    !! noisy/ZNE stage skipped: {type(exc).__name__}: {exc}")
            return None, None, None, None

        job_record = {"job_name": pending["job_name"], "project_name": pending["project_name"],
                      "step_counts": list(step_counts), "fold_factors": list(fold_factors),
                      "fit_deg": fit_deg, **meta}
        persistence.save_stage_results("dynamics_zne_job", job_record)

        try:
            batch = collect_zne_batch(pending, timeout=timeout)
        except TimeoutError:
            print(f"    !! client-side timeout after {timeout:g}s. The Nexus job "
                  f"'{pending['job_name']}' IS STILL RUNNING and the quota is already "
                  f"spent -- do NOT resubmit. Once it finishes, recover it with:")
            print(f"         python -c \"import fh_zne; fh_zne.resume_noisy_zne()\"")
            print(f"       or raise cfg.NOISY_TIMEOUT / shrink the grid "
                  f"(NOISY_MAX_STEPS, NOISY_TIME_STRIDE, NOISY_SHOTS, FH_ZNE_FOLD_FACTORS).")
            return None, None, None, None
        except Exception as exc:                  # noqa: BLE001
            print(f"    !! noisy/ZNE stage skipped: {type(exc).__name__}: {exc}")
            return None, None, None, None
    else:
        try:
            batch = submit_fn(lat, t, U, dt, step_counts, fold_factors, shots,
                              initial_state=initial_state, order=order,
                              device_name=device_name, project_name=cfg.H2_PROJECT_NAME,
                              job_name="fh-zne-quench", noise_scale=noise_scale, seed=seed)
        except Exception as exc:                  # noqa: BLE001
            print(f"    !! noisy/ZNE stage skipped: {type(exc).__name__}: {exc}")
            return None, None, None, None

    _save_raw(batch, step_counts, fold_factors, meta)
    return _postprocess(batch, lat, step_counts, fold_factors, dt, fit_deg, raw_idx,
                        seed=seed, n_boot=n_boot, meta=meta)


def resume_noisy_zne(job_name=None, timeout=None, seed=0, n_boot=200, replot=True):
    """Recover a ZNE job that was started but whose results were never collected
    (client-side TimeoutError, crash, interrupted run) and rebuild the figure.
    Costs NO additional quota -- the job already ran on Nexus.

    With no arguments it reads the job record fh_zne wrote at submission time
    (data_fh/dynamics_zne_job.json), so the usual recovery is simply:

        python -c "import fh_zne; fh_zne.resume_noisy_zne()"

    Returns (noisy, noisy_err, zne, zne_err).
    """
    from fh_qnexus_backend import resume_zne_batch

    record = persistence.load_stage_results("dynamics_zne_job")
    if record is None and job_name is None:
        raise RuntimeError("no saved ZNE job record (data_fh/dynamics_zne_job.json) "
                           "and no job_name given")
    record = record or {}
    job_name = job_name or record["job_name"]
    step_counts = list(record["step_counts"])
    fold_factors = tuple(record["fold_factors"])
    fit_deg = record.get("fit_deg", cfg.FH_ZNE_FIT_DEG)
    dt = record.get("dt", cfg.QUENCH_DT)
    timeout = getattr(cfg, "NOISY_TIMEOUT", 1800.0) if timeout is None else timeout

    from fh_lattice import HubbardLattice
    Lx, Ly = record.get("lattice", [cfg.LX, cfg.LY])
    lat = HubbardLattice(Lx, Ly)

    batch = resume_zne_batch(job_name, step_counts, fold_factors,
                             project_name=record.get("project_name", cfg.H2_PROJECT_NAME),
                             timeout=timeout)
    meta = {k: record[k] for k in ("backend", "device", "shots", "noise_scale",
                                   "t", "U", "dt", "order", "initial_state")
            if k in record}
    _save_raw(batch, step_counts, fold_factors, meta)
    raw_idx = list(fold_factors).index(1)
    out = _postprocess(batch, lat, step_counts, fold_factors, dt, fit_deg, raw_idx,
                       seed=seed, n_boot=n_boot, meta=meta)

    if replot:
        # Re-run the (free, classical) ED/Trotter/noiseless parts and redraw fig2
        # with the recovered noisy + ZNE series.
        import run_fh_dynamics
        run_fh_dynamics.run(with_noise=False, _noisy_override=out)
    return out


def compare_to_ed(ed, traj, key):
    """Max %-deviation of a (possibly subsampled) trajectory from ED, comparing
    only at the time points the trajectory actually has. Mirrors
    fh_trotter_simulation.max_percent_deviation, which assumes the full grid."""
    idx = np.asarray(traj["step_counts"], dtype=int) - 1
    a = np.asarray(traj[key], dtype=float)
    b = np.asarray(ed[key], dtype=float)[idx]
    denom = np.max(np.abs(np.asarray(ed[key], dtype=float)))
    if denom == 0:
        return 0.0
    return float(np.max(np.abs(a - b)) / denom * 100)


def zne_improvement(ed, noisy, zne, key):
    """How many time points ZNE moved closer to ED than the raw noisy value --
    the headline 'did mitigation help' number, same check run_zne.py prints."""
    idx = np.asarray(noisy["step_counts"], dtype=int) - 1
    e = np.asarray(ed[key], dtype=float)[idx]
    r = np.asarray(noisy[key], dtype=float)
    z = np.asarray(zne[key], dtype=float)
    return int(np.sum(np.abs(z - e) < np.abs(r - e))), len(e)
