"""
run_h2_emulator.py

Capstone section 6/6: TFIM Trotter quench circuit run against a Quantinuum
H2 device, with the circuit built in pytket (see tket_circuit.py). Two
interchangeable execution backends, selected via each function's `local`
argument (also `--local` on the command line):

- qnexus_backend (default, local=False): submits over the network through
  qnexus. Requires a live qnexus login and costs against a metered usage
  quota -- config.RUN_ON_H2_EMULATOR gates every call in this mode. This is
  the one section ftim_main.py will still print-and-skip unless that flag
  is flipped; run standalone with `python run_h2_emulator.py` once you're
  ready to spend quota.
- local_emulator_backend (local=True): runs the same circuits locally via
  pytket-quantinuum + pytket-pecos (`pip install "pytket-quantinuum[pecos]"`).
  H2-1LE is exact noiseless state-vector emulation (only shot noise, no
  physical noise model -- see qnexus_backend.py's docstring for the source
  quote), so this is the *same* computation as the qnexus path, just free
  and instant instead of network-queued and quota-metered. Not gated by
  RUN_ON_H2_EMULATOR since there's no quota to protect. Results are saved
  under a "_local" stage suffix so they never overwrite a real qnexus run's
  saved data.

Each of run()/run_phase_transition()/run_vqe() also takes a `noisy` argument
(also `--noisy` on the command line) -- the noisy counterpart to the default
H2_DEVICE_NAME (H2-1LE, noiseless). Passing noisy=True submits the exact
same circuits to config.H2_DEVICE_NAME_NOISY (H2-Emulator) instead, which includes
Quantinuum's real gate/SPAM/crosstalk noise model. Unlike local vs. qnexus,
this isn't a free/paid pair of equivalent computations -- H2-Emulator carries
real published noise_specs (gate/SPAM/crosstalk/dephasing error rates,
not exact state-vector emulation), and it's Nexus-hosted rather than a
device pytket-pecos can run locally, so it's only reachable through
qnexus. noisy=True has no effect when local=True (there's no way to
inject H2-Emulator's noise model into the local pytket-pecos path). noisy=True
still costs qnexus quota and is still gated by config.RUN_ON_H2_EMULATOR --
it does not submit anything by itself, it just points the same
submit_*_batch calls at the noisy device. Results are saved under a
"_noisy" stage suffix so they never overwrite a H2-1LE run's saved data.

Because every qnexus submission costs quota, raw shot data is persisted
(data/h2_emulator_raw_*.json) immediately after each job returns -- before
any postprocessing -- so a bug in bitstrings_to_observables (or anything
downstream) never means re-submitting to recover results that already came
back. The same persist-immediately pattern is kept for local runs too, for
consistency, even though there's no quota risk there.
"""
import config
from exact_diagonalization import ed_baseline, ed_time_evolution_exact
from persistence import save_stage_results, load_latest
from plotting import plot_h2_vs_ed_time, plot_h2_phase_transition, plot_vqe_convergence
from shot_observables import (
    bitstrings_to_observables, bootstrap_observable_errors,
    bitstrings_to_mx, bootstrap_mx_error,
)

from vqe import run_vqe_h2
from local_emulator_backend import submit_vqe_batch_job as local_submit
from qnexus_backend import submit_vqe_batch_job as nexus_submit


def run(local=False, noisy=False):
    device_name = config.H2_DEVICE_NAME_NOISY if (noisy and not local) else config.H2_DEVICE_NAME
    print("=" * 60)
    print(f"STAGE 6/6: QUANTINUUM {device_name} EMULATOR "
          f"({'local pytket-quantinuum/pecos' if local else 'qnexus/pytket'})")
    print("=" * 60)

    if noisy and local:
        print("Note: noisy=True has no effect when local=True -- pytket-pecos's "
              "local emulator only runs H2-*LE (noiseless) devices; H2-Emulator's noise "
              "model is only reachable through qnexus. Running local H2-1LE as usual.")

    if not local and not config.RUN_ON_H2_EMULATOR:
        print("Skipped (config.RUN_ON_H2_EMULATOR = False). Enable it to submit "
              "jobs to qnexus -- this consumes a metered usage quota. "
              "(Or pass local=True / --local to run for free instead.)")
        return None

    if local:
        from local_emulator_backend import submit_quench_batch
    else:
        from qnexus_backend import submit_quench_batch
    stage_suffix = "_local" if local else ("_noisy" if noisy else "")
    raw_stage = f"h2_emulator_raw{stage_suffix}"
    results_stage = f"h2_emulator{stage_suffix}"

    # qnexus/H2 only returns final measurement shots -- no mid-circuit
    # statevector readout -- so a <Z>/<X>/<Zi Zi+1> vs. time curve needs
    # one circuit per (h, step_count) snapshot, per basis. submit_quench_batch
    # submits the whole step_count curve for one h (both bases) as a single
    # compile/execute call (rather than one queue-wait per step) since
    # Quantinuum's own hardware-only batching feature doesn't apply to the
    # emulator devices.
    raw_by_h = {}
    results = {}
    step_counts = list(range(1, config.H2_STEPS + 1))
    for h in config.H2_H_VALUES:
        print(f"\nSubmitting N={config.H2_N}, h/J={h:.2f}, dt={config.H2_DT}, "
              f"steps=1..{config.H2_STEPS} (batched), shots={config.H2_SHOTS} "
              f"to {device_name} ...")

        batch_results = submit_quench_batch(
            config.H2_N, h, config.J, config.H2_DT, step_counts,
            config.H2_SHOTS, device_name=device_name,
            project_name=config.H2_PROJECT_NAME,
        )

        # Persist the raw hardware result immediately -- before any
        # postprocessing -- so it survives even if bitstrings_to_observables
        # (or anything after it) turns out to be buggy.
        raw_by_h[h] = batch_results
        save_stage_results(raw_stage, raw_by_h)

        times, z_h2, z_err, x_h2, x_err, mzz_h2, mzz_err = [], [], [], [], [], [], []
        for step_count in step_counts:
            job_result = batch_results[step_count]
            z_rms, mzz = bitstrings_to_observables(job_result["bitstrings"], config.H2_N)
            z_se, mzz_se = bootstrap_observable_errors(job_result["bitstrings"], config.H2_N)
            # <X> does not vanish by symmetry the way <Z> does (the -h*X
            # field polarizes the ground state along +X), so it's a plain
            # signed mean over shots -- see bitstrings_to_mx -- not the
            # RMS formula bitstrings_to_observables uses for <Z>.
            x_mean = bitstrings_to_mx(job_result["bitstrings_x"], config.H2_N)
            x_se = bootstrap_mx_error(job_result["bitstrings_x"], config.H2_N)

            times.append(step_count * config.H2_DT)
            z_h2.append(z_rms)
            z_err.append(z_se)
            x_h2.append(x_mean)
            x_err.append(x_se)
            mzz_h2.append(mzz)
            mzz_err.append(mzz_se)

        _, z_ed, mzz_ed, x_ed = ed_time_evolution_exact(
            config.H2_N, h, config.J, config.H2_DT, config.H2_STEPS
        )

        max_pct_z = max(abs(a - b) / abs(b) * 100 if b != 0 else 0
                         for a, b in zip(z_h2, z_ed))
        max_pct_x = max(abs(a - b) / abs(b) * 100 if b != 0 else 0
                         for a, b in zip(x_h2, x_ed))
        max_pct_mzz = max(abs(a - b) / abs(b) * 100 if b != 0 else 0
                           for a, b in zip(mzz_h2, mzz_ed))
        print(f"\n  h/J={h:.2f}: max deviation <Z> = {max_pct_z:.2f}%, "
              f"max deviation <X> = {max_pct_x:.2f}%, "
              f"max deviation <Zi Zi+1> = {max_pct_mzz:.2f}%")

        results[h] = {
            'times': times, 'z_h2': z_h2, 'z_err': z_err,
            'x_h2': x_h2, 'x_err': x_err,
            'mzz_h2': mzz_h2, 'mzz_err': mzz_err,
            'z_ed': list(z_ed), 'x_ed': list(x_ed), 'mzz_ed': list(mzz_ed),
        }

    save_stage_results(results_stage, results)

    plot_h2_vs_ed_time(
        config.H2_H_VALUES, results, save_dir=config.PLOT_SAVE_DIR,
        filename=f"h2_vs_ed_time{stage_suffix}.png",
    )

    return results


def run_phase_transition(local=False, noisy=False):
    """Adiabatic-ramp sweep on H2: the phase-transition signal (<Z>/<X>/
    <Zi Zi+1> vs. h/J across the h/J=1 critical point) reproduced on
    hardware, rather than the fixed-h quench-vs-time protocol in run().

    Not called from run() or from ftim_main.py -- invoke explicitly (e.g.
    `python run_h2_emulator.py --phase-transition`) since it's a separate,
    additional quota cost on top of run() when local=False. Pass
    local=True (or --local --phase-transition on the command line) to run
    against local_emulator_backend instead -- free and instant, same
    computation, results saved under a "_local" stage suffix so they never
    overwrite a real qnexus run. Gated by config.RUN_ON_H2_EMULATOR only
    when local=False.

    Ramp length is scaled per h target via sweep_schedule.steps_for_target
    (targeting |dh/dt| ~= config.H2_ADIABATIC_RATE_REF, same logic as the
    local pipeline's adiabatic sweep), capped at config.H2_ADIABATIC_MAX_STEPS
    -- a single flat step count for every target was tried first and
    over-Trotterized targets close to H_INIT without any adiabaticity
    benefit (see config.py's H2_ADIABATIC_RATE_REF comment for the numbers).

    The critical point h/J=1 is pinned to
    H2_ADIABATIC_MAX_STEPS * H2_ADIABATIC_CRITICAL_TIME_FACTOR steps at the
    *same* dt as every other target -- i.e. a longer total ramp time, not
    finer Trotter resolution (textbook critical slowing down: the
    adiabatic theorem needs more *time* as the gap closes, not finer
    time-resolution -- see config.py's H2_ADIABATIC_CRITICAL_TIME_FACTOR
    comment for the numbers that pinned this down).

    A finer-dt-at-fixed-time variant was tried first (since 2000 shots
    showed h/J=1's bias was statistically real, not noise) but barely moved
    the deviation; a local_emulator_backend test isolating time vs.
    resolution (free to run at high shot counts) showed doubling the total
    ramp time at the original dt fixed it (~6.5% -> ~2.15%), while doubling
    resolution at fixed time did not -- see config.py's
    H2_ADIABATIC_CRITICAL_TIME_FACTOR comment for the numbers. Textbook
    critical slowing down: the adiabatic theorem needs more *time* as the
    gap closes, not finer time-resolution.

    Any other target whose ramp *passes through* h/J=1 without landing
    there (e.g. h/J=0.5 with H_INIT=4.0 sweeps down through 1.0 on the way
    to 0.5) gets a related but larger treatment
    (H2_ADIABATIC_TRANSIT_TIME_FACTOR, not H2_ADIABATIC_CRITICAL_TIME_FACTOR)
    -- see _ramp_steps below. With H_INIT=4.0, that's h/J=0.5 in addition
    to h/J=1.0 itself; h/J=2.0's ramp (4.0 -> 2.0) never crosses 1.0 so it
    keeps the plain rate_ref-based step count. See
    H2_ADIABATIC_TRANSIT_TIME_FACTOR's comment in config.py for the local
    sweep that picked its value.
    """
    device_name = config.H2_DEVICE_NAME_NOISY if (noisy and not local) else config.H2_DEVICE_NAME
    print("=" * 60)
    print(f"H2 ADIABATIC SWEEP (phase-transition signal, {device_name}, "
          f"{'local pytket-quantinuum/pecos' if local else 'qnexus/pytket'})")
    print("=" * 60)

    if noisy and local:
        print("Note: noisy=True has no effect when local=True -- see run()'s "
              "module-level note. Running local H2-1LE as usual.")

    if not local and not config.RUN_ON_H2_EMULATOR:
        print("Skipped (config.RUN_ON_H2_EMULATOR = False). Enable it to submit "
              "jobs to qnexus -- this consumes a metered usage quota. "
              "(Or pass local=True / --local to run for free instead.)")
        return None

    if local:
        from local_emulator_backend import submit_adiabatic_batch
    else:
        from qnexus_backend import submit_adiabatic_batch
    from sweep_schedule import steps_for_target
    stage_suffix = "_local" if local else ("_noisy" if noisy else "")
    raw_stage = f"h2_adiabatic_raw{stage_suffix}"
    results_stage = f"h2_adiabatic{stage_suffix}"

    ed_results = ed_baseline(config.H2_ADIABATIC_N, config.H2_H_VALUES, J=config.J)

    h_values = list(config.H2_H_VALUES)
    dt_by_target = [config.H2_ADIABATIC_DT for _ in h_values]  # same dt for every target
    # A target doesn't have to *end* at h/J=1 to need the critical-slowing-down
    # treatment -- a linear ramp from H_INIT down to a target below 1 (e.g.
    # h/J=0.5, with H_INIT=4.0) passes *through* h/J=1 partway along the ramp,
    # at the same constant rate as the rest of the sweep. That transit needs
    # even *more* total time than landing on h/J=1 itself (see
    # H2_ADIABATIC_TRANSIT_TIME_FACTOR's comment in config.py for the sweep
    # that pinned this down): landing on h/J=1 only has to be adiabatic
    # approaching the gapless point, while a target past it (h/J=0.5) has to
    # stay adiabatic approaching AND leaving the gapless point within the
    # same ramp. (Confirmed empirically: h/J=0.5's <X> was 21.6% off ED on
    # the real qnexus run at the rate_ref-only step count, and still 9.00%
    # at h/J=1's own 200-step treatment, dropping to 0.43% at 400 steps;
    # h/J=2.0, whose ramp from H_INIT never crosses 1.0, was under 0.5% at
    # the plain rate_ref-based count throughout.)
    def _ramp_steps(h):
        if h == config.J:
            return round(config.H2_ADIABATIC_MAX_STEPS * config.H2_ADIABATIC_CRITICAL_TIME_FACTOR)
        if min(config.H_INIT, h) <= config.J <= max(config.H_INIT, h):
            return round(config.H2_ADIABATIC_MAX_STEPS * config.H2_ADIABATIC_TRANSIT_TIME_FACTOR)
        return min(
            steps_for_target(h, config.H2_ADIABATIC_DT, config.H2_ADIABATIC_RATE_REF, h_init=config.H_INIT),
            config.H2_ADIABATIC_MAX_STEPS,
        )

    ramp_steps_by_target = [_ramp_steps(h) for h in h_values]
    print(f"\nSubmitting N={config.H2_ADIABATIC_N}, h_init={config.H_INIT} -> "
          f"h_target/J in {h_values} (batched), dt={dt_by_target}, "
          f"ramp_steps={ramp_steps_by_target} (rate_ref={config.H2_ADIABATIC_RATE_REF}, "
          f"capped at {config.H2_ADIABATIC_MAX_STEPS}), shots={config.H2_ADIABATIC_SHOTS} "
          f"to {device_name} ...")

    raw_by_h = submit_adiabatic_batch(
        config.H2_ADIABATIC_N, h_values, config.J, ramp_steps_by_target,
        dt_by_target, config.H2_ADIABATIC_SHOTS, config.H_INIT,
        device_name=device_name, project_name=config.H2_PROJECT_NAME,
    )

    save_stage_results(raw_stage, raw_by_h)

    results = {}
    for h in h_values:
        job_result = raw_by_h[h]
        z_rms, mzz = bitstrings_to_observables(job_result["bitstrings"], config.H2_ADIABATIC_N)
        z_se, mzz_se = bootstrap_observable_errors(job_result["bitstrings"], config.H2_ADIABATIC_N)
        # <X> is a signed mean, not an RMS -- see the comment in run()
        # above and bitstrings_to_mx's docstring.
        x_mean = bitstrings_to_mx(job_result["bitstrings_x"], config.H2_ADIABATIC_N)
        x_se = bootstrap_mx_error(job_result["bitstrings_x"], config.H2_ADIABATIC_N)

        ed_z = next(r['mz_rms'] for r in ed_results if r['h'] == h)
        ed_x = next(r['mx'] for r in ed_results if r['h'] == h)
        ed_mzz = next(r['mzz'] for r in ed_results if r['h'] == h)
        pct_z = abs(z_rms - ed_z) / ed_z * 100 if ed_z != 0 else 0
        pct_x = abs(x_mean - ed_x) / abs(ed_x) * 100 if ed_x != 0 else 0
        pct_mzz = abs(mzz - ed_mzz) / abs(ed_mzz) * 100 if ed_mzz != 0 else 0
        print(f"  h/J={h:.2f}: H2 <Z>       = {z_rms:.4f}  (ED = {ed_z:.4f}, {pct_z:.2f}% diff)")
        print(f"           H2 <X>       = {x_mean:.4f}  (ED = {ed_x:.4f}, {pct_x:.2f}% diff)")
        print(f"           H2 <Zi Zi+1> = {mzz:.4f}  (ED = {ed_mzz:.4f}, {pct_mzz:.2f}% diff)")

        results[h] = {
            'z_h2': z_rms, 'z_err': z_se,
            'x_h2': x_mean, 'x_err': x_se,
            'mzz_h2': mzz, 'mzz_err': mzz_se,
        }

    save_stage_results(results_stage, results)

    plot_h2_phase_transition(
        config.H2_H_VALUES, results, ed_results, save_dir=config.PLOT_SAVE_DIR,
        filename=f"h2_phase_transition{stage_suffix}.png",
    )

    return results


def run_vqe(local=False, noisy=False):
    """VQE ground-state search on H2: config.H2_VQE_ANSATZ (default "hva",
    the Hamiltonian Variational Ansatz) optimized via gradient-free COBYLA
    (see vqe.run_vqe_h2), one independent optimization per h/J target --
    finds the ground state directly rather than following a time-dependent
    path, so it isn't subject to run_phase_transition()'s diabatic-
    transition problem near h/J=1.

    Not called from run() or from ftim_main.py -- invoke explicitly (e.g.
    `python run_h2_emulator.py --vqe`) since it's a separate, additional
    cost on top of run() when local=False. Pass local=True (or --local
    --vqe on the command line) to run against local_emulator_backend
    instead -- free and instant, same computation. Gated by
    config.RUN_ON_H2_EMULATOR only when local=False.
    """
    device_name = config.H2_DEVICE_NAME_NOISY if (noisy and not local) else config.H2_DEVICE_NAME
    print("=" * 60)
    print(f"H2 VQE GROUND-STATE SEARCH ({device_name}, "
          f"{'local pytket-quantinuum/pecos' if local else 'qnexus/pytket'})")
    print("=" * 60)

    if noisy and local:
        print("Note: noisy=True has no effect when local=True -- see run()'s "
              "module-level note. Running local H2-1LE as usual.")

    if not local and not config.RUN_ON_H2_EMULATOR:
        print("Skipped (config.RUN_ON_H2_EMULATOR = False). Enable it to submit "
              "jobs to qnexus -- this consumes a metered usage quota. "
              "(Or pass local=True / --local to run for free instead.)")
        return None

    from vqe import run_vqe_h2
    if local:
        from local_emulator_backend import submit_vqe_batch_job
    else:
        from qnexus_backend import submit_vqe_batch_job
    stage_suffix = "_local" if local else ("_noisy" if noisy else "")
    raw_stage = f"h2_vqe_raw{stage_suffix}"
    results_stage = f"h2_vqe{stage_suffix}"

    shots = config.H2_VQE_SHOTS_LOCAL if local else config.H2_VQE_SHOTS
    max_iters = config.H2_VQE_MAX_ITERS_LOCAL if local else config.H2_VQE_MAX_ITERS

    ed_results = ed_baseline(config.H2_VQE_N, config.H2_H_VALUES, J=config.J)

    results = {}
    for h in config.H2_H_VALUES:
        print(f"\nRunning VQE: N={config.H2_VQE_N}, h/J={h:.2f}, "
              f"max_iters={max_iters}, shots={shots} "
              f"on {device_name} ...")

        vqe_result = run_vqe_h2(
            config.H2_VQE_N, h, config.J, shots,
            max_iters, config.H2_VQE_TOL, config.H2_VQE_SEED,
            device_name=device_name, project_name=config.H2_PROJECT_NAME,
            submit_fn=submit_vqe_batch_job, raw_stage=raw_stage,
            ansatz=config.H2_VQE_ANSATZ, p=config.H2_VQE_P,
        )

        ed_energy = next(r['energy'] for r in ed_results if r['h'] == h) * config.H2_VQE_N
        ed_z = next(r['mz_rms'] for r in ed_results if r['h'] == h)
        ed_mzz = next(r['mzz'] for r in ed_results if r['h'] == h)
        pct_energy = abs(vqe_result['final_energy'] - ed_energy) / abs(ed_energy) * 100 if ed_energy != 0 else 0
        pct_z = abs(vqe_result['final_z_rms'] - ed_z) / ed_z * 100 if ed_z != 0 else 0
        pct_mzz = abs(vqe_result['final_mzz'] - ed_mzz) / abs(ed_mzz) * 100 if ed_mzz != 0 else 0
        print(f"  VQE energy   = {vqe_result['final_energy']:.4f}  (ED = {ed_energy:.4f}, {pct_energy:.2f}% diff)")
        print(f"  VQE <Z>      = {vqe_result['final_z_rms']:.4f}  (ED = {ed_z:.4f}, {pct_z:.2f}% diff)")
        print(f"  VQE <Zi Zi+1> = {vqe_result['final_mzz']:.4f}  (ED = {ed_mzz:.4f}, {pct_mzz:.2f}% diff)")

        results[h] = vqe_result

    save_stage_results(results_stage, results)

    plot_vqe_convergence(
        config.H2_H_VALUES, results, ed_results, save_dir=config.PLOT_SAVE_DIR,
        filename=f"vqe_convergence{stage_suffix}.png",
    )

    phase_transition_data = {
        h: {
            'z_h2': r['final_z_rms'], 'z_err': r['final_z_err'],
            'mzz_h2': r['final_mzz'], 'mzz_err': r['final_mzz_err'],
        }
        for h, r in results.items()
    }
    plot_h2_phase_transition(
        config.H2_H_VALUES, phase_transition_data, ed_results, save_dir=config.PLOT_SAVE_DIR,
        method_label="VQE",
        filename=f"h2_phase_transition_vqe{stage_suffix}.png",
    )

    return results



def run_vqe_hybrid(noisy = False):
    """Hybrid VQE: converge locally for free, the confirm with one real submission to Nexus at the best parameters found."""

    if not config.RUN_ON_H2_EMULATOR:
        print("Skipped confirmation step (config.RUN_ON_H2_EMULATOR = False). Enable it first.")
        return None 

    device_name = config.H2_DEVICE_NAME_NOISY if noisy else config.H2_DEVICE_NAME
    print("="*60)
    print(f"H2 VQE HYBRID (local optimization -> {device_name} confirmation)")
    print("=" * 60)

    ed_results = ed_baseline(config.H2_VQE_N, config.H2_H_VALUES, J = config.J)

    results = {}
    for h in config.H2_H_VALUES:
        print(f"\n[1/2] Converging locally: N={config.H2_VQE_N}, h/J={h:.2f}, "
              f"max_iters={config.H2_VQE_MAX_ITERS_LOCAL} (free, local)...")
        local_result = run_vqe_h2(
            config.H2_VQE_N, h, config.J, config.H2_VQE_SHOTS_LOCAL,
            config.H2_VQE_MAX_ITERS_LOCAL, config.H2_VQE_TOL, config.H2_VQE_SEED,
            submit_fn=local_submit, raw_stage="h2_vqe_raw_local",
            ansatz=config.H2_VQE_ANSATZ, p=config.H2_VQE_P,
            )

        print(f"      Local best energy = {local_result['final_energy']:.4f}")

        print(f"[2/2] Confirming on {device_name} (real, ONE submission)...")
        confirm_result = run_vqe_h2(
            config.H2_VQE_N, h, config.J, config.H2_VQE_SHOTS,
            1, config.H2_VQE_TOL, config.H2_VQE_SEED,
            device_name=device_name, project_name=config.H2_PROJECT_NAME,
            submit_fn=nexus_submit, raw_stage="h2_vqe_raw_hybrid",
            ansatz=config.H2_VQE_ANSATZ, p=config.H2_VQE_P,
            fixed_params=local_result['final_params'],
        )

        ed_energy = next(r['energy'] for r in ed_results if r['h'] == h)* config.H2_VQE_N
        pct_energy = abs(confirm_result['final_energy'] - ed_energy) / abs(ed_energy) * 100 if ed_energy != 0 else 0
        print(f"      Confirmed energy = {confirm_result['final_energy']:.4f}  "
              f"(ED = {ed_energy:.4f}, {pct_energy:.2f}% diff)")

        results[h] = confirm_result

    save_stage_results("h2_vqe_hybrid", results)
    return results





def load_last_run(local=False, noisy=False):
    """Convenience accessor for the most recent saved H2 results (raw shots
    and processed observables), without spending any quota. Pass
    local=True to read back local_emulator_backend runs instead of qnexus
    ones, or noisy=True to read back H2-Emulator (noisy qnexus) runs instead
    of H2-1LE ones -- each combination is saved under its own suffixed
    stage name. noisy is ignored when local=True (see run()'s module-level
    note: there is no local-noisy combination)."""
    suffix = "_local" if local else ("_noisy" if noisy else "")
    return {
        "raw": load_latest(f"h2_emulator_raw{suffix}"),
        "processed": load_latest(f"h2_emulator{suffix}"),
        "adiabatic_raw": load_latest(f"h2_adiabatic_raw{suffix}"),
        "adiabatic": load_latest(f"h2_adiabatic{suffix}"),
        "vqe_raw": load_latest(f"h2_vqe_raw{suffix}"),
        "vqe": load_latest(f"h2_vqe{suffix}"),
    }


if __name__ == "__main__":
    import sys
    is_local = "--local" in sys.argv
    is_noisy = "--noisy" in sys.argv
    if "--phase-transition" in sys.argv:
        run_phase_transition(local=is_local, noisy=is_noisy)
    elif "--vqe" in sys.argv and "--hybrid" in sys.argv:
        run_vqe_hybrid(noisy=is_noisy)
    elif "--vqe" in sys.argv:
        run_vqe(local=is_local, noisy=is_noisy)
    else:
        run(local=is_local, noisy=is_noisy)
