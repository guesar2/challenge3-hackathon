"""
run_h2_emulator.py

Capstone section 4/4: TFIM Trotter quench circuit run against H2-1LE, with
the circuit built in pytket (see tket_circuit.py). Two interchangeable
backends, selected via each function's `local` argument (also `--local` on
the command line):

- qnexus_backend (default, local=False): submits over the network through
  qnexus. Requires a live qnexus login and costs against a metered usage
  quota -- config.RUN_ON_H2_EMULATOR gates every call in this mode. This is
  the one section main.py / ftim_main.py will still print-and-skip unless
  that flag is flipped; run standalone with `python run_h2_emulator.py`
  once you're ready to spend quota.
- local_emulator_backend (local=True): runs the same circuits locally via
  pytket-quantinuum + pytket-pecos (`pip install "pytket-quantinuum[pecos]"`).
  H2-1LE is exact noiseless state-vector emulation (only shot noise, no
  physical noise model -- see qnexus_backend.py's docstring for the source
  quote), so this is the *same* computation as the qnexus path, just free
  and instant instead of network-queued and quota-metered. Not gated by
  RUN_ON_H2_EMULATOR since there's no quota to protect. Results are saved
  under a "_local" stage suffix so they never overwrite a real qnexus run's
  saved data.

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
from shot_observables import bitstrings_to_observables, bootstrap_observable_errors


def run(local=False):
    print("=" * 60)
    print(f"STAGE 4/4: QUANTINUUM {config.H2_DEVICE_NAME} EMULATOR "
          f"({'local pytket-quantinuum/pecos' if local else 'qnexus/pytket'})")
    print("=" * 60)

    if not local and not config.RUN_ON_H2_EMULATOR:
        print("Skipped (config.RUN_ON_H2_EMULATOR = False). Enable it to submit "
              "jobs to qnexus -- this consumes a metered usage quota. "
              "(Or pass local=True / --local to run for free instead.)")
        return None

    if local:
        from local_emulator_backend import submit_quench_batch
    else:
        from qnexus_backend import submit_quench_batch
    raw_stage = "h2_emulator_raw_local" if local else "h2_emulator_raw"
    results_stage = "h2_emulator_local" if local else "h2_emulator"

    # qnexus/H2 only returns final measurement shots -- no mid-circuit
    # statevector readout -- so a <Z>/<Zi Zi+1> vs. time curve needs one
    # circuit per (h, step_count) snapshot. submit_quench_batch submits the
    # whole step_count curve for one h as a single compile/execute call
    # (rather than one queue-wait per step) since Quantinuum's own
    # hardware-only batching feature doesn't apply to the H2-1LE emulator.
    raw_by_h = {}
    results = {}
    step_counts = list(range(1, config.H2_STEPS + 1))
    for h in config.H2_H_VALUES:
        print(f"\nSubmitting N={config.H2_N}, h/J={h:.2f}, dt={config.H2_DT}, "
              f"steps=1..{config.H2_STEPS} (batched), shots={config.H2_SHOTS} "
              f"to {config.H2_DEVICE_NAME} ...")

        batch_results = submit_quench_batch(
            config.H2_N, h, config.J, config.H2_DT, step_counts,
            config.H2_SHOTS, device_name=config.H2_DEVICE_NAME,
            project_name=config.H2_PROJECT_NAME,
        )

        # Persist the raw hardware result immediately -- before any
        # postprocessing -- so it survives even if bitstrings_to_observables
        # (or anything after it) turns out to be buggy.
        raw_by_h[h] = batch_results
        save_stage_results(raw_stage, raw_by_h)

        times, z_h2, z_err, mzz_h2, mzz_err = [], [], [], [], []
        for step_count in step_counts:
            job_result = batch_results[step_count]
            z_rms, mzz = bitstrings_to_observables(job_result["bitstrings"], config.H2_N)
            z_se, mzz_se = bootstrap_observable_errors(job_result["bitstrings"], config.H2_N)

            times.append(step_count * config.H2_DT)
            z_h2.append(z_rms)
            z_err.append(z_se)
            mzz_h2.append(mzz)
            mzz_err.append(mzz_se)

        _, z_ed, mzz_ed, _ = ed_time_evolution_exact(
            config.H2_N, h, config.J, config.H2_DT, config.H2_STEPS
        )

        max_pct_z = max(abs(a - b) / abs(b) * 100 if b != 0 else 0
                         for a, b in zip(z_h2, z_ed))
        max_pct_mzz = max(abs(a - b) / abs(b) * 100 if b != 0 else 0
                           for a, b in zip(mzz_h2, mzz_ed))
        print(f"\n  h/J={h:.2f}: max deviation <Z> = {max_pct_z:.2f}%, "
              f"max deviation <Zi Zi+1> = {max_pct_mzz:.2f}%")

        results[h] = {
            'times': times, 'z_h2': z_h2, 'z_err': z_err,
            'mzz_h2': mzz_h2, 'mzz_err': mzz_err,
            'z_ed': list(z_ed), 'mzz_ed': list(mzz_ed),
        }

    save_stage_results(results_stage, results)

    plot_h2_vs_ed_time(
        config.H2_H_VALUES, results, save_dir=config.PLOT_SAVE_DIR,
        filename="h2_vs_ed_time_local.png" if local else "h2_vs_ed_time.png",
    )

    return results


def run_phase_transition(local=False):
    """Adiabatic-ramp sweep on H2: the phase-transition signal (<Z>/<Zi
    Zi+1> vs. h/J across the h/J=1 critical point) reproduced on hardware,
    rather than the fixed-h quench-vs-time protocol in run().

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

    The critical point h/J=1 is pinned to H2_ADIABATIC_MAX_STEPS regardless
    of |h_target - h_init|: |Delta h|-based scaling assumes ramp difficulty
    depends only on distance from h_init, but h/J=1 is where the TFIM gap
    closes, so critical slowing down demands a long ramp independent of
    that distance. Testing bore this out -- scaling h/J=1 down to 86 steps
    (its |Delta h|-based value) made its <Zi Zi+1> deviation *worse* than
    the flat-100-steps baseline, the only target of the three where scaling
    down hurt rather than helped.

    h/J=1 also uses H2_ADIABATIC_CRITICAL_TIME_FACTOR * H2_ADIABATIC_MAX_STEPS
    steps at the *same* dt as every other target -- i.e. a longer total
    ramp time, not finer Trotter resolution. A finer-dt-at-fixed-time
    variant was tried first (since 2000 shots showed h/J=1's bias was
    statistically real, not noise) but barely moved the deviation; a
    local_emulator_backend test isolating time vs. resolution (free to run
    at high shot counts) showed doubling the total ramp time at the
    original dt fixed it (~6.5% -> ~2.15%), while doubling resolution at
    fixed time did not -- see config.py's H2_ADIABATIC_CRITICAL_TIME_FACTOR
    comment for the numbers. Textbook critical slowing down: the adiabatic
    theorem needs more *time* as the gap closes, not finer time-resolution.
    """
    print("=" * 60)
    print(f"H2 ADIABATIC SWEEP (phase-transition signal, {config.H2_DEVICE_NAME}, "
          f"{'local pytket-quantinuum/pecos' if local else 'qnexus/pytket'})")
    print("=" * 60)

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
    raw_stage = "h2_adiabatic_raw_local" if local else "h2_adiabatic_raw"
    results_stage = "h2_adiabatic_local" if local else "h2_adiabatic"

    ed_results = ed_baseline(config.H2_ADIABATIC_N, config.H2_H_VALUES, J=config.J)

    h_values = list(config.H2_H_VALUES)
    dt_by_target = [config.H2_ADIABATIC_DT for _ in h_values]  # same dt for every target
    ramp_steps_by_target = [
        # h/J=1: longer total ramp time (more steps at the same dt), not
        # finer resolution -- see docstring above.
        round(config.H2_ADIABATIC_MAX_STEPS * config.H2_ADIABATIC_CRITICAL_TIME_FACTOR) if h == config.J
        else min(
            steps_for_target(h, config.H2_ADIABATIC_DT, config.H2_ADIABATIC_RATE_REF, h_init=config.H_INIT),
            config.H2_ADIABATIC_MAX_STEPS,
        )
        for h in h_values
    ]
    print(f"\nSubmitting N={config.H2_ADIABATIC_N}, h_init={config.H_INIT} -> "
          f"h_target/J in {h_values} (batched), dt={dt_by_target}, "
          f"ramp_steps={ramp_steps_by_target} (rate_ref={config.H2_ADIABATIC_RATE_REF}, "
          f"capped at {config.H2_ADIABATIC_MAX_STEPS}), shots={config.H2_ADIABATIC_SHOTS} "
          f"to {config.H2_DEVICE_NAME} ...")

    # One compile/execute call for the whole h/J sweep instead of one per h
    # -- see submit_adiabatic_batch's docstring for why (Quantinuum's own
    # batching feature is hardware-only, doesn't apply to H2-1LE).
    raw_by_h = submit_adiabatic_batch(
        config.H2_ADIABATIC_N, h_values, config.J, ramp_steps_by_target,
        dt_by_target, config.H2_ADIABATIC_SHOTS, config.H_INIT,
        device_name=config.H2_DEVICE_NAME, project_name=config.H2_PROJECT_NAME,
    )

    # Persist the raw hardware result immediately -- before any
    # postprocessing -- so it survives even if bitstrings_to_observables
    # (or anything after it) turns out to be buggy.
    save_stage_results(raw_stage, raw_by_h)

    results = {}
    for h in h_values:
        job_result = raw_by_h[h]
        z_rms, mzz = bitstrings_to_observables(job_result["bitstrings"], config.H2_ADIABATIC_N)
        z_se, mzz_se = bootstrap_observable_errors(job_result["bitstrings"], config.H2_ADIABATIC_N)

        ed_z = next(r['mz_rms'] for r in ed_results if r['h'] == h)
        ed_mzz = next(r['mzz'] for r in ed_results if r['h'] == h)
        pct_z = abs(z_rms - ed_z) / ed_z * 100 if ed_z != 0 else 0
        pct_mzz = abs(mzz - ed_mzz) / abs(ed_mzz) * 100 if ed_mzz != 0 else 0
        print(f"  h/J={h:.2f}: H2 <Z>       = {z_rms:.4f}  (ED = {ed_z:.4f}, {pct_z:.2f}% diff)")
        print(f"           H2 <Zi Zi+1> = {mzz:.4f}  (ED = {ed_mzz:.4f}, {pct_mzz:.2f}% diff)")

        results[h] = {
            'z_h2': z_rms, 'z_err': z_se,
            'mzz_h2': mzz, 'mzz_err': mzz_se,
        }

    save_stage_results(results_stage, results)

    plot_h2_phase_transition(
        config.H2_H_VALUES, results, ed_results, save_dir=config.PLOT_SAVE_DIR,
        filename="h2_phase_transition_local.png" if local else "h2_phase_transition.png",
    )

    return results


def run_vqe(local=False):
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
    print("=" * 60)
    print(f"H2 VQE GROUND-STATE SEARCH ({config.H2_DEVICE_NAME}, "
          f"{'local pytket-quantinuum/pecos' if local else 'qnexus/pytket'})")
    print("=" * 60)

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
    raw_stage = "h2_vqe_raw_local" if local else "h2_vqe_raw"
    results_stage = "h2_vqe_local" if local else "h2_vqe"

    # Local runs cost no quota, so they can afford the much larger shots/iters
    # budget COBYLA actually needs to converge a 36-parameter HEA (see
    # config.py's H2_VQE_SHOTS_LOCAL/H2_VQE_MAX_ITERS_LOCAL comment) --
    # qnexus keeps the small, quota-safe defaults.
    shots = config.H2_VQE_SHOTS_LOCAL if local else config.H2_VQE_SHOTS
    max_iters = config.H2_VQE_MAX_ITERS_LOCAL if local else config.H2_VQE_MAX_ITERS

    ed_results = ed_baseline(config.H2_VQE_N, config.H2_H_VALUES, J=config.J)

    results = {}
    for h in config.H2_H_VALUES:
        print(f"\nRunning VQE: N={config.H2_VQE_N}, h/J={h:.2f}, "
              f"max_iters={max_iters}, shots={shots} "
              f"on {config.H2_DEVICE_NAME} ...")

        vqe_result = run_vqe_h2(
            config.H2_VQE_N, h, config.J, shots,
            max_iters, config.H2_VQE_TOL, config.H2_VQE_SEED,
            device_name=config.H2_DEVICE_NAME, project_name=config.H2_PROJECT_NAME,
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
        filename="vqe_convergence_local.png" if local else "vqe_convergence.png",
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
        filename="h2_phase_transition_vqe_local.png" if local else "h2_phase_transition_vqe.png",
    )

    return results


def load_last_run(local=False):
    """Convenience accessor for the most recent saved H2 results (raw shots
    and processed observables), without spending any quota. Pass
    local=True to read back local_emulator_backend runs instead of qnexus
    ones (they're saved under separate "_local"-suffixed stage names)."""
    suffix = "_local" if local else ""
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
    if "--phase-transition" in sys.argv:
        run_phase_transition(local=is_local)
    elif "--vqe" in sys.argv:
        run_vqe(local=is_local)
    else:
        run(local=is_local)
