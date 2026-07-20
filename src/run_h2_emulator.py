"""
run_h2_emulator.py

Capstone section 4/4: TFIM Trotter quench circuit run on a Quantinuum-hosted
backend (default: the H2-1LE noiseless-leakage emulator) via qnexus, with
the circuit built in pytket (see tket_circuit.py).

Off by default -- config.RUN_ON_H2_EMULATOR gates every call in this module,
since it requires a live qnexus login and costs against a metered usage
quota. This is the one section main.py / ftim_main.py will still print-and-
skip unless the flag is flipped; run standalone with
`python run_h2_emulator.py` once you're ready to spend quota.

Because every submission here costs quota, raw shot data is persisted
(data/h2_emulator_raw_*.json) immediately after each job returns -- before
any postprocessing -- so a bug in bitstrings_to_observables (or anything
downstream) never means re-submitting to recover results that already came
back from the hardware.
"""
import config
from exact_diagonalization import ed_baseline, ed_time_evolution_exact
from persistence import save_stage_results, load_latest
from plotting import plot_h2_vs_ed_time, plot_h2_phase_transition, plot_vqe_convergence


def run():
    print("=" * 60)
    print(f"STAGE 4/4: QUANTINUUM {config.H2_DEVICE_NAME} EMULATOR (qnexus/pytket)")
    print("=" * 60)

    if not config.RUN_ON_H2_EMULATOR:
        print("Skipped (config.RUN_ON_H2_EMULATOR = False). Enable it to submit "
              "jobs to qnexus -- this consumes a metered usage quota.")
        return None

    from qnexus_backend import submit_quench_job, bitstrings_to_observables, bootstrap_observable_errors

    # qnexus/H2 only returns final measurement shots -- no mid-circuit
    # statevector readout -- so a <Z>/<Zi Zi+1> vs. time curve needs one job
    # per (h, step_count) snapshot rather than one job per h.
    raw_by_h = {}
    results = {}
    for h in config.H2_H_VALUES:
        raw_by_h[h] = {}
        times, z_h2, z_err, mzz_h2, mzz_err = [], [], [], [], []

        for step_count in range(1, config.H2_STEPS + 1):
            print(f"\nSubmitting N={config.H2_N}, h/J={h:.2f}, dt={config.H2_DT}, "
                  f"steps={step_count}/{config.H2_STEPS}, shots={config.H2_SHOTS} "
                  f"to {config.H2_DEVICE_NAME} ...")

            job_result = submit_quench_job(
                config.H2_N, h, config.J, config.H2_DT, step_count,
                config.H2_SHOTS, device_name=config.H2_DEVICE_NAME,
                project_name=config.H2_PROJECT_NAME,
            )

            # Persist the raw hardware result immediately -- before any
            # postprocessing -- so it survives even if bitstrings_to_observables
            # (or anything after it) turns out to be buggy.
            raw_by_h[h][step_count] = job_result
            save_stage_results("h2_emulator_raw", raw_by_h)

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

    save_stage_results("h2_emulator", results)

    plot_h2_vs_ed_time(config.H2_H_VALUES, results, save_dir=config.PLOT_SAVE_DIR)

    return results


def run_phase_transition():
    """Adiabatic-ramp sweep on H2: the phase-transition signal (<Z>/<Zi
    Zi+1> vs. h/J across the h/J=1 critical point) reproduced on hardware,
    rather than the fixed-h quench-vs-time protocol in run().

    Not called from run() or from ftim_main.py -- invoke explicitly (e.g.
    `python -c "import run_h2_emulator; run_h2_emulator.run_phase_transition()"`)
    since it's a separate, additional quota cost on top of run(). Still
    gated by config.RUN_ON_H2_EMULATOR.

    Uses a fixed ramp length (config.H2_ADIABATIC_STEPS) for every h
    target rather than sweep_schedule.steps_for_target's rate-based
    scaling, which would need thousands of steps per job -- infeasible to
    submit to qnexus.
    """
    print("=" * 60)
    print(f"H2 ADIABATIC SWEEP (phase-transition signal, {config.H2_DEVICE_NAME})")
    print("=" * 60)

    if not config.RUN_ON_H2_EMULATOR:
        print("Skipped (config.RUN_ON_H2_EMULATOR = False). Enable it to submit "
              "jobs to qnexus -- this consumes a metered usage quota.")
        return None

    from qnexus_backend import submit_adiabatic_job, bitstrings_to_observables, bootstrap_observable_errors

    ed_results = ed_baseline(config.H2_ADIABATIC_N, config.H2_H_VALUES, J=config.J)

    raw_by_h = {}
    results = {}
    for h in config.H2_H_VALUES:
        print(f"\nSubmitting N={config.H2_ADIABATIC_N}, h_init={config.H_INIT} -> "
              f"h_target/J={h:.2f}, dt={config.H2_ADIABATIC_DT}, "
              f"ramp_steps={config.H2_ADIABATIC_STEPS}, shots={config.H2_ADIABATIC_SHOTS} "
              f"to {config.H2_DEVICE_NAME} ...")

        job_result = submit_adiabatic_job(
            config.H2_ADIABATIC_N, h, config.J, config.H2_ADIABATIC_STEPS,
            config.H2_ADIABATIC_DT, config.H2_ADIABATIC_SHOTS, config.H_INIT,
            device_name=config.H2_DEVICE_NAME, project_name=config.H2_PROJECT_NAME,
        )

        # Persist the raw hardware result immediately -- before any
        # postprocessing -- so it survives even if bitstrings_to_observables
        # (or anything after it) turns out to be buggy.
        raw_by_h[h] = job_result
        save_stage_results("h2_adiabatic_raw", raw_by_h)

        z_rms, mzz = bitstrings_to_observables(job_result["bitstrings"], config.H2_ADIABATIC_N)
        z_se, mzz_se = bootstrap_observable_errors(job_result["bitstrings"], config.H2_ADIABATIC_N)

        ed_z = next(r['mz_rms'] for r in ed_results if r['h'] == h)
        ed_mzz = next(r['mzz'] for r in ed_results if r['h'] == h)
        pct_z = abs(z_rms - ed_z) / ed_z * 100 if ed_z != 0 else 0
        pct_mzz = abs(mzz - ed_mzz) / abs(ed_mzz) * 100 if ed_mzz != 0 else 0
        print(f"  H2 <Z>       = {z_rms:.4f}  (ED = {ed_z:.4f}, {pct_z:.2f}% diff)")
        print(f"  H2 <Zi Zi+1> = {mzz:.4f}  (ED = {ed_mzz:.4f}, {pct_mzz:.2f}% diff)")

        results[h] = {
            'z_h2': z_rms, 'z_err': z_se,
            'mzz_h2': mzz, 'mzz_err': mzz_se,
        }

    save_stage_results("h2_adiabatic", results)

    plot_h2_phase_transition(config.H2_H_VALUES, results, ed_results, save_dir=config.PLOT_SAVE_DIR)

    return results


def run_vqe():
    """VQE ground-state search on H2: a hardware-efficient ansatz optimized
    via gradient-free COBYLA (see vqe.run_vqe_h2), one independent
    optimization per h/J target -- finds the ground state directly rather
    than following a time-dependent path, so it isn't subject to
    run_phase_transition()'s diabatic-transition problem near h/J=1.

    Not called from run() or from ftim_main.py -- invoke explicitly (e.g.
    `python run_h2_emulator.py --vqe`) since it's a separate, additional
    cost on top of run(). Still gated by config.RUN_ON_H2_EMULATOR.
    """
    print("=" * 60)
    print(f"H2 VQE GROUND-STATE SEARCH ({config.H2_DEVICE_NAME})")
    print("=" * 60)

    if not config.RUN_ON_H2_EMULATOR:
        print("Skipped (config.RUN_ON_H2_EMULATOR = False). Enable it to submit "
              "jobs to qnexus -- this consumes a metered usage quota.")
        return None

    from vqe import run_vqe_h2

    ed_results = ed_baseline(config.H2_VQE_N, config.H2_H_VALUES, J=config.J)

    results = {}
    for h in config.H2_H_VALUES:
        print(f"\nRunning VQE: N={config.H2_VQE_N}, h/J={h:.2f}, "
              f"max_iters={config.H2_VQE_MAX_ITERS}, shots={config.H2_VQE_SHOTS} "
              f"on {config.H2_DEVICE_NAME} ...")

        vqe_result = run_vqe_h2(
            config.H2_VQE_N, h, config.J, config.H2_VQE_SHOTS,
            config.H2_VQE_MAX_ITERS, config.H2_VQE_TOL, config.H2_VQE_SEED,
            device_name=config.H2_DEVICE_NAME, project_name=config.H2_PROJECT_NAME,
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

    save_stage_results("h2_vqe", results)

    plot_vqe_convergence(config.H2_H_VALUES, results, ed_results, save_dir=config.PLOT_SAVE_DIR)

    phase_transition_data = {
        h: {
            'z_h2': r['final_z_rms'], 'z_err': r['final_z_err'],
            'mzz_h2': r['final_mzz'], 'mzz_err': r['final_mzz_err'],
        }
        for h, r in results.items()
    }
    plot_h2_phase_transition(
        config.H2_H_VALUES, phase_transition_data, ed_results, save_dir=config.PLOT_SAVE_DIR,
        method_label="VQE", filename="h2_phase_transition_vqe.png",
    )

    return results


def load_last_run():
    """Convenience accessor for the most recent saved H2 results (raw shots
    and processed observables), without spending any quota."""
    return {
        "raw": load_latest("h2_emulator_raw"),
        "processed": load_latest("h2_emulator"),
        "adiabatic_raw": load_latest("h2_adiabatic_raw"),
        "adiabatic": load_latest("h2_adiabatic"),
        "vqe_raw": load_latest("h2_vqe_raw"),
        "vqe": load_latest("h2_vqe"),
    }


if __name__ == "__main__":
    import sys
    if "--phase-transition" in sys.argv:
        run_phase_transition()
    elif "--vqe" in sys.argv:
        run_vqe()
    else:
        run()
