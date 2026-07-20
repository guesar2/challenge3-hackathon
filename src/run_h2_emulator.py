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
from exact_diagonalization import ed_time_evolution_exact
from persistence import save_stage_results, load_latest
from plotting import plot_h2_vs_ed_time


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


def load_last_run():
    """Convenience accessor for the most recent saved H2 results (raw shots
    and processed observables), without spending any quota."""
    return {
        "raw": load_latest("h2_emulator_raw"),
        "processed": load_latest("h2_emulator"),
    }


if __name__ == "__main__":
    run()
