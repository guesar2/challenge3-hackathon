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
from plotting import plot_h2_vs_ed


def run():
    print("=" * 60)
    print(f"STAGE 4/4: QUANTINUUM {config.H2_DEVICE_NAME} EMULATOR (qnexus/pytket)")
    print("=" * 60)

    if not config.RUN_ON_H2_EMULATOR:
        print("Skipped (config.RUN_ON_H2_EMULATOR = False). Enable it to submit "
              "jobs to qnexus -- this consumes a metered usage quota.")
        return None

    from qnexus_backend import submit_quench_job, bitstrings_to_observables, assemble_h2_vs_ed

    raw_by_h = {}
    results = {}
    for h in config.H2_H_VALUES:
        print(f"\nSubmitting N={config.H2_N}, h/J={h:.2f}, dt={config.H2_DT}, "
              f"steps={config.H2_STEPS}, shots={config.H2_SHOTS} "
              f"to {config.H2_DEVICE_NAME} ...")

        job_result = submit_quench_job(
            config.H2_N, h, config.J, config.H2_DT, config.H2_STEPS,
            config.H2_SHOTS, device_name=config.H2_DEVICE_NAME,
            project_name=config.H2_PROJECT_NAME,
        )

        # Persist the raw hardware result immediately -- before any
        # postprocessing -- so it survives even if bitstrings_to_observables
        # (or anything after it) turns out to be buggy.
        raw_by_h[h] = job_result
        save_stage_results("h2_emulator_raw", raw_by_h)

        z_rms_h2, mzz_h2 = bitstrings_to_observables(job_result["bitstrings"], config.H2_N)

        _, z_ed, mzz_ed, _ = ed_time_evolution_exact(
            config.H2_N, h, config.J, config.H2_DT, config.H2_STEPS
        )
        pct_z = abs(z_rms_h2 - z_ed[-1]) / z_ed[-1] * 100 if z_ed[-1] != 0 else 0
        pct_mzz = abs(mzz_h2 - mzz_ed[-1]) / abs(mzz_ed[-1]) * 100 if mzz_ed[-1] != 0 else 0
        print(f"  H2 <Z>_rms    = {z_rms_h2:.4f}  (ED = {z_ed[-1]:.4f}, {pct_z:.2f}% diff)")
        print(f"  H2 <Zi Zi+1>  = {mzz_h2:.4f}  (ED = {mzz_ed[-1]:.4f}, {pct_mzz:.2f}% diff)")

        results[h] = {
            'circuit_ref_id': job_result['circuit_ref_id'],
            'job_name': job_result['job_name'],
            'z_rms_h2': z_rms_h2, 'mzz_h2': mzz_h2,
            'z_ed': z_ed[-1], 'mzz_ed': mzz_ed[-1],
            'pct_diff_z': pct_z, 'pct_diff_mzz': pct_mzz,
        }

    save_stage_results("h2_emulator", results)

    plot_data = assemble_h2_vs_ed(results, raw_by_h)
    plot_h2_vs_ed(
        plot_data["h_values"], plot_data["z_h2"], plot_data["z_err"],
        plot_data["mzz_h2"], plot_data["mzz_err"], plot_data["z_ed"], plot_data["mzz_ed"],
        save_dir=config.PLOT_SAVE_DIR,
    )

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
