"""
run_iceberg_sweep.py

Multi-point counterpart to run_iceberg_qec.py's run_iceberg_noisy(): runs
several step counts (submit_iceberg_quench_batch already batches a list of
step_counts sharing one shot count into a single qnx.execute call) and
merges the results into one "points" list, in the same shape
plot_iceberg_comparison.py expects, so a real depth sweep -- not just a
single pilot point -- can be overlaid on the ED/raw-noisy/ZNE comparison.

Costs against the qnexus usage quota -- gated by
config.ICEBERG_RUN_ON_H2_EMULATOR, same convention as run_iceberg_qec.py.
Standalone: `python run_iceberg_sweep.py`.
"""
import config
from iceberg_decode import decode_shots
from persistence import save_stage_results
from shot_observables import bitstrings_to_observables, bootstrap_observable_errors


def run_iceberg_sweep(k, h_field, J, dt, step_shot_pairs, device_name=None,
                       early_exit=None, syndrome_every=1, timeout=1800.0):
    """step_shot_pairs: list of (steps, n_shots) pairs. Pairs sharing the
    same n_shots are grouped into a single submit_iceberg_quench_batch
    call (one qnx.execute submission for the whole group) rather than one
    call per step count.
    """
    device_name = device_name or config.ICEBERG_DEVICE_NAME
    early_exit = config.ICEBERG_EARLY_EXIT if early_exit is None else early_exit

    if not config.ICEBERG_RUN_ON_H2_EMULATOR:
        print("\n[Iceberg QEC sweep] Skipped (config.ICEBERG_RUN_ON_H2_EMULATOR = False).")
        return None

    from qnexus_backend import submit_iceberg_quench_batch

    groups = {}
    for steps, shots in step_shot_pairs:
        groups.setdefault(shots, []).append(steps)

    print("\n" + "=" * 70)
    print(f"RUNNING ICEBERG-ENCODED TFIM QUENCH SWEEP on {device_name}")
    print(f"(k={k}, h/J={h_field:.2f}, dt={dt}, syndrome_every={syndrome_every}, "
          f"early_exit={early_exit}, points={step_shot_pairs})")
    print("=" * 70)

    points = []
    for shots, steps_list in groups.items():
        print(f"\nSubmitting steps={steps_list} at shots={shots} ...")
        batch = submit_iceberg_quench_batch(
            k, h_field, J, dt, steps_list, shots,
            device_name=device_name, early_exit=early_exit,
            syndrome_every=syndrome_every, project_name=config.H2_PROJECT_NAME,
            timeout=timeout,
        )
        for steps in steps_list:
            entry = batch[steps]
            raw_shots = list(zip(entry["flags_bitstrings"], entry["data_bitstrings"]))
            kept, discard_rate = decode_shots(raw_shots, k)
            point = {
                "t": steps * dt, "steps": steps,
                "discard_rate": discard_rate, "n_kept": len(kept), "n_shots": shots,
                "z": None, "z_err": None, "mzz": None, "mzz_err": None,
            }
            if kept:
                z_rms, mzz = bitstrings_to_observables(kept, k)
                z_err, mzz_err = bootstrap_observable_errors(kept, k)
                point.update(z=z_rms, z_err=z_err, mzz=mzz, mzz_err=mzz_err)
                print(f"  steps={steps} (t={point['t']:.2f}): "
                      f"discard={discard_rate:.1%} ({len(kept)}/{shots} kept), "
                      f"z={z_rms:.4f}+/-{z_err:.4f}, mzz={mzz:.4f}+/-{mzz_err:.4f}")
            else:
                print(f"  steps={steps} (t={point['t']:.2f}): "
                      f"discard=100% -- no shots survived post-selection.")
            points.append(point)

    points.sort(key=lambda p: p["steps"])
    result = {
        "k": k, "h_field": h_field, "J": J, "dt": dt, "device": device_name,
        "early_exit": early_exit, "syndrome_every": syndrome_every, "points": points,
    }
    save_stage_results("iceberg_qec_sweep", result)
    return result


if __name__ == "__main__":
    run_iceberg_sweep(
        k=8, h_field=0.5, J=config.J, dt=0.1,
        step_shot_pairs=[(8, 500), (15, 500), (20, 500), (25, 500), (30, 200)],
        syndrome_every=None,
    )
