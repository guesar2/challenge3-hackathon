"""
run_phase_transition_zne.py

ZNE-mitigates run_h2_emulator.run_phase_transition()'s noisy H2-Emulator
adiabatic sweep -- the deviations visible in figures/h2_phase_transition_noisy.png
(H2-Emulator's real gate/SPAM/crosstalk/dephasing noise vs. ED, especially
at h/J=0.5/1.0 -- see run_h2_emulator.py's h2_phase_transition_noisy figure
and the checklist's noted <X> gap).

Same protocol, same ramp schedule as that run -- ramp_steps_by_target comes
from run_h2_emulator.ramp_steps_for_targets(h_values), the exact function
run_phase_transition() itself uses, so any change in deviation from ED is
attributable to ZNE, not a different adiabatic schedule. Each h_target's
*unmeasured* adiabatic ansatz (tket_circuit.build_adiabatic_ansatz_circuit)
is folded at config.H2_ZNE_FOLD_FACTORS (qermit's Folding.circuit -- ODD
integers only) via qnexus_backend.submit_adiabatic_zne_batch, then
extrapolated to the zero-noise limit per h/J and per observable via
zne_fit.zne_extrapolate -- the adiabatic-ramp/phase-transition analog of
what run_zne.py already does for the fixed-h quench-vs-time protocol.

fold_factor=1 IS the raw-noisy baseline (Folding.circuit performs zero fold
iterations at noise_scaling=1), so this one batch covers both the
"raw-noisy" and "ZNE-mitigated" series -- no separate run_phase_transition
call is needed to get the comparison point.

Standalone: `python run_phase_transition_zne.py`. Costs qnexus quota
(gated by config.RUN_ON_H2_EMULATOR) -- one batch of
len(h_values) * len(fold_factors) * 2 (bases) adiabatic-ramp circuits.
"""
import os
import sys

import config
from exact_diagonalization import ed_baseline
from persistence import save_stage_results
from plotting import plot_h2_phase_transition_zne
from qnexus_backend import submit_adiabatic_zne_batch
from run_h2_emulator import ramp_steps_for_targets
from shot_observables import (
    bitstrings_to_observables, bitstrings_to_mx,
    bootstrap_observable_errors, bootstrap_mx_error,
)
from zne_fit import zne_extrapolate

# Always resolve figures/data relative to the repo root -- same pitfall
# run_zne.py/run_noise_scaling.py guard against (a plain relative
# PLOT_SAVE_DIR silently writes to src/figures/ instead of the top-level
# figures/ when run from src/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_REPO_ROOT)


def run(h_values=None, fold_factors=None, shots=None, fit_deg=None, noise_scale=1.0, timeout=None):
    """h_values/fold_factors/shots/fit_deg: override config.H2_H_VALUES /
    config.H2_ZNE_FOLD_FACTORS / config.H2_ADIABATIC_SHOTS /
    config.H2_ZNE_FIT_DEG for this call -- pass a single h_values entry
    for a cheap trial run before committing to the full sweep.

    noise_scale: linear multiplier on H2-Emulator's default error rates
    (see qnexus_backend.submit_adiabatic_zne_batch's docstring) -- defaults
    to 1.0, the device's real, unscaled published noise. Pass None to skip
    constructing UserErrorParams entirely (functionally identical to 1.0).

    Gated by config.RUN_ON_H2_EMULATOR, like every other real qnexus call
    in this project.
    """
    if not config.RUN_ON_H2_EMULATOR:
        print("Skipped (config.RUN_ON_H2_EMULATOR = False). Enable it to submit "
              "jobs to qnexus -- this consumes a metered usage quota.")
        return None

    h_values = list(config.H2_H_VALUES) if h_values is None else list(h_values)
    fold_factors = config.H2_ZNE_FOLD_FACTORS if fold_factors is None else fold_factors
    shots = config.H2_ADIABATIC_SHOTS if shots is None else shots
    fit_deg = config.H2_ZNE_FIT_DEG if fit_deg is None else fit_deg
    raw_idx = list(fold_factors).index(1)  # fold_factor=1 IS the raw-noisy baseline

    N = config.H2_ADIABATIC_N
    ramp_steps_by_target = ramp_steps_for_targets(h_values)
    dt_by_target = [config.H2_ADIABATIC_DT for _ in h_values]

    print("=" * 60)
    print(f"PHASE-TRANSITION ZNE: N={N}, h_init={config.H_INIT}, h_values={h_values}, "
          f"ramp_steps={ramp_steps_by_target}, fold_factors={fold_factors}, "
          f"shots/point={shots}, noise_scale={noise_scale}")
    print("=" * 60)

    ed_results = ed_baseline(N, h_values, J=config.J)

    batch_kwargs = {}
    if noise_scale is not None:
        batch_kwargs["noise_scale"] = noise_scale
    if timeout is not None:
        batch_kwargs["timeout"] = timeout

    print(f"\nSubmitting N={N}, h_values={h_values}, ramp_steps={ramp_steps_by_target}, "
          f"fold_factors={fold_factors}, shots={shots} to {config.H2_DEVICE_NAME_NOISY} ...")
    batch = submit_adiabatic_zne_batch(
        N, h_values, config.J, ramp_steps_by_target, dt_by_target, fold_factors, shots,
        config.H_INIT, device_name=config.H2_DEVICE_NAME_NOISY, project_name=config.H2_PROJECT_NAME,
        **batch_kwargs,
    )

    # Persist raw shots before any postprocessing -- crash-safety, same
    # pattern as run_zne.py / run_h2_emulator.run().
    save_stage_results("h2_phase_transition_zne_raw", batch)

    results = {}
    for h in h_values:
        ed_z = next(r['mz_rms'] for r in ed_results if r['h'] == h)
        ed_x = next(r['mx'] for r in ed_results if r['h'] == h)
        ed_mzz = next(r['mzz'] for r in ed_results if r['h'] == h)

        z_vals, z_errs, mzz_vals, mzz_errs, x_vals, x_errs = [], [], [], [], [], []
        for fold in fold_factors:
            b = batch[h][fold]
            z_rms, mzz = bitstrings_to_observables(b["bitstrings"], N)
            z_se, mzz_se = bootstrap_observable_errors(b["bitstrings"], N)
            x_mean = bitstrings_to_mx(b["bitstrings_x"], N)
            x_se = bootstrap_mx_error(b["bitstrings_x"], N)
            z_vals.append(z_rms); z_errs.append(z_se)
            mzz_vals.append(mzz); mzz_errs.append(mzz_se)
            x_vals.append(x_mean); x_errs.append(x_se)

        z_zne, z_zne_err = zne_extrapolate(fold_factors, z_vals, z_errs, deg=fit_deg)
        mzz_zne, mzz_zne_err = zne_extrapolate(fold_factors, mzz_vals, mzz_errs, deg=fit_deg)
        x_zne, x_zne_err = zne_extrapolate(fold_factors, x_vals, x_errs, deg=fit_deg)

        results[h] = {
            'z_ed': ed_z, 'z_raw': z_vals[raw_idx], 'z_raw_err': z_errs[raw_idx],
            'z_zne': z_zne, 'z_zne_err': z_zne_err,
            'x_ed': ed_x, 'x_raw': x_vals[raw_idx], 'x_raw_err': x_errs[raw_idx],
            'x_zne': x_zne, 'x_zne_err': x_zne_err,
            'mzz_ed': ed_mzz, 'mzz_raw': mzz_vals[raw_idx], 'mzz_raw_err': mzz_errs[raw_idx],
            'mzz_zne': mzz_zne, 'mzz_zne_err': mzz_zne_err,
        }

        for key, label in (('z', '<Z>  '), ('x', '<X>  '), ('mzz', '<ZZ> ')):
            raw, raw_err = results[h][f'{key}_raw'], results[h][f'{key}_raw_err']
            zne, zne_err = results[h][f'{key}_zne'], results[h][f'{key}_zne_err']
            ed = results[h][f'{key}_ed']
            closer = "closer to ED" if abs(zne - ed) < abs(raw - ed) else "NOT closer to ED"
            print(f"  h/J={h:.2f} {label}: raw={raw:.4f}+/-{raw_err:.4f}  "
                  f"ZNE={zne:.4f}+/-{zne_err:.4f}  ED={ed:.4f}  ({closer})")

    save_stage_results("h2_phase_transition_zne", results)

    plot_h2_phase_transition_zne(
        h_values, results, save_dir=config.PLOT_SAVE_DIR, n=N,
        filename=f"h2_phase_transition_zne_N{N}.png",
    )

    return results


if __name__ == "__main__":
    h_values_override = None
    if "--h-values" in sys.argv:
        raw = sys.argv[sys.argv.index("--h-values") + 1]
        h_values_override = tuple(float(h) for h in raw.split(","))
    shots_override = None
    if "--shots" in sys.argv:
        shots_override = int(sys.argv[sys.argv.index("--shots") + 1])
    noise_scale_override = 1.0
    if "--noise-scale" in sys.argv:
        noise_scale_override = float(sys.argv[sys.argv.index("--noise-scale") + 1])
    fold_factors_override = None
    if "--fold-factors" in sys.argv:
        raw = sys.argv[sys.argv.index("--fold-factors") + 1]
        fold_factors_override = tuple(int(f) for f in raw.split(","))
    timeout_override = None
    if "--timeout" in sys.argv:
        timeout_override = float(sys.argv[sys.argv.index("--timeout") + 1])

    run(h_values=h_values_override, shots=shots_override,
        noise_scale=noise_scale_override, fold_factors=fold_factors_override,
        timeout=timeout_override)
