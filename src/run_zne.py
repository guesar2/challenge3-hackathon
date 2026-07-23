"""
run_zne.py

Zero-Noise Extrapolation (ZNE) against the real noisy H2-Emulator, via
qermit's Folding.circuit (qnexus_backend.submit_zne_batch) plus this
project's own submission/observable/error pipeline -- not qermit's
gen_ZNE_MitEx/MitEx machinery, which needs a real pytket.backends.Backend
object (this project submits via the qnexus cloud client's own async job
functions, not a Backend) and returns a bare point estimate with no
uncertainty channel. See qnexus_backend.submit_zne_batch's docstring for
the folding details.

Produces a full quench-vs-time curve (one point per Trotter step, times =
step_count * H2_DT), matching run_h2_emulator.run()'s/plot_h2_noise_
comparison's time-series convention, rather than a single fixed-depth
point. One qnexus batch per h covers every (step_count, fold_factor,
basis) combination, including fold_factor=1 -- which IS the plain
raw-noisy circuit at that step count (Folding.circuit performs zero fold
iterations at noise_scaling=1, verified), so no separate raw-noisy
baseline submission is needed. Each (step_count, fold_factor)'s
bitstrings are converted to observables via shot_observables.py's
existing functions, with real bootstrap standard errors; zne_fit.
zne_extrapolate then fits each observable's (fold_factor, value, error)
series back to the zero-noise limit at that step count, propagating
those bootstrap errors into a real error bar on each ZNE-mitigated point.

Targets run_noise_scaling.SHORT_TIME_N/SHORT_TIME_STEPS (N=8, steps=5,
T=0.5) by default -- the same short-time point already characterized in
run_noise_scaling.py's noise-scale comparison -- at noise_scale=1 (the
device's real, unscaled published noise). Costs scale as
len(step_counts) * len(fold_factors) * len(bases) circuits per h -- pass
h_values/bases to restrict to a cheap trial (e.g. a single h, Z-basis
only for <Zi Zi+1>) before committing to the full config.H2_H_VALUES x
both-bases sweep.
"""
import os
import sys

import config
from exact_diagonalization import ed_time_evolution_exact
from persistence import save_stage_results
from plotting import plot_zne_comparison
from qnexus_backend import submit_zne_batch
from shot_observables import (
    bitstrings_to_observables, bitstrings_to_mx,
    bootstrap_observable_errors, bootstrap_mx_error,
)
from zne_fit import zne_extrapolate

# run_noise_scaling.py itself os.chdir()s to the repo root as an import-time
# side effect (see below) -- imported last among local modules so every
# other plain src/-relative import above resolves before cwd changes.
from run_noise_scaling import SHORT_TIME_N, SHORT_TIME_STEPS

# Always resolve figures/data relative to the repo root, matching
# run_noise_scaling.py's own convention (a prior run from src/ silently
# wrote to src/figures/ instead of the top-level figures/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_REPO_ROOT)


def run(n=None, steps=None, shots=None, fold_factors=None, fit_deg=None, noise_scale=1.0,
        timeout=None, h_values=None, bases=None):
    """n/steps/shots/fold_factors/fit_deg: override config.H2_N-equivalent
    defaults (see below) / config.H2_ZNE_SHOTS / config.H2_ZNE_FOLD_FACTORS /
    config.H2_ZNE_FIT_DEG for this call.

    n/steps default to run_noise_scaling.SHORT_TIME_N/SHORT_TIME_STEPS
    (N=8, steps=5, T=0.5) rather than config.H2_N/H2_STEPS -- this is
    deliberately the same short-time point already characterized in
    run_noise_scaling.py's noise-scale comparison, so ZNE's result is
    directly comparable to it. Produces the whole 1..steps curve (times =
    step_count * H2_DT), not just the final point.

    noise_scale: linear multiplier on H2-Emulator's default error rates
    (see qnexus_backend.submit_quench_batch's docstring for the mechanism)
    -- defaults to 1.0, i.e. the device's real, unscaled published noise
    (UserErrorParams(scale=1.0) leaves error rates unchanged from their
    default), NOT the artificially amplified 3x/5x levels used in this
    branch's earlier noise-scan work. This is a distinct knob from
    fold_factors: noise_scale amplifies the *device's* noise model,
    fold_factors amplifies noise via *circuit* folding (qermit's
    Folding.circuit) -- ZNE always needs the latter to build its
    noise-scaling series; noise_scale is an independent multiplier on top
    of whichever noise level fold_factors probes at each point. Pass
    noise_scale=None to skip constructing UserErrorParams entirely
    (functionally identical to 1.0 per the noise-model docs, but 1.0 is
    used as the explicit default so it's visible in logs/filenames which
    regime was run).

    h_values: override config.H2_H_VALUES -- restrict to a subset (e.g. a
    single h) for a cheap trial run before committing to the full sweep.

    bases: override which measurement bases to submit -- ("z", "x") by
    default (all three observables: <Z>, <X>, <Zi Zi+1>). Pass ("z",)
    alone to compute only <Z>/<Zi Zi+1> (skip <X> entirely -- roughly
    halves circuit count) for a cheap trial run; the resulting `results`
    entries simply omit the x_* keys, and plot_zne_comparison drops the
    <X> column automatically when it's absent.

    Gated by config.RUN_ON_H2_EMULATOR, like every other real qnexus call
    in this project.
    """
    if not config.RUN_ON_H2_EMULATOR:
        print("Skipped (config.RUN_ON_H2_EMULATOR = False). Enable it to submit "
              "jobs to qnexus -- this consumes a metered usage quota.")
        return None

    n = SHORT_TIME_N if n is None else n
    steps = SHORT_TIME_STEPS if steps is None else steps
    shots = config.H2_ZNE_SHOTS if shots is None else shots
    fold_factors = config.H2_ZNE_FOLD_FACTORS if fold_factors is None else fold_factors
    fit_deg = config.H2_ZNE_FIT_DEG if fit_deg is None else fit_deg
    h_values = config.H2_H_VALUES if h_values is None else h_values
    bases = ("z", "x") if bases is None else tuple(bases)

    step_counts = list(range(1, steps + 1))
    raw_idx = list(fold_factors).index(1)  # fold_factor=1 IS the raw-noisy baseline

    print("=" * 60)
    print(f"ZERO-NOISE EXTRAPOLATION: N={n}, steps=1..{steps} (T={steps * config.H2_DT:.2g}), "
          f"h_values={h_values}, bases={bases}, fold_factors={fold_factors}, "
          f"shots/point={shots}, noise_scale={noise_scale}")
    print("=" * 60)

    results = {}
    for h in h_values:
        print(f"\nSubmitting N={n}, h/J={h:.2f}, dt={config.H2_DT}, steps=1..{steps} (batched), "
              f"fold_factors={fold_factors}, bases={bases}, shots={shots} "
              f"to {config.H2_DEVICE_NAME_NOISY} ...")

        batch_kwargs = {}
        if noise_scale is not None:
            batch_kwargs["noise_scale"] = noise_scale
        if timeout is not None:
            batch_kwargs["timeout"] = timeout
        batch = submit_zne_batch(
            n, h, config.J, config.H2_DT, step_counts, fold_factors, shots,
            device_name=config.H2_DEVICE_NAME_NOISY, project_name=config.H2_PROJECT_NAME,
            bases=bases, **batch_kwargs,
        )

        # Persist raw shots before any postprocessing -- crash-safety, same
        # pattern as run_h2_emulator.run().
        save_stage_results(f"h2_zne_raw_h{h:.2f}", batch)

        _, z_ed_arr, mzz_ed_arr, x_ed_arr = ed_time_evolution_exact(n, h, config.J, config.H2_DT, steps)
        times = [sc * config.H2_DT for sc in step_counts]

        entry = {'times': times}
        if "z" in bases:
            entry.update({k: [] for k in (
                'z_ed', 'z_raw', 'z_raw_err', 'z_zne', 'z_zne_err',
                'mzz_ed', 'mzz_raw', 'mzz_raw_err', 'mzz_zne', 'mzz_zne_err',
            )})
        if "x" in bases:
            entry.update({k: [] for k in ('x_ed', 'x_raw', 'x_raw_err', 'x_zne', 'x_zne_err')})

        for si, sc in enumerate(step_counts):
            if "z" in bases:
                z_vals, z_errs, mzz_vals, mzz_errs = [], [], [], []
                for fold in fold_factors:
                    b = batch[sc][fold]
                    z_rms, mzz = bitstrings_to_observables(b["bitstrings"], n)
                    z_se, mzz_se = bootstrap_observable_errors(b["bitstrings"], n)
                    z_vals.append(z_rms); z_errs.append(z_se)
                    mzz_vals.append(mzz); mzz_errs.append(mzz_se)

                z_zne, z_zne_err = zne_extrapolate(fold_factors, z_vals, z_errs, deg=fit_deg)
                mzz_zne, mzz_zne_err = zne_extrapolate(fold_factors, mzz_vals, mzz_errs, deg=fit_deg)

                entry['z_ed'].append(float(z_ed_arr[si]))
                entry['z_raw'].append(z_vals[raw_idx]); entry['z_raw_err'].append(z_errs[raw_idx])
                entry['z_zne'].append(z_zne); entry['z_zne_err'].append(z_zne_err)
                entry['mzz_ed'].append(float(mzz_ed_arr[si]))
                entry['mzz_raw'].append(mzz_vals[raw_idx]); entry['mzz_raw_err'].append(mzz_errs[raw_idx])
                entry['mzz_zne'].append(mzz_zne); entry['mzz_zne_err'].append(mzz_zne_err)

            if "x" in bases:
                x_vals, x_errs = [], []
                for fold in fold_factors:
                    b = batch[sc][fold]
                    x_mean = bitstrings_to_mx(b["bitstrings_x"], n)
                    x_se = bootstrap_mx_error(b["bitstrings_x"], n)
                    x_vals.append(x_mean); x_errs.append(x_se)

                x_zne, x_zne_err = zne_extrapolate(fold_factors, x_vals, x_errs, deg=fit_deg)

                entry['x_ed'].append(float(x_ed_arr[si]))
                entry['x_raw'].append(x_vals[raw_idx]); entry['x_raw_err'].append(x_errs[raw_idx])
                entry['x_zne'].append(x_zne); entry['x_zne_err'].append(x_zne_err)

        results[h] = entry

        for key, label in (('z', '<Z>  '), ('x', '<X>  '), ('mzz', '<ZZ> ')):
            if f'{key}_raw' not in entry:
                continue
            closer = sum(
                abs(zne - ed) < abs(raw - ed)
                for zne, raw, ed in zip(entry[f'{key}_zne'], entry[f'{key}_raw'], entry[f'{key}_ed'])
            )
            print(f"  h/J={h:.2f}: {label} ZNE closer to ED than raw in {closer}/{len(step_counts)} steps "
                  f"(final step: raw={entry[f'{key}_raw'][-1]:.4f}+/-{entry[f'{key}_raw_err'][-1]:.4f}  "
                  f"ZNE={entry[f'{key}_zne'][-1]:.4f}+/-{entry[f'{key}_zne_err'][-1]:.4f}  "
                  f"ED={entry[f'{key}_ed'][-1]:.4f})")

    save_stage_results("h2_zne", results)

    plot_zne_comparison(
        list(h_values), results, save_dir=config.PLOT_SAVE_DIR, n=n,
        filename=f"h2_zne_comparison_N{n}_S{steps}_scale{noise_scale}.png",
    )

    return results


if __name__ == "__main__":
    n_override = None
    if "--n" in sys.argv:
        n_override = int(sys.argv[sys.argv.index("--n") + 1])
    steps_override = None
    if "--steps" in sys.argv:
        steps_override = int(sys.argv[sys.argv.index("--steps") + 1])
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
    h_values_override = None
    if "--h-values" in sys.argv:
        raw = sys.argv[sys.argv.index("--h-values") + 1]
        h_values_override = tuple(float(h) for h in raw.split(","))
    bases_override = None
    if "--bases" in sys.argv:
        raw = sys.argv[sys.argv.index("--bases") + 1]
        bases_override = tuple(raw.split(","))

    run(n=n_override, steps=steps_override, shots=shots_override,
        fold_factors=fold_factors_override, noise_scale=noise_scale_override,
        h_values=h_values_override, bases=bases_override)
