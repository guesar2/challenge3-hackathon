"""
run_noise_scaling.py

First diagnostic step towards QEC / error mitigation: characterize how
Quantinuum H2-Emulator's real noise model (gate/SPAM/crosstalk/dephasing)
scales with (a) spin-chain size N and (b) Trotter circuit depth, before
writing any mitigation or QEC-encoding code.

Two scans, both submitted to qnexus (cloud) for both H2-1LE (noiseless) and
H2-Emulator (noisy) -- per project decision, local=False throughout (local
pytket-pecos is slow in practice and Nexus quota is currently unlimited):

- run_noise_scaling(): sweeps N in {4, 6, 8, 12, 16, 20, 26} at the default
  H2_STEPS=5 -- 26 is the device's real qubit ceiling. At N<=8, hardware
  noise stayed in the same ~1-8% band, i.e. these shallow (~20-40 two-qubit
  gate) circuits aren't yet deep enough to show noise growing with N --
  consistent with the hackathon brief's claim that circuits need >~50 gates
  before noise dominates. N=12/16/20/26 push past config.H2_ED_MAX_N, where
  run() no longer has a dense-ED reference to compare against (see its
  docstring) -- the noisy-vs-noiseless comparison (hardware noise, Trotter
  error cancelled) doesn't need ED and still runs at every N here; only the
  vs-ED Trotter-error check is unavailable past N=12.
- run_depth_scaling(): holds N fixed and instead sweeps circuit depth
  (Trotter step count) via run()'s `steps=` override, to test that
  threshold directly. Since gate count per step scales with N, N is fixed
  at the scan's largest useful value (default 8) so the sweep pushes gate
  count past ~50 within a single run() call (run() already returns every
  step 1..`steps` in one batch, so one call sweeps the whole depth axis).
- run_noise_scale_comparison(): the orthogonal knob to the two scans above
  -- instead of growing the *circuit* (more N or more steps) to make
  hardware noise visible, this holds the circuit fixed at N=8/steps=5
  (T=0.5, deliberately short -- run_noise_scaling()'s own N=8 point at this
  same depth is where the ~1-8% noise band was too close to shot noise to
  separate cleanly) and instead scales H2-Emulator's *noise model* itself
  up via qnexus_backend.submit_quench_batch's noise_scale (UserErrorParams'
  linear 'scale' multiplier on gate/SPAM/crosstalk/dephasing rates -- see
  its docstring and docs.quantinuum.com/systems/user_guide/emulator_user_guide/noise_model.html).

Reports BOTH absolute and relative deviations for <Z>, <X>, <Zi Zi+1> --
not just <Z> -- since relative error blows up near <X>'s zero-crossing at
shallow depth (confirmed: <X>_ED ~ 0.02 at step 1, so a shot-noise-sized
absolute gap reads as >100% relative deviation there; this is a denominator
artifact, not a real noise effect -- see the module-level check that
disambiguated it against the saved N=8/N=4 h=0.5 JSON before this script
was written this way).
"""
import os
import sys

import config
from plotting import plot_h2_noise_comparison
from run_h2_emulator import run

# Always resolve figures/data relative to the repo root, regardless of the
# cwd this script is invoked from -- a prior run from src/ silently wrote
# to src/figures/ instead of the top-level figures/ due to PLOT_SAVE_DIR
# being a plain relative path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_REPO_ROOT)

# Spins to scan (fixed depth). N=4 matches the historical default
# (config.H2_N); 6 and 8 extend it -- 8 also matches the PDF's "good
# enough" noiseless-Trotter benchmark size, and ED stays cheap
# (2^8 = 256 states) at all three. 12 is the last point with an ED
# reference (config.H2_ED_MAX_N); 16/20/26 push past the classical wall to
# probe hardware noise up to the device's real qubit ceiling (26) -- run()
# skips the ED comparison there automatically, but still returns z_h2/x_h2/
# mzz_h2 for both devices, so the noisy-vs-noiseless comparison below
# (hardware noise, Trotter error cancelled) still works at every N here.
NOISE_SCALING_N_VALUES = (4, 6, 8, 12, 16, 20, 26)

# qnx.execute()'s own default wait (300s) was enough through N=8, but timed
# out client-side submitting N=12's noisy (H2-Emulator) leg -- larger
# circuits and the noisy device's real error-model simulation (heavier than
# H2-1LE's plain statevector) push past it as N grows. Reuse
# DEPTH_SCALING_TIMEOUT's value (see its comment below) for every N here.
NOISE_SCALING_TIMEOUT = 1800.0

# Depth scan: fixed N, sweep Trotter steps far enough that gate count
# (~N two-qubit gates/step) crosses the brief's ~50-gate noise-dominated
# threshold. At N=8, 30 steps -> ~240 two-qubit gates alone.
DEPTH_SCALING_N = 8
DEPTH_SCALING_STEPS = 30
# Shots bumped from the default 200: config.py's own H2_ADIABATIC_SHOTS
# comment found 200-500 shots gave a bootstrap SE comparable in size to the
# 1-5% deviations being resolved, which is exactly the regime we're in here
# -- 2000 keeps SE ~4x smaller (SE ~ 1/sqrt(shots)).
DEPTH_SCALING_SHOTS = 2000
# qnx.execute()'s own default wait is 300s; a 30-step x 2000-shot x 2-basis
# batch measured taking longer than that and hit TimeoutError client-side
# (the job itself keeps running server-side -- see qnexus_backend.py's
# submit_quench_batch timeout docstring).
DEPTH_SCALING_TIMEOUT = 1800.0

# Below this |ED value|, relative-error % is not reported (denominator
# artifact -- see module docstring); absolute difference and shot-noise
# sigma are reported instead.
_REL_ERROR_FLOOR = 0.05


def _deviation(sim, ed, err=None):
    """Per-point (abs_diff, rel_pct-or-None, sigma-or-None) for a single
    observable's time/step series. rel_pct is None when |ed| is too close
    to zero for a percentage to be meaningful (see _REL_ERROR_FLOOR);
    sigma is abs_diff/err when a shot-noise error bar is available.
    """
    out = []
    for i, (s, e) in enumerate(zip(sim, ed)):
        abs_diff = abs(s - e)
        rel_pct = abs_diff / abs(e) * 100 if abs(e) > _REL_ERROR_FLOOR else None
        sigma = abs_diff / err[i] if err is not None and err[i] > 0 else None
        out.append((abs_diff, rel_pct, sigma))
    return out


def _summarize_observable(name, sim, ed, err):
    devs = _deviation(sim, ed, err)
    max_abs = max(d[0] for d in devs)
    rel_vals = [d[1] for d in devs if d[1] is not None]
    max_rel = max(rel_vals) if rel_vals else None
    max_sigma = max((d[2] for d in devs if d[2] is not None), default=None)
    rel_str = f"{max_rel:.2f}%" if max_rel is not None else "n/a (|ED|<%.2f throughout)" % _REL_ERROR_FLOOR
    sigma_str = f"{max_sigma:.1f}sigma" if max_sigma is not None else "n/a"
    print(f"    <{name}>: max abs diff = {max_abs:.4f}, max rel = {rel_str}, max = {sigma_str}")
    return {'max_abs': max_abs, 'max_rel_pct': max_rel, 'max_sigma': max_sigma}


def _compare(label, sim_data, ref_data, ref_is_ed):
    """sim_data: dict h -> per-run results (as returned by run()) to report
    deviations for. ref_data: dict h -> results to compare against -- when
    ref_is_ed is True, pulls the reference's z_ed/x_ed/mzz_ed (exact
    diagonalization) fields (Trotter-error comparison); when False, pulls
    its z_h2/x_h2/mzz_h2 fields instead, i.e. comparing two *simulated*
    curves against each other (e.g. noisy vs. noiseless -- isolates
    hardware noise since both ran the identical circuit, cancelling
    Trotter error).
    """
    print(f"  {label}:")
    summary = {}
    for h, r in sim_data.items():
        ref = ref_data[h]
        suffix = '_ed' if ref_is_ed else '_h2'
        print(f"  h/J={h:.2f}")
        summary[h] = {
            'z': _summarize_observable('Z', r['z_h2'], ref[f'z{suffix}'], r['z_err']),
            'x': _summarize_observable('X', r['x_h2'], ref[f'x{suffix}'], r['x_err']),
            'mzz': _summarize_observable('ZZ', r['mzz_h2'], ref[f'mzz{suffix}'], r['mzz_err']),
        }
    return summary


def run_noise_scaling(n_values=NOISE_SCALING_N_VALUES):
    """N-scan at the default circuit depth (config.H2_STEPS).

    Always runs config.H2_H_VALUES in full -- run() (which this calls)
    loops over config.H2_H_VALUES directly and has no per-h override. Use
    run_noise_scaling_async's h_values param for a cheaper single-h sweep.
    """
    all_summaries = {}
    for n in n_values:
        print("\n" + "#" * 60)
        print(f"# NOISE SCALING: N={n}")
        print("#" * 60)

        noiseless_results = run(local=False, noisy=False, n=n, timeout=NOISE_SCALING_TIMEOUT)
        noisy_results = run(local=False, noisy=True, n=n, timeout=NOISE_SCALING_TIMEOUT)
        if noiseless_results is None or noisy_results is None:
            print(f"N={n}: skipped (config.RUN_ON_H2_EMULATOR is False) -- "
                  "enable it in config.py to actually submit to qnexus.")
            continue

        filename = f"h2_noise_comparison_N{n}.png"
        plot_h2_noise_comparison(
            config.H2_H_VALUES, noiseless_results, noisy_results,
            save_dir=config.PLOT_SAVE_DIR, n=n, filename=filename,
        )

        if n <= config.H2_ED_MAX_N:
            print(f"\nN={n} -- Trotter error (noiseless vs ED):")
            trotter = _compare("noiseless vs ED", noiseless_results, noiseless_results, ref_is_ed=True)
        else:
            print(f"\nN={n} -- Trotter error vs ED skipped (n > config.H2_ED_MAX_N="
                  f"{config.H2_ED_MAX_N}, dense ED infeasible at this N)")
            trotter = None
        print(f"\nN={n} -- Hardware noise (noisy vs noiseless, same circuit):")
        hw_noise = _compare("noisy vs noiseless", noisy_results, noiseless_results, ref_is_ed=False)
        all_summaries[n] = {'trotter_error': trotter, 'hardware_noise': hw_noise}

    return all_summaries


def run_noise_scaling_async(n_values=NOISE_SCALING_N_VALUES, timeout=NOISE_SCALING_TIMEOUT,
                             h_values=None):
    """Async counterpart to run_noise_scaling(): submits every (N, device,
    h) combination in n_values UP FRONT via qnexus_backend.start_quench_batch
    (non-blocking), so they all queue concurrently on Nexus, THEN collects
    every one via collect_quench_batch. run_noise_scaling() submits-and-
    waits one (N, device, h) batch at a time -- with len(n_values) * 2 *
    len(h_values) batches total, that means paying every batch's
    queue-wait in sequence; submitting first decouples "how long one job
    waits in Nexus's queue" from "how many jobs we're running", so total
    wall time tracks the slowest single job, not the sum of all of them.

    h_values: overrides config.H2_H_VALUES for this call only (e.g. a
    single h/J to cut the sweep's batch count -- and quota/time cost -- by
    len(config.H2_H_VALUES)x, without touching the global config every
    other stage reads). Unlike run_noise_scaling() (which calls run(), and
    run() always loops over config.H2_H_VALUES with no override), this
    function builds circuits directly, so it can honor h_values cleanly.

    Same N range (modulo h_values), same H2_ED_MAX_N-gated ED comparison
    (see run()'s and _process_h_batch's docstrings), same figures/summaries/
    saved-stage names as run_noise_scaling() -- this changes submission
    order (and optionally h coverage) only, not how a given h is measured
    or reported. Reuses run_h2_emulator's _process_h_batch/_stage_suffix so
    both submission paths postprocess and name files identically.

    timeout: passed to each job's collect_quench_batch call individually
    (see its docstring) -- since jobs are submitted independently, one
    slow job timing out doesn't block collecting the others before it
    (though a KeyError/exception in one collect call still stops this
    function the same way an exception anywhere in run_noise_scaling()
    would; each batch's raw results are still saved as they're collected,
    same persist-immediately guarantee run() gives its own caller).
    """
    from run_h2_emulator import _process_h_batch, _stage_suffix
    from persistence import save_stage_results
    from plotting import plot_h2_vs_ed_time
    from qnexus_backend import start_quench_batch, collect_quench_batch

    if not config.RUN_ON_H2_EMULATOR:
        print("Skipped (config.RUN_ON_H2_EMULATOR = False). Enable it to submit "
              "jobs to qnexus -- this consumes a metered usage quota.")
        return None

    h_values = config.H2_H_VALUES if h_values is None else h_values
    step_counts = list(range(1, config.H2_STEPS + 1))
    total_batches = len(n_values) * 2 * len(h_values)

    print("\n" + "#" * 60)
    print(f"# SUBMITTING {total_batches} batches ({len(n_values)} N-values x 2 devices x "
          f"{len(h_values)} h-values) -- not waiting for results yet")
    print("#" * 60)
    pending = {}  # (n, noisy, h) -> start_quench_batch's pending handle
    for n in n_values:
        for noisy in (False, True):
            device_name = config.H2_DEVICE_NAME_NOISY if noisy else config.H2_DEVICE_NAME
            for h in h_values:
                print(f"  submitting N={n}, noisy={noisy}, h/J={h:.2f} to {device_name} ...")
                pending[(n, noisy, h)] = start_quench_batch(
                    n, h, config.J, config.H2_DT, step_counts, config.H2_SHOTS,
                    device_name=device_name, project_name=config.H2_PROJECT_NAME,
                )
    print(f"\nAll {total_batches} batches submitted -- collecting results now.")

    all_summaries = {}
    for n in n_values:
        print("\n" + "#" * 60)
        print(f"# COLLECTING: N={n}")
        print("#" * 60)

        results_by_noisy = {}
        for noisy in (False, True):
            stage_suffix = _stage_suffix(n, config.H2_STEPS, local=False, noisy=noisy)
            raw_by_h = {}
            results = {}
            for h in h_values:
                batch_results = collect_quench_batch(pending[(n, noisy, h)], timeout=timeout)
                # Persist immediately, same as run() -- a bug downstream in
                # _process_h_batch should never mean resubmitting to recover
                # raw data that already came back.
                raw_by_h[h] = batch_results
                save_stage_results(f"h2_emulator_raw{stage_suffix}", raw_by_h)
                results[h] = _process_h_batch(n, h, config.H2_STEPS, batch_results)
            save_stage_results(f"h2_emulator{stage_suffix}", results)
            if n <= config.H2_ED_MAX_N:
                plot_h2_vs_ed_time(
                    h_values, results, save_dir=config.PLOT_SAVE_DIR,
                    filename=f"h2_vs_ed_time{stage_suffix}.png",
                )
            results_by_noisy[noisy] = results

        noiseless_results, noisy_results = results_by_noisy[False], results_by_noisy[True]
        filename = f"h2_noise_comparison_N{n}.png"
        plot_h2_noise_comparison(
            h_values, noiseless_results, noisy_results,
            save_dir=config.PLOT_SAVE_DIR, n=n, filename=filename,
        )

        if n <= config.H2_ED_MAX_N:
            print(f"\nN={n} -- Trotter error (noiseless vs ED):")
            trotter = _compare("noiseless vs ED", noiseless_results, noiseless_results, ref_is_ed=True)
        else:
            print(f"\nN={n} -- Trotter error vs ED skipped (n > config.H2_ED_MAX_N="
                  f"{config.H2_ED_MAX_N}, dense ED infeasible at this N)")
            trotter = None
        print(f"\nN={n} -- Hardware noise (noisy vs noiseless, same circuit):")
        hw_noise = _compare("noisy vs noiseless", noisy_results, noiseless_results, ref_is_ed=False)
        all_summaries[n] = {'trotter_error': trotter, 'hardware_noise': hw_noise}

    return all_summaries


def run_depth_scaling(n=DEPTH_SCALING_N, steps=DEPTH_SCALING_STEPS, shots=DEPTH_SCALING_SHOTS,
                       timeout=DEPTH_SCALING_TIMEOUT):
    """Fixed-N depth scan: sweeps Trotter step count (not N) to test whether
    noise grows once circuit depth crosses the brief's ~50-gate
    noise-dominated threshold -- the N-scan alone (run_noise_scaling, fixed
    at 5 steps) stayed too shallow to show that.
    """
    print("\n" + "#" * 60)
    print(f"# DEPTH SCALING: N={n}, steps=1..{steps}, shots={shots}")
    print("#" * 60)

    noiseless_results = run(local=False, noisy=False, n=n, steps=steps, shots=shots, timeout=timeout)
    noisy_results = run(local=False, noisy=True, n=n, steps=steps, shots=shots, timeout=timeout)
    if noiseless_results is None or noisy_results is None:
        print("Skipped (config.RUN_ON_H2_EMULATOR is False).")
        return None

    filename = f"h2_noise_comparison_N{n}_S{steps}.png"
    plot_h2_noise_comparison(
        config.H2_H_VALUES, noiseless_results, noisy_results,
        save_dir=config.PLOT_SAVE_DIR, n=n, filename=filename,
    )

    print(f"\nN={n}, steps=1..{steps} -- Trotter error (noiseless vs ED):")
    trotter = _compare("noiseless vs ED", noiseless_results, noiseless_results, ref_is_ed=True)
    print(f"\nN={n}, steps=1..{steps} -- Hardware noise (noisy vs noiseless, same circuit):")
    hw_noise = _compare("noisy vs noiseless", noisy_results, noiseless_results, ref_is_ed=False)
    return {'trotter_error': trotter, 'hardware_noise': hw_noise}


# Short-time, amplified-noise scan. Fixed at N=8/steps=5 -- the same point
# already covered by run_noise_scaling()'s N-sweep at config.H2_STEPS's
# default depth (T = steps * H2_DT = 0.5, see figures/h2_noise_comparison_N8.png)
# -- but instead of varying N or depth, this holds the circuit fixed and
# turns up H2-Emulator's own noise model via qnexus_backend.submit_quench_batch's
# noise_scale (UserErrorParams' 'scale' knob -- a linear multiplier on
# H2-Emulator's default gate/SPAM/crosstalk/dephasing error rates, see
# docs.quantinuum.com/systems/user_guide/emulator_user_guide/noise_model.html).
# At T=0.5 the *real* (scale=1) H2-Emulator noise is comparable to shot
# noise (see run_noise_scaling()'s N=8 finding, ~1-8% band, same order as
# shot-noise-sized deviations) -- scaling it up is a deliberate way to make
# the noisy-vs-noiseless gap visually obvious at this short a time, not a
# claim about the real device's actual error rates.
SHORT_TIME_N = DEPTH_SCALING_N  # =8, matches run_depth_scaling()'s N and the N8_S30 reference figure
SHORT_TIME_STEPS = 5            # dt=H2_DT=0.1 -> T=0.5 ("shorter" vs. N8_S30's T=3.0)
SHORT_TIME_SHOTS = DEPTH_SCALING_SHOTS  # =2000, same shot-noise rationale as run_depth_scaling
SHORT_TIME_NOISE_SCALES = (3.0, 5.0)    # amplify the default (scale=1.0) H2-Emulator error rates


def run_noise_scale_comparison(n=SHORT_TIME_N, steps=SHORT_TIME_STEPS, shots=SHORT_TIME_SHOTS,
                                noise_scales=SHORT_TIME_NOISE_SCALES):
    """Fixed-circuit (N/steps held constant, short T=steps*H2_DT) noise-scale
    scan: submits the H2-1LE noiseless reference once and reuses it against
    an H2-Emulator run per entry in `noise_scales`, producing one
    h2_noise_comparison_N{n}_S{steps}_scale{scale}.png per scale (styled
    like h2_noise_comparison_N8_S30.png / run_depth_scaling()'s output).

    Costs against the qnexus usage quota: one noiseless batch plus one
    noisy batch per noise_scales entry, each shots x len(H2_H_VALUES) x 2
    bases circuits (same per-call cost as a single run() call -- see its
    docstring).
    """
    print("\n" + "#" * 60)
    print(f"# NOISE-SCALE COMPARISON: N={n}, steps=1..{steps} "
          f"(T={steps * config.H2_DT:.2g}), scales={noise_scales}, shots={shots}")
    print("#" * 60)

    noiseless_results = run(local=False, noisy=False, n=n, steps=steps, shots=shots)
    if noiseless_results is None:
        print(f"Skipped (config.RUN_ON_H2_EMULATOR is False) -- "
              "enable it in config.py to actually submit to qnexus.")
        return None

    all_summaries = {}
    for scale in noise_scales:
        print(f"\n--- noise_scale={scale:g} ---")
        noisy_results = run(local=False, noisy=True, n=n, steps=steps, shots=shots, noise_scale=scale)

        filename = f"h2_noise_comparison_N{n}_S{steps}_scale{scale:g}.png"
        plot_h2_noise_comparison(
            config.H2_H_VALUES, noiseless_results, noisy_results,
            save_dir=config.PLOT_SAVE_DIR, n=n, filename=filename,
        )

        print(f"\nN={n}, steps=1..{steps}, noise_scale={scale:g} -- "
              f"Hardware noise (noisy vs noiseless, same circuit):")
        hw_noise = _compare("noisy vs noiseless", noisy_results, noiseless_results, ref_is_ed=False)
        all_summaries[scale] = {'hardware_noise': hw_noise}

    return all_summaries


if __name__ == "__main__":
    if "--depth" in sys.argv:
        run_depth_scaling()
    elif "--async" in sys.argv:
        run_noise_scaling_async()
    elif "--noise-scale" in sys.argv:
        run_noise_scale_comparison()
    else:
        run_noise_scaling()
