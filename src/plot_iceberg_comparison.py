"""
plot_iceberg_comparison.py

Plots every persisted Iceberg-encoded qnexus run against this repo's
existing ED/raw-noisy/ZNE-mitigated comparison (data/h2_zne_latest.json,
from run_zne.py) at the matching h/J -- see plotting.plot_iceberg_comparison's
docstring for why these two datasets line up point-for-point (same N, same
dt, same h/J). Overlays every Iceberg dataset present across three stages,
each as its own labeled marker/color series (plotting.ICEBERG_SERIES_STYLES)
so different syndrome-checking schedules can be told apart on the same
figure instead of one clobbering the other:
  - "iceberg_qec_sweep_dense": the original syndrome_every=1 (checked every
    half-step) reference depth sweep.
  - "iceberg_qec": a single pilot point (run_iceberg_qec.py's
    run_iceberg_noisy()).
  - "iceberg_qec_sweep": the most recent depth sweep (run_iceberg_sweep.py's
    run_iceberg_sweep()) -- whatever syndrome_every it was run with.
Missing stages are skipped; whichever subset exists gets plotted.

Reads only what's already on disk -- doesn't touch qnexus, costs no
quota, and can be re-run any number of times (e.g. to restyle the
figures) without resubmitting anything.
"""
import config
from persistence import load_latest
from plotting import plot_iceberg_comparison, plot_iceberg_discard_rate


def _syndrome_label(syndrome_every):
    if syndrome_every is None:
        return "Iceberg (1 mid-round + final only)"
    if syndrome_every == 1:
        return "Iceberg (checked every half-step)"
    return f"Iceberg (checked every {syndrome_every} half-steps)"


def run():
    dense_reference = load_latest("iceberg_qec_sweep_dense")
    single = load_latest("iceberg_qec")
    sweep = load_latest("iceberg_qec_sweep")
    zne = load_latest("h2_zne")

    if dense_reference is None and single is None and sweep is None:
        print("No persisted Iceberg run found. Run run_iceberg_sweep.py or "
              "run_iceberg_qec.py (with config.ICEBERG_RUN_ON_H2_EMULATOR = True) first.")
        return None
    if zne is None:
        print("No persisted ZNE comparison found (data/h2_zne_latest.json) -- "
              "run run_zne.py first, at the same h/J as the Iceberg run(s), "
              "to get a matching ED/raw-noisy/ZNE reference curve.")
        return None

    # Dense reference first, then single pilot, then latest sweep -- keeps
    # each technique's marker/color identity (plotting.ICEBERG_SERIES_STYLES)
    # stable across re-runs regardless of which datasets happen to exist.
    datasets = [d for d in (dense_reference, single, sweep) if d is not None]
    h_values = {d["results"]["h_field"] for d in datasets}
    if len(h_values) > 1:
        print(f"Iceberg datasets disagree on h/J ({h_values}) -- plotting each "
              f"against its own h/J would need separate figures; aborting.")
        return None
    h = h_values.pop()
    k_values = {d["results"]["k"] for d in datasets}
    k = k_values.pop() if len(k_values) == 1 else "/".join(str(kv) for kv in sorted(k_values))

    zne_data_for_h = zne["results"][f"{h}"]

    iceberg_series = []
    for d in datasets:
        r = d["results"]
        # syndrome_every=1 is the original single-point pilot's implicit
        # value, since it predates this field.
        label = _syndrome_label(r.get("syndrome_every", 1))
        iceberg_series.append({"label": label, "points": r["points"]})

    # A point with 0 kept shots (100% discard) has z/mzz == None -- drop it
    # from the observable overlay (discard-rate plot below still shows it,
    # since that plot only needs discard_rate, not z/mzz).
    comparison_series = [
        {"label": s["label"], "points": [p for p in s["points"] if p.get("z") is not None]}
        for s in iceberg_series
    ]

    fig1 = plot_iceberg_comparison(
        h, zne_data_for_h, comparison_series, save_dir=config.PLOT_SAVE_DIR, n=k,
        filename=f"h2_iceberg_comparison_N{k}_hJ{h:.2f}.png",
    )
    fig2 = plot_iceberg_discard_rate(
        iceberg_series, save_dir=config.PLOT_SAVE_DIR, n=k,
        filename=f"h2_iceberg_discard_rate_N{k}_hJ{h:.2f}.png",
    )
    return fig1, fig2


if __name__ == "__main__":
    run()
