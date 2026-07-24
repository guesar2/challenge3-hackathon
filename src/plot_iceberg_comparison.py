"""
plot_iceberg_comparison.py

Plots the most recently persisted Iceberg-encoded qnexus pilot sweep
(data/iceberg_qec_sweep_latest.json, written by run_iceberg_qec.py-style
runs) against this repo's existing ED/raw-noisy/ZNE-mitigated comparison
(data/h2_zne_latest.json, from run_zne.py) at the matching h/J -- see
plotting.plot_iceberg_comparison's docstring for why these two datasets
line up point-for-point (same N, same dt, same h/J).

Reads only what's already on disk -- doesn't touch qnexus, costs no
quota, and can be re-run any number of times (e.g. to restyle the
figures) without resubmitting anything.
"""
import config
from persistence import load_latest
from plotting import plot_iceberg_comparison, plot_iceberg_discard_rate


def run():
    iceberg = load_latest("iceberg_qec_sweep")
    zne = load_latest("h2_zne")
    if iceberg is None:
        print("No persisted Iceberg sweep found. Run run_iceberg_qec.py "
              "(with config.ICEBERG_RUN_ON_H2_EMULATOR = True) first.")
        return None
    if zne is None:
        print("No persisted ZNE comparison found (data/h2_zne_latest.json) -- "
              "run run_zne.py first, at the same h/J as the Iceberg sweep, "
              "to get a matching ED/raw-noisy/ZNE reference curve.")
        return None

    r = iceberg["results"]
    h = r["h_field"]
    k = r["k"]
    zne_data_for_h = zne["results"][f"{h}"]

    fig1 = plot_iceberg_comparison(
        h, zne_data_for_h, r["points"], save_dir=config.PLOT_SAVE_DIR, n=k,
        filename=f"h2_iceberg_comparison_N{k}_hJ{h:.2f}.png",
    )
    fig2 = plot_iceberg_discard_rate(
        r["points"], save_dir=config.PLOT_SAVE_DIR, n=k,
        filename=f"h2_iceberg_discard_rate_N{k}_hJ{h:.2f}.png",
    )
    return fig1, fig2


if __name__ == "__main__":
    run()
