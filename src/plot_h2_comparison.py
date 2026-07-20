"""
plot_h2_comparison.py

Plots the most recently persisted Quantinuum H2 emulator run
(data/h2_emulator_latest.json + data/h2_emulator_raw_latest.json, written by
run_h2_emulator.py) against the ED ground state, with shot-noise error bars
bootstrapped from the raw bitstrings.

Reads only what's already on disk -- doesn't touch qnexus, costs no quota,
and can be re-run any number of times (e.g. to restyle the figure) without
resubmitting anything.
"""
import config
from persistence import load_latest
from plotting import plot_h2_vs_ed
from qnexus_backend import assemble_h2_vs_ed


def run():
    processed = load_latest("h2_emulator")
    raw = load_latest("h2_emulator_raw")
    if processed is None or raw is None:
        print("No persisted H2 emulator results found. Run run_h2_emulator.py "
              "first (with config.RUN_ON_H2_EMULATOR = True) to produce "
              "data/h2_emulator_latest.json.")
        return None

    plot_data = assemble_h2_vs_ed(processed["results"], raw["results"])
    fig = plot_h2_vs_ed(
        plot_data["h_values"], plot_data["z_h2"], plot_data["z_err"],
        plot_data["mzz_h2"], plot_data["mzz_err"], plot_data["z_ed"], plot_data["mzz_ed"],
        save_dir=config.PLOT_SAVE_DIR, saved_at=processed["saved_at"],
    )
    return fig


if __name__ == "__main__":
    run()
