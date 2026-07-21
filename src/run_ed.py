"""
run_ed.py

Capstone section 1/5: exact diagonalization (ED) baseline for the TFIM.
Ground-state observables (<Z>, <X>, <Zi Zi+1>, energy/N) across
config.H_VALUES at config.N -- and also at each N in config.ED_EXTRA_N_VALUES
(e.g. N=8), so the baseline table is available at more than one system size
for comparison, printed the same way for each -- via sparse eigsh.

Standalone: `python run_ed.py` (from src/, or with src/ on sys.path).
Purely classical -- no Trotter simulation, no qnexus -- so it's the fastest
section to check in isolation.
"""
import config
from exact_diagonalization import ed_baseline
from persistence import save_stage_results


def run():
    print("=" * 60)
    print("STAGE 1/5: EXACT DIAGONALIZATION (ED) BASELINE")
    print("=" * 60)

    # config.N first, then any extra N values (skipping duplicates) so the
    # console output always shows the primary N used by every other stage,
    # followed by the extra comparison sizes.
    N_values = [config.N] + [n for n in config.ED_EXTRA_N_VALUES if n != config.N]

    ed_results_by_N = {}
    for N in N_values:
        ed_results_by_N[N] = ed_baseline(N, config.H_VALUES, J=config.J)

    save_stage_results("ed", {
        "N_values": N_values, "J": config.J, "ed_results_by_N": ed_results_by_N,
    })
    return ed_results_by_N


if __name__ == "__main__":
    run()