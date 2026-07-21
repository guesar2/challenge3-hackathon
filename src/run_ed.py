"""
run_ed.py

Capstone section 1/5: exact diagonalization (ED) baseline for the TFIM.
Ground-state observables (<Z>, <X>, <Zi Zi+1>, energy/N) across
config.H_VALUES at config.N -- plus the classical scaling comparison
(Comparación y escalado: 2+ problem sizes):

- Observable scaling: ed_baseline run at every N in config.ED_EXTRA_N_VALUES
  (includes N=8, which the challenge doc's "Línea base clásica" section
  asks for specifically at h/J in {0.5, 1.0, 2.0}), plotted together.
- Runtime scaling: wall-clock cost of the Hamiltonian build + eigsh vs. N
  at config.N_RUNTIME_SCALING_VALUES, extrapolated (clearly marked, not
  actually run) out to config.N_RUNTIME_SCALING_EXTRAPOLATE_TO -- the
  "honest extrapolation: state where classical methods still win" evidence.

Standalone: `python run_ed.py` (from src/, or with src/ on sys.path).
Purely classical -- no Trotter simulation, no qnexus -- so it's the fastest
section to check in isolation. The runtime-scaling benchmark adds real wall
time to this stage (~2 minutes on the machine this was tuned on, dominated
by N=12 -- see config.py's N_RUNTIME_SCALING_VALUES comment) since it's
measuring actual cost, not simulating a result.
"""
import time

import config
from exact_diagonalization import ed_baseline
from persistence import save_stage_results
from plotting import plot_ed_scaling, plot_ed_runtime_scaling


def run():
    print("=" * 60)
    print("STAGE 1/5: EXACT DIAGONALIZATION (ED) BASELINE")
    print("=" * 60)
    ed_results = ed_baseline(config.N, config.H_VALUES, J=config.J)
    save_stage_results("ed", {"N": config.N, "J": config.J, "ed_results": ed_results})

    # Observable scaling across 2+ sizes.
    ed_results_by_N = {config.N: ed_results}
    for N in config.ED_EXTRA_N_VALUES:
        if N == config.N:
            continue
        ed_results_by_N[N] = ed_baseline(N, config.H_VALUES, J=config.J)
    save_stage_results("ed_scaling", {
        "N_values": sorted(ed_results_by_N),
        "J": config.J,
        "ed_results_by_N": ed_results_by_N,
    })
    plot_ed_scaling(config.H_VALUES, ed_results_by_N, save_dir=config.PLOT_SAVE_DIR)

    # Wall-clock runtime scaling -- the classical-cost side of the scaling
    # comparison, and the "honest extrapolation" evidence: measures actual
    # time on this machine, only extrapolating (dashed, clearly labeled) past
    # what was actually run.
    print(f"\nMeasuring ED wall-clock cost vs. N (h/J=1.0, Hamiltonian build + eigsh)...")
    timings = []
    for N in config.N_RUNTIME_SCALING_VALUES:
        t0 = time.perf_counter()
        ed_baseline(N, (1.0,), J=config.J, verbose=False)
        dt = time.perf_counter() - t0
        timings.append({"N": N, "dim": 2 ** N, "time_s": dt})
        print(f"  N={N:2d} (dim=2^{N}={2 ** N:6d}): {dt:8.3f}s")
    save_stage_results("ed_runtime_scaling", {"timings": timings})
    plot_ed_runtime_scaling(
        timings, extrapolate_to=config.N_RUNTIME_SCALING_EXTRAPOLATE_TO,
        save_dir=config.PLOT_SAVE_DIR,
    )

    return ed_results


if __name__ == "__main__":
    run()