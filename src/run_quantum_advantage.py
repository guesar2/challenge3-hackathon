"""
run_quantum_advantage.py

Answers the presentation's "do we observe a quantum advantage?" question
with a figure instead of an assertion: overlays classical ED's measured
wall-clock cost vs. N (same measurement as run_ed.py's runtime-scaling
stage, log-linear-extrapolated past what was actually run) against the
Trotter circuit's own cost -- gate count / depth vs. N -- which is cheap
to compute up to N=20 since it only requires building the (unexecuted)
circuit, not simulating or diagonalizing anything.

Answer this project's data supports: no quantum advantage is observed, and
none is expected to appear at these sizes -- classical ED stays under a
tenth of a second through N=12 (this machine, this implementation) while
the Trotter circuit's gate count/depth grows only mildly with N (the TFIM
chain is local and edge-colored into O(1) layers/step). The two curves
aren't racing toward a crossover: ED's cost is what actually explodes
(log-linear fit crosses 1 hour/1 day well beyond N=20), and the "quantum
side" here is a shallow, honestly-cheap circuit that current devices could
run all day without hitting the ~50-gate noise-dominated threshold the
hackathon brief warns about (docs/hackathon.pdf, "Honest limitations" /
"Limitaciones honestas") -- consistent with the brief's own statement that
near-term quantum advantage for TFIM is not established at these N.

Standalone: `python run_quantum_advantage.py`. Purely classical -- no
qnexus, no quota cost. Reuses config.N_RUNTIME_SCALING_VALUES (the same
grid run_ed.py measures) for the ED timings and config.N_SCALING_VALUES
(4..20) for the circuit cost, so the two curves are drawn from the same
grids used elsewhere in this project rather than a new ad hoc choice.
"""
import os
import time

import config
from circuits import build_chain_color_edges, build_single_layer_circuit
from exact_diagonalization import ed_baseline
from persistence import save_stage_results
from plotting import plot_quantum_advantage_scaling

# Always resolve figures/data relative to the repo root, regardless of the
# cwd this script is invoked from -- same pitfall run_noise_scaling.py and
# run_zne.py guard against (a plain relative PLOT_SAVE_DIR silently writes
# to src/figures/ instead of the top-level figures/ when run from src/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_REPO_ROOT)


def _trotter_circuit_cost(N, dt, h, J):
    """Depth/gate count of a single Trotter layer at this N -- mirrors
    run_n_scaling._trotter_circuit_cost (cost doesn't depend on dt/h/J,
    only the gate structure via the edge coloring, but built the same way
    the real circuit is to keep this honest rather than assumed)."""
    color_edges = build_chain_color_edges(N)
    theta_x = -2 * h * dt
    theta_zz = -2 * J * dt
    layer = build_single_layer_circuit(N, color_edges, theta_x, theta_zz, mirror=True)
    return layer.depth(), layer.size()


def run():
    print("=" * 60)
    print("QUANTUM ADVANTAGE SCALING: classical ED cost vs. Trotter circuit cost")
    print("=" * 60)

    print(f"\nMeasuring ED wall-clock cost vs. N (h/J=1.0, Hamiltonian build + eigsh)...")
    ed_timings = []
    for N in config.N_RUNTIME_SCALING_VALUES:
        t0 = time.perf_counter()
        ed_baseline(N, (1.0,), J=config.J, verbose=False)
        dt = time.perf_counter() - t0
        ed_timings.append({"N": N, "dim": 2 ** N, "time_s": dt})
        print(f"  N={N:2d} (dim=2^{N}={2 ** N:6d}): {dt:8.4f}s")

    print(f"\nComputing Trotter circuit cost vs. N (one layer, gate count independent of h/dt)...")
    circuit_costs = []
    for N in config.N_SCALING_VALUES:
        depth, gate_count = _trotter_circuit_cost(N, config.QUENCH_DT, 1.0, config.J)
        circuit_costs.append({"N": N, "depth": depth, "gate_count": gate_count})
        print(f"  N={N:2d}: depth={depth:3d}, gate_count={gate_count:4d}")

    save_stage_results("quantum_advantage_scaling", {
        "ed_timings": ed_timings,
        "circuit_costs": circuit_costs,
        "extrapolate_to": config.N_RUNTIME_SCALING_EXTRAPOLATE_TO,
    })

    plot_quantum_advantage_scaling(
        ed_timings, circuit_costs,
        extrapolate_to=config.N_RUNTIME_SCALING_EXTRAPOLATE_TO,
        save_dir=config.PLOT_SAVE_DIR,
    )

    return ed_timings, circuit_costs


if __name__ == "__main__":
    run()
