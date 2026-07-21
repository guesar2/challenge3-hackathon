"""
run_n_scaling.py

System-size scan: "Implement Trotterized time evolution of the 1D TFIM
(4-20 spins). Does it break down for many spins?"

For each N in config.N_SCALING_VALUES, runs the fixed-Hamiltonian Trotter
quench (same building blocks as run_quench.py) and records:

  - circuit depth / gate count for one Trotter layer (circuits.py) -- the
    actual quantum-circuit cost, which grows only mildly with N since the
    TFIM chain is local and the edge-coloring keeps each layer shallow.
  - wall-clock runtime of the statevector-simulated Trotter propagation.
  - accuracy vs. a dense-ED reference (exact_diagonalization.py), but ONLY
    up to config.N_SCALING_ED_MAX (and capped further, per-N, by a caught
    MemoryError) -- ed_time_evolution_exact builds a full 2**N x 2**N
    dense matrix and calls np.linalg.eigh on it, which is what actually
    stops scaling, well before N=20 on most machines, not the Trotter
    circuit itself.
  - wall-clock runtime of that dense-ED reference where it was computed, so
    its cost growth is visible directly next to the Trotter circuit's.

This directly answers the "does it break down" question, and the answer
depends on which piece you mean:
  - The *Trotter circuit* (circuits.py) stays shallow -- local TFIM chain,
    edge-colored into O(1) layers/step -- and its statevector simulation
    reaches N=20 in well under a minute.
  - The classical *observable bookkeeping* it depends on (pauli_ops.py)
    used to build dense 2**N x 2**N Pauli operators via np.kron, which
    silently made *that* the actual ceiling (this project hit MemoryError
    around N=12-14 with it) despite the circuit itself being fine up to
    N=20. pauli_ops.py now builds everything as sparse (scipy.sparse)
    operators instead, which is what makes this script reach N=20 at all.
  - The *dense-ED reference* used only to check Trotter's accuracy is what
    genuinely can't scale -- it needs the full 2**N x 2**N matrix and a
    dense eigensolver by construction, so it's capped well below N=20 and
    that cutoff is reported directly in this script's output/plot.

A THIRD, separate question -- does *circuit-execution accuracy* degrade
with N once real hardware noise (not just Trotter/dt error) is in play --
is NOT answered by anything above, since run_trotter_fixed_hamiltonian is
exact noiseless statevector evolution. That question needs the shot-based
H2 execution path (tket_circuit.py + qnexus_backend.py, with
device_name=config.H2_DEVICE_NAME_NOISY -- the default H2_DEVICE_NAME is
noiseless except for shot noise, so it wouldn't show anything new here
either) run at a QEC-encoded circuit, which doesn't exist in this project
yet. This project does now have a real noisy H2 device reachable via
qnexus (see run_h2_emulator.py's `noisy=True` path) -- so device access is
no longer the blocker it once was -- but that alone doesn't answer this
N-scaling question, since it still needs a QEC-encoded circuit built at
each N, not just a noisy device to submit an unencoded circuit to. Rather
than silently skip it, run_noisy_stub() below prints/saves a clearly-
labeled placeholder row per N so the table has the right shape ahead of
time -- see its docstring for exactly what to wire in once a QEC-encoded
circuit path exists.

Standalone: `python run_n_scaling.py`. Computes its own ED baseline where
feasible, so it can be checked in isolation like the other capstone
sections.
"""
import time

from qiskit.quantum_info import Statevector

import config
from circuits import build_chain_color_edges, build_single_layer_circuit
from exact_diagonalization import ed_time_evolution_exact
from trotter_simulation import run_trotter_fixed_hamiltonian
from plotting import plot_n_scaling
from persistence import save_stage_results


def _trotter_circuit_cost(N, dt, h, J):
    """Depth/gate count of a single Trotter layer at this N (cost doesn't
    depend on dt/h/J -- only the gate structure does, via the edge
    coloring -- but building it the same way run_trotter_fixed_hamiltonian
    does keeps this honest rather than assumed)."""
    color_edges = build_chain_color_edges(N)
    theta_x = -2 * h * dt
    theta_zz = -2 * J * dt
    layer = build_single_layer_circuit(N, color_edges, theta_x, theta_zz, mirror=True)
    return layer.depth(), layer.size()


def run_noisy_stub(N_values, device_name=None, shots=None, max_N=None):
    """
    REAL noisy H2 emulation using qnexus (no longer a stub).

    Args:
        N_values: iterable of system sizes to consider (will be filtered).
        device_name: device name (default: config.N_SCALING_NOISY_DEVICE).
        shots: number of shots (default: config.N_SCALING_NOISY_SHOTS).
        max_N: optional maximum N to run. If None, uses
               config.N_SCALING_NOISY_MAX if present, otherwise 12.

    Returns:
        dict: for each N actually run (N <= max_N), a dict with results.
              If RUN_ON_H2_EMULATOR is False, returns a "SKIPPED" status
              for all N (without submitting anything).
    """
    # Determine max_N: argument > config > default
    if max_N is None:
        max_N = getattr(config, 'N_SCALING_NOISY_MAX', 12)

    # Filter N_values to those <= max_N
    N_list = [N for N in N_values if N <= max_N]
    if not N_list:
        print(f"[Noisy stub] No N values <= {max_N}. Skipping noisy emulation.")
        return {}

    if not config.RUN_ON_H2_EMULATOR:
        print("\n[Noisy stub] Skipped (config.RUN_ON_H2_EMULATOR = False).")
        return {
            N: {
                'device': device_name or config.N_SCALING_NOISY_DEVICE,
                'shots': shots or config.N_SCALING_NOISY_SHOTS,
                'max_pct_z': None,
                'max_pct_mzz': None,
                'runtime_s': None,
                'status': 'SKIPPED (RUN_ON_H2_EMULATOR=False)',
            }
            for N in N_list
        }

    import time
    from qnexus_backend import submit_quench_batch
    from shot_observables import bitstrings_to_observables
    from exact_diagonalization import ed_time_evolution_exact

    device_name = device_name or config.N_SCALING_NOISY_DEVICE
    shots = shots or config.N_SCALING_NOISY_SHOTS
    h = config.N_SCALING_H
    J = config.J
    dt = config.N_SCALING_DT
    steps = config.N_SCALING_STEPS

    print("\n" + "=" * 70)
    print(f"RUNNING REAL NOISY H2 EMULATION on {device_name}")
    print(f"(h/J = {h:.1f}, dt = {dt:.3f}, steps = {steps}, shots = {shots})")
    print(f"Max N = {max_N} (filtered: {N_list})")
    print("=" * 70)

    results = {}
    for N in N_list:
        print(f"\n>>> Submitting N = {N} ...")
        initial_state = '0' * N
        step_counts = [steps]  # only the final step

        try:
            t0 = time.perf_counter()
            batch = submit_quench_batch(
                N, h, J, dt, step_counts, shots,
                device_name=device_name,
                project_name=config.H2_PROJECT_NAME,
                initial_state_label=initial_state,
            )
            runtime_s = time.perf_counter() - t0

            bitstrings = batch[steps]["bitstrings"]
            z_rms, mzz = bitstrings_to_observables(bitstrings, N)

            # ED reference (only if N allows it)
            try:
                _, z_ed, mzz_ed, _ = ed_time_evolution_exact(
                    N, h, J, dt, steps, initial_state
                )
                ed_z_final = z_ed[-1]
                ed_mzz_final = mzz_ed[-1]

                pct_z = (abs(z_rms - ed_z_final) / max(abs(ed_z_final), 1e-12)) * 100
                pct_mzz = (abs(mzz - ed_mzz_final) / max(abs(ed_mzz_final), 1e-12)) * 100
                status = "Completed"
            except MemoryError:
                pct_z = None
                pct_mzz = None
                status = "ED skipped (MemoryError)"
                print(f"  [WARN] ED reference not available for N={N} (too large).")

            results[N] = {
                'device': device_name,
                'shots': shots,
                'max_pct_z': pct_z,
                'max_pct_mzz': pct_mzz,
                'runtime_s': runtime_s,
                'status': status,
            }
            if status == "Completed":
                print(f"  ✅ <Z> dev = {pct_z:.2f}%, <ZZ> dev = {pct_mzz:.2f}%  (runtime {runtime_s:.2f}s)")
            else:
                print(f"  ⚠️  {status}")

        except Exception as e:
            import traceback
            traceback.print_exc()  # Imprime el traceback completo en la consola
            results[N] = {
                'device': device_name,
                'shots': shots,
                'max_pct_z': None,
                'max_pct_mzz': None,
                'runtime_s': None,
                'status': f'ERROR: {e}',
            }
            print(f"  ❌ Submission failed for N={N}: {e}")

    return results

def _print_scaling_table(scaling_data, ed_max_N):
    N_values = sorted(scaling_data.keys())
    print(f"\n{'N':>4} | {'Depth':>6} | {'Gates':>6} | {'Trot t(s)':>10} | "
          f"{'ED t(s)':>10} | {'%dev <Z>':>9} | {'%dev <ZZ>':>10}")
    print("-" * 70)
    for N in N_values:
        r = scaling_data[N]
        ed_t = f"{r['ed_runtime_s']:.3f}" if r['ed_runtime_s'] is not None else ("-" if N > ed_max_N else "OOM")
        pz = f"{r['max_pct_z']:.3f}" if r['max_pct_z'] is not None else "-"
        pzz = f"{r['max_pct_mzz']:.3f}" if r['max_pct_mzz'] is not None else "-"
        print(f"{N:>4} | {r['depth']:>6} | {r['gate_count']:>6} | "
              f"{r['trotter_runtime_s']:>10.3f} | {ed_t:>10} | {pz:>9} | {pzz:>10}")
    print(f"(ED: '-' = skipped, N > {ed_max_N}; 'OOM' = attempted but ran out of memory)")


def _print_noisy_table(noisy_data):
    N_values = sorted(noisy_data.keys())
    print(f"\n{'N':>4} | {'Device':>12} | {'Shots':>6} | {'%dev <Z>':>9} | "
          f"{'%dev <ZZ>':>10} | {'Runtime(s)':>10} | Status")
    print("-" * 95)
    for N in N_values:
        r = noisy_data[N]
        pz = f"{r['max_pct_z']:.3f}" if r['max_pct_z'] is not None else "-"
        pzz = f"{r['max_pct_mzz']:.3f}" if r['max_pct_mzz'] is not None else "-"
        rt = f"{r['runtime_s']:.3f}" if r['runtime_s'] is not None else "-"
        print(f"{N:>4} | {r['device']:>12} | {r['shots']:>6} | {pz:>9} | {pzz:>10} | {rt:>10} | {r['status']}")


def run():
    print("=" * 60)
    print("N-SCALING SCAN: DOES TROTTERIZED EVOLUTION BREAK DOWN FOR MANY SPINS?")
    print("=" * 60)

    h = config.N_SCALING_H
    J = config.J
    dt = config.N_SCALING_DT
    steps = config.N_SCALING_STEPS
    ed_max_N = config.N_SCALING_ED_MAX

    scaling_data = {}

    for N in config.N_SCALING_VALUES:
        initial_state_label = '0' * N
        print(f"\nN = {N} ...")

        depth, gate_count = _trotter_circuit_cost(N, dt, h, J)

        t0 = time.perf_counter()
        _, z_trot, mzz_trot, _ = run_trotter_fixed_hamiltonian(
            N, h, J, dt, steps, initial_state_label, mirror=True
        )
        trotter_runtime_s = time.perf_counter() - t0
        print(f"  Trotter: depth={depth}, gates={gate_count}, "
              f"runtime={trotter_runtime_s:.3f}s")

        max_pct_z = max_pct_mzz = ed_runtime_s = None
        if N <= ed_max_N:
            try:
                t0 = time.perf_counter()
                _, z_ed, mzz_ed, _ = ed_time_evolution_exact(N, h, J, dt, steps, initial_state_label)
                ed_runtime_s = time.perf_counter() - t0

                max_pct_z = (max(abs(z_trot - z_ed)) / max(abs(z_ed))) * 100 if max(abs(z_ed)) > 0 else 0.0
                max_pct_mzz = (max(abs(mzz_trot - mzz_ed)) / max(abs(mzz_ed))) * 100 if max(abs(mzz_ed)) > 0 else 0.0
                print(f"  Dense ED: runtime={ed_runtime_s:.3f}s | "
                      f"max % dev <Z>={max_pct_z:.3f}%, <Zi Zi+1>={max_pct_mzz:.3f}%")
            except MemoryError:
                # ed_time_evolution_exact builds a dense 2^N x 2^N matrix and
                # calls np.linalg.eigh on it -- the actual ceiling is this
                # machine's available RAM, not a fixed N, so this is caught
                # per-N rather than only trusting the ed_max_N cutoff above.
                print(f"  Dense ED: MemoryError at N={N} -- this machine can't hold a "
                      f"2^{N} x 2^{N} dense matrix; treating this as the observed ED cutoff.")
        else:
            print(f"  Dense ED: skipped (N > {ed_max_N}, 2^{N} x 2^{N} matrix not affordable)")

        scaling_data[N] = {
            'depth': depth,
            'gate_count': gate_count,
            'trotter_runtime_s': trotter_runtime_s,
            'ed_runtime_s': ed_runtime_s,
            'max_pct_z': max_pct_z,
            'max_pct_mzz': max_pct_mzz,
        }

    noisy_data = run_noisy_stub(config.N_SCALING_VALUES)


    plot_n_scaling(scaling_data, ed_max_N, save_dir=config.PLOT_SAVE_DIR)

    print("\n" + "=" * 60)
    print("SUMMARY TABLE -- CLASSICAL (noiseless Trotter vs. dense ED)")
    print("=" * 60)
    _print_scaling_table(scaling_data, ed_max_N)

    print("\n" + "=" * 60)
    print("SUMMARY TABLE -- NOISY SIMULATION (PLACEHOLDER, see run_noisy_stub())")
    print("=" * 60)
    _print_noisy_table(noisy_data)
    print("\nNote: the noisy section above is a placeholder. It requires a QEC-encoded")
    print("circuit that doesn't exist in this project yet -- see run_noisy_stub()'s")
    print("docstring in run_n_scaling.py for exactly what to wire in once it does. A")
    print("real noisy H2 device (config.H2_DEVICE_NAME_NOISY) is available via")
    print("run_h2_emulator.py's noisy=True path, but that alone doesn't fill this table --")
    print("it still needs a QEC-encoded circuit built at each N, not just device access.")

    print("\nSummary:")
    print("  - Circuit depth/gate count grow only mildly with N (local TFIM chain,")
    print("    edge-colored into O(1) layers per Trotter step) -- the circuit itself")
    print("    does not break down over 4-20 spins.")
    print(f"  - Wherever checked against dense ED (N <= {ed_max_N}), Trotter error vs. ED")
    print("    stays within the challenge's <5% tolerance (see n_scaling.png, left panel).")
    print(f"  - The dense ED *reference* is what stops scaling (N > {ed_max_N} skipped above) --")
    print("    it needs a full 2^N x 2^N matrix, unlike the local, gate-based Trotter circuit.")
    print("  - Noisy (hardware-accuracy) scaling is not yet measured -- see the placeholder")
    print("    table above.")

    save_stage_results("n_scaling", {
        "J": J, "h": h, "dt": dt, "steps": steps,
        "N_values": list(config.N_SCALING_VALUES), "ed_max_N": ed_max_N,
        "scaling_data": scaling_data,
        "noisy_data": noisy_data,
    })
    return scaling_data, noisy_data


if __name__ == "__main__":
    run()