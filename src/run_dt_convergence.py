"""
run_dt_convergence.py

Capstone section 4/5: Trotter step-size convergence analysis. For each
h/J in config.H_VALUES, runs the fixed-Hamiltonian quench (same setup as
run_quench.py) at a sequence of Trotter step sizes obtained by
halving/doubling config.QUENCH_DT, holding the total evolution time
T = dt * steps fixed, and compares each run's max % deviation from ED
against dt.

The symmetrized Rx(theta/2)-Rzz(theta)-Rx(theta/2) layer (circuits.py) is
a 2nd-order Trotter-Suzuki step, so error should shrink roughly as dt^2 --
halving dt should cut the error ~4x. This is the "Errores comunes"
dt-halving check the challenge doc asks for, previously only present as
dead code in ed_figures.py (never wired into any runnable script) --
this script is the live equivalent.

Standalone: `python run_dt_convergence.py`. Computes its own ED baseline
rather than depending on run_ed.py/run_quench.py, so it can be checked in
isolation like the other capstone sections.
"""
import config
from exact_diagonalization import ed_baseline, ed_time_evolution_exact
from trotter_simulation import run_trotter_fixed_hamiltonian
from plotting import plot_dt_convergence
from persistence import save_stage_results

# Halvings/doublings of the production QUENCH_DT, holding total time
# T = QUENCH_DT * QUENCH_STEPS fixed so every dt evolves to the same
# physical time and only the Trotter step resolution changes.
DT_FACTORS = (4, 2, 1, 0.5, 0.25)


def run():
    print("=" * 60)
    print("TROTTER dt-CONVERGENCE (halving dt at fixed total evolution time)")
    print("=" * 60)

    ed_results = ed_baseline(config.N, config.H_VALUES, J=config.J)

    total_time = config.QUENCH_DT * config.QUENCH_STEPS
    dt_values = sorted(config.QUENCH_DT * factor for factor in DT_FACTORS)
    initial_state = config.QUENCH_INITIAL_STATE or '0' * config.N

    error_data = {}
    for h in config.H_VALUES:
        print(f"\nh/J = {h:.1f} (total evolution time T = {total_time:.2f}):")
        dts, max_pct_z, max_pct_mzz = [], [], []

        for dt in dt_values:
            steps = round(total_time / dt)

            _, z_ed, mzz_ed, _ = ed_time_evolution_exact(
                config.N, h, config.J, dt, steps, initial_state
            )
            _, z_trot, mzz_trot, _ = run_trotter_fixed_hamiltonian(
                config.N, h, config.J, dt, steps, initial_state, mirror=True
            )

            pct_z = (max(abs(z_trot - z_ed)) / max(abs(z_ed))) * 100 if max(abs(z_ed)) > 0 else 0
            pct_mzz = (max(abs(mzz_trot - mzz_ed)) / max(abs(mzz_ed))) * 100 if max(abs(mzz_ed)) > 0 else 0
            print(f"  dt={dt:.4f} (steps={steps}): max % dev <Z>={pct_z:.3f}%, "
                  f"<Zi Zi+1>={pct_mzz:.3f}%")

            dts.append(dt)
            max_pct_z.append(pct_z)
            max_pct_mzz.append(pct_mzz)

        # Report the halving ratio between the two finest dt's as a sanity
        # check on the expected O(dt^2) scaling (ratio ~= 4 for a clean
        # 2nd-order Trotter step; shot/observable-magnitude noise floors
        # can flatten this at very small dt).
        if max_pct_z[0] > 0:
            print(f"  error ratio (dt={dts[0]:.4f} -> dt={dts[1]:.4f}), <Z>: "
                  f"{max_pct_z[1] / max_pct_z[0]:.2f}x (expect ~4x for O(dt^2))")

        error_data[h] = {'dt_values': dts, 'max_pct_z': max_pct_z, 'max_pct_mzz': max_pct_mzz}

    plot_dt_convergence(config.H_VALUES, dt_values, error_data, save_dir=config.PLOT_SAVE_DIR)

    save_stage_results("dt_convergence", {
        "N": config.N, "J": config.J, "total_time": total_time,
        "dt_values": dt_values, "initial_state": initial_state,
        "error_data": error_data,
    })
    return error_data


if __name__ == "__main__":
    run()
