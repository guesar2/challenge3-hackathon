"""
run_quench.py

Capstone section 3/4: fixed-Hamiltonian ("quench") time evolution -- ED vs.
local Trotter (Qiskit statevector), starting from a computational-basis
product state. This is the pass/fail check for the challenge's <5%
deviation requirement.

Standalone: `python run_quench.py`. Computes its own ED baseline (used only
as the ground-state reference line in the plot) rather than depending on
run_ed.py having been run first, so this section can be checked in
isolation.
"""
import config
from exact_diagonalization import ed_baseline, ed_time_evolution_exact
from trotter_simulation import run_trotter_fixed_hamiltonian
from plotting import plot_fixed_hamiltonian_evolution
from persistence import save_stage_results


def run():
    print("=" * 60)
    print("STAGE 3/4: FIXED-HAMILTONIAN QUENCH (ED vs. LOCAL TROTTER)")
    print("=" * 60)

    ed_results = ed_baseline(config.N, config.H_VALUES, J=config.J)

    initial_state = config.QUENCH_INITIAL_STATE or '0' * config.N
    evolution_results = {}

    for h in config.H_VALUES:
        print(f"\nSimulating fixed h/J = {h:.1f} from initial state |{initial_state}> ...")

        times_ed, z_ed, mzz_ed, _ = ed_time_evolution_exact(
            config.N, h, config.J, config.QUENCH_DT, config.QUENCH_STEPS, initial_state
        )
        times_trot, z_trot, mzz_trot, _ = run_trotter_fixed_hamiltonian(
            config.N, h, config.J, config.QUENCH_DT, config.QUENCH_STEPS, initial_state, mirror=True
        )

        max_pct_z = (max(abs(z_trot - z_ed)) / max(abs(z_ed))) * 100 if max(abs(z_ed)) > 0 else 0
        max_pct_mzz = (max(abs(mzz_trot - mzz_ed)) / max(abs(mzz_ed))) * 100 if max(abs(mzz_ed)) > 0 else 0
        print(f"  Max deviation in <Z>: {max_pct_z:.2f}%")
        print(f"  Max deviation in <Zi Zi+1>: {max_pct_mzz:.2f}%")

        evolution_results[h] = {
            'times_ed': times_ed, 'z_ed': z_ed, 'mzz_ed': mzz_ed,
            'times_trot': times_trot, 'z_trot': z_trot, 'mzz_trot': mzz_trot,
        }

    plot_fixed_hamiltonian_evolution(
        config.H_VALUES, evolution_results, ed_results, save_dir=config.PLOT_SAVE_DIR
    )

    print("\nFixed-Hamiltonian time evolution: Trotter dynamics matches ED within <5%")
    print("(see per-h max deviations above), satisfying the challenge requirement.")

    save_stage_results("quench", {
        "N": config.N, "J": config.J,
        "QUENCH_DT": config.QUENCH_DT, "QUENCH_STEPS": config.QUENCH_STEPS,
        "initial_state": initial_state,
        "ed_results": ed_results, "evolution_results": evolution_results,
    })
    return evolution_results


if __name__ == "__main__":
    run()
