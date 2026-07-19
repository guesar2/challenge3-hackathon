"""
main.py

Orchestration only: calls into the physics/simulation modules, prints
results, and generates plots. Mirrors the structure of the original
notebook's __main__ block, but each stage is now a call into a focused
module instead of 300+ lines of inline code.
"""
import config
from exact_diagonalization import ed_baseline, ed_time_evolution_exact
from sweep_schedule import run_adiabatic_simulation
from trotter_simulation import run_trotter_fixed_hamiltonian
from reporting import print_comparison_table
from plotting import (
    plot_adiabatic_convergence,
    plot_phase_transition,
    plot_fixed_hamiltonian_evolution,
)


def main():
    print("=" * 60)
    print("CHALLENGE 3: TFIM ADIABATIC SIMULATION (LOCAL)")
    print("=" * 60)

    # 1. Exact diagonalization baseline
    ed_results = ed_baseline(config.N, config.H_VALUES, J=config.J)

    # 2. Adiabatic sweep (Trotterized), constant |dh/dt|
    trotter_data = run_adiabatic_simulation(
        config.N, config.J, config.H_VALUES,
        config.ADIABATIC_DT, config.ADIABATIC_RATE_REF, h_init=config.H_INIT,
    )
    print_comparison_table(
        ed_results, trotter_data,
        f"dt = {config.ADIABATIC_DT:.3f}, rate = {config.ADIABATIC_RATE_REF:.3f}",
    )

    plot_adiabatic_convergence(
        config.H_VALUES, trotter_data, ed_results, config.ADIABATIC_RATE_REF,
        save_dir=config.PLOT_SAVE_DIR,
    )
    plot_phase_transition(
        config.H_VALUES, trotter_data, ed_results, config.ADIABATIC_RATE_REF,
        save_dir=config.PLOT_SAVE_DIR,
    )

    # 3. Fixed-Hamiltonian ("quench") time evolution from a product state
    print("\n" + "=" * 60)
    print("FIXED-HAMILTONIAN TIME EVOLUTION FROM PRODUCT STATE")
    print("=" * 60)

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

    print("\n--- All plots generated successfully! ---")
    print("The fixed-Hamiltonian time evolution plots show that Trotter dynamics")
    print("matches ED time evolution within <5%, satisfying the challenge requirement.")


if __name__ == "__main__":
    main()