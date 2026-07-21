"""
run_adiabatic.py

Capstone section 2/5: adiabatic Trotter sweep -- the "quantum implementation"
run locally as a noiseless Qiskit statevector simulation. Ramps h from
config.H_INIT to each config.H_VALUES target, compares the final state
against the ED ground state, and saves adiabatic_convergence.png /
phase_transition.png.

Standalone: `python run_adiabatic.py`. Computes its own ED baseline for
comparison rather than depending on run_ed.py having been run first, so this
section can be checked in isolation.
"""
import config
from exact_diagonalization import ed_baseline
from sweep_schedule import run_adiabatic_simulation
from reporting import print_comparison_table
from plotting import plot_adiabatic_convergence, plot_phase_transition
from persistence import save_stage_results


def run():
    print("=" * 60)
    print("STAGE 2/5: ADIABATIC TROTTER SWEEP (LOCAL STATEVECTOR)")
    print("=" * 60)

    ed_results = ed_baseline(config.N, config.H_VALUES, J=config.J)

    trotter_data = run_adiabatic_simulation(
        config.N, config.J, config.H_VALUES,
        config.ADIABATIC_DT, config.ADIABATIC_RATE_REF, h_init=config.H_INIT,
        hold_steps=config.ADIABATIC_HOLD_STEPS,
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

    save_stage_results("adiabatic", {
        "N": config.N, "J": config.J, "H_INIT": config.H_INIT,
        "ADIABATIC_DT": config.ADIABATIC_DT, "ADIABATIC_RATE_REF": config.ADIABATIC_RATE_REF,
        "ed_results": ed_results, "trotter_data": trotter_data,
    })
    return ed_results, trotter_data


if __name__ == "__main__":
    run()
