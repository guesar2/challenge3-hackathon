"""
ftim_main.py

Runs every capstone section of the TFIM pipeline in sequence: ED baseline,
adiabatic Trotter sweep, fixed-Hamiltonian quench, Trotter dt-convergence
analysis, N-scaling scan (4-20 spins), and (only if
config.RUN_ON_H2_EMULATOR is set) the Quantinuum H2 emulator run.

Each section is also independently runnable -- see run_ed.py,
run_adiabatic.py, run_quench.py, run_dt_convergence.py, run_n_scaling.py,
run_h2_emulator.py -- so a single section can be checked without paying
the cost (in time or, for the H2 stage, quota) of running everything else.
"""
import run_ed
import run_adiabatic
import run_quench
import run_dt_convergence
import run_n_scaling
import run_h2_emulator


def main():
    print("=" * 60)
    print("CHALLENGE 3: TFIM -- FULL PIPELINE (ED + LOCAL TROTTER + H2)")
    print("=" * 60)

    run_ed.run()
    run_adiabatic.run()
    run_quench.run()
    run_dt_convergence.run()
    run_n_scaling.run()
    run_h2_emulator.run()

    print("\n--- All stages complete. ---")


if __name__ == "__main__":
    main()