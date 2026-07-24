"""
ftim_main.py

Runs every section of the TFIM pipeline in sequence: ED baseline, adiabatic
Trotter sweep, fixed-Hamiltonian quench, Trotter dt-convergence analysis,
N-scaling scan (4-20 spins), classical-vs-quantum cost comparison, and
(only if config.RUN_ON_H2_EMULATOR is set) the Quantinuum H2 emulator run,
Zero-Noise Extrapolation, an Iceberg QEC pilot run (separately gated by
config.ICEBERG_RUN_ON_H2_EMULATOR, off by default), the noise-vs-N/depth
characterization scan, and the Iceberg-vs-ED/ZNE comparison plot.

Each section is also independently runnable -- see run_ed.py,
run_adiabatic.py, run_quench.py, run_dt_convergence.py, run_n_scaling.py,
run_quantum_advantage.py, run_h2_emulator.py, run_zne.py,
run_iceberg_qec.py, run_noise_scaling.py, plot_iceberg_comparison.py -- so
a single section can be checked without paying the cost (in time or, for
the H2/ZNE/Iceberg/noise-scaling stages, quota) of running everything else.
"""
import run_ed
import run_adiabatic
import run_quench
import run_dt_convergence
import run_n_scaling
import run_quantum_advantage
import run_h2_emulator
import run_zne
import run_iceberg_qec
import run_noise_scaling
import plot_iceberg_comparison


def main():
    print("=" * 60)
    print("CHALLENGE 3: TFIM -- FULL PIPELINE (ED + LOCAL TROTTER + H2)")
    print("=" * 60)

    run_ed.run()
    run_adiabatic.run()
    run_quench.run()
    run_dt_convergence.run()
    run_n_scaling.run()
    run_quantum_advantage.run()
    run_h2_emulator.run()
    run_zne.run()
    run_iceberg_qec.run_iceberg_noisy()
    run_noise_scaling.run_noise_scaling()
    plot_iceberg_comparison.run()

    print("\n--- All stages complete. ---")


if __name__ == "__main__":
    main()