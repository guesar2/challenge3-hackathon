"""
ftim_main.py

Runs every section of the TFIM pipeline in sequence: ED baseline, adiabatic
Trotter sweep, fixed-Hamiltonian quench, Trotter dt-convergence analysis,
N-scaling scan (4-20 spins), classical-vs-quantum cost comparison, and
(only if config.RUN_ON_H2_EMULATOR is set) the Quantinuum H2 emulator run,
Zero-Noise Extrapolation, an Iceberg QEC depth sweep (separately gated by
config.ICEBERG_RUN_ON_H2_EMULATOR, off by default), and the Iceberg-vs-ED/ZNE
comparison plot.

Deliberately NOT run here: run_noise_scaling.py's noise-vs-N/depth
characterization (N up to 26) -- the single most expensive qnexus stage in
the whole pipeline (two H2 submissions per N). Run it separately with
`python run_noise_scaling.py` if you want that scan; main.py skips it to
keep the full pipeline's runtime/quota cost down.

Each section is also independently runnable -- see run_ed.py,
run_adiabatic.py, run_quench.py, run_dt_convergence.py, run_n_scaling.py,
run_quantum_advantage.py, run_h2_emulator.py, run_zne.py,
run_iceberg_qec.py, run_iceberg_sweep.py, run_noise_scaling.py,
plot_iceberg_comparison.py -- so a single section can be checked without
paying the cost (in time or, for the H2/ZNE/Iceberg stages, quota) of
running everything else.
"""
import config
import run_ed
import run_adiabatic
import run_quench
import run_dt_convergence
import run_n_scaling
import run_quantum_advantage
import run_h2_emulator
import run_zne
from run_iceberg_sweep import run_iceberg_sweep
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
    run_iceberg_sweep(
        k=config.ICEBERG_SWEEP_K, h_field=config.ICEBERG_SWEEP_H, J=config.J,
        dt=config.ICEBERG_SWEEP_DT, step_shot_pairs=config.ICEBERG_SWEEP_STEP_SHOT_PAIRS,
        syndrome_every=config.ICEBERG_SYNDROME_EVERY,
    )
    plot_iceberg_comparison.run()

    print("\n--- All stages complete. ---")


if __name__ == "__main__":
    main()