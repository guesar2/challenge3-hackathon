"""
fh_main.py  --  single entry point for the 2D Fermi-Hubbard deliverable.

Runs the whole pipeline from a clean environment and reproduces every figure and
headline number:

  0. self-checks         (JW consistency, free-fermion cross-check, half-filling)
  1. circuit checks      (PauliExpBox angle, bit order, circuit vs Trotter vs ED)
  2. ED ground state     -> fig1_groundstate_vs_U.png
  3. N-scaling           -> fig5_n_scaling.png
  4. dt-convergence      -> fig4_dt_convergence.png
  5. quench dynamics     -> fig2_quench_dynamics.png, fig3_density_heatmaps.png,
                            fig7_ed_vs_circuit.png
  6. H2 emulator run     -> fig8_h2_run.png  (free local sampler by default)
  7. VQE (optional)      -> fig6_vqe.png

Usage:
    python fh_main.py               # everything
    python fh_main.py --quick       # skip VQE and the multi-size scans
    python fh_main.py --no-vqe      # everything except VQE
"""
from __future__ import annotations

import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip VQE and multi-size scans")
    ap.add_argument("--no-vqe", action="store_true", help="skip VQE stage")
    ap.add_argument("--no-shots", action="store_true", help="skip emulator-shot trajectories")
    args = ap.parse_args()

    print("=" * 70)
    print("2D FERMI-HUBBARD  --  Quantathon CR 2026 Challenge 3 (optional model)")
    print("=" * 70)

    print("\n[0/7] Self-checks (Jordan-Wigner correctness)")
    import run_fh_selfcheck
    run_fh_selfcheck.check_consistency()
    run_fh_selfcheck.check_free_fermions()
    run_fh_selfcheck.check_half_filling_mu()

    print("\n[1/7] Circuit checks (quantum path)")
    import run_fh_circuit_check
    run_fh_circuit_check.check_angle_convention()
    run_fh_circuit_check.check_bit_order()
    run_fh_circuit_check.check_full_path()

    print("\n[2/7] ED ground state vs U/t")
    import run_fh_ed
    run_fh_ed.run()

    if not args.quick:
        print("\n[3/7] N-scaling and qubit budget")
        import run_fh_n_scaling
        run_fh_n_scaling.run()

    print("\n[4/7] Trotter dt-convergence")
    import run_fh_dt_convergence
    run_fh_dt_convergence.run()

    print("\n[5/7] Quench dynamics (ED vs Trotter vs shots)")
    import run_fh_dynamics
    run_fh_dynamics.run(with_shots=not args.no_shots)

    print("\n[6/7] H2 emulator run")
    import run_fh_h2
    run_fh_h2.run()

    if not (args.quick or args.no_vqe):
        print("\n[7/7] VQE (optional extension)")
        import run_fh_vqe
        run_fh_vqe.run()
    else:
        print("\n[7/7] VQE skipped")

    print("\nDone. Figures are in ./figures_fh/, cached data in ./data_fh/.")


if __name__ == "__main__":
    sys.exit(main())