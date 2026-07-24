"""
fh_main.py  --  entry point for the optional 2D Fermi-Hubbard extension.

This is a separate, self-contained package (independent of the repo's main
`main.py` / TFIM 1D pipeline). Runs the whole Fermi-Hubbard pipeline from a
clean environment and reproduces every figure and headline number:

  0. self-checks        (JW consistency, free-fermion cross-check,
                         sector engine vs full-space engine)
  1. ED ground state    -> fig1_groundstate_vs_U.png      (2x2 and 3x4, U=0,1,4,8)
  2. dt-convergence     -> fig4_dt_convergence.png        (second order only)
  3. quench dynamics    -> fig2_quench_dynamics.png       (ED vs Trotter vs
                         noiseless shots vs raw noisy shots vs ZNE-mitigated)
  4. density heatmap    -> fig3_density_heatmaps.png      (3x4 stripe quench)
  5. H2 emulator run    -> fig8_h2_run.png                (free local sampler by default)
  6. VQE (optional)     -> fig6_vqe.png
  7. summary tables     -> figures_fh/summary_tables.txt  (one per U)

Usage:
    python fh_main.py               # everything
    python fh_main.py --quick       # skip VQE and the summary tables
    python fh_main.py --no-vqe      # everything except VQE and the tables
    python fh_main.py --no-shots    # skip emulator-shot trajectories
    python fh_main.py --no-noise    # skip the noisy + ZNE curves in fig2
    python fh_main.py --no-heatmap  # skip the 3x4 stage (the slow one)
"""
from __future__ import annotations

import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip VQE and the summary tables")
    ap.add_argument("--no-vqe", action="store_true", help="skip VQE and the summary tables")
    ap.add_argument("--no-shots", action="store_true", help="skip emulator-shot trajectories")
    ap.add_argument("--no-noise", action="store_true",
                    help="skip the noisy + ZNE curves in the quench figure")
    ap.add_argument("--no-heatmap", action="store_true", help="skip the 3x4 heatmap stage")
    args = ap.parse_args()

    print("=" * 70)
    print("2D FERMI-HUBBARD  --  Quantathon CR 2026 Challenge 3 (optional model)")
    print("periodic boundaries, half filling, second-order Trotter")
    print("=" * 70)

    print("\n[0/7] Self-checks")
    import run_fh_selfcheck
    run_fh_selfcheck.check_consistency()
    run_fh_selfcheck.check_free_fermions()
    run_fh_selfcheck.check_sector_vs_full_space()
    run_fh_selfcheck.check_sector_sizes()

    print("\n[1/7] ED ground state vs U/t  (2x2 and 3x4)")
    import run_fh_ed
    run_fh_ed.run()

    print("\n[2/7] Trotter dt-convergence (second order)")
    import run_fh_dt_convergence
    run_fh_dt_convergence.run()

    print("\n[3/7] Quench dynamics (ED vs Trotter vs noiseless vs noisy vs ZNE)")
    import run_fh_dynamics
    run_fh_dynamics.run(with_shots=not args.no_shots,
                        with_noise=(None if not args.no_noise else False))

    if not args.no_heatmap:
        print("\n[4/7] Per-site density heatmap (3x4 stripe quench)")
        import run_fh_heatmap
        run_fh_heatmap.run()
    else:
        print("\n[4/7] Density heatmap skipped")

    print("\n[5/7] H2 emulator run")
    import run_fh_h2
    run_fh_h2.run()

    if not (args.quick or args.no_vqe):
        print("\n[6/7] VQE (optional extension)")
        import run_fh_vqe
        run_fh_vqe.run()

        print("\n[7/7] Summary tables (one per U)")
        import fh_tables
        fh_tables.run()
    else:
        print("\n[6/7] VQE skipped")
        print("[7/7] Summary tables skipped (they need the VQE row)")

    print("\nDone. Figures are in ./figures_fh/, cached data in ./data_fh/.")


if __name__ == "__main__":
    sys.exit(main())
