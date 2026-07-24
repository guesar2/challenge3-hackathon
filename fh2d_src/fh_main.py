"""
fh_main.py  --  single entry point for the 2D Fermi-Hubbard deliverable.

Runs the whole pipeline from a clean environment and reproduces every figure and
headline number:

  0. self-checks        (JW consistency, free-fermion cross-check,
                         sector engine vs full-space engine)
  1. ED ground state    -> fig1_groundstate_vs_U.png      (2x2 and 3x4, U=0,1,4,8)
  2. quench dynamics    -> fig2_quench_dynamics.png       (ED vs Trotter vs
                         noiseless shots vs raw noisy shots vs ZNE-mitigated)
  3. per-site heatmap   -> fig3_density_heatmaps.png      (see fh_config.HEATMAP_*)
  4. H2 emulator run    -> fig8_h2_run.png                (free local sampler by default)
  5. VQE (optional)     -> fig6_vqe.png
  6. summary tables     -> figures_fh/summary_tables.txt  (one per U)

Removed relative to earlier versions: the Trotter dt-convergence stage and its
figure, the N-scaling / qubit-budget figure, the ED-vs-emulator bar chart, the
VQE bar panel, all first-order-Trotter curves, and the circuit/angle-convention
checks.

Usage:
    python fh_main.py               # everything
    python fh_main.py --quick       # skip VQE and the summary tables
    python fh_main.py --no-vqe      # everything except VQE and the tables
    python fh_main.py --no-shots    # skip emulator-shot trajectories
    python fh_main.py --no-noise    # skip the noisy + ZNE curves in fig2
    python fh_main.py --no-heatmap  # skip the per-site heatmap stage
                                    # (or set fh_config.RUN_HEATMAP = False)
"""
from __future__ import annotations

import argparse
import sys


def fh_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip VQE and the summary tables")
    ap.add_argument("--no-vqe", action="store_true", help="skip VQE and the summary tables")
    ap.add_argument("--no-shots", action="store_true", help="skip emulator-shot trajectories")
    ap.add_argument("--no-noise", action="store_true",
                    help="skip the noisy + ZNE curves in the quench figure")
    ap.add_argument("--no-heatmap", action="store_true", help="skip the per-site heatmap stage")
    args = ap.parse_args()

    import fh_config as cfg

    print("=" * 70)
    print("2D FERMI-HUBBARD  --  Quantathon CR 2026 Challenge 3 (optional model)")
    print("periodic boundaries, half filling, second-order Trotter")
    print("=" * 70)

    print("\n[0/6] Self-checks")
    import run_fh_selfcheck
    run_fh_selfcheck.check_consistency()
    run_fh_selfcheck.check_free_fermions()
    run_fh_selfcheck.check_sector_vs_full_space()
    run_fh_selfcheck.check_sector_sizes()

    print("\n[1/6] ED ground state vs U/t  (2x2 and 3x4)")
    import run_fh_ed
    run_fh_ed.run()

    print("\n[2/6] Quench dynamics (ED vs Trotter vs noiseless vs noisy vs ZNE)")
    import run_fh_dynamics
    run_fh_dynamics.run(with_shots=not args.no_shots,
                        with_noise=(None if not args.no_noise else False))

    # Skipped by EITHER the command-line flag or cfg.RUN_HEATMAP = False.
    if getattr(cfg, "RUN_HEATMAP", True) and not args.no_heatmap:
        print("\n[3/6] Per-site heatmap")
        import run_fh_heatmap
        run_fh_heatmap.run()
    else:
        print("\n[3/6] Per-site heatmap skipped")

    print("\n[4/6] H2 emulator run")
    import run_fh_h2
    run_fh_h2.run()

    if not (args.quick or args.no_vqe):
        print("\n[5/6] VQE (optional extension)")
        import run_fh_vqe
        run_fh_vqe.run()

        print("\n[6/6] Summary tables (one per U)")
        import fh_tables
        fh_tables.run()
    else:
        print("\n[5/6] VQE skipped")
        print("[6/6] Summary tables skipped (they need the VQE row)")

    print("\nDone. Figures are in ./figures_fh/, cached data in ./data_fh/.")


