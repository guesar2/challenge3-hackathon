#!/usr/bin/env python3
"""
Quantathon CR 2026 · Challenge 3
Single entry point that reproduces every reported figure and number.

    python main.py

Runs, in sequence (each is also independently runnable, see src/run_*.py):
    1.  run_ed.py                 -- exact diagonalization (ED) baseline
    2.  run_adiabatic.py          -- adiabatic Trotter sweep (local statevector)
    3.  run_quench.py             -- fixed-Hamiltonian quench (ED vs. local Trotter)
    4.  run_dt_convergence.py     -- Trotter dt-convergence check (O(dt^2) scaling)
    5.  run_n_scaling.py          -- Trotter-vs-ED breakdown scan, N=4..20
    6.  run_quantum_advantage.py  -- classical ED cost vs. Trotter circuit cost
                                     (purely classical, no qnexus, no quota)
    7.  run_h2_emulator.py        -- Quantinuum H2 emulator run (qnexus/pytket) --
                                     see config.RUN_ON_H2_EMULATOR (currently True
                                     by default: this step costs a metered qnexus
                                     usage quota every run unless set to False)
    8.  run_zne.py                -- Zero-Noise Extrapolation against the real
                                     noisy H2-Emulator (qermit Folding.circuit) --
                                     gated by the same config.RUN_ON_H2_EMULATOR;
                                     additional qnexus quota cost per run
    9.  run_iceberg_qec.py        -- Iceberg [[k+2,k,2]] QEC pilot run against
                                     the real noisy H2-Emulator -- separately
                                     gated by config.ICEBERG_RUN_ON_H2_EMULATOR
                                     (False by default: no-ops unless enabled)
    10. run_noise_scaling.py      -- noise-vs-N and noise-vs-depth characterization
                                     against the real H2-Emulator -- gated by
                                     config.RUN_ON_H2_EMULATOR; the largest qnexus
                                     quota cost in this pipeline (N up to 26, two
                                     H2 submissions per N)
    11. plot_iceberg_comparison.py -- regenerates the Iceberg-vs-ED/ZNE comparison
                                     figure from step 9's saved data (no quota)
    12. fh2d/fh_main.py           -- optional 2D Fermi-Hubbard extension, its own
                                     self-contained package (same qnexus-quota
                                     caveat -- see fh2d/fh_config.py)

To check a single section without running the rest, e.g.:
    cd src && python run_ed.py
    cd fh2d && python fh_main.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
# src/ and fh2d/ both import their own modules by bare name (e.g.
# "zne_fit" exists in both); put fh2d/ first so its modules resolve
# from there first if a name is ever imported from both packages.
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "fh2d"))

import ftim_main
import fh_main

if __name__ == "__main__":
    ftim_main.main()
    fh_main.main(argv=[])
