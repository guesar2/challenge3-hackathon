#!/usr/bin/env python3
"""
Quantathon CR 2026 · Challenge 3
Single entry point that reproduces every reported figure and number.

    python main.py

Runs, in sequence (each is also independently runnable, see src/run_*.py):
    1. run_ed.py          -- exact diagonalization (ED) baseline
    2. run_adiabatic.py   -- adiabatic Trotter sweep (local statevector)
    3. run_quench.py      -- fixed-Hamiltonian quench (ED vs. local Trotter)
    4. run_h2_emulator.py -- Quantinuum H2 emulator run (qnexus/pytket;
                              off by default, see config.RUN_ON_H2_EMULATOR
                              -- costs against a metered usage quota)

To check a single section without running the rest, e.g.:
    cd src && python run_ed.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from ftim_main import main

if __name__ == "__main__":
    main()
