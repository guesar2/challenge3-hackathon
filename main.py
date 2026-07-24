#!/usr/bin/env python3
"""
Quantathon CR 2026 · Challenge 3
Single entry point that reproduces every reported figure and number.

    python main.py

Runs, in sequence (each is also independently runnable, see src/run_*.py):
    1. run_ed.py             -- exact diagonalization (ED) baseline
    2. run_adiabatic.py      -- adiabatic Trotter sweep (local statevector)
    3. run_quench.py         -- fixed-Hamiltonian quench (ED vs. local Trotter)
    4. run_dt_convergence.py -- Trotter dt-convergence check (O(dt^2) scaling)
    5. run_n_scaling.py      -- Trotter-vs-ED breakdown scan, N=4..20
    6. run_h2_emulator.py    -- Quantinuum H2 emulator run (qnexus/pytket) --
                                 see config.RUN_ON_H2_EMULATOR (currently True
                                 by default: this step costs a metered qnexus
                                 usage quota every run unless set to False)
    7. fh2d/fh2d/fh_main.py  -- optional 2D Fermi-Hubbard extension, its own
                                 self-contained package (same qnexus-quota
                                 caveat -- see fh2d/fh2d/fh_config.py)

Additional standalone analyses not run by default (see each script's
docstring): run_zne.py (zero-noise extrapolation), run_iceberg_qec.py +
plot_iceberg_comparison.py (Iceberg QEC pilot), run_quantum_advantage.py
(classical-vs-quantum cost comparison), run_noise_scaling.py (noise-model
characterization).

To check a single section without running the rest, e.g.:
    cd src && python run_ed.py
    cd fh2d/fh2d && python fh_main.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
# src/ and fh2d/fh2d/ both import their own modules by bare name (e.g.
# "zne_fit" exists in both); put fh2d/fh2d/ first so its modules resolve
# from there first if a name is ever imported from both packages.
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "fh2d", "fh2d"))

import ftim_main
import fh_main

if __name__ == "__main__":
    ftim_main.main()
    fh_main.main(argv=[])
