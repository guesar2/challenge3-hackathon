"""
config.py

Centralized simulation parameters. 
"""

N = 6                       # number of spins in the periodic chain
J = 1.0                     # ZZ coupling strength
H_VALUES = (0.5, 1.0, 2.0)  # target transverse fields to probe (critical point at h/J=1)

H_INIT = 4.0                # starting transverse field for adiabatic sweeps (deep paramagnetic)

# Adiabatic sweep
ADIABATIC_DT = 0.02
ADIABATIC_RATE_REF = 0.022   # target |dh/dt|, sets sweep duration per h_target
ADIABATIC_HOLD_STEPS = 0   # extra steps at fixed (h_target, J) after the ramp,
                             # to check the state has actually settled (see plot_adiabatic_convergence)

# Fixed-Hamiltonian time evolution (quench dynamics from a product state)
QUENCH_DT = 0.05
QUENCH_STEPS = 400          # total evolution time = QUENCH_DT * QUENCH_STEPS
QUENCH_INITIAL_STATE = None  # None -> defaults to |00...0>

# Where to save figures. Set to None to skip saving (only meaningful if
# your backend is interactive and plt.show() actually opens a window).
PLOT_SAVE_DIR = "figures"

# Quantinuum H2 emulator run (pytket circuit submitted via qnexus).
# OFF by default: requires a live qnexus login and costs against a metered
# usage quota. Flip to True only with explicit approval to spend quota.
RUN_ON_H2_EMULATOR = True
H2_DEVICE_NAME = "H2-1LE"    # H2 noiseless-leakage emulator (cheapest H2-family target)
H2_PROJECT_NAME = "ftim-hackathon"
H2_N = 4                     # small chain -- keep circuit width/cost modest
H2_H_VALUES = (0.5, 1.0, 2.0)         # single point at criticality by default
H2_STEPS = 5                 # deliberately shallow (few Trotter steps -> lower cost)
H2_DT = 0.1
H2_SHOTS = 200