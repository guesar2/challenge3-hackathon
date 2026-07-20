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

# H2 adiabatic sweep (phase-transition signal on hardware). Independent of
# the quench pipeline above -- uses a fixed, modest ramp length regardless
# of h_target (unlike sweep_schedule.steps_for_target's rate-based scaling,
# which would need thousands of steps and is infeasible for a single
# qnexus job).
H2_ADIABATIC_N = 6           # matches the N required for the quantum-vs-ED comparison
H2_ADIABATIC_STEPS = 15      # fixed ramp length for every h target
H2_ADIABATIC_DT = 0.1
H2_ADIABATIC_SHOTS = 200

# H2 VQE ground-state search (hardware-efficient ansatz + COBYLA, following
# Quantinuum's own batched-variational-experiment pattern). Independent of
# the adiabatic/quench pipelines above -- finds the ground state directly
# per h target instead of following a time-dependent path, so it isn't
# subject to the adiabatic ramp's diabatic-transition problem near h/J=1.
H2_VQE_N = 6                 # reuses the N used for H2_ADIABATIC_N
H2_VQE_SHOTS = 200
H2_VQE_MAX_ITERS = 15        # COBYLA iterations -- ~15 real batch round-trips per h/J
H2_VQE_TOL = 1e-2            # matches Quantinuum's reference tol
H2_VQE_SEED = 10             # matches the reference snippet's random.seed(a=10)