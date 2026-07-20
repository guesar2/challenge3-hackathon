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
# the quench pipeline above. Ramp length is scaled per h_target via
# sweep_schedule.steps_for_target (same |dh/dt| ~= rate_ref logic as the
# local ADIABATIC_RATE_REF), capped at H2_ADIABATIC_MAX_STEPS -- a flat
# step count for every target was tried first and over-Trotterized targets
# close to H_INIT (h/J=2.0's <Zi Zi+1> got *worse* going from 15 to 100
# steps, 9.5% -> 12%, and a 2.5x shot increase didn't fix it, ruling out
# shot noise -- consistent with excess Trotter error from evolving far
# longer than that target's smaller |Delta h| actually needs).
H2_ADIABATIC_N = 6           # matches the N required for the quantum-vs-ED comparison
H2_ADIABATIC_DT = 0.1
H2_ADIABATIC_RATE_REF = 0.35  # target |dh/dt| -- anchored so the largest |Delta h| in
                              # H2_H_VALUES (h/J=0.5, Delta h=3.5) gets ~100 steps, the
                              # value that worked well (0.6-2.0% deviation) in testing
H2_ADIABATIC_MAX_STEPS = 100  # hard cap regardless of rate_ref -- keeps circuit depth/cost
                              # bounded even if H2_H_VALUES later adds a target far from H_INIT
H2_ADIABATIC_CRITICAL_DT_FACTOR = 0.5  # at h/J=1 (gap closes -> critical slowing down), use
                              # dt * this factor with steps scaled up by 1/factor to keep the
                              # same total ramp time -- finer Trotter resolution rather than a
                              # longer ramp, since pinning h/J=1 to H2_ADIABATIC_MAX_STEPS alone
                              # left a confirmed (~4.7 sigma) systematic bias in <Zi Zi+1> that
                              # more shots couldn't shrink, pointing at Trotter-step-size error
                              # rather than an insufficiently adiabatic ramp
H2_ADIABATIC_SHOTS = 2000    # bumped from 500 -- bootstrap SE at 500 shots was ~0.02 on
                             # observables of magnitude ~0.6-0.9, comparable in size to the
                             # 5% deviation target itself, making individual runs bounce
                             # between "pass" and "fail" on noise alone (e.g. h/J=2.0's
                             # <Zi Zi+1> read 4% one run, 12% the next, same config) --
                             # SE ~ 1/sqrt(shots), so 4x shots ~ 2x smaller error bars

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