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
H2_ADIABATIC_CRITICAL_TIME_FACTOR = 2  # at h/J=1 (gap closes -> critical slowing down), use
                              # steps = H2_ADIABATIC_MAX_STEPS * this factor at the *same* dt
                              # -- i.e. a longer total ramp time, not finer resolution. Pinning
                              # h/J=1 to H2_ADIABATIC_MAX_STEPS alone left a confirmed (~14-17
                              # sigma, 20000 shots) systematic bias of ~6.5% in <Zi Zi+1>; a
                              # local_emulator_backend test (free, 5000 shots) that separately
                              # varied resolution (dt) and total ramp time (T = steps*dt) at
                              # h/J=1 showed doubling dt resolution at fixed T barely moved the
                              # bias, but doubling T at fixed (coarse) dt alone dropped it from
                              # ~6.5% to ~2.15% -- confirming it's ramp *time*, not Trotter
                              # step-size, that was insufficient (textbook critical slowing
                              # down: the adiabatic theorem needs T -> large as the gap closes,
                              # independent of how finely that time is resolved)
H2_ADIABATIC_TRANSIT_TIME_FACTOR = 4  # for a target whose ramp *passes through* h/J=1 without
                              # landing there (e.g. h/J=0.5 with H_INIT=4.0 sweeps down through
                              # 1.0 on the way to 0.5) -- steps = H2_ADIABATIC_MAX_STEPS * this
                              # factor, separate from H2_ADIABATIC_CRITICAL_TIME_FACTOR since a
                              # mid-ramp transit needed more total time than landing exactly on
                              # the critical point did. A local_emulator_backend sweep at
                              # h/J=0.5 (free, 2000 shots) found <X> deviation of 9.00% at
                              # CRITICAL_TIME_FACTOR's 200 steps, 2.34% at 300, 0.43% at 400,
                              # and 3.22% at 500 (500's rise is shot noise, not a systematic
                              # trend -- 2000 shots gives an <X> standard error comparable to
                              # these percentages at this magnitude) -- 400 was the clear best
                              # and is comfortably under the 5% target with margin for shot noise.
H2_ADIABATIC_SHOTS = 2000    # bumped from 500 -- bootstrap SE at 500 shots was ~0.02 on
                             # observables of magnitude ~0.6-0.9, comparable in size to the
                             # 5% deviation target itself, making individual runs bounce
                             # between "pass" and "fail" on noise alone (e.g. h/J=2.0's
                             # <Zi Zi+1> read 4% one run, 12% the next, same config) --
                             # SE ~ 1/sqrt(shots), so 4x shots ~ 2x smaller error bars

# H2 VQE ground-state search (COBYLA over a variational ansatz, following
# Quantinuum's own batched-variational-experiment pattern). Independent of
# the adiabatic/quench pipelines above -- finds the ground state directly
# per h target instead of following a time-dependent path, so it isn't
# subject to the adiabatic ramp's diabatic-transition problem near h/J=1.
H2_VQE_N = 6                 # reuses the N used for H2_ADIABATIC_N

# ansatz="hva" (Hamiltonian Variational Ansatz, build_hva_ansatz_circuit):
# p layers of (ZZPhase, Rx) mirroring the TFIM's own term structure -- only
# 2*p parameters vs. the alternative "hea" (hardware-efficient ansatz,
# build_hea_ansatz_circuit)'s problem-agnostic 6*N. Noiseless-statevector
# test (scipy COBYLA, no shot noise) hit the *exact* ED ground energy at
# h/J=0.5/1.0/2.0 with p=4 (8 params), where the 36-param HEA only reached
# ~2-3% of ED even given ~500 evaluations -- and against the real (noisy,
# shot-sampled) local emulator, HVA's COBYLA self-terminates
# (trust-region radius hits its lower bound) in ~36-46 evaluations, vs.
# needing hundreds for the HEA to get anywhere close. Set ansatz="hea" and
# see vqe.run_vqe_h2's docstring to switch back.
H2_VQE_ANSATZ = "hva"
H2_VQE_P = 4                 # HVA layers -- ignored when H2_VQE_ANSATZ="hea"

H2_VQE_SHOTS = 200
H2_VQE_MAX_ITERS = 15        # COBYLA iterations -- ~15 real batch round-trips per h/J
                             # (kept small: this is what actually submits to real
                             # H2-1LE via qnexus and costs quota). NOTE: HVA's
                             # measured ~36-46 evaluations to self-converge means
                             # 15 is *below* what it needs -- a real qnexus run at
                             # this setting will still hit the iteration cap before
                             # converging. Bump this deliberately (and accept the
                             # added quota cost) before an actual qnexus submission.
H2_VQE_TOL = 1e-2            # matches Quantinuum's reference tol
H2_VQE_SEED = 10             # matches the reference snippet's random.seed(a=10)

# Local-only VQE budget (local_emulator_backend, no quota cost). Generous on
# purpose -- COBYLA self-terminates once converged regardless of this cap
# (measured ~36-46 evals for HVA, ~500 for the HEA fallback), so a high
# ceiling costs nothing but a little wasted time if never reached.
H2_VQE_SHOTS_LOCAL = 4000
H2_VQE_MAX_ITERS_LOCAL = 500