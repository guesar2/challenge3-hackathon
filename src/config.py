"""
config.py

Centralized simulation parameters.

Note on naming: `N_SCALING_VALUES` is the 4..20 spin range used by
run_n_scaling.py's Trotter-vs-ED scan. The ED baseline's own observable-
scaling plot uses a separate name, `ED_EXTRA_N_VALUES` (config.N plus a
few extra sizes), so both scripts have a single, unambiguous source of
truth for which N's they check.
"""

N = 6                       # number of spins in the periodic chain
J = 1.0                     # ZZ coupling strength
H_VALUES = (0.5, 1.0, 2.0)  # target transverse fields to probe (critical point at h/J=1)

# Extra system sizes checked by the ED baseline (run_ed.py), alongside N
# above -- both for the printed comparison tables and for the observable-
# scaling plot (plot_ed_scaling). Doesn't affect any other stage --
# adiabatic/quench/dt_convergence/n_scaling still run at N only.
ED_EXTRA_N_VALUES = (8, 10)  # add more sizes here to extend the scaling comparison

# System sizes for the ED wall-clock *runtime* scaling benchmark
# (run_ed.py) -- separate from ED_EXTRA_N_VALUES above since this one
# costs real wall time (it actually runs ed_baseline once per N to time
# it) rather than being "free" extra table rows. Kept modest since
# pauli_ops.py builds every operator sparse in this merged codebase (see
# that module's docstring) -- this benchmark will therefore run faster,
# and the classical wall will sit further out, than it did against the
# original dense-Pauli-operator implementation; raise these if your
# machine has time/RAM to spare and you want to push the wall out further.
# Measured on this machine: N=6/8/10/12 took 0.03s/0.12s/2.0s/104s
# respectively -- note the N=10->12 jump (52x, not the ~4x that 2^N alone
# would predict) comes largely from pauli_ops.py building each Pauli term
# as a DENSE 2^N x 2^N matrix (np.kron) before sparsifying, and
# build_collective_observables' Mz/Mx/Mzz operators staying dense
# throughout -- i.e. this specific implementation's classical wall is hit
# earlier than the Hilbert space's fundamental 2^N scaling alone would
# force; a sparse-native rewrite would push it out further, but the
# exponential trend itself doesn't go away, which is the actual point of
# the comparison. N capped at 12 here -- N=14's dense 2^14 x 2^14
# complex128 intermediate alone is ~4.3GB and, extrapolating the measured
# trend, would take on the order of an hour, not worth actually running.
N_RUNTIME_SCALING_VALUES = (6, 8, 10, 12)
N_RUNTIME_SCALING_EXTRAPOLATE_TO = 20  # plot_ed_runtime_scaling fits the measured points and
                             # projects (dashed, clearly marked "not run") out to this N, to
                             # make the "where classical still wins" wall visually concrete.

H_INIT = 4.0                # starting transverse field for adiabatic sweeps (deep paramagnetic)

# Adiabatic sweep
ADIABATIC_DT = 0.02
ADIABATIC_RATE_REF = 0.022   # target |dh/dt|, sets sweep duration per h_target
ADIABATIC_HOLD_STEPS = 0    # extra steps at fixed (h_target, J) after the ramp,
                             # to check the state has actually settled (see plot_adiabatic_convergence)

# Fixed-Hamiltonian time evolution (quench dynamics from a product state)
QUENCH_DT = 0.05
QUENCH_STEPS = 400          # total evolution time = QUENCH_DT * QUENCH_STEPS
QUENCH_INITIAL_STATE = None  # None -> defaults to |00...0>

# Where to save figures. Set to None to skip saving (only meaningful if
# your backend is interactive and plt.show() actually opens a window).
PLOT_SAVE_DIR = "figures"

# System-size ("does it break down for many spins?") scan -- the 4..20
# spin Trotter-vs-ED breakdown study (run_n_scaling.py). N_SCALING_VALUES
# sets the range probed; the dense ED time-evolution reference
# (exact_diagonalization.ed_time_evolution_exact, which builds a full
# 2^N x 2^N dense matrix) is only requested up to N_SCALING_ED_MAX, since
# it becomes memory/time-prohibitive well before N=20 -- that cutoff is
# itself part of what the scan is meant to show.
N_SCALING_VALUES = tuple(range(4, 21, 2))    # 4, 6, 8, ..., 20
N_SCALING_ED_MAX = 12                        # dense ED reference only up to this N as a
                                              # starting point -- run_n_scaling.py also
                                              # catches MemoryError per-N and skips ED past
                                              # whatever this machine can actually hold,
                                              # since the real ceiling is RAM-dependent
                                              # (dense eigh on a 2^14 x 2^14 matrix already
                                              # raised MemoryError on a 4GB test machine)
N_SCALING_NOISY_MAX = 8   # max N for the noisy H2 emulation in run_n_scaling.py;
                           # None uses all of N_SCALING_VALUES
N_SCALING_H = 1.0                            # h/J probed at each N (critical point)
N_SCALING_DT = 0.05
N_SCALING_STEPS = 20                       # tuned for this scan; adjust freely

# Noisy-simulation section of run_n_scaling.py's table -- currently a
# placeholder (run_noisy_stub()) until a QEC-encoded circuit exists. These
# just set what the eventual real run would use. Note this project *does*
# now have a real noisy H2 device reachable via qnexus (H2_DEVICE_NAME_NOISY
# below, "H2-Emulator") -- the earlier placeholder referenced an invalid
# device name ("H2-1E"), fixed here to reuse the same real device name.
# What's still missing to make run_noisy_stub real is a QEC-encoded
# circuit (see its docstring in run_n_scaling.py) -- noisy-device access
# by itself doesn't answer the "does encoded error correction scale"
# question that stub is a placeholder for.
N_SCALING_NOISY_DEVICE = "H2-Emulator"
N_SCALING_NOISY_SHOTS = 200 # tuned for this scan; adjust freely

# Iceberg [[k+2,k,2]] error-detection code (iceberg_code.py,
# iceberg_circuits.py, iceberg_tfim_circuit.py) -- fills in the QEC-encoded
# circuit run_n_scaling.py's run_noisy_stub() docstring says is missing.
# k plays N's role (must be even, like every other N/H2_* value above).
# OFF by default and gated SEPARATELY from RUN_ON_H2_EMULATOR above,
# since it costs its own qnexus quota when enabled.
ICEBERG_RUN_ON_H2_EMULATOR = False
ICEBERG_K = 4                       # even; small to start -- see docs/ICEBERG_QEC_PLAN.md
ICEBERG_DEVICE_NAME = "H2-Emulator"  # noisy -- the whole point is to see it catch real
                                     # noise; H2-1LE would trivially show a 0% discard rate
ICEBERG_H = 1.0                      # h/J probed (critical point, matches N_SCALING_H)
ICEBERG_DT = 0.05
ICEBERG_STEPS = 5                   # deliberately shallow to start -- each step compiles to
                                     # 2 syndrome-measurement rounds (see iceberg_tfim_circuit.py),
                                     # so circuit depth/cost grows faster with steps here than
                                     # in the unencoded pipeline; raise only after a successful
                                     # small pilot run
ICEBERG_SHOTS = 200
ICEBERG_EARLY_EXIT = True           # native condition= gating skips remaining two-qubit gates
                                     # once a shot is already flagged for discard; False -> plain
                                     # post-selection-only circuit (useful as an A/B comparison)

# Quantinuum H2 emulator run (pytket circuit submitted via qnexus).
# Requires a live qnexus login and costs against a metered usage quota.
RUN_ON_H2_EMULATOR = True
H2_DEVICE_NAME = "H2-1LE"    # H2 noiseless-local emulator (cheapest H2-family target)
H2_DEVICE_NAME_NOISY = "H2-Emulator"  # H2's real noisy emulator counterpart to
                             # H2_DEVICE_NAME -- carries Quantinuum's published noise_specs
                             # (gate/SPAM/crosstalk/dephasing error rates), unlike H2-1LE's
                             # exact noiseless state-vector emulation (shot noise only -- see
                             # qnexus_backend.py's module docstring). 
H2_PROJECT_NAME = "ftim-hackathon"
H2_N = 4                     # small chain -- keep circuit width/cost modest
H2_H_VALUES = (0.5, 1.0, 2.0)         # single point at criticality by default
H2_STEPS = 5                 # deliberately shallow (few Trotter steps -> lower cost)
H2_DT = 0.1
H2_SHOTS = 200

# Default client-side wait (seconds) for a qnexus submission before
# qnx.jobs.wait_for()/qnx.execute() raise TimeoutError -- the job itself
# keeps running server-side; this only bounds how long the *client* blocks.
# qnexus_backend.py's submit_quench_batch/submit_zne_batch/collect_quench_batch
# all default to this value (still overridable per-call via their own
# `timeout` kwarg, e.g. run_h2_emulator.run(timeout=...) / run_zne.run(timeout=...)).
# run_noise_scaling.py's NOISE_SCALING_TIMEOUT/DEPTH_SCALING_TIMEOUT keep their
# own independently-measured 1800s override for its N/step sweeps regardless.
#
# Bumped from 300s: that was too short for real H2-Emulator queue/run times
# once submissions stack up across the full pipeline -- a full `python
# main.py` run against the real emulator (RUN_ON_H2_EMULATOR = True) is
# expected to take several hours end-to-end, not minutes.
QNEXUS_TIMEOUT = 3600.0

# Above this N, run_h2_emulator.run() skips the dense-ED reference curve --
# same 2^N x 2^N dense-diagonalization wall as N_SCALING_ED_MAX above, just
# named separately since it gates a different script. Needed for an N-sweep
# that goes past the classical wall (e.g. a hardware-noise-vs-N scan up to
# the device's real qubit count) -- run() still returns z_h2/x_h2/mzz_h2
# either way, so a noisy-vs-noiseless comparison (same circuit on both
# devices, no ED needed) still works past this cutoff; only the vs-ED
# Trotter-error check and the h2_vs_ed_time plot are skipped.
H2_ED_MAX_N = 12

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
                             # H2-1LE via qnexus and costs quota). The HVA ansatz
                             # measures ~36-46 evaluations to self-converge, so this
                             # cap is below what it needs; a real qnexus run at this
                             # setting hits the iteration cap before converging.
                             # Raise it (at the cost of more qnexus quota) for a
                             # fully-converged hardware VQE run.
H2_VQE_TOL = 1e-2            # matches Quantinuum's reference tol
H2_VQE_SEED = 10             # matches the reference snippet's random.seed(a=10)

# Local-only VQE budget (local_emulator_backend, no quota cost). Generous on
# purpose -- COBYLA self-terminates once converged regardless of this cap
# (measured ~36-46 evals for HVA, ~500 for the HEA fallback), so a high
# ceiling costs nothing but a little wasted time if never reached.
H2_VQE_SHOTS_LOCAL = 4000

# H2 Zero-Noise Extrapolation (run_zne.py), via qermit's Folding.circuit
# (qnexus_backend.submit_zne_batch) -- amplifies noise by folding the
# circuit (C -> C C^-1 C ...) rather than the device's own error-rate
# scale knob (config's noise_scale kwarg elsewhere), then extrapolates
# back to the zero-noise limit (zne_fit.zne_extrapolate). Targets
# run_noise_scaling.SHORT_TIME_N/SHORT_TIME_STEPS (N=8, steps=5, T=0.5) by
# default -- see run_zne.run()'s docstring.
H2_ZNE_FOLD_FACTORS = (1, 3, 5)   # ODD integers only -- Folding.circuit raises
                             # otherwise. fold_factor=1 performs zero fold
                             # iterations (verified) -- i.e. it IS the plain
                             # raw-noisy circuit, not a separate submission.
H2_ZNE_SHOTS = 2000          # shots PER fold factor -- same shot-noise-vs-
                             # signal-size rationale as run_noise_scaling.
                             # DEPTH_SCALING_SHOTS (total shots per h is
                             # H2_ZNE_SHOTS * len(H2_ZNE_FOLD_FACTORS))
H2_ZNE_FIT_DEG = 1           # zero-noise extrapolation polynomial degree
                             # (1 = linear fit, ZNE's most common choice;
                             # must be < len(H2_ZNE_FOLD_FACTORS))
H2_VQE_MAX_ITERS_LOCAL = 500