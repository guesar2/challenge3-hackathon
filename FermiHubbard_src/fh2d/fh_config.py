"""
fh_config.py

Centralised parameters for the 2D Fermi-Hubbard pipeline.

BOUNDARY CONDITIONS
  Periodic in both directions, always. There is no open-boundary switch; see
  fh_lattice.py for the one caveat (an L=2 direction has no distinct wrap bond,
  so a 2x2 "periodic" cluster is the same 4-site ring as the open one).

SIZE / QUBIT BUDGET (read this before changing any lattice)
  Jordan-Wigner uses 2 qubits per site, so Lx*Ly needs 2*Lx*Ly qubits:
      2x2 -> 8q,  2x3 -> 12q,  2x4 -> 16q,  3x3 -> 18q,  3x4 / 2x6 -> 24q,
      4x4 -> 32q  (EXCEEDS Quantinuum's 26-qubit H2 exact emulator).
  3x4 = 24 qubits is the LARGEST lattice that still fits H2.

  Classically, 24 qubits is far past what a full-Hilbert-space solver can do:
  dim 2^24 = 16.7M, and the sparse Hamiltonian alone would need ~32 GB. All 3x4
  results therefore come from fh_sector.py, which works inside the half-filling
  Sz=0 sector (dimension 853 776) where the same physics costs ~14 MB. The
  8-qubit 2x2 lattice is small enough for BOTH engines, and they are checked
  against each other to machine precision in run_fh_selfcheck.

  Consequence: the quantum-circuit path (pytket / Trotter statevector / shots)
  still runs on 2x2 only. A 3x4 Trotter circuit is perfectly well defined and
  fits H2, but simulating it classically on the full 2^24 space is not possible
  here -- that asymmetry is exactly the honest scaling statement the rubric asks
  for, so it is reported rather than hidden.
"""

# ---- primary benchmark lattice (quantum-circuit path: ED vs Trotter vs shots)
LX = 2
LY = 2

# ---- Hamiltonian ----
T_HOP = 1.0                        # hopping amplitude t (energy unit)
U_VALUES = (0.0, 1.0, 4.0, 8.0)    # non-interacting -> weak -> intermediate ->
                                   # strong. U/t=1 and U/t=8 are the two regimes
                                   # named in the challenge; U=0 anchors the
                                   # free-fermion limit and U=4 sits near the
                                   # half-filled crossover.

# ---- ground-state scan (fig1) ----
GS_LATTICES = ((2, 2), (3, 4))     # 8 qubits and 24 qubits: the two problem
                                   # sizes used for the scaling comparison.

# ---- quench (fixed-Hamiltonian) time dynamics, 2x2 ----
QUENCH_U = 8.0              # which interaction regime to quench in
QUENCH_DT = 0.05            # Trotter / sampling step (units of 1/t). 0.05 keeps
                            # second-order Trotter < 5% of ED on every
                            # observable (dt=0.1 already exceeds 5% on <D> --
                            # see the dt-convergence study, which is exactly the
                            # "halve dt and confirm convergence" analysis the
                            # rubric asks for).
QUENCH_STEPS = 20           # total evolved time = QUENCH_DT * QUENCH_STEPS = 1.0
TROTTER_ORDER = 2           # second-order (symmetric Strang) throughout.
                            # First-order is no longer run anywhere.
# Initial state: "neel"   -> half-filling staggered product state, UNIFORM
#                            density: the right probe for m_stag.
#                "stripe" -> half-filling charge-imbalanced product state:
#                            the right probe for the per-site density map.
# Anything else is treated as an explicit occupation bitstring of length
# 2*n_sites.
QUENCH_INITIAL_STATE = "neel"

# ---- per-site density heatmap (fig3) ----
# The Neel state has uniform density and stays uniform, which is why the old
# heatmap was a flat field of 1.00. The heatmap therefore runs its own quench
# from the charge-imbalanced "stripe" state, on the big lattice.
HEATMAP_LATTICE = (3, 4)
HEATMAP_U = 4.0             # U/t=4: the stripe visibly MELTS toward uniform
                            # density within t ~ 2/t. For contrast: U/t=0 gives
                            # coherent ballistic oscillation with a near-perfect
                            # revival (the stripe reappears), and U/t=8 relaxes
                            # more slowly because doublons are energetically
                            # blocked. Change this one number to see any of them.
HEATMAP_INITIAL_STATE = "stripe"
HEATMAP_DT = 0.2
HEATMAP_STEPS = 10          # total evolved time = 2.0/t
HEATMAP_SNAPSHOTS = 3       # number of time slices drawn

# ---- Trotter dt-convergence study (second order only) ----
DT_CONVERGENCE_VALUES = (0.4, 0.2, 0.1, 0.05)  # halving dt, cf. common pitfalls
DT_CONVERGENCE_TOTAL_TIME = 1.0                # fixed physical time to compare at

# ---- summary tables (one per U in U_VALUES) ----
TABLE_LATTICE = (2, 2)      # must be a lattice the FULL pipeline runs on, i.e.
                            # one where ED, statevector Trotter and VQE are all
                            # available. 3x4 has ED only (see the size note).
TABLE_VQE_LAYERS = 3        # HVA depth p used for the table's VQE row
TABLE_VQE_RESTARTS = 4

# ---- H2 emulator run (Quantinuum, via qnexus or local pytket-quantinuum) ----
RUN_ON_H2_EMULATOR = True   # False -> free local statevector sampler
H2_DEVICE_NAME = "H2-1LE"    # noiseless state-vector emulator (shot noise only)
H2_DEVICE_NAME_NOISY = "H2-Emulator"  # real published noise model (qnexus only)
H2_PROJECT_NAME = "fermi-hubbard-hackathon"
H2_LATTICE = (2, 2)          # 8 qubits -- smallest, cheapest circuit
H2_U = 8.0
H2_DT = 0.1
H2_STEPS = 4                 # shallow: few Trotter steps -> lower cost / noise
H2_TROTTER_ORDER = 2
H2_SHOTS = 500
H2_INITIAL_STATE = "neel"

# ---- VQE ground-state search (optional extension) ----
VQE_LATTICE = (2, 2)
VQE_U = 8.0
VQE_LAYERS = 2               # Hamiltonian-variational-ansatz depth p
VQE_MAXITER_LOCAL = 1500     # COBYLA budget on the free statevector backend
VQE_MAXITER_H2 = 20          # small budget when it actually costs quota
VQE_SHOTS = 2000
VQE_SEED = 7

# ---- output ----
PLOT_SAVE_DIR = "figures_fh"
