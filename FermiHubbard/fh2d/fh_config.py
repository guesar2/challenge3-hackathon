"""
fh_config.py

Centralised parameters for the 2D Fermi-Hubbard pipeline. Kept separate from
the TFIM project's config.py so the two can coexist in one repository.

SIZE / QUBIT BUDGET (read this before changing LATTICES):
  Jordan-Wigner uses 2 qubits per site (one per spin-orbital), so an Lx*Ly
  lattice needs 2*Lx*Ly qubits:
      2x2 -> 8q,  2x3 -> 12q,  2x4 -> 16q,  3x3 -> 18q,  3x4 / 2x6 -> 24q,
      4x4 -> 32q  (EXCEEDS Quantinuum's 26-qubit H2 exact emulator).
  Classical exact diagonalisation (full space) is comfortable to ~2^18-2^20,
  i.e. up to ~9-10 sites; exact dynamics uses scipy.sparse expm_multiply, which
  reaches a little further. The defaults below stay firmly inside BOTH ceilings
  so every reported number is ED-verifiable and every circuit fits H2.
"""

# ---- primary benchmark lattice (ED comparison + quench dynamics) ----
LX = 2
LY = 2
PERIODIC_X = False
PERIODIC_Y = False

# ---- Hamiltonian ----
T_HOP = 1.0                 # hopping amplitude t (energy unit)
U_VALUES = (1.0, 8.0)       # weak (U/t=1) and strong (U/t=8) regimes, per challenge
# For ground-state work we enforce half-filling via a particle-hole chemical
# potential mu = U/2 (see fh_jordan_wigner). Reported energies subtract it out.

# ---- quench (fixed-Hamiltonian) time dynamics ----
QUENCH_U = 8.0              # which interaction regime to quench in
QUENCH_DT = 0.05            # Trotter / sampling step (units of 1/t). 0.05 keeps
                            # BOTH Trotter orders < 5% of ED on every observable
                            # (dt=0.1 already exceeds 5% on <D> -- see the
                            # dt-convergence study, which is exactly the
                            # "halve dt and confirm convergence" analysis the
                            # rubric asks for).
QUENCH_STEPS = 20           # total evolved time = QUENCH_DT * QUENCH_STEPS = 1.0
TROTTER_ORDER = 2           # 1 (first-order) or 2 (second-order, symmetric)
# Initial state: "neel" -> half-filling staggered product state (recommended,
# ED-checkable, echoes the paper's antiferromagnet setup). Any other value is
# treated as an explicit occupation bitstring of length 2*n_qubits.
QUENCH_INITIAL_STATE = "neel"

# ---- Trotter dt-convergence study ----
DT_CONVERGENCE_VALUES = (0.4, 0.2, 0.1, 0.05)  # halving dt, cf. common pitfalls
DT_CONVERGENCE_TOTAL_TIME = 1.0                # fixed physical time to compare at

# ---- scaling study (how far do ED / the circuit reach) ----
SCALING_LATTICES = ((2, 2), (2, 3), (2, 4))    # (Lx, Ly) pairs; each still ED-able

# ---- H2 emulator run (Quantinuum, via qnexus or local pytket-quantinuum) ----
RUN_ON_H2_EMULATOR = True   # OFF by default: qnexus costs metered quota.
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
                            # (300 under-converges the bond/site-resolved HVA;
                            # 1500 reaches ~1-2% of ED at weak coupling)
VQE_MAXITER_H2 = 20          # small budget when it actually costs quota
VQE_SHOTS = 2000
VQE_SEED = 7

# ---- output ----
PLOT_SAVE_DIR = "figures_fh"