"""
shot_observables.py

Converts Z-basis and X-basis measurement bitstrings (from either qnexus_backend
or local_emulator_backend) into TFIM observables: <Z>_rms, <Zi Zi+1>, and <X>.
Kept separate from backend modules so neither needs to import the other's
dependencies just to reuse this postprocessing.

Includes bootstrap error estimation for shot-noise standard errors.
"""

import numpy as np


# ---------- internal helpers ----------

def _observables_from_shots(shots, N):
    """Compute <Z>_rms per site and <Zi Zi+1> per bond from an array of spin
    values (±1 for each qubit, shape (n_shots, N)).
    """
    mz = shots.sum(axis=1)                     # sum of Z eigenvalues per shot
    z_rms = np.sqrt(np.mean(mz ** 2)) / N
    mzz_per_shot = sum(shots[:, i] * shots[:, (i + 1) % N] for i in range(N))
    mzz = np.mean(mzz_per_shot) / N
    return z_rms, mzz


# ---------- Z-basis observables ----------

def bitstrings_to_observables(bitstrings, N):
    """Convert Z-basis measurement bitstrings ('0'/'1' strings) into
    (<Z>_rms per site, <Zi Zi+1> per bond). Uses RMS magnetization because
    <Z> vanishes by Z2 symmetry in the TFIM without a longitudinal field.
    """
    shots = np.array([[1 - 2 * int(bit) for bit in bitstr] for bitstr in bitstrings])
    return _observables_from_shots(shots, N)


def bootstrap_observable_errors(bitstrings, N, n_boot=1000, seed=0):
    """Bootstrap standard errors for (<Z>_rms, <Zi Zi+1>) from Z-basis shots.
    Both are nonlinear functions of the per-shot bits, so bootstrap is used.
    """
    rng = np.random.default_rng(seed)
    shots = np.array([[1 - 2 * int(bit) for bit in bitstr] for bitstr in bitstrings])
    n_shots = shots.shape[0]

    z_samples = np.empty(n_boot)
    mzz_samples = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_shots, size=n_shots)
        z_samples[b], mzz_samples[b] = _observables_from_shots(shots[idx], N)
    return z_samples.std(ddof=1), mzz_samples.std(ddof=1)


# ---------- X-basis observables ----------

def bitstrings_to_mx(bitstrings, N):
    """Convert X-basis measurement bitstrings into <X> per site (plain mean
    over shots and sites). Unlike <Z>, <X> does NOT vanish by symmetry;
    it is a signed mean, matching pauli_ops.expectation_values's x_exp.
    """
    shots = np.array([[1 - 2 * int(bit) for bit in bitstr] for bitstr in bitstrings])
    return shots.mean()


def bootstrap_mx_error(bitstrings, N, n_boot=1000, seed=0):
    """Bootstrap standard error for <X> (mean, not RMS) from X-basis shots.
    """
    rng = np.random.default_rng(seed)
    shots = np.array([[1 - 2 * int(bit) for bit in bitstr] for bitstr in bitstrings])
    n_shots = shots.shape[0]

    x_samples = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_shots, size=n_shots)
        x_samples[b] = shots[idx].mean()
    return x_samples.std(ddof=1)